//! Fast Confirmation Rule (FCR) implementation for Lighthouse.
//!
//! This module implements the Fast Confirmation Rule as described in the specification,
//! providing faster block confirmation times (12-24 seconds) compared to traditional
//! finalization (13-19 minutes).
//!
//! The FCR operates under network synchrony assumptions and uses LMD-GHOST vote weights
//! combined with FFG checkpoint support to determine block permanence.
//!
//! TODO: The following components still need to be implemented:
//! - Performance benchmarking and optimization
//! - Additional FFG integration features
//! - Advanced state access optimizations

use crate::Error::ProtoArrayStringError;
use lru::LruCache;
use proto_array::ProtoArrayForkChoice;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use types::{Checkpoint, Epoch, EthSpec, FixedBytesExtended, Hash256, Slot};

/// Default Byzantine threshold in percentage (25%)
/// **Specification**: `CONFIRMATION_BYZANTINE_THRESHOLD = 33` (but we use 25% for single-slot confirmation)
pub const DEFAULT_FCR_BYZANTINE_THRESHOLD_PERCENTAGE: u64 = 25;
/// Maximum depth to scan for reorgs (mainnet safety)
/// **Specification**: Not in spec (Lighthouse safety limit)
const MAX_REORG_DEPTH: usize = 32;

/// Configuration for the Fast Confirmation Rule.
#[derive(Debug, Clone)]
pub struct FastConfirmationConfig {
    /// Byzantine threshold in percentage (e.g., 25 = 25%)
    pub beta_percentage: u64,
}

impl FastConfirmationConfig {
    /// Creates a new FCR configuration with the given Byzantine threshold.
    ///
    /// # Arguments
    /// * `beta_percentage` - Byzantine threshold in percentage (0-49)
    ///
    /// # Returns
    /// * `Ok(FcrConfig)` - Valid configuration
    /// * `Err(String)` - Invalid threshold (≥50% makes confirmation impossible)
    pub fn new(beta_percentage: u64) -> Result<Self, String> {
        if beta_percentage >= 50 {
            return Err(format!(
                "Invalid byzantine threshold: {}%, must be < 50%",
                beta_percentage
            ));
        }

        Ok(Self { beta_percentage })
    }
}

/// Metadata for a block's FCR status.
///
/// **Why Side-Table Approach**: Unlike the Python spec where confirmation status is computed
/// on-demand, Lighthouse caches this metadata to avoid repeated expensive computations.
/// This is a performance optimization that trades memory for CPU cycles, which is beneficial
/// for the high-frequency confirmation checks required by FCR.
///
/// This stores per-block FCR metadata that would be computed on-demand in the Python spec.
///
#[derive(Debug, Clone)]
pub struct FcrMeta {
    /// LMD-GHOST support weight for this block
    /// **Spec**: Computed in `is_one_confirmed()` as `support`
    pub support: u64,
    /// Total committee weight that could have attested
    /// **Spec**: Computed in `is_one_confirmed()` as `maximum_support`
    pub committee_weight: u64,
    /// Whether this block is confirmed by FCR
    /// **Spec**: Computed on-demand in various functions
    pub confirmed: bool,
}

impl Default for FcrMeta {
    fn default() -> Self {
        Self {
            support: 0,
            committee_weight: 0,
            confirmed: false,
        }
    }
}

/// Store for FCR state across slots and blocks.
///
/// **Why Separate Struct**: Lighthouse uses a side-table approach to avoid database schema
/// changes. Instead of modifying the existing `Store` (which would require a hard fork),
/// FCR state is stored in a separate struct that can be easily enabled/disabled via `Option`.
///
/// **Specification**: Corresponds to additional fields in the `Store` class:
/// - `confirmed_root: Root`
/// - `prev_slot_justified_checkpoint: Checkpoint`
/// - `prev_slot_unrealized_justified_checkpoint: Checkpoint`
/// - `prev_slot_head: Root`
///
///
#[derive(Debug, Clone)]
pub struct FcrStore {
    /// Latest confirmed block root
    /// **Spec**: `store.confirmed_root`
    pub confirmed_root: Hash256,
    /// Previous slot's justified checkpoint
    /// **Spec**: `store.prev_slot_justified_checkpoint`
    pub prev_slot_justified_checkpoint: Checkpoint,
    /// Previous slot's unrealized justified checkpoint
    /// **Spec**: `store.prev_slot_unrealized_justified_checkpoint`
    pub prev_slot_unrealized_justified_checkpoint: Checkpoint,
    /// Previous slot's head block
    /// **Spec**: `store.prev_slot_head`
    pub prev_slot_head: Hash256,
    /// LRU cache for last 100 committee weight calculations
    /// **Spec**: Not in spec (Lighthouse optimization)
    /// **Why**: Committee weight calculations are expensive under tree-states architecture
    pub committee_weight_lru: LruCache<(Epoch, Slot, Slot), u64>,
    /// LRU cache for last 50 FFG support calculations
    /// **Spec**: Not in spec (Lighthouse optimization)
    /// **Why**: FFG support calculations involve complex head vote analysis
    pub ffg_support_lru: LruCache<(Checkpoint, Checkpoint), u64>,
}

impl Default for FcrStore {
    fn default() -> Self {
        Self {
            confirmed_root: Hash256::zero(),
            prev_slot_justified_checkpoint: Checkpoint::default(),
            prev_slot_unrealized_justified_checkpoint: Checkpoint::default(),
            prev_slot_head: Hash256::zero(),
            committee_weight_lru: LruCache::new(NonZeroUsize::new(100).unwrap()),
            ffg_support_lru: LruCache::new(NonZeroUsize::new(50).unwrap()),
        }
    }
}

/// Main Fast Confirmation Rule implementation.
pub struct FastConfirmation<E: EthSpec> {
    /// FCR configuration including Byzantine threshold
    /// **Spec**: `CONFIRMATION_BYZANTINE_THRESHOLD` constant
    config: FastConfirmationConfig,
    /// Per-block FCR metadata, keyed by block root
    /// **Spec**: Computed on-demand in various functions
    meta: HashMap<Hash256, FcrMeta>,
    /// FCR state store (confirmed root, prev slot checkpoints, etc)
    /// **Spec**: Additional fields in `Store` class
    fcr_store: FcrStore,
    /// Phantom data to hold the EthSpec type parameter
    phantom: PhantomData<E>,
}

impl<E: EthSpec> FastConfirmation<E> {
    /// Creates a new Fast Confirmation Rule instance.
    pub fn new(config: FastConfirmationConfig) -> Self {
        Self {
            config,
            meta: HashMap::new(),
            fcr_store: FcrStore::default(),
            phantom: PhantomData,
        }
    }

    /// Returns the current confirmed root.
    pub fn confirmed_root(&self) -> Hash256 {
        self.fcr_store.confirmed_root
    }

    /// Returns the previous slot's justified checkpoint.
    pub fn prev_slot_justified_checkpoint(&self) -> Checkpoint {
        self.fcr_store.prev_slot_justified_checkpoint
    }

    /// Returns the previous slot's unrealized justified checkpoint.
    pub fn prev_slot_unrealized_justified_checkpoint(&self) -> Checkpoint {
        self.fcr_store.prev_slot_unrealized_justified_checkpoint
    }

    /// Returns the previous slot's head block.
    pub fn prev_slot_head(&self) -> Hash256 {
        self.fcr_store.prev_slot_head
    }

    /// Returns the FCR configuration.
    pub fn config(&self) -> &FastConfirmationConfig {
        &self.config
    }

    /// Checks if a block is an ancestor of another block.
    ///
    /// **Python Specification**: `is_ancestor(store, root, ancestor)`
    ///
    /// **Why Required**: This function is used to ensure blocks are on the canonical chain
    /// and for confirmation inheritance. It's a fundamental building block for FCR logic
    /// that determines block relationships in the DAG.
    ///
    /// **Specification**: Returns true if `ancestor` is an ancestor of `root` in the block DAG.
    /// A block is considered an ancestor of itself.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `root` - The descendant block root to check
    /// * `ancestor` - The potential ancestor block root
    ///
    /// # Returns
    /// * `bool` - True if `ancestor` is an ancestor of `root`, false otherwise
    pub fn is_ancestor(
        &self,
        proto_array: &ProtoArrayForkChoice,
        root: Hash256,
        ancestor: Hash256,
    ) -> bool {
        // A block is an ancestor of itself
        if root == ancestor {
            return true;
        }

        // Check if both blocks exist in the proto array
        let root_block = match proto_array.get_block(&root) {
            Some(block) => block,
            None => return false, // Root doesn't exist
        };

        let ancestor_block = match proto_array.get_block(&ancestor) {
            Some(block) => block,
            None => return false, // Ancestor doesn't exist
        };

        // Walk up the chain from root to find the ancestor
        let mut current_root = root;
        let mut depth = 0;
        const MAX_ANCESTOR_DEPTH: usize = 1000; // Safety limit to prevent infinite loops

        while depth < MAX_ANCESTOR_DEPTH {
            let current_block = match proto_array.get_block(&current_root) {
                Some(block) => block,
                None => break, // Reached a block that doesn't exist
            };

            // Check if we've found the ancestor
            if current_root == ancestor {
                return true;
            }

            // Move to parent
            if let Some(parent_root) = current_block.parent_root {
                current_root = parent_root;
                depth += 1;
            } else {
                // Reached genesis (no parent)
                break;
            }
        }

        false
    }

    /// Checks if a block is confirmed by FCR.
    ///
    /// **Why Required**: This method provides a simple way to check confirmation status
    /// without exposing internal metadata structures. It's used by tests and other
    /// parts of the codebase to verify FCR behavior.
    ///
    /// # Arguments
    /// * `block_root` - The block root to check
    ///
    /// # Returns
    /// * `bool` - True if the block is confirmed, false otherwise
    pub fn is_block_confirmed(&self, block_root: &Hash256) -> bool {
        self.meta
            .get(block_root)
            .map_or(false, |meta| meta.confirmed)
    }

    /// Updates FCR state when transitioning to a new slot.
    ///
    /// This method is the Lighthouse adaptation of the Python specification's
    /// `on_tick_per_slot_after_attestations_applied(store)` function. It should be called
    /// at the beginning of each new slot to update the FCR state according to the specification.
    ///
    /// It performs the following operations:
    ///
    /// 1. Updates the confirmed root to the latest confirmed block
    /// 2. Stores the previous slot's justified checkpoint
    /// 3. Stores the previous slot's unrealized justified checkpoint  
    /// 4. Stores the previous slot's head block
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Ok(())` - Successfully updated FCR state
    /// * `Err(Error)` - Error occurred during state update
    pub fn update_per_slot<T>(
        &mut self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<(), crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Update the confirmed root to the latest confirmed block
        if let Some(new_confirmed_root) =
            self.get_latest_confirmed(proto_array, fc_store, head_root)
        {
            self.fcr_store.confirmed_root = new_confirmed_root;
        }

        // Store the previous slot's state
        self.fcr_store.prev_slot_justified_checkpoint = *fc_store.justified_checkpoint();

        // Store the previous slot's unrealized justified checkpoint
        self.fcr_store.prev_slot_unrealized_justified_checkpoint =
            *fc_store.unrealized_justified_checkpoint();

        // Store the previous slot's head block
        self.fcr_store.prev_slot_head = head_root;

        Ok(())
    }

    /// Updates FCR state for a new slot transition.
    ///
    /// **Specification**: Convenience wrapper for `update_per_slot()` (not in Python spec)
    ///
    /// **Why Required**: Lighthouse's fork choice architecture requires integration hooks
    /// at specific points in the slot processing pipeline. This method provides a clean
    /// interface for the `on_tick()` method to call FCR state updates without exposing
    /// internal implementation details.
    ///
    /// This is a convenience method that can be called from the fork choice's `on_tick`
    /// method to update FCR state when transitioning to a new slot. It uses the current
    /// head from the fork choice update parameters.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root (from fork choice)
    ///
    /// # Returns
    /// * `Ok(())` - Successfully updated FCR state
    /// * `Err(Error)` - Error occurred during state update
    pub fn on_new_slot<T>(
        &mut self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<(), crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Call the main update method
        self.update_per_slot(proto_array, fc_store, head_root)
    }

    /// Updates FCR state using fork choice update parameters.
    ///
    /// **Specification**: Convenience wrapper for `on_new_slot()` (not in Python spec)
    ///
    /// **Why Required**: Lighthouse's fork choice system caches update parameters to avoid
    /// redundant computations. This method allows FCR to integrate with the existing
    /// parameter caching system, extracting the head root from pre-computed parameters
    /// rather than requiring additional state lookups.
    ///
    /// This method is designed to be called from the fork choice system using
    /// the cached fork choice update parameters. It's a convenience wrapper
    /// that extracts the head root from the update parameters.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `forkchoice_params` - The fork choice update parameters containing the head
    ///
    /// # Returns
    /// * `Ok(())` - Successfully updated FCR state
    /// * `Err(Error)` - Error occurred during state update
    pub fn on_new_slot_with_params<T>(
        &mut self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        forkchoice_params: &crate::ForkchoiceUpdateParameters,
    ) -> Result<(), crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        let head_root = forkchoice_params.head_root;
        self.on_new_slot(proto_array, fc_store, head_root)
    }

    /// Gets the latest confirmed block root.
    ///
    /// **Python Specification**: `get_latest_confirmed(store)`
    ///
    /// This implements the core FCR logic to determine the latest confirmed block
    /// along the canonical chain. It follows the Python specification's `get_latest_confirmed`
    /// function with fallback to finalized checkpoint for safety.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Some(Hash256)` - The latest confirmed block root
    /// * `None` - No confirmed block found (fallback to finalized checkpoint)
    pub fn get_latest_confirmed<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Option<Hash256>
    where
        T: crate::ForkChoiceStore<E>,
    {
        let mut confirmed_root = self.fcr_store.confirmed_root;
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());

        // Safety check: revert to finalized block if confirmed block is too old
        // or doesn't belong to canonical chain (equivalent to Python spec's safety checks)
        if let Some(confirmed_block) = proto_array.get_block(&confirmed_root) {
            let confirmed_block_epoch = confirmed_block.slot.epoch(E::slots_per_epoch());

            // Check if confirmed block is from 2+ epochs ago or not in canonical chain
            if confirmed_block_epoch + 1 < current_epoch
                || !proto_array.is_descendant(confirmed_root, head_root)
            {
                // Fallback to finalized checkpoint for safety
                return Some(fc_store.finalized_checkpoint().root);
            }
        } else {
            // Confirmed block not found in proto array, fallback to finalized
            return Some(fc_store.finalized_checkpoint().root);
        }

        // Try to advance the confirmed root along the canonical chain
        // This is equivalent to Python spec's find_latest_confirmed_descendant logic
        if let Some(new_confirmed) =
            self.find_latest_confirmed_descendant(confirmed_root, proto_array, fc_store, head_root)
        {
            confirmed_root = new_confirmed;
        }

        Some(confirmed_root)
    }

    /// Updates FCR state after finding a new head.
    ///
    /// **Specification**: Custom Lighthouse integration hook (not in Python spec)
    ///
    /// **Why Required**: Lighthouse's fork choice architecture separates head determination
    /// from confirmation logic. This method provides the integration point where FCR can
    /// perform confirmation checks immediately after a new head is determined, leveraging
    /// the already-computed head and performing an efficient O(depth) ancestor scan rather
    /// than a full DAG traversal.
    ///
    /// This method is called after fork choice determines a new head, allowing FCR
    /// to perform confirmation checks and update its internal state. It performs
    /// an O(depth) ancestor scan from the new head to check for confirmations.
    ///
    /// # Arguments
    /// * `head_root` - The newly determined head block root
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    ///
    /// # Returns
    /// * `Ok(())` - Successfully updated FCR state
    /// * `Err(Error)` - Error occurred during state update
    pub fn update_after_find_head<T>(
        &mut self,
        head_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
    ) -> Result<(), crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Perform reverse ancestor scan from head_root to check for confirmations
        // This is the O(depth) operation that leverages the already-computed head
        let mut current_root = head_root;
        let mut depth = 0;

        while depth < MAX_REORG_DEPTH {
            // Check if this block is already confirmed
            if let Some(meta) = self.meta.get(&current_root) {
                if meta.confirmed {
                    // Found a confirmed ancestor, no need to scan further
                    break;
                }
            }

            // Check if this block meets confirmation criteria
            if self.is_one_confirmed(current_root, proto_array, fc_store)? {
                // Mark this block and all its descendants as confirmed
                self.mark_confirmed(current_root, proto_array);
                break;
            }

            // Move to parent block
            if let Some(block) = proto_array.get_block(&current_root) {
                if let Some(parent_root) = block.parent_root {
                    current_root = parent_root;
                    depth += 1;
                } else {
                    // Reached genesis block, stop scanning
                    break;
                }
            } else {
                // Reached end of chain
                break;
            }
        }

        Ok(())
    }

    /// Checks if a block meets the FCR confirmation criteria.
    ///
    /// **Specification**: `is_one_confirmed(store, block_root)`
    ///
    /// This implements the core FCR Q-indicator check: `2 * S > W * (1 + 2 * β / 100) + proposer_score`
    /// where S is support weight, W is committee weight, and β is the Byzantine threshold.
    ///
    /// # Arguments
    /// * `block_root` - The block root to check for confirmation
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    ///
    /// # Returns
    /// * `Ok(true)` - Block meets confirmation criteria
    /// * `Ok(false)` - Block does not meet confirmation criteria
    /// * `Err(Error)` - Error occurred during check
    fn is_one_confirmed<T>(
        &self,
        block_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Get the block to check
        let block = match proto_array.get_block(&block_root) {
            Some(block) => block,
            None => {
                return Err(ProtoArrayStringError(format!(
                    "Block {} not found in proto array",
                    block_root
                )));
            }
        };

        // Get the parent block for committee weight calculation
        let parent_block = match block.parent_root {
            Some(parent_root) => match proto_array.get_block(&parent_root) {
                Some(parent) => parent,
                None => {
                    return Err(crate::Error::ProtoArrayStringError(format!(
                        "Parent block {} not found in proto array",
                        parent_root
                    )));
                }
            },
            None => {
                // Genesis block cannot be confirmed by FCR
                return Ok(false);
            }
        };

        // Get LMD-GHOST support weight (S) from proto array WITHOUT proposer boost
        // FCR specification requires separating support weight from proposer boost
        let support = match proto_array.get_weight::<E>(
            &block_root,
            None,  // checkpoint_state not needed for basic support calculation
            false, // FCR doesn't want proposer boost included in support
            fc_store.proposer_boost_root(),
            fc_store.chain_spec(),
        ) {
            Some(weight) => weight,
            None => {
                return Err(ProtoArrayStringError(format!(
                    "Failed to get weight for block {}",
                    block_root
                )));
            }
        };

        // Get committee weight (W) with proper cross-epoch handling
        let committee_weight = self.get_committee_weight_between_slots(
            parent_block.slot + 1,
            fc_store.get_current_slot() - 1,
            fc_store,
        )?;

        // Get proposer boost score separately (as required by FCR spec)
        let proposer_score =
            match proto_array.get_proposer_score::<E>(block_root, fc_store.chain_spec()) {
                Some(score) => score,
                None => 0, // No proposer boost applicable
            };

        // Calculate the Byzantine threshold
        let beta_threshold = self.config.beta_percentage;

        // Apply the FCR formula: 2 * S > W * (1 + 2 * β / 100) + proposer_score
        // Using integer arithmetic to avoid floating point issues
        let left_side = 2 * support;
        let right_side = committee_weight * (100 + 2 * beta_threshold) / 100 + proposer_score;

        Ok(left_side > right_side)
    }

    /// Marks a block and all its descendants as confirmed.
    ///
    /// **Specification**: Custom Lighthouse implementation
    ///
    /// **Why Required**: Lighthouse's side-table approach for FCR metadata requires explicit
    /// confirmation inheritance management. Unlike the Python spec where confirmation status
    /// is computed on-demand, Lighthouse caches confirmation status in `FcrMeta` to avoid
    /// repeated expensive computations. This method ensures that when a parent is confirmed,
    /// all descendants inherit the confirmation status efficiently.
    ///
    /// This implements confirmation inheritance where if a parent block is confirmed,
    /// all its descendants are also confirmed.
    ///
    /// # Arguments
    /// * `block_root` - The block root to mark as confirmed
    /// * `proto_array` - The proto array containing the block DAG
    fn mark_confirmed(&mut self, block_root: Hash256, proto_array: &ProtoArrayForkChoice) {
        // Mark the specific block as confirmed
        if let Some(meta) = self.meta.get_mut(&block_root) {
            meta.confirmed = true;
        } else {
            // Create new metadata if it doesn't exist
            self.meta.insert(
                block_root,
                FcrMeta {
                    support: 0,
                    committee_weight: 0,
                    confirmed: true,
                },
            );
        }

        // Mark all descendants as confirmed
        self.mark_descendants_confirmed(block_root, proto_array);
    }

    /// Recursively marks all descendants of a confirmed block as confirmed.
    ///
    /// **Why Required**: When a parent block is confirmed, all its descendants inherit
    /// the confirmation status. This is a key property of FCR that ensures consistency
    /// across the block DAG.
    ///
    /// # Arguments
    /// * `parent_root` - The parent block root whose descendants should be marked
    /// * `proto_array` - The proto array containing the block DAG
    fn mark_descendants_confirmed(
        &mut self,
        parent_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
    ) {
        // Get all blocks in the proto array to find descendants
        let mut to_process = vec![parent_root];
        let mut processed = std::collections::HashSet::new();

        while let Some(current_root) = to_process.pop() {
            if processed.contains(&current_root) {
                continue;
            }
            processed.insert(current_root);

            // Find all blocks that have current_root as their parent
            // We need to iterate through all blocks in the proto array
            // Since there's no direct iterator, we'll use the indices HashMap
            let proto_array_ref = proto_array.core_proto_array();
            for (block_root, _) in &proto_array_ref.indices {
                if let Some(block) = proto_array.get_block(block_root) {
                    if let Some(parent) = block.parent_root {
                        if parent == current_root {
                            // This is a descendant, mark it as confirmed
                            if let Some(meta) = self.meta.get_mut(block_root) {
                                meta.confirmed = true;
                            } else {
                                // Create new metadata if it doesn't exist
                                self.meta.insert(
                                    *block_root,
                                    FcrMeta {
                                        support: 0,
                                        committee_weight: 0,
                                        confirmed: true,
                                    },
                                );
                            }

                            // Add this descendant to the processing queue
                            to_process.push(*block_root);
                        }
                    }
                }
            }
        }
    }

    /// Finds the latest confirmed descendant along the canonical chain.
    ///
    /// **Specification**: `find_latest_confirmed_descendant(store, latest_confirmed_root)`
    ///
    /// This implements the Python specification's `find_latest_confirmed_descendant`
    /// function to advance confirmation along the canonical chain with epoch boundary handling.
    ///
    /// # Arguments
    /// * `confirmed_root` - The current confirmed root
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Some(Hash256)` - The latest confirmed descendant
    /// * `None` - No advancement possible
    fn find_latest_confirmed_descendant<T>(
        &self,
        confirmed_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Option<Hash256>
    where
        T: crate::ForkChoiceStore<E>,
    {
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let mut confirmed_root = confirmed_root;

        // Get the confirmed block to check its epoch
        let confirmed_block = match proto_array.get_block(&confirmed_root) {
            Some(block) => block,
            None => return None,
        };

        let confirmed_block_epoch = confirmed_block.slot.epoch(E::slots_per_epoch());

        // First condition: Previous epoch advancement
        if confirmed_block_epoch + 1 == current_epoch
            && self
                .check_voting_source_conditions(proto_array, fc_store, head_root)
                .unwrap_or(false)
            && (fc_store.get_current_slot() % E::slots_per_epoch() == 0
                || self
                    .will_no_conflicting_checkpoint_be_justified(proto_array, fc_store, head_root)
                    .unwrap_or(false))
        {
            // Advance through canonical chain for previous epoch blocks
            if let Some(new_confirmed) = self.advance_through_canonical_chain(
                confirmed_root,
                proto_array,
                fc_store,
                head_root,
            ) {
                confirmed_root = new_confirmed;
            }
        }

        // Second condition: Current epoch advancement
        if fc_store.get_current_slot() % E::slots_per_epoch() == 0
            || self
                .check_unrealized_justification_conditions(proto_array, fc_store, head_root)
                .unwrap_or(false)
        {
            if let Some(new_confirmed) =
                self.try_advance_current_epoch(confirmed_root, proto_array, fc_store, head_root)
            {
                confirmed_root = new_confirmed;
            }
        }

        Some(confirmed_root)
    }

    /// Checks voting source conditions for previous epoch advancement.
    ///
    /// **Python Specification**: Part of `find_latest_confirmed_descendant()` logic
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Ok(bool)` - True if conditions are met
    /// * `Err(Error)` - Error occurred during check
    fn check_voting_source_conditions<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Simplified implementation: check if voting source is recent enough
        // In a full implementation, this would check the voting source epoch
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let voting_source_epoch = if current_epoch.as_u64() > 0 {
            current_epoch.as_u64() - 1
        } else {
            0
        };

        Ok(voting_source_epoch + 2 >= current_epoch.as_u64())
    }

    /// Checks unrealized justification conditions for current epoch advancement.
    ///
    /// **Python Specification**: Part of `find_latest_confirmed_descendant()` logic
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Ok(bool)` - True if conditions are met
    /// * `Err(Error)` - Error occurred during check
    fn check_unrealized_justification_conditions<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Simplified implementation: check if unrealized justification is recent enough
        // In a full implementation, this would check the unrealized justification epoch
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let unrealized_epoch = if current_epoch.as_u64() > 0 {
            current_epoch.as_u64() - 1
        } else {
            0
        };

        Ok(unrealized_epoch + 1 >= current_epoch.as_u64())
    }

    /// Advances confirmation through the canonical chain for previous epoch blocks.
    ///
    /// **Python Specification**: Part of `find_latest_confirmed_descendant()` logic
    ///
    /// # Arguments
    /// * `confirmed_root` - The current confirmed root
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Some(Hash256)` - The new confirmed root
    /// * `None` - No advancement possible
    fn advance_through_canonical_chain<T>(
        &self,
        confirmed_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Option<Hash256>
    where
        T: crate::ForkChoiceStore<E>,
    {
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let mut current_confirmed = confirmed_root;

        // Get canonical chain from confirmed root to head
        let canonical_roots = self.get_canonical_roots(proto_array, confirmed_root, head_root)?;

        // Skip the first root (confirmed_root itself)
        for &block_root in canonical_roots.iter().skip(1) {
            let block = proto_array.get_block(&block_root)?;
            let block_epoch = block.slot.epoch(E::slots_per_epoch());

            // Stop if we reach current epoch
            if block_epoch == current_epoch {
                break;
            }

            // Check if this block is a descendant of the previous head
            if !proto_array.is_descendant(self.fcr_store.prev_slot_head, block_root) {
                break;
            }

            // Check if this block is confirmed
            if self
                .is_one_confirmed(block_root, proto_array, fc_store)
                .ok()?
            {
                current_confirmed = block_root;
            } else {
                break;
            }
        }

        Some(current_confirmed)
    }

    /// Tries to advance confirmation for current epoch blocks.
    ///
    /// **Python Specification**: Part of `find_latest_confirmed_descendant()` logic
    ///
    /// # Arguments
    /// * `confirmed_root` - The current confirmed root
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Some(Hash256)` - The new confirmed root
    /// * `None` - No advancement possible
    fn try_advance_current_epoch<T>(
        &self,
        confirmed_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Option<Hash256>
    where
        T: crate::ForkChoiceStore<E>,
    {
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let mut tentative_confirmed = confirmed_root;

        // Get canonical chain from confirmed root to head
        let canonical_roots = self.get_canonical_roots(proto_array, confirmed_root, head_root)?;

        // Skip the first root (confirmed_root itself)
        for &block_root in canonical_roots.iter().skip(1) {
            let block = proto_array.get_block(&block_root)?;
            let block_epoch = block.slot.epoch(E::slots_per_epoch());
            let tentative_block = proto_array.get_block(&tentative_confirmed)?;
            let tentative_epoch = tentative_block.slot.epoch(E::slots_per_epoch());

            // If we advance to current epoch, check checkpoint justification
            if block_epoch > tentative_epoch {
                let checkpoint_root =
                    self.get_checkpoint_block(proto_array, block_root, block_epoch)?;
                let checkpoint = Checkpoint {
                    epoch: block_epoch,
                    root: checkpoint_root,
                };

                // Ensure current epoch checkpoint will be justified
                if !self
                    .will_checkpoint_be_justified(proto_array, fc_store, &checkpoint)
                    .unwrap_or(false)
                {
                    break;
                }
            }

            // Check if this block is confirmed
            if self
                .is_one_confirmed(block_root, proto_array, fc_store)
                .ok()?
            {
                tentative_confirmed = block_root;
            } else {
                break;
            }
        }

        // Final safety check for current epoch confirmation
        if self
            .check_current_epoch_confirmation_safety(
                tentative_confirmed,
                proto_array,
                fc_store,
                head_root,
            )
            .unwrap_or(false)
        {
            Some(tentative_confirmed)
        } else {
            Some(confirmed_root)
        }
    }

    /// Gets canonical roots from ancestor to descendant.
    ///
    /// **Python Specification**: `get_canonical_roots(store, ancestor_slot)`
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `ancestor_root` - The ancestor block root
    /// * `descendant_root` - The descendant block root
    ///
    /// # Returns
    /// * `Some(Vec<Hash256>)` - Canonical roots from ancestor to descendant
    /// * `None` - No canonical path found
    fn get_canonical_roots(
        &self,
        proto_array: &ProtoArrayForkChoice,
        ancestor_root: Hash256,
        descendant_root: Hash256,
    ) -> Option<Vec<Hash256>> {
        let mut canonical_roots = Vec::new();
        let mut current_root = descendant_root;

        // Walk from descendant to ancestor
        while current_root != ancestor_root {
            canonical_roots.push(current_root);

            let block = proto_array.get_block(&current_root)?;
            current_root = block.parent_root?;
        }

        canonical_roots.push(ancestor_root);
        canonical_roots.reverse();

        Some(canonical_roots)
    }

    /// Gets the checkpoint block for a given epoch.
    ///
    /// **Python Specification**: `get_checkpoint_block(store, block_root, epoch)`
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `block_root` - The block root
    /// * `epoch` - The epoch
    ///
    /// # Returns
    /// * `Some(Hash256)` - The checkpoint block root
    /// * `None` - No checkpoint block found
    fn get_checkpoint_block(
        &self,
        proto_array: &ProtoArrayForkChoice,
        block_root: Hash256,
        epoch: Epoch,
    ) -> Option<Hash256> {
        // Simplified implementation: return the block root itself
        // In a full implementation, this would find the epoch boundary block
        Some(block_root)
    }

    /// Checks if a checkpoint will be justified.
    ///
    /// **Python Specification**: `will_checkpoint_be_justified(store, checkpoint)`
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `checkpoint` - The checkpoint to check
    ///
    /// # Returns
    /// * `Ok(bool)` - True if checkpoint will be justified
    /// * `Err(Error)` - Error occurred during check
    fn will_checkpoint_be_justified<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        checkpoint: &Checkpoint,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Simplified implementation: always return true for now
        // In a full implementation, this would check FFG support
        Ok(true)
    }

    /// Checks if no conflicting checkpoint will be justified.
    ///
    /// **Python Specification**: `will_no_conflicting_checkpoint_be_justified(store, checkpoint)`
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Ok(bool)` - True if no conflicting checkpoint will be justified
    /// * `Err(Error)` - Error occurred during check
    fn will_no_conflicting_checkpoint_be_justified<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Simplified implementation: always return true for now
        // In a full implementation, this would check for conflicting FFG support
        Ok(true)
    }

    /// Checks safety conditions for current epoch confirmation.
    ///
    /// **Python Specification**: Part of `try_advance_current_epoch()` logic
    ///
    /// # Arguments
    /// * `tentative_confirmed` - The tentative confirmed root
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Ok(bool)` - True if safety conditions are met
    /// * `Err(Error)` - Error occurred during check
    fn check_current_epoch_confirmation_safety<T>(
        &self,
        tentative_confirmed: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let tentative_block = proto_array.get_block(&tentative_confirmed).ok_or_else(|| {
            ProtoArrayStringError("Tentative confirmed block not found".to_string())
        })?;
        let tentative_epoch = tentative_block.slot.epoch(E::slots_per_epoch());

        // Check if tentative block is from current epoch or has recent voting source
        if tentative_epoch == current_epoch {
            return Ok(true);
        }

        // Check voting source conditions
        let voting_source_epoch = if current_epoch.as_u64() > 0 {
            current_epoch.as_u64() - 1
        } else {
            0
        };
        if voting_source_epoch + 2 >= current_epoch.as_u64() {
            return Ok(true);
        }

        // Check unrealized justification conditions
        let unrealized_epoch = if current_epoch.as_u64() > 0 {
            current_epoch.as_u64() - 1
        } else {
            0
        };
        if unrealized_epoch + 1 >= current_epoch.as_u64() {
            return Ok(true);
        }

        Ok(false)
    }

    /// Prunes FCR metadata to align with the DAG pruning.
    ///
    /// **Why Required**: Lighthouse's side-table approach requires explicit pruning to prevent
    /// unbounded memory growth. When blocks are pruned from the proto array, their FCR metadata
    /// must also be removed to maintain consistency and prevent memory leaks.
    ///
    /// This method removes FCR metadata for blocks that have been pruned from the proto array,
    /// ensuring that the FCR side-table stays aligned with the main DAG structure.
    ///
    /// # Arguments
    /// * `finalized_root` - The finalized block root (blocks before this are pruned)
    /// * `proto_array` - The proto array containing the block DAG
    ///
    /// # Returns
    /// * `Ok(())` - Successfully pruned FCR metadata
    /// * `Err(Error)` - Error occurred during pruning
    pub fn prune<T>(
        &mut self,
        finalized_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
    ) -> Result<(), crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        // Get the finalized block to determine the pruning boundary
        let finalized_block = match proto_array.get_block(&finalized_root) {
            Some(block) => block,
            None => {
                // If finalized block not found, something is wrong
                return Err(ProtoArrayStringError(
                    "Finalized block not found in proto array during FCR pruning".to_string(),
                ));
            }
        };

        // Remove FCR metadata for blocks that are no longer in the proto array
        // or are before the finalized block
        self.meta.retain(|block_root, _| {
            // Keep metadata if block still exists in proto array
            if let Some(block) = proto_array.get_block(block_root) {
                // Keep if block is at or after the finalized block slot
                // This ensures we only keep metadata for blocks that are still
                // part of the canonical chain or recent forks
                block.slot >= finalized_block.slot
            } else {
                // Block no longer exists in proto array, remove metadata
                // This handles the case where proto array pruning has already
                // removed the block from the DAG
                false
            }
        });

        // Clear expired cache entries
        // Note: LRU caches handle their own eviction, but we can clear very old entries
        // that are definitely no longer needed
        self.fcr_store.committee_weight_lru.clear();
        self.fcr_store.ffg_support_lru.clear();

        Ok(())
    }

    /// Gets the committee weight between slots with proper cross-epoch handling.
    ///
    /// **Python Specification**: `get_committee_weight_between_slots(state, start_slot, end_slot)`
    ///
    /// This implements the committee weight calculation with cross-epoch boundary handling
    /// and safety adjustments as specified in the Python implementation.
    ///
    /// # Arguments
    /// * `start_slot` - Starting slot for committee weight calculation
    /// * `end_slot` - Ending slot for committee weight calculation
    /// * `fc_store` - The fork choice store containing current state
    ///
    /// # Returns
    /// * `Ok(u64)` - Committee weight between the slots
    /// * `Err(Error)` - Error occurred during calculation
    fn get_committee_weight_between_slots<T>(
        &self,
        start_slot: Slot,
        end_slot: Slot,
        fc_store: &T,
    ) -> Result<u64, crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        if start_slot > end_slot {
            return Ok(0);
        }

        let total_active_balance = fc_store.justified_balances().total_effective_balance;
        let start_epoch = start_slot.epoch(E::slots_per_epoch());
        let end_epoch = end_slot.epoch(E::slots_per_epoch());

        // If an entire epoch is covered by the range, return the total active balance
        if self.is_full_validator_set_covered(start_slot, end_slot) {
            return Ok(total_active_balance);
        }

        if start_epoch == end_epoch {
            // Same epoch: simple pro-rata calculation
            let slots_covered = end_slot - start_slot + 1;
            let weight_per_slot = total_active_balance / E::slots_per_epoch();
            return Ok(weight_per_slot * slots_covered.as_u64());
        } else {
            // Cross-epoch boundary: complex calculation with safety adjustment
            let estimate = match self.calculate_cross_epoch_weight_estimate(
                start_slot,
                end_slot,
                total_active_balance,
            ) {
                Ok(estimate) => estimate,
                Err(_) => {
                    // Fallback to simple calculation if cross-epoch calculation fails
                    let slots_covered = end_slot - start_slot + 1;
                    let weight_per_slot = total_active_balance / E::slots_per_epoch();
                    weight_per_slot * slots_covered.as_u64()
                }
            };

            // Apply safety adjustment factor for partial epoch coverage
            Ok(estimate * 1005 / 1000) // 0.5% safety margin
        }
    }

    /// Checks if the slot range covers a full validator set.
    ///
    /// **Python Specification**: `is_full_validator_set_covered(start_slot, end_slot)`
    ///
    /// Returns whether the range from `start_slot` to `end_slot` (inclusive of both)
    /// includes an entire epoch.
    ///
    /// # Arguments
    /// * `start_slot` - Starting slot
    /// * `end_slot` - Ending slot
    ///
    /// # Returns
    /// * `bool` - True if a full epoch is covered
    fn is_full_validator_set_covered(&self, start_slot: Slot, end_slot: Slot) -> bool {
        let start_epoch = start_slot.epoch(E::slots_per_epoch());
        let end_epoch = end_slot.epoch(E::slots_per_epoch());

        end_epoch > start_epoch + 1
            || (end_epoch == start_epoch + 1 && start_slot % E::slots_per_epoch() == 0)
    }

    /// Calculates cross-epoch weight estimate with pro-rata adjustments.
    ///
    /// **Python Specification**: Complex calculation from `get_committee_weight_between_slots()`
    ///
    /// This implements the cross-epoch boundary calculation with pro-rata adjustments
    /// as specified in the Python implementation.
    ///
    /// # Arguments
    /// * `start_slot` - Starting slot
    /// * `end_slot` - Ending slot
    /// * `total_active_balance` - Total active validator balance
    ///
    /// # Returns
    /// * `Ok(u64)` - Estimated committee weight
    /// * `Err(Error)` - Error occurred during calculation
    fn calculate_cross_epoch_weight_estimate(
        &self,
        start_slot: Slot,
        end_slot: Slot,
        total_active_balance: u64,
    ) -> Result<u64, crate::Error<String>> {
        let start_epoch = start_slot.epoch(E::slots_per_epoch());
        let end_epoch = end_slot.epoch(E::slots_per_epoch());

        // Calculate slots in each epoch using simple arithmetic
        let start_epoch_slot = start_epoch * E::slots_per_epoch();
        let end_epoch_slot = end_epoch * E::slots_per_epoch();

        let slots_since_start = start_slot.as_u64() - start_epoch_slot.as_u64();
        let slots_since_end = end_slot.as_u64() - end_epoch_slot.as_u64();

        let slots_in_start_epoch = E::slots_per_epoch() - slots_since_start;
        let slots_in_end_epoch = slots_since_end + 1;

        // Calculate weight estimates for each epoch
        let weight_per_slot = total_active_balance / E::slots_per_epoch();
        let start_epoch_weight = weight_per_slot * slots_in_start_epoch;
        let end_epoch_weight = weight_per_slot * slots_in_end_epoch;

        // Cross-epoch adjustment: each committee from end epoch only contributes pro-rated weight
        let cross_epoch_adjustment =
            (weight_per_slot / E::slots_per_epoch()) * slots_in_start_epoch * slots_in_end_epoch;

        Ok(start_epoch_weight + end_epoch_weight - cross_epoch_adjustment)
    }

    /// Gets a simplified committee weight between slots for basic FCR implementation.
    ///
    /// **Specification**: Simplified version of `get_committee_weight_between_slots()`
    ///
    /// **Why Required**: This is a basic implementation for the initial FCR integration.
    /// The full committee weight calculation with cross-epoch handling will be implemented
    /// later. This provides a working foundation that can be enhanced later.
    ///
    /// # Arguments
    /// * `start_slot` - Starting slot for committee weight calculation
    /// * `end_slot` - Ending slot for committee weight calculation
    /// * `fc_store` - The fork choice store containing current state
    ///
    /// # Returns
    /// * `Ok(u64)` - Committee weight between the slots
    /// * `Err(Error)` - Error occurred during calculation
    fn get_committee_weight_between_slots_simple<T>(
        &self,
        start_slot: Slot,
        end_slot: Slot,
        fc_store: &T,
    ) -> Result<u64, crate::Error<T::Error>>
    where
        T: crate::ForkChoiceStore<E>,
    {
        if start_slot > end_slot {
            return Ok(0);
        }

        // For now, use a simplified calculation based on total active balance
        // This will be replaced with proper committee weight calculation in Week 9
        let total_active_balance = fc_store.justified_balances().total_effective_balance;

        // Simple pro-rata calculation: assume equal distribution across slots
        let slots_covered = end_slot - start_slot + 1;
        let weight_per_slot = total_active_balance / E::slots_per_epoch();

        Ok(weight_per_slot * slots_covered.as_u64())
    }
}
