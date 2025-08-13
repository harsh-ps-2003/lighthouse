//! Fast Confirmation Rule (FCR) implementation for Lighthouse.
//!
//! This module implements the Fast Confirmation Rule as described in the specification,
//! providing faster block confirmation times (12-24 seconds) compared to traditional
//! finalization (13-19 minutes).
//!
//! The FCR operates under network synchrony assumptions and uses LMD-GHOST vote weights
//! combined with FFG checkpoint support to determine block permanence.
//!
use crate::Error::ProtoArrayStringError;
use crate::ForkChoiceStore;
use lru::LruCache;
use proto_array::ProtoArrayForkChoice;
use std::collections::HashMap;
use std::error::Error;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tracing::{debug, info, warn};
use types::{
    BeaconState, ChainSpec, Checkpoint, Epoch, EthSpec, FixedBytesExtended, Hash256, Slot,
};

/// Default Byzantine threshold percentage for FCR
/// **Python Specification**: `CONFIRMATION_BYZANTINE_THRESHOLD = 33`
/// **Why**: This is the maximum fraction of Byzantine stake that FCR assumes
/// can be controlled by an adversary. The 33% threshold provides a balance
/// between confirmation speed and safety guarantees.
pub const DEFAULT_FCR_BYZANTINE_THRESHOLD_PERCENTAGE: u64 = 33;

/// Default slashing threshold percentage for FCR
/// **Python Specification**: `CONFIRMATION_SLASHING_THRESHOLD = 33`
/// **Why**: This is the maximum fraction of stake that can be slashed due to
/// equivocation or other slashable offenses. Used in FFG analysis to calculate
/// minimum honest support.
pub const DEFAULT_FCR_SLASHING_THRESHOLD_PERCENTAGE: u64 = 33;
/// Maximum depth to scan for reorgs (mainnet safety)
/// **Specification**: Not in spec (Lighthouse safety limit)
const MAX_REORG_DEPTH: usize = 32;
/// Committee weight estimation adjustment factor for safety
/// **Specification**: `COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR = 5` (0.5%)
/// **Why**: Adds a small safety margin to committee weight estimates to ensure
/// FCR safety guarantees are maintained even with estimation errors
const COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR: u64 = 5;

/// Trait for accessing historical checkpoint states required for FFG analysis.
///
/// This trait abstracts access to historical beacon states needed for FFG
/// checkpoint justification analysis. It allows the fork choice crate to remain
/// independent of the store crate while providing access to the necessary state data.
///
/// **Why Required**: FFG analysis requires access to historical checkpoint states
/// to calculate validator balances and voting patterns across epochs. The
/// `ForkChoiceStore` trait only provides current justified balances, not historical
/// states needed for complete FFG analysis.
pub trait StateProvider<E: EthSpec> {
    /// Error type for state access operations
    type Error: Error + Send + Sync + 'static;

    /// Gets the checkpoint state for a given checkpoint.
    ///
    /// This method should return the beacon state at the checkpoint's epoch boundary.
    /// The state is used for FFG weight calculations and validator analysis.
    ///
    /// # Arguments
    /// * `checkpoint` - The checkpoint to get the state for
    ///
    /// # Returns
    /// * `Ok(Option<Arc<BeaconState<E>>>)` - The checkpoint state if available
    /// * `Err(Self::Error)` - Error occurred during state access
    fn get_checkpoint_state(
        &self,
        checkpoint: &Checkpoint,
    ) -> Result<Option<Arc<BeaconState<E>>>, Self::Error>;

    /// Gets the total active balance at a given epoch.
    ///
    /// This method provides the total effective balance of all active validators
    /// at the specified epoch, which is needed for FFG weight calculations.
    ///
    /// # Arguments
    /// * `epoch` - The epoch to get the total active balance for
    ///
    /// # Returns
    /// * `Ok(u64)` - The total active balance in Gwei
    /// * `Err(Self::Error)` - Error occurred during balance calculation
    fn get_total_active_balance_at_epoch(&self, epoch: Epoch) -> Result<u64, Self::Error>;

    /// Gets the chain specification.
    ///
    /// This method provides access to the chain specification which is needed
    /// for various FFG calculations and state transitions.
    ///
    /// # Returns
    /// * `&ChainSpec` - The chain specification
    fn chain_spec(&self) -> &ChainSpec;
}

/// Configuration for the Fast Confirmation Rule.
#[derive(Debug, Clone)]
pub struct FastConfirmationConfig {
    /// Byzantine threshold in percentage (e.g., 25 = 25%)
    /// **Python Specification**: `CONFIRMATION_BYZANTINE_THRESHOLD`
    pub beta_percentage: u64,
    /// Slashing threshold in percentage (e.g., 33 = 33%)
    /// **Python Specification**: `CONFIRMATION_SLASHING_THRESHOLD`
    pub slashing_percentage: u64,
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
        Self::new_with_slashing(beta_percentage, DEFAULT_FCR_SLASHING_THRESHOLD_PERCENTAGE)
    }

    /// Creates a new FCR configuration with the given Byzantine and slashing thresholds.
    ///
    /// # Arguments
    /// * `beta_percentage` - Byzantine threshold in percentage (0-49)
    /// * `slashing_percentage` - Slashing threshold in percentage (0-100)
    ///
    /// # Returns
    /// * `Ok(FcrConfig)` - Valid configuration
    /// * `Err(String)` - Invalid threshold (≥50% makes confirmation impossible)
    pub fn new_with_slashing(
        beta_percentage: u64,
        slashing_percentage: u64,
    ) -> Result<Self, String> {
        if beta_percentage >= 50 {
            return Err(format!(
                "Invalid byzantine threshold: {}%, must be < 50%",
                beta_percentage
            ));
        }

        if slashing_percentage > 100 {
            return Err(format!(
                "Invalid slashing threshold: {}%, must be <= 100%",
                slashing_percentage
            ));
        }

        Ok(Self {
            beta_percentage,
            slashing_percentage,
        })
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
#[derive(Debug, Clone, Default)]
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
}

impl Default for FcrStore {
    fn default() -> Self {
        Self {
            confirmed_root: Hash256::zero(),
            prev_slot_justified_checkpoint: Checkpoint::default(),
            prev_slot_unrealized_justified_checkpoint: Checkpoint::default(),
            prev_slot_head: Hash256::zero(),
            committee_weight_lru: LruCache::new(NonZeroUsize::new(100).unwrap()),
        }
    }
}

/// Main Fast Confirmation Rule implementation.
pub struct FastConfirmation<E: EthSpec, S: StateProvider<E>> {
    /// FCR configuration including Byzantine threshold
    /// **Spec**: `CONFIRMATION_BYZANTINE_THRESHOLD` constant
    config: FastConfirmationConfig,
    /// Per-block FCR metadata, keyed by block root
    /// **Spec**: Computed on-demand in various functions
    meta: HashMap<Hash256, FcrMeta>,
    /// FCR state store (confirmed root, prev slot checkpoints, etc)
    /// **Spec**: Additional fields in `Store` class
    fcr_store: FcrStore,
    /// State provider for accessing historical checkpoint states
    /// **Why Required**: FFG analysis requires access to historical states
    state_provider: S,
    /// Phantom data to hold the EthSpec type parameter
    phantom: PhantomData<E>,
}

impl<E: EthSpec, S: StateProvider<E>> FastConfirmation<E, S> {
    /// Creates a new Fast Confirmation Rule instance.
    pub fn new(config: FastConfirmationConfig, state_provider: S) -> Self {
        info!(
            beta = config.beta_percentage,
            slashing = config.slashing_percentage,
            "Fast Confirmation Rule enabled"
        );
        Self {
            config,
            meta: HashMap::new(),
            fcr_store: FcrStore::default(),
            state_provider,
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
    /// **Implementation**: Uses the existing `is_descendant` method with swapped arguments,
    /// following the same pattern as the fork choice's `is_ancestor` method.
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

        // Use the existing is_descendant method with swapped arguments
        // is_descendant(ancestor, root) checks if root is a descendant of ancestor
        // which is equivalent to ancestor being an ancestor of root
        proto_array.is_descendant(ancestor, root)
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
        self.meta.get(block_root).is_some_and(|meta| meta.confirmed)
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
        T: ForkChoiceStore<E>,
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
        T: ForkChoiceStore<E>,
    {
        debug!(
            slot = fc_store.get_current_slot().as_u64(),
            head = %head_root,
            "FCR on_new_slot"
        );
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
        T: ForkChoiceStore<E>,
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
        T: ForkChoiceStore<E>,
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
                let finalized = fc_store.finalized_checkpoint().root;
                warn!(
                    current_epoch = current_epoch.as_u64(),
                    confirmed = %confirmed_root,
                    finalized = %finalized,
                    head = %head_root,
                    "FCR: falling back to finalized checkpoint for safety"
                );
                return Some(finalized);
            }
        } else {
            // Confirmed block not found in proto array, fallback to finalized
            let finalized = fc_store.finalized_checkpoint().root;
            warn!(confirmed_missing = %confirmed_root, finalized = %finalized, "FCR: confirmed root missing, using finalized");
            return Some(finalized);
        }

        // At the start of an epoch, if the prev-slot unrealized justified checkpoint
        // belongs to the previous epoch and is later than the current confirmed,
        // promote confirmed to that checkpoint (spec-aligned safety uplift),
        // then continue to attempt further advancement below.
        if fc_store.get_current_slot() % E::slots_per_epoch() == 0 {
            let prev_uj = *fc_store.unrealized_justified_checkpoint();
            let prev_uj_epoch = prev_uj.epoch;
            if prev_uj_epoch + 1 == current_epoch {
                if let (Some(confirmed_block), Some(prev_uj_block)) = (
                    proto_array.get_block(&confirmed_root),
                    proto_array.get_block(&prev_uj.root),
                ) {
                    if confirmed_block.slot < prev_uj_block.slot {
                        confirmed_root = prev_uj.root;
                    }
                }
            }
        }

        // Try to advance the confirmed root along the canonical chain
        // This is equivalent to Python spec's find_latest_confirmed_descendant logic
        if let Some(new_confirmed) =
            self.find_latest_confirmed_descendant(confirmed_root, proto_array, fc_store, head_root)
        {
            if new_confirmed != confirmed_root {
                info!(
                    old = %confirmed_root,
                    new = %new_confirmed,
                    head = %head_root,
                    "FCR: advanced latest confirmed descendant"
                );
            }
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
        T: ForkChoiceStore<E>,
    {
        // Perform reverse ancestor scan from head_root to check for confirmations
        // This is the O(depth) operation that leverages the already-computed head
        let mut current_root = head_root;
        let mut depth = 0;

        let mut found_confirmed: Option<Hash256> = None;
        while depth < MAX_REORG_DEPTH {
            // Check if this block is already confirmed
            if let Some(meta) = self.meta.get(&current_root) {
                if meta.confirmed {
                    // Found a confirmed ancestor, no need to scan further
                    found_confirmed = Some(current_root);
                    break;
                }
            }

            // Check if this block meets confirmation criteria
            if self.is_one_confirmed(current_root, proto_array, fc_store)? {
                // Mark this block and all its descendants as confirmed
                self.mark_confirmed(current_root, proto_array);
                found_confirmed = Some(current_root);
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

        match found_confirmed {
            Some(root) => {
                debug!(head = %head_root, confirmed_ancestor = %root, depth, "FCR update_after_find_head: confirmed ancestor")
            }
            None => {
                debug!(head = %head_root, depth, "FCR update_after_find_head: no confirmed ancestor within depth")
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
        T: ForkChoiceStore<E>,
    {
        // Get the block to check
        let Some(block) = proto_array.get_block(&block_root) else {
            return Err(ProtoArrayStringError(format!(
                "Block {} not found in proto array",
                block_root
            )));
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

        // Use weighting checkpoint state for both S and W when available
        // If current slot is at epoch boundary, use prev_slot_unrealized_justified_checkpoint,
        // otherwise use prev_slot_justified_checkpoint.
        let weighting_checkpoint = if fc_store.get_current_slot() % E::slots_per_epoch() == 0 {
            self.fcr_store.prev_slot_unrealized_justified_checkpoint
        } else {
            self.fcr_store.prev_slot_justified_checkpoint
        };

        // Try to obtain the checkpoint state via the StateProvider. If unavailable, fall back
        // to pre-existing behavior that uses no checkpoint state and justified_balances.
        let weighting_checkpoint_state_opt = self
            .state_provider
            .get_checkpoint_state(&weighting_checkpoint)
            .ok()
            .flatten();

        // Get LMD-GHOST support weight (S) from proto array WITHOUT proposer boost.
        // Pass the weighting checkpoint state to align with the spec signature. The
        // ProtoArray implementation will currently ignore it, but this preserves API
        // compatibility and allows future refinement without changing FCR.
        let Some(support) = proto_array.get_weight::<E>(
            &block_root,
            weighting_checkpoint_state_opt.as_deref(),
            false, // FCR doesn't want proposer boost included in support
            fc_store.proposer_boost_root(),
            fc_store.chain_spec(),
        ) else {
            return Err(ProtoArrayStringError(format!(
                "Failed to get weight for block {}",
                block_root
            )));
        };

        // Compute committee weight (W) using the weighting checkpoint state if present,
        // otherwise fall back to the existing fc_store-based calculation.
        let start_slot = parent_block.slot + 1;
        let end_slot = fc_store.get_current_slot() - 1;

        let committee_weight = if let Some(weighting_state) = weighting_checkpoint_state_opt {
            // Use TAB derived from the weighting checkpoint state, mirroring the Python spec
            let total_active_balance = weighting_state
                .get_total_active_balance()
                .unwrap_or(fc_store.justified_balances().total_effective_balance);

            if start_slot > end_slot {
                0
            } else if self.is_full_validator_set_covered(start_slot, end_slot) {
                total_active_balance
            } else {
                let start_epoch = start_slot.epoch(E::slots_per_epoch());
                let end_epoch = end_slot.epoch(E::slots_per_epoch());
                if start_epoch == end_epoch {
                    let slots_covered = end_slot - start_slot + 1;
                    let weight_per_slot = total_active_balance / E::slots_per_epoch();
                    weight_per_slot * slots_covered.as_u64()
                } else {
                    // Cross-epoch boundary calculation with safety adjustment
                    let estimate = match self.calculate_cross_epoch_weight_estimate(
                        start_slot,
                        end_slot,
                        total_active_balance,
                    ) {
                        Ok(estimate) => estimate,
                        Err(_) => {
                            // Conservative fallback
                            let slots_covered = end_slot - start_slot + 1;
                            let weight_per_slot = total_active_balance / E::slots_per_epoch();
                            weight_per_slot * slots_covered.as_u64()
                        }
                    };
                    self.adjust_committee_weight_estimate_to_ensure_safety(estimate)
                }
            }
        } else {
            // Fallback: use the existing method which relies on justified_balances
            self.get_committee_weight_between_slots(start_slot, end_slot, fc_store)?
        };

        // Get proposer boost score separately (as required by FCR spec)
        let proposer_score = proto_array
            .get_proposer_score::<E>(block_root, fc_store.chain_spec())
            .unwrap_or_default();

        // Calculate the Byzantine threshold
        let beta_threshold = self.config.beta_percentage;

        // Apply the FCR formula: 2 * S > W + W // 50 * CONFIRMATION_BYZANTINE_THRESHOLD + proposer_score
        // **Python Specification**: 2 * support > maximum_support + maximum_support // 50 * CONFIRMATION_BYZANTINE_THRESHOLD + proposer_score
        // Using integer arithmetic to avoid floating point issues
        let left_side = 2 * support;
        let right_side = committee_weight + committee_weight / 50 * beta_threshold + proposer_score;

        // Check LMD-GHOST confirmation (Q-indicator)
        let lmd_confirmed = left_side > right_side;

        if !lmd_confirmed {
            return Ok(false);
        }

        // Check FFG confirmation (checkpoint justification)
        // Get the checkpoint for this block's epoch
        let block_epoch = block.slot.epoch(E::slots_per_epoch());

        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());

        // Use current_epoch for epoch boundary checks
        if block_epoch > current_epoch {
            // Block is from a future epoch, cannot be confirmed yet
            return Ok(false);
        }

        let checkpoint_root = self
            .get_checkpoint_block(proto_array, block_root, block_epoch)
            .unwrap_or(block_root); // Fallback to block root if no checkpoint block found

        let checkpoint = Checkpoint {
            epoch: block_epoch,
            root: checkpoint_root,
        };

        // Check if the checkpoint will be justified using FFG analysis
        let ffg_confirmed =
            self.will_checkpoint_be_justified(proto_array, fc_store, &checkpoint)?;

        Ok(lmd_confirmed && ffg_confirmed)
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
            for block_root in proto_array_ref.indices.keys() {
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
        T: ForkChoiceStore<E>,
    {
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let mut confirmed_root = confirmed_root;

        // Get the confirmed block to check its epoch
        let Some(confirmed_block) = proto_array.get_block(&confirmed_root) else {
            return None;
        };

        let confirmed_block_epoch = confirmed_block.slot.epoch(E::slots_per_epoch());

        // First condition: Previous epoch advancement
        if confirmed_block_epoch + 1 == current_epoch
            && self
                .check_voting_source_conditions(proto_array, fc_store, head_root)
                .unwrap_or(false)
            && {
                // boundary OR (no_conflict AND (uj_prev OR uj_head))
                let boundary = fc_store.get_current_slot() % E::slots_per_epoch() == 0;
                let no_conflict = self
                    .will_no_conflicting_checkpoint_be_justified(proto_array, fc_store, head_root)
                    .unwrap_or(false);
                let uj_prev = self
                    .get_unrealized_justification_epoch(proto_array, self.fcr_store.prev_slot_head)
                    .ok()
                    .flatten()
                    .map(|e| e + 1 >= current_epoch)
                    .unwrap_or(false);
                let uj_head = self
                    .get_unrealized_justification_epoch(proto_array, head_root)
                    .ok()
                    .flatten()
                    .map(|e| e + 1 >= current_epoch)
                    .unwrap_or(false);
                boundary || (no_conflict && (uj_prev || uj_head))
            }
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
        if fc_store.get_current_slot() % E::slots_per_epoch() == 0 || {
            // Stricter unrealized-justification gate per Python spec:
            // require that either prev_slot_head or head has an unrealized justification epoch
            // at least current_epoch - 1.
            let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
            let head = head_root;
            let prev_head = self.fcr_store.prev_slot_head;
            let cond_prev = self
                .get_unrealized_justification_epoch(proto_array, prev_head)
                .ok()
                .flatten()
                .map(|e| e + 1 >= current_epoch)
                .unwrap_or(false);
            let cond_head = self
                .get_unrealized_justification_epoch(proto_array, head)
                .ok()
                .flatten()
                .map(|e| e + 1 >= current_epoch)
                .unwrap_or(false);
            cond_prev || cond_head
        } {
            if let Some(new_confirmed) =
                self.try_advance_current_epoch(confirmed_root, proto_array, fc_store, head_root)
            {
                confirmed_root = new_confirmed;
            }
        }

        Some(confirmed_root)
    }

    /// Checks voting source conditions for confirmation advancement.
    ///
    /// **Python Specification**: Part of `find_latest_confirmed_descendant()` logic
    ///
    /// This function checks if the voting source conditions are met for advancing
    /// confirmation through the canonical chain. It ensures that the previous
    /// slot's head is properly connected to the current confirmation logic.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Ok(bool)` - True if voting source conditions are met
    /// * `Err(Error)` - Error occurred during check
    fn check_voting_source_conditions<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());

        // Check if the voting source epoch is within the required range
        // This ensures that the previous slot's head is recent enough
        let voting_source_epoch = self
            .get_voting_source_epoch(proto_array, fc_store, head_root)
            .unwrap();

        Ok(voting_source_epoch + 2 >= current_epoch)
    }

    /// Gets the voting source epoch for a block.
    ///
    /// **Python Specification**: Helper function for voting source conditions
    ///
    /// This function determines the epoch of the voting source for a given block.
    /// The voting source is the block that validators are voting for when they
    /// vote for the given block.
    ///
    /// # Returns
    /// * `Ok(Epoch)` - The voting source epoch
    /// * `Err(Error)` - Error occurred during calculation
    fn get_voting_source_epoch<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        block_root: Hash256,
    ) -> Result<Epoch, crate::Error<String>>
    where
        T: ForkChoiceStore<E>,
    {
        use std::collections::HashMap;
        let balances = &fc_store.justified_balances().effective_balances;

        let mut epoch_to_weight: HashMap<Epoch, u128> = HashMap::new();
        for (validator_index, &eb) in balances.iter().enumerate() {
            if eb == 0 {
                continue;
            }
            if let Some((vote_root, vote_epoch)) = proto_array.latest_message(validator_index) {
                // Consider votes that support (are descendants of) the given block_root
                if proto_array.is_descendant(block_root, vote_root) {
                    *epoch_to_weight.entry(vote_epoch).or_insert(0) += eb as u128;
                }
            }
        }

        if let Some((best_epoch, _)) = epoch_to_weight.into_iter().max_by_key(|(_, w)| *w) {
            return Ok(best_epoch);
        }

        // Fallback: use prev_slot_unrealized_justified_checkpoint epoch as conservative proxy
        Ok(self
            .fcr_store
            .prev_slot_unrealized_justified_checkpoint
            .epoch)
    }

    /// Returns the unrealized justification epoch for a given block if available.
    fn get_unrealized_justification_epoch(
        &self,
        proto_array: &ProtoArrayForkChoice,
        block_root: Hash256,
    ) -> Result<Option<Epoch>, crate::Error<String>> {
        let Some(block) = proto_array.get_block(&block_root) else {
            return Err(ProtoArrayStringError(
                "Block not found while reading unrealized justification".to_string(),
            ));
        };
        Ok(block
            .unrealized_justified_checkpoint
            .as_ref()
            .map(|cp| cp.epoch))
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
        T: ForkChoiceStore<E>,
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
        T: ForkChoiceStore<E>,
    {
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
        // Find the checkpoint block for the given epoch
        let mut current_root = block_root;
        let mut depth = 0;
        const MAX_DEPTH: usize = 1000; // Safety limit

        while depth < MAX_DEPTH {
            let Some(current_block) = proto_array.get_block(&current_root) else {
                break;
            };

            let current_epoch = current_block.slot.epoch(E::slots_per_epoch());
            if current_epoch == epoch {
                return Some(current_root);
            }

            if let Some(parent_root) = current_block.parent_root {
                current_root = parent_root;
                depth += 1;
            } else {
                break; // Reached genesis
            }
        }

        None
    }

    /// Checks if a checkpoint will be justified.
    ///
    /// **Python Specification**: `will_checkpoint_be_justified(store, checkpoint)`
    ///
    /// This function determines if a checkpoint will be justified based on current
    /// vote patterns and FFG analysis. It handles both current epoch and previous
    /// epoch checkpoints appropriately.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `checkpoint` - The checkpoint to check justification for
    ///
    /// # Returns
    /// * `Ok(bool)` - True if the checkpoint will be justified
    /// * `Err(Error)` - Error occurred during analysis
    fn will_checkpoint_be_justified<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        checkpoint: &Checkpoint,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());

        // If checkpoint is already justified, return true
        if checkpoint == fc_store.justified_checkpoint() {
            return Ok(true);
        }

        // If checkpoint is the unrealized justified checkpoint, return true
        if checkpoint == fc_store.unrealized_justified_checkpoint() {
            return Ok(true);
        }

        // If checkpoint is from current epoch, use current epoch analysis
        if checkpoint.epoch == current_epoch {
            return self.will_current_epoch_checkpoint_be_justified(
                proto_array,
                fc_store,
                checkpoint,
            );
        }

        // For previous epoch checkpoints, assume they won't be justified
        // This is a conservative approach for safety
        Ok(false)
    }

    /// Checks safety conditions for current epoch confirmation.
    ///
    /// **Python Specification**: Part of `find_latest_confirmed_descendant()` logic
    ///
    /// This function checks if it's safe to confirm a block from the current epoch.
    /// It ensures that the confirmation won't be reorged out in either the current
    /// or next epoch.
    ///
    /// # Arguments
    /// * `tentative_confirmed` - The tentative confirmed block root
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Ok(bool)` - True if current epoch confirmation is safe
    /// * `Err(Error)` - Error occurred during check
    fn check_current_epoch_confirmation_safety<T>(
        &self,
        tentative_confirmed: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let tentative_block = proto_array.get_block(&tentative_confirmed).ok_or_else(|| {
            crate::Error::ProtoArrayStringError("Tentative confirmed block not found".to_string())
        })?;
        let tentative_epoch = tentative_block.slot.epoch(E::slots_per_epoch());

        // If the tentative confirmed block is from the current epoch
        if tentative_epoch == current_epoch {
            // Check if the voting source epoch is recent enough
            let voting_source_epoch = self
                .get_voting_source_epoch(proto_array, fc_store, tentative_confirmed)
                .unwrap();
            return Ok(voting_source_epoch + 2 >= current_epoch);
        }

        // For blocks from previous epochs, check if we're at epoch boundary
        // and no conflicting checkpoint will be justified
        if fc_store.get_current_slot() % E::slots_per_epoch() == 0 {
            return self.will_no_conflicting_checkpoint_be_justified(
                proto_array,
                fc_store,
                head_root,
            );
        }

        Ok(false)
    }

    /// Gets the checkpoint weight for FFG analysis.
    ///
    /// **Python Specification**: `get_checkpoint_weight(store, checkpoint, checkpoint_state)`
    ///
    /// This function calculates the FFG support weight for a given checkpoint by analyzing
    /// validator votes. It uses LMD-GHOST votes to estimate FFG support, as validators
    /// voting for blocks descended from a checkpoint implicitly support that checkpoint.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `checkpoint` - The checkpoint to calculate weight for
    /// * `fc_store` - The fork choice store containing current state
    ///
    /// # Returns
    /// * `Ok(u64)` - The checkpoint weight in Gwei
    /// * `Err(Error)` - Error occurred during calculation

    /// Gets the checkpoint weight for FFG analysis using a provided checkpoint state.
    ///
    /// **Python Specification**: `get_checkpoint_weight(store, checkpoint, checkpoint_state)`
    ///
    /// This function calculates the FFG support weight for a given checkpoint by analyzing
    /// validator votes using a provided checkpoint state. It uses LMD-GHOST votes to estimate
    /// FFG support, as validators voting for blocks descended from a checkpoint implicitly
    /// support that checkpoint.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `checkpoint` - The checkpoint to calculate weight for
    /// * `fc_store` - The fork choice store containing current state
    /// * `checkpoint_state` - The checkpoint state to use for analysis
    ///
    /// # Returns
    /// * `Ok(u64)` - The checkpoint weight in Gwei
    /// * `Err(Error)` - Error occurred during calculation
    fn get_checkpoint_weight_with_state<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        checkpoint: &Checkpoint,
        _fc_store: &T,
        checkpoint_state: &BeaconState<E>,
    ) -> Result<u64, crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let mut checkpoint_weight = 0u64;

        // Iterate through all validators and check their votes
        for validator_index in 0..checkpoint_state.validators().len() {
            // Get the validator's latest message
            if let Some((vote_root, vote_epoch)) = proto_array.latest_message(validator_index) {
                // Only votes from the same epoch support the checkpoint
                if vote_epoch != checkpoint.epoch {
                    continue;
                }
                // Derive the vote target checkpoint per spec
                if let Some(vote_target_root) =
                    self.get_checkpoint_block(proto_array, vote_root, vote_epoch)
                {
                    if vote_target_root == checkpoint.root {
                        let effective_balance = checkpoint_state
                            .get_effective_balance(validator_index)
                            .unwrap_or(0);
                        checkpoint_weight = checkpoint_weight.saturating_add(effective_balance);
                    }
                }
            }
        }

        Ok(checkpoint_weight)
    }

    /// Gets the FFG weight up to a specific slot.
    ///
    /// **Python Specification**: `get_ffg_weight_till_slot(slot, epoch, total_active_balance)`
    ///
    /// This function calculates the total FFG weight that could have been cast
    /// up to a specific slot within an epoch. It's used for FFG justification analysis.
    ///
    /// # Arguments
    /// * `slot` - The slot to calculate weight up to
    /// * `epoch` - The epoch containing the slot
    /// * `total_active_balance` - The total active validator balance
    ///
    /// # Returns
    /// * `u64` - The FFG weight up to the slot
    fn get_ffg_weight_till_slot(&self, slot: Slot, epoch: Epoch, total_active_balance: u64) -> u64 {
        let epoch_start_slot = epoch.start_slot(E::slots_per_epoch());
        let next_epoch_start = (epoch + 1).start_slot(E::slots_per_epoch());

        if slot <= epoch_start_slot {
            0
        } else if slot >= next_epoch_start {
            total_active_balance
        } else {
            // Calculate pro-rata weight for slots within the epoch
            let slots_passed = slot.as_u64() - epoch_start_slot.as_u64();
            let slots_per_epoch = E::slots_per_epoch();
            total_active_balance / slots_per_epoch * slots_passed
        }
    }

    /// Checks if the current epoch checkpoint will be justified.
    ///
    /// **Python Specification**: `will_current_epoch_checkpoint_be_justified(store, checkpoint)`
    ///
    /// This function predicts whether a current epoch checkpoint will be justified
    /// based on current vote patterns and remaining honest weight. It implements
    /// the FFG justification analysis from the Python specification.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `checkpoint` - The checkpoint to check justification for
    ///
    /// # Returns
    /// * `Ok(bool)` - True if the checkpoint will be justified
    /// * `Err(Error)` - Error occurred during analysis
    fn will_current_epoch_checkpoint_be_justified<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        checkpoint: &Checkpoint,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let current_slot = fc_store.get_current_slot();
        let current_epoch = current_slot.epoch(E::slots_per_epoch());

        // Ensure this is a current epoch checkpoint
        if checkpoint.epoch != current_epoch {
            return Err(crate::Error::ProtoArrayStringError(
                "Checkpoint is not from current epoch".to_string(),
            ));
        }

        // Get the checkpoint state for analysis
        let checkpoint_state = match self.state_provider.get_checkpoint_state(checkpoint) {
            Ok(Some(state)) => state,
            Ok(None) => {
                // If checkpoint state is not available, assume it won't be justified
                return Ok(false);
            }
            Err(_) => {
                // If we can't access the checkpoint state, assume it won't be justified
                return Ok(false);
            }
        };

        // Use TAB from the checkpoint state to avoid extra provider lookups
        let total_active_balance = checkpoint_state.get_total_active_balance().map_err(|_| {
            crate::Error::ProtoArrayStringError(
                "Failed to get total active balance from checkpoint state".to_string(),
            )
        })?;

        // Calculate FFG support for the checkpoint using the checkpoint state
        let ffg_support_for_checkpoint = self.get_checkpoint_weight_with_state(
            proto_array,
            checkpoint,
            fc_store,
            checkpoint_state.as_ref(),
        )?;

        // Calculate total FFG weight till current slot
        let ffg_weight_till_now =
            self.get_ffg_weight_till_slot(current_slot, current_epoch, total_active_balance);

        // Calculate remaining honest FFG weight
        let remaining_ffg_weight = total_active_balance - ffg_weight_till_now;
        let remaining_honest_ffg_weight =
            remaining_ffg_weight / 100 * (100 - self.config.beta_percentage);

        // Calculate minimum honest FFG support
        // **Python Specification**: min_honest_ffg_support = ffg_support_for_checkpoint - min(
        //     ffg_weight_till_now // 100 * CONFIRMATION_BYZANTINE_THRESHOLD,
        //     ffg_weight_till_now // 100 * CONFIRMATION_SLASHING_THRESHOLD,
        //     ffg_support_for_checkpoint
        // )
        let byzantine_weight = ffg_weight_till_now / 100 * self.config.beta_percentage;
        let slashing_weight = ffg_weight_till_now / 100 * self.config.slashing_percentage;
        let min_byzantine_weight = std::cmp::min(byzantine_weight, slashing_weight);
        let min_byzantine_weight = std::cmp::min(min_byzantine_weight, ffg_support_for_checkpoint);

        let min_honest_ffg_support = ffg_support_for_checkpoint - min_byzantine_weight;

        // **Python Specification**: 3 * (min_honest_ffg_support + remaining_honest_ffg_weight) >= 2 * total_active_balance
        let left_side = 3 * (min_honest_ffg_support + remaining_honest_ffg_weight);
        let right_side = 2 * total_active_balance;

        Ok(left_side >= right_side)
    }

    /// Checks if no conflicting checkpoint will be justified.
    ///
    /// **Python Specification**: `will_no_conflicting_checkpoint_be_justified(store, checkpoint)`
    ///
    /// This function checks if any conflicting checkpoint could be justified,
    /// ensuring that advancing confirmation is safe at epoch boundaries.
    /// It's a safety check for FFG confirmation logic.
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    /// * `fc_store` - The fork choice store containing current state
    /// * `head_root` - The current head block root
    ///
    /// # Returns
    /// * `Ok(bool)` - True if no conflicting checkpoint will be justified
    /// * `Err(Error)` - Error occurred during analysis
    fn will_no_conflicting_checkpoint_be_justified<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let current_slot = fc_store.get_current_slot();
        let current_epoch = current_slot.epoch(E::slots_per_epoch());

        // Get the checkpoint for the current epoch
        let checkpoint_root = self
            .get_checkpoint_block(proto_array, head_root, current_epoch)
            .unwrap_or(head_root);
        let checkpoint = Checkpoint {
            epoch: current_epoch,
            root: checkpoint_root,
        };

        // **Python Specification**: This function uses the same logic as will_current_epoch_checkpoint_be_justified
        // but with a different threshold: 3 * (min_honest_ffg_support + remaining_honest_ffg_weight) >= total_active_balance
        // instead of >= 2 * total_active_balance

        // Get the checkpoint state for analysis
        let checkpoint_state = match self.state_provider.get_checkpoint_state(&checkpoint) {
            Ok(Some(state)) => state,
            Ok(None) => {
                // If checkpoint state is not available, assume it won't be justified
                return Ok(false);
            }
            Err(_) => {
                // If we can't access the checkpoint state, assume it won't be justified
                return Ok(false);
            }
        };

        // Use TAB from the checkpoint state to avoid extra provider lookups.
        let total_active_balance = checkpoint_state.get_total_active_balance().map_err(|_| {
            crate::Error::ProtoArrayStringError(
                "Failed to get total active balance from checkpoint state".to_string(),
            )
        })?;

        // Calculate FFG support for the checkpoint using the checkpoint state
        let ffg_support_for_checkpoint = self.get_checkpoint_weight_with_state(
            proto_array,
            &checkpoint,
            fc_store,
            checkpoint_state.as_ref(),
        )?;

        // Calculate total FFG weight till current slot
        let ffg_weight_till_now =
            self.get_ffg_weight_till_slot(current_slot, current_epoch, total_active_balance);

        // Calculate remaining honest FFG weight
        let remaining_ffg_weight = total_active_balance - ffg_weight_till_now;
        let remaining_honest_ffg_weight =
            remaining_ffg_weight / 100 * (100 - self.config.beta_percentage);

        // Calculate minimum honest FFG support
        // **Python Specification**: min_honest_ffg_support = ffg_support_for_checkpoint - min(
        //     ffg_weight_till_now // 100 * CONFIRMATION_BYZANTINE_THRESHOLD,
        //     ffg_weight_till_now // 100 * CONFIRMATION_SLASHING_THRESHOLD,
        //     ffg_support_for_checkpoint
        // )
        let byzantine_weight = ffg_weight_till_now / 100 * self.config.beta_percentage;
        let slashing_weight = ffg_weight_till_now / 100 * self.config.slashing_percentage;
        let min_byzantine_weight = std::cmp::min(byzantine_weight, slashing_weight);
        let min_byzantine_weight = std::cmp::min(min_byzantine_weight, ffg_support_for_checkpoint);

        let min_honest_ffg_support = ffg_support_for_checkpoint - min_byzantine_weight;

        // **Python Specification**: 3 * (min_honest_ffg_support + remaining_honest_ffg_weight) >= total_active_balance
        // Note: This is different from will_current_epoch_checkpoint_be_justified which uses >= 2 * total_active_balance
        let left_side = 3 * (min_honest_ffg_support + remaining_honest_ffg_weight);
        let right_side = total_active_balance;

        Ok(left_side >= right_side)
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
        T: ForkChoiceStore<E>,
    {
        // Get the finalized block to determine the pruning boundary
        let Some(finalized_block) = proto_array.get_block(&finalized_root) else {
            // If finalized block not found, something is wrong
            return Err(ProtoArrayStringError(
                "Finalized block not found in proto array during FCR pruning".to_string(),
            ));
        };

        let before = self.meta.len();
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

        let after = self.meta.len();
        debug!(
            finalized = %finalized_root,
            pruned = (before as i64 - after as i64),
            remaining = after,
            "FCR: pruned side-table metadata"
        );

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
        T: ForkChoiceStore<E>,
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
            Ok(weight_per_slot * slots_covered.as_u64())
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
            Ok(self.adjust_committee_weight_estimate_to_ensure_safety(estimate))
            // 0.5% safety margin
        }
    }

    /// Adjusts committee weight estimates to ensure safety.
    ///
    /// **Python Specification**: `adjust_committee_weight_estimate_to_ensure_safety(estimate)`
    ///
    /// **Why Required**: Committee weight estimation can have small errors due to
    /// cross-epoch calculations and validator set changes. This function adds a
    /// small safety margin to ensure FCR safety guarantees are maintained even
    /// with estimation errors.
    ///
    /// **Specification**: Multiplies the estimate by `(1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR) / 1000`
    /// to add a small safety margin (0.5% for the default factor of 5).
    ///
    /// # Arguments
    /// * `estimate` - The raw committee weight estimate
    ///
    /// # Returns
    /// * `u64` - The adjusted estimate with safety margin
    fn adjust_committee_weight_estimate_to_ensure_safety(&self, estimate: u64) -> u64 {
        // Apply safety adjustment: estimate * (1000 + adjustment_factor) / 1000
        // This adds a small safety margin to ensure FCR safety guarantees
        estimate * (1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR) / 1000
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
        let slots_per_epoch = E::slots_per_epoch();
        let start_epoch = start_slot.epoch(slots_per_epoch);
        let end_epoch = end_slot.epoch(slots_per_epoch);

        // End epoch component
        let num_slots_in_end_epoch =
            end_slot.as_u64() - end_epoch.start_slot(slots_per_epoch).as_u64() + 1;
        let end_epoch_weight_estimate =
            (total_active_balance / slots_per_epoch).saturating_mul(num_slots_in_end_epoch);

        // Start epoch component (pro-rated)
        let num_slots_in_start_epoch = slots_per_epoch
            - (start_slot.as_u64() - start_epoch.start_slot(slots_per_epoch).as_u64());
        let remaining_slots_in_end_epoch =
            slots_per_epoch - (num_slots_in_end_epoch % slots_per_epoch);

        let start_epoch_weight_estimate =
            (total_active_balance / slots_per_epoch / slots_per_epoch)
                .saturating_mul(num_slots_in_start_epoch)
                .saturating_mul(remaining_slots_in_end_epoch);

        Ok(start_epoch_weight_estimate.saturating_add(end_epoch_weight_estimate))
    }
}
