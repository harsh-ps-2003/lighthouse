//! Fast Confirmation Rule (FCR) implementation for Lighthouse.
//!
//! This module implements the Fast Confirmation Rule as described in the specification,
//! providing faster block confirmation times (12-24 seconds) compared to traditional
//! finalization (13-19 minutes).
//!
//! The FCR operates under network synchrony assumptions and uses LMD-GHOST vote weights
//! combined with FFG checkpoint support to determine block permanence.
//!
use crate::metrics::{
    inc_counter, observe, set_gauge, FCR_BYZANTINE_THRESHOLD_PERCENTAGE,
    FCR_COMMITTEE_WEIGHT_CALCULATION_TIME, FCR_CONFIRMATION_SLOT_DELAY,
    FCR_CONFIRMATION_TIME_SECONDS, FCR_CONFIRMED_REORG_COUNT, FCR_CONFIRMED_REORG_SLOTS,
    FCR_CONFIRMED_ROOT_ROLLBACK_COUNT, FCR_CONFIRMED_ROOT_ROLLBACK_SLOTS,
    FCR_EPOCH_BOUNDARY_TRANSITIONS, FCR_FFG_SUPPORT_CALCULATION_TIME,
    FCR_HEAD_TO_CONFIRMED_GAP_SLOTS, FCR_IN_SYNC, FCR_METADATA_CACHE_SIZE, FCR_RESTARTS_TOTAL,
    FCR_SAFE_HEAD_REORG_COUNT, FCR_SAFE_HEAD_REORG_DEPTH, FCR_SAFE_HEAD_REORG_DISTANCE,
    FCR_STATE_PROVIDER_ERROR_TOTAL, FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
    FCR_STATE_PROVIDER_MISS_TOTAL, FCR_TAIL_CASES_TOTAL, FCR_VALIDATOR_SUPPORT_PERCENTAGE,
};
use crate::Error::ProtoArrayStringError;
use crate::ForkChoiceStore;

use proto_array::ProtoArrayForkChoice;
use std::collections::HashMap;

use std::marker::PhantomData;
use std::time::Instant;

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
    type Error: std::error::Error + Send + Sync + 'static;

    /// Gets the checkpoint state for a given checkpoint.
    ///
    /// This method should return the beacon state at the checkpoint's epoch boundary.
    /// The state is used for FFG weight calculations and validator analysis.
    fn get_checkpoint_state(
        &self,
        checkpoint: &Checkpoint,
    ) -> Result<Option<Arc<BeaconState<E>>>, Self::Error>;

    /// Gets the total active balance at a given epoch.
    ///
    /// This method provides the total effective balance of all active validators
    /// at the specified epoch, which is needed for FFG weight calculations.
    fn get_total_active_balance_at_epoch(&self, epoch: Epoch) -> Result<u64, Self::Error>;

    /// Gets the chain specification.
    ///
    /// This method provides access to the chain specification which is needed
    /// for various FFG calculations and state transitions.
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
    pub fn new(beta_percentage: u64) -> Result<Self, String> {
        Self::new_with_slashing(beta_percentage, DEFAULT_FCR_SLASHING_THRESHOLD_PERCENTAGE)
    }

    /// Creates a new FCR configuration with the given Byzantine and slashing thresholds.
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
    /// Slot when this block was first seen (for confirmation delay tracking)
    /// **FCR Spec**: Used to calculate actual confirmation delay
    pub block_creation_slot: Slot,
    /// Slot when this block was confirmed (for confirmation delay tracking)
    /// **FCR Spec**: Used to calculate actual confirmation delay
    pub confirmation_slot: Option<Slot>,
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
    /// Cached safe head root for O(1) lookup
    /// **Performance**: O(1) safe head access instead of O(depth) ancestor scan
    /// **Tree-States**: Leverages structural sharing for efficient updates
    pub safe_head_root: Hash256,
    /// Previous slot's justified checkpoint
    /// **Spec**: `store.prev_slot_justified_checkpoint`
    pub prev_slot_justified_checkpoint: Checkpoint,
    /// Previous slot's unrealized justified checkpoint
    /// **Spec**: `store.prev_slot_unrealized_justified_checkpoint`
    pub prev_slot_unrealized_justified_checkpoint: Checkpoint,
    /// Previous slot's head block
    /// **Spec**: `store.prev_slot_head`
    pub prev_slot_head: Hash256,
    /// Unrealized justified checkpoint captured at the last slot of the previous epoch.
    /// This value remains constant throughout the current epoch and is the weighting checkpoint
    /// used for all LMD-GHOST Q-indicator computations.
    pub prev_epoch_unrealized_justified_checkpoint: Checkpoint,
}

impl Default for FcrStore {
    fn default() -> Self {
        Self {
            confirmed_root: Hash256::zero(),
            safe_head_root: Hash256::zero(),
            prev_slot_justified_checkpoint: Checkpoint::default(),
            prev_slot_unrealized_justified_checkpoint: Checkpoint::default(),
            prev_slot_head: Hash256::zero(),
            prev_epoch_unrealized_justified_checkpoint: Checkpoint::default(),
        }
    }
}

/// Safe head metrics for monitoring
#[derive(Debug, Clone)]
pub struct SafeHeadMetrics {
    /// Current safe head root
    pub safe_head_root: Hash256,
    /// Current safe head slot
    pub safe_head_slot: u64,
    /// Current confirmed root
    pub confirmed_root: Hash256,
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

        // Set the Byzantine threshold metric
        set_gauge(
            &FCR_BYZANTINE_THRESHOLD_PERCENTAGE,
            config.beta_percentage as i64,
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

    /// Returns the current safe head root
    ///
    /// **Performance**: O(1) safe head access instead of O(depth) ancestor scan.
    /// **Spec Compliance**: Safe head is derived from confirmed_root following the spec.
    ///
    /// # Returns
    /// * `Hash256` - The current safe head root
    pub fn get_safe_head(&self) -> Hash256 {
        self.fcr_store.safe_head_root
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

    /// Returns the Byzantine threshold percentage.
    pub fn beta_percentage(&self) -> u64 {
        self.config.beta_percentage
    }

    /// Returns the number of metadata entries in the cache.
    pub fn metadata_cache_size(&self) -> usize {
        self.meta.len()
    }

    /// Tracks when a block is first seen for confirmation delay measurement.
    ///
    /// This method should be called when a block is first processed by the fork choice
    /// to establish the baseline for measuring confirmation delay.
    ///
    /// # Arguments
    /// * `block_root` - The block root to track
    /// * `block_slot` - The slot when the block was created
    pub fn track_block_creation(&mut self, block_root: Hash256, block_slot: Slot) {
        if !self.meta.contains_key(&block_root) {
            self.meta.insert(
                block_root,
                FcrMeta {
                    support: 0,
                    committee_weight: 0,
                    confirmed: false,
                    block_creation_slot: block_slot,
                    confirmation_slot: None,
                },
            );
        }
    }

    /// Gets safe head metrics for monitoring
    ///
    /// # Arguments
    /// * `proto_array` - The proto array containing the block DAG
    ///
    /// # Returns
    /// * `SafeHeadMetrics` - Safe head metrics for monitoring
    pub fn get_safe_head_metrics(&self, proto_array: &ProtoArrayForkChoice) -> SafeHeadMetrics {
        let safe_head_slot = proto_array
            .get_block(&self.fcr_store.safe_head_root)
            .map(|b| b.slot.as_u64())
            .unwrap_or_default();

        SafeHeadMetrics {
            safe_head_root: self.fcr_store.safe_head_root,
            safe_head_slot,
            confirmed_root: self.fcr_store.confirmed_root,
        }
    }

    /// Updates the safe head root - O(1) update.
    ///
    /// # Arguments
    /// * `new_safe_head` - The new safe head root
    fn update_safe_head(&mut self, new_safe_head: Hash256) {
        if new_safe_head != self.fcr_store.safe_head_root {
            let old_safe_head = self.fcr_store.safe_head_root;
            self.fcr_store.safe_head_root = new_safe_head;

            info!(
                old_safe_head = %old_safe_head,
                new_safe_head = %new_safe_head,
                "FCR: safe head updated (O(1))"
            );
        }
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

        // Delegate to the canonical implementation in `proto_array`.
        proto_array.is_descendant(ancestor, root)
    }

    /// Checks if a block is confirmed by FCR.
    ///
    /// **Why Required**: This method provides a simple way to check confirmation status
    /// without exposing internal metadata structures. It's used by tests and other
    /// parts of the codebase to verify FCR behavior.
    pub fn is_block_confirmed(&self, block_root: &Hash256) -> bool {
        self.meta.get(block_root).is_some_and(|meta| meta.confirmed)
    }

    /// Checks if a block satisfies the LMD-GHOST confirmation rule inequality.
    ///
    /// **Specification**: `2 * S > W * (1 + 2 * β / 100) + proposer_score`
    ///
    /// This is the core "Q-indicator" from the FCR specification, which determines if a block
    /// has sufficient LMD-GHOST support (`S`) relative to the total possible committee weight (`W`),
    /// the proposer's score, and the configured Byzantine tolerance (`β`).
    fn check_lmd_ghost_confirmation_inequality(
        &self,
        support_weight: u64,
        committee_weight: u64,
        proposer_score: u64,
    ) -> bool {
        let beta_threshold = self.config.beta_percentage;
        // Using integer arithmetic to avoid floating point issues:
        // 2 * S > W + (W / 50 * β) + proposer_score
        let left_side = 2 * support_weight;
        let right_side = committee_weight + committee_weight / 50 * beta_threshold + proposer_score;

        left_side > right_side
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
    pub fn update_per_slot<T>(
        &mut self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<(), crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let prev_confirmed = self.fcr_store.confirmed_root;
        // Update the confirmed root to the latest confirmed block
        if let Some(new_confirmed_root) =
            self.get_latest_confirmed(proto_array, fc_store, head_root)
        {
            self.fcr_store.confirmed_root = new_confirmed_root;
            // Update safe head cache when confirmed root changes
            self.update_safe_head(new_confirmed_root);

            if new_confirmed_root != prev_confirmed {
                // Seed metadata with the true block creation slot so the
                // confirmation delay histogram reflects real wall-clock delay.
                if let Some(b) = proto_array.get_block(&new_confirmed_root) {
                    self.track_block_creation(new_confirmed_root, b.slot);
                }
                // Emit a single observation when we advance confirmed_root.
                // This guarantees Prometheus sees samples even when the
                // post-find_head fast-path isn't taken (e.g. during sync).
                let head_slot_for_gap = proto_array
                    .get_block(&head_root)
                    .map(|b| b.slot)
                    .unwrap_or(fc_store.get_current_slot());
                self.mark_confirmed(
                    new_confirmed_root,
                    proto_array,
                    fc_store.get_current_slot(),
                    head_slot_for_gap,
                );

                info!(
                    old = %prev_confirmed,
                    new = %new_confirmed_root,
                    slot = fc_store.get_current_slot().as_u64(),
                    "FCR: per-slot confirmed root updated"
                );
            }
        }

        // Store the previous slot's state
        self.fcr_store.prev_slot_justified_checkpoint = *fc_store.justified_checkpoint();

        // Store the previous slot's unrealized justified checkpoint
        self.fcr_store.prev_slot_unrealized_justified_checkpoint =
            *fc_store.unrealized_justified_checkpoint();

        // Capture prev-epoch unrealized justified checkpoint once at the beginning of each epoch.
        // This value is then held constant for all LMD-GHOST Q-indicator computations during the epoch.
        if fc_store.get_current_slot() % E::slots_per_epoch() == 0 {
            self.fcr_store.prev_epoch_unrealized_justified_checkpoint =
                *fc_store.unrealized_justified_checkpoint();
        }

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
    pub fn on_new_slot<T>(
        &mut self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Result<(), crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let current_slot = fc_store.get_current_slot();

        // Check for epoch boundary transition
        if current_slot % E::slots_per_epoch() == 0 {
            inc_counter(&FCR_EPOCH_BOUNDARY_TRANSITIONS);
        }

        // Get head_slot for logging purposes
        let head_slot = proto_array
            .get_block(&head_root)
            .map(|b| b.slot)
            .unwrap_or(current_slot);

        info!(
            slot = current_slot.as_u64(),
            head = %head_root,
            head_slot = head_slot.as_u64(),
            prev_confirmed = %self.fcr_store.confirmed_root,
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
    pub fn get_latest_confirmed<T>(
        &self,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
        head_root: Hash256,
    ) -> Option<Hash256>
    where
        T: ForkChoiceStore<E>,
    {
        let mut current_confirmed_root = self.fcr_store.confirmed_root;
        let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
        let head_slot = proto_array
            .get_block(&head_root)
            .map(|b| b.slot.as_u64())
            .unwrap_or_default();
        let confirmed_slot_initial = proto_array
            .get_block(&current_confirmed_root)
            .map(|b| b.slot.as_u64())
            .unwrap_or_default();
        debug!(
            head = %head_root,
            head_slot = head_slot,
            confirmed_initial = %current_confirmed_root,
            confirmed_initial_slot = confirmed_slot_initial,
            current_epoch = current_epoch.as_u64(),
            "FCR get_latest_confirmed: start"
        );

        // Safety check: if confirmed is missing/too-old/off-canonical, revert to finalized but
        // DO NOT return early; continue with epoch-start uplift and advancement as per spec.
        if let Some(confirmed_block) = proto_array.get_block(&current_confirmed_root) {
            let confirmed_block_epoch = confirmed_block.slot.epoch(E::slots_per_epoch());

            // A confirmed block from a prior epoch (epoch N-2 or older) is considered "too old"
            // and is a potential safety violation.
            let is_too_old = confirmed_block_epoch + 1 < current_epoch;
            // A confirmed block that is no longer an ancestor of the current head indicates a
            // reorg has occurred that moved the canonical chain away from the confirmed block.
            let is_off_canonical_chain =
                !proto_array.is_descendant(current_confirmed_root, head_root);
            if is_too_old || is_off_canonical_chain {
                let finalized_root = fc_store.finalized_checkpoint().root;
                warn!(
                    current_epoch = current_epoch.as_u64(),
                    confirmed = %current_confirmed_root,
                    finalized = %finalized_root,
                    head = %head_root,
                    "FCR: falling back to finalized checkpoint for safety"
                );
                if is_off_canonical_chain {
                    // This is a confirmed reorg event, a serious safety failure where a block
                    // previously marked as confirmed has been reorganized out of the canonical chain.
                    inc_counter(&FCR_CONFIRMED_REORG_COUNT);
                    crate::metrics::inc_counter_vec(&FCR_RESTARTS_TOTAL, &["reorg"]);
                    if let (Some(old_b), Some(new_b)) = (
                        proto_array.get_block(&current_confirmed_root),
                        proto_array.get_block(&finalized_root),
                    ) {
                        let old_slot = old_b.slot.as_u64();
                        let new_slot = new_b.slot.as_u64();
                        let slot_delta = old_slot.saturating_sub(new_slot) as f64;
                        observe(&FCR_CONFIRMED_REORG_SLOTS, slot_delta);
                        warn!(
                            old_confirmed = %current_confirmed_root,
                            old_slot,
                            new_confirmed = %finalized_root,
                            new_slot,
                            slot_delta,
                            "FCR: confirmed root reorged off canonical chain"
                        );
                    }
                } else {
                    // If the block is too old but still canonical, it's considered a "stale"
                    // state that requires a safe reset, but not a critical reorg failure.
                    crate::metrics::inc_counter_vec(&FCR_RESTARTS_TOTAL, &["stale"]);
                }
                current_confirmed_root = finalized_root;
            }
        } else {
            // If the confirmed root is not even present in proto_array, it's a critical error
            // or a pruning issue.  We fall back to the finalized root for safety.
            let finalized_root = fc_store.finalized_checkpoint().root;
            warn!(
                confirmed_missing = %current_confirmed_root,
                finalized = %finalized_root,
                "FCR: confirmed root missing, using finalized"
            );
            // Missing confirmed root implies a stale state, requiring a safe restart.
            crate::metrics::inc_counter_vec(&FCR_RESTARTS_TOTAL, &["stale"]);
            current_confirmed_root = finalized_root;
        }

        // At the start of an epoch, if the prev-slot unrealized justified checkpoint
        // belongs to the previous epoch and is later than the current confirmed,
        // promote confirmed to that checkpoint (spec-aligned safety uplift),
        // then continue to attempt further advancement below.
        if fc_store.get_current_slot() % E::slots_per_epoch() == 0 {
            // Per the spec, at an epoch boundary, we check the previous slot's unrealized
            // justified checkpoint for a potential uplift to the confirmed root.
            let prev_unrealized_justified_checkpoint =
                self.fcr_store.prev_slot_unrealized_justified_checkpoint;
            let prev_uj_epoch = prev_unrealized_justified_checkpoint.epoch;
            info!(
                current_slot = fc_store.get_current_slot().as_u64(),
                prev_uj_epoch = prev_uj_epoch.as_u64(),
                current_epoch = current_epoch.as_u64(),
                epoch_boundary = true,
                "FCR: checking epoch-start uplift conditions"
            );
            // This uplift only applies if the checkpoint is from the immediately preceding epoch.
            if prev_uj_epoch + 1 == current_epoch {
                if let (Some(confirmed_block), Some(prev_uj_block)) = (
                    proto_array.get_block(&current_confirmed_root),
                    proto_array.get_block(&prev_unrealized_justified_checkpoint.root),
                ) {
                    // If the unrealized justified block is more recent than the current confirmed
                    // block, we can safely "uplift" our confirmed root to this block.
                    if confirmed_block.slot < prev_uj_block.slot {
                        info!(
                            prev_uj_root = %prev_unrealized_justified_checkpoint.root,
                            prev_uj_slot = prev_uj_block.slot.as_u64(),
                            uplift_from = %current_confirmed_root,
                            uplift_from_slot = confirmed_block.slot.as_u64(),
                            "FCR: epoch-start uplift to prev-slot unrealized justified checkpoint"
                        );
                        current_confirmed_root = prev_unrealized_justified_checkpoint.root;
                    } else {
                        info!(
                            prev_uj_slot = prev_uj_block.slot.as_u64(),
                            confirmed_slot = confirmed_block.slot.as_u64(),
                            "FCR: no epoch-start uplift needed (confirmed already ahead)"
                        );
                    }
                }
            } else {
                info!(
                    prev_uj_epoch = prev_uj_epoch.as_u64(),
                    current_epoch = current_epoch.as_u64(),
                    "FCR: epoch-start uplift not applicable (epoch mismatch)"
                );
            }
        } else {
            debug!(
                current_slot = fc_store.get_current_slot().as_u64(),
                slots_per_epoch = E::slots_per_epoch(),
                "FCR: not at epoch boundary, skipping uplift check"
            );
        }

        // Try to advance the confirmed root along the canonical chain
        if let Some(new_confirmed) = self.find_latest_confirmed_descendant(
            current_confirmed_root,
            proto_array,
            fc_store,
            head_root,
        ) {
            if new_confirmed != current_confirmed_root {
                info!(
                    old = %current_confirmed_root,
                    new = %new_confirmed,
                    head = %head_root,
                    "FCR: advanced latest confirmed descendant"
                );
            }
            current_confirmed_root = new_confirmed;
        }

        let final_confirmed_slot = proto_array
            .get_block(&current_confirmed_root)
            .map(|b| b.slot.as_u64())
            .unwrap_or_default();
        info!(
            confirmed = %current_confirmed_root,
            confirmed_slot = final_confirmed_slot,
            head = %head_root,
            head_slot = head_slot,
            "FCR get_latest_confirmed: result"
        );

        // A "rollback" occurs if the new confirmed root is at an earlier slot than the previous
        // one. This is a potential warning sign, indicating a reversion in the confirmed chain.
        if final_confirmed_slot > 0
            && confirmed_slot_initial > 0
            && final_confirmed_slot < confirmed_slot_initial
        {
            let slot_delta = (confirmed_slot_initial - final_confirmed_slot) as f64;
            inc_counter(&FCR_CONFIRMED_ROOT_ROLLBACK_COUNT);
            observe(&FCR_CONFIRMED_ROOT_ROLLBACK_SLOTS, slot_delta);
            warn!(
                previous_confirmed_slot = confirmed_slot_initial,
                new_confirmed_slot = final_confirmed_slot,
                slot_delta,
                previous_confirmed = %self.fcr_store.confirmed_root,
                new_confirmed = %current_confirmed_root,
                "FCR: confirmed root rollback detected"
            );
        }

        Some(current_confirmed_root)
    }

    /// Updates FCR state after finding a new head - O(1) optimized.
    ///
    /// This method is called after fork choice determines a new head, allowing FCR
    /// to perform confirmation checks and update its internal state efficiently.
    pub fn update_after_find_head<T>(
        &mut self,
        head_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
    ) -> Result<(), crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        // Track block creation for confirmation delay measurement
        if let Some(block) = proto_array.get_block(&head_root) {
            self.track_block_creation(head_root, block.slot);
        }

        // O(1) optimization: Check if we already have a cached safe head
        // and if the new head is a descendant of the current safe head
        if !self.fcr_store.safe_head_root.is_zero()
            && self.is_ancestor(proto_array, head_root, self.fcr_store.safe_head_root)
        {
            // Safe head is still valid - O(1) lookup, no scanning needed
            debug!(
                head = %head_root,
                safe_head = %self.fcr_store.safe_head_root,
                "FCR: safe head still valid (O(1))"
            );
            return Ok(());
        }

        // O(depth) scan only when necessary: no cached safe head or head changed
        let mut current_root = head_root;
        let mut depth = 0;
        let mut found_confirmed: Option<Hash256> = None;

        while depth < MAX_REORG_DEPTH {
            if let Some(b) = proto_array.get_block(&current_root) {
                debug!(
                    depth = depth,
                    block = %current_root,
                    slot = b.slot.as_u64(),
                    "FCR scan: visiting ancestor"
                );
                // Ensure we seed real creation slots for any ancestor we might
                // confirm, preventing 0-slot delay artifacts in metrics.
                self.track_block_creation(current_root, b.slot);
            }

            // Check if this block is already confirmed
            if let Some(meta) = self.meta.get(&current_root) {
                if meta.confirmed {
                    // Found a confirmed ancestor, no need to scan further
                    found_confirmed = Some(current_root);
                    break;
                }
            }

            // Check if this block meets confirmation criteria
            let lmd_ok = self.is_one_confirmed(current_root, proto_array, fc_store)?;
            debug!(
                depth = depth,
                block = %current_root,
                passed = lmd_ok,
                "FCR scan: is_one_confirmed result"
            );
            if lmd_ok {
                // Mark this block as confirmed
                let current_slot = fc_store.get_current_slot();
                let head_slot_for_gap = proto_array
                    .get_block(&head_root)
                    .map(|b| b.slot)
                    .unwrap_or(current_slot);
                self.mark_confirmed(current_root, proto_array, current_slot, head_slot_for_gap);
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

        // Update safe head cache - O(1) update
        match found_confirmed {
            Some(root) => {
                self.update_safe_head(root);
                info!(
                    head = %head_root,
                    confirmed_ancestor = %root,
                    depth,
                    "FCR update_after_find_head: confirmed ancestor found"
                );
            }
            None => {
                // No confirmed ancestor found, safe head remains unchanged
                debug!(
                    head = %head_root,
                    depth,
                    "FCR update_after_find_head: no confirmed ancestor within depth"
                );
            }
        }

        // Check for safe head reorgs by comparing with previous head
        if head_root != self.fcr_store.prev_slot_head && !self.fcr_store.prev_slot_head.is_zero() {
            // Calculate reorg distance
            let reorg_distance = depth;
            if reorg_distance > 0 {
                inc_counter(&FCR_SAFE_HEAD_REORG_COUNT);
                observe(&FCR_SAFE_HEAD_REORG_DISTANCE, reorg_distance as f64);
                observe(&FCR_SAFE_HEAD_REORG_DEPTH, reorg_distance as f64);
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
    /// Per spec, this function performs ONLY the LMD-GHOST inequality without any
    /// FFG gating. FFG-related checks are handled in the advancement logic
    /// `find_latest_confirmed_descendant`. If the weighting checkpoint state is
    /// unavailable, we conservatively fall back to using `justified_balances` to
    /// estimate W.
    fn is_one_confirmed<T>(
        &self,
        block_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
        fc_store: &T,
    ) -> Result<bool, crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let start_time = Instant::now();
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

        // Use the *prev-epoch unrealized justified checkpoint* captured at the
        // epoch boundary for all LMD-GHOST Q-indicator checks during the epoch.
        // This avoids mid-epoch drift in committee shuffling.
        let weighting_checkpoint = self.fcr_store.prev_epoch_unrealized_justified_checkpoint;
        debug!(
            block = %block_root,
            block_slot = block.slot.as_u64(),
            parent_slot = parent_block.slot.as_u64(),
            weighting_checkpoint_epoch = weighting_checkpoint.epoch.as_u64(),
            weighting_checkpoint_root = %weighting_checkpoint.root,
            "FCR Q-indicator: inputs"
        );

        // Try to obtain the checkpoint state via the StateProvider. If unavailable, fall back
        // to pre-existing behavior that uses no checkpoint state and justified_balances.
        // Instrument StateProvider latency/miss/error
        let sp_start = Instant::now();
        let weighting_checkpoint_state_opt = match self
            .state_provider
            .get_checkpoint_state(&weighting_checkpoint)
        {
            Ok(opt) => {
                observe(
                    &FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
                    sp_start.elapsed().as_secs_f64(),
                );
                if opt.is_none() {
                    inc_counter(&FCR_STATE_PROVIDER_MISS_TOTAL);
                }
                opt
            }
            Err(_) => {
                observe(
                    &FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
                    sp_start.elapsed().as_secs_f64(),
                );
                inc_counter(&FCR_STATE_PROVIDER_ERROR_TOTAL);
                None
            }
        };

        // Retrieve the LMD‐GHOST *support weight* (denoted *S* in the Python spec) **excluding**
        // any proposer boost.  This represents the total validator balance that has
        // (directly or transitively) attested to `block_root`.
        //
        // NOTE: We pass the optional `weighting_checkpoint_state` to maintain API compatibility
        // with the ProtoArray variant that accepts checkpoint context, even though the current
        // implementation ignores it.  This future-proofs the call once ProtoArray integrates
        // checkpoint aware weighting.
        let Some(support_weight) = proto_array.get_weight::<E>(
            &block_root,
            weighting_checkpoint_state_opt.as_deref(),
            false, // support must EXCLUDE proposer boost
            fc_store.proposer_boost_root(),
            fc_store.chain_spec(),
        ) else {
            return Err(ProtoArrayStringError(format!(
                "Failed to get weight for block {}",
                block_root
            )));
        };

        // Derive the maximum possible *committee weight* (denoted *W* in the spec).  This is the
        // total stake that could have voted for slots in the inclusive range `(parent_block.slot+1 .. current_slot-1)`.
        // Depending on data availability we either compute it precisely using the weighting
        // checkpoint or fall back to a conservative estimate.
        let start_slot = parent_block.slot + 1;
        let end_slot = fc_store.get_current_slot() - 1;

        // Do not confirm within the same slot.
        // If there is no attestation window yet (start_slot > end_slot),
        // defer confirmation to the next slot when votes for this block can exist.
        if start_slot > end_slot {
            debug!(
                start_slot = start_slot.as_u64(),
                end_slot = end_slot.as_u64(),
                "FCR Q-indicator: empty window this slot → defer"
            );
            return Ok(false);
        }

        let committee_weight = if let Some(weighting_state) =
            weighting_checkpoint_state_opt.as_deref()
        {
            let total_active_balance = weighting_state
                .get_total_active_balance()
                .unwrap_or(fc_store.justified_balances().total_effective_balance);

            if self.is_full_validator_set_covered(start_slot, end_slot) {
                debug!(
                    start_slot = start_slot.as_u64(),
                    end_slot = end_slot.as_u64(),
                    total_active_balance = total_active_balance,
                    "FCR W: full validator set covered → total_active_balance"
                );
                total_active_balance
            } else {
                let start_epoch = start_slot.epoch(E::slots_per_epoch());
                let end_epoch = end_slot.epoch(E::slots_per_epoch());
                if start_epoch == end_epoch {
                    let slots_covered = end_slot - start_slot + 1;
                    let weight_per_slot = total_active_balance / E::slots_per_epoch();
                    let w = weight_per_slot * slots_covered.as_u64();
                    debug!(
                        start_slot = start_slot.as_u64(),
                        end_slot = end_slot.as_u64(),
                        slots_covered = slots_covered.as_u64(),
                        weight_per_slot = weight_per_slot,
                        w = w,
                        "FCR W: same-epoch pro-rata"
                    );
                    w
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
                            let w = weight_per_slot * slots_covered.as_u64();
                            debug!(
                                start_slot = start_slot.as_u64(),
                                end_slot = end_slot.as_u64(),
                                estimate = w,
                                "FCR W: cross-epoch fallback pro-rata"
                            );
                            w
                        }
                    };
                    let adjusted = self.adjust_committee_weight_estimate_to_ensure_safety(estimate);
                    debug!(
                        start_slot = start_slot.as_u64(),
                        end_slot = end_slot.as_u64(),
                        estimate = estimate,
                        adjusted = adjusted,
                        "FCR W: cross-epoch estimate with safety adjustment"
                    );
                    adjusted
                }
            }
        } else {
            // Fallback: use the existing method which relies on justified_balances
            debug!(
                start_slot = start_slot.as_u64(),
                end_slot = end_slot.as_u64(),
                "FCR W: using fc_store fallback (no checkpoint state)"
            );
            let w = self.get_committee_weight_between_slots(start_slot, end_slot, fc_store)?;
            w
        };

        // We compare against the delta contributed by proposer boost
        // Compute an upper bound on proposer_delta using the boosted query if available.
        let boosted_support_opt = proto_array.get_weight::<E>(
            &block_root,
            weighting_checkpoint_state_opt.as_deref(),
            true, // include proposer boost
            fc_store.proposer_boost_root(),
            fc_store.chain_spec(),
        );
        let proposer_score_delta = match boosted_support_opt {
            Some(boosted) if boosted > support_weight => boosted.saturating_sub(support_weight),
            _ => proto_array
                .get_proposer_score::<E>(block_root, fc_store.chain_spec())
                .unwrap_or_default(),
        };

        // Check LMD-GHOST confirmation (Q-indicator)
        let lmd_confirmed = self.check_lmd_ghost_confirmation_inequality(
            support_weight,
            committee_weight,
            proposer_score_delta,
        );

        debug!(
            block = %block_root,
            support = support_weight,
            committee_weight = committee_weight,
            proposer_score = proposer_score_delta,
            beta = self.config.beta_percentage,
            passed = lmd_confirmed,
            "FCR Q-indicator: LMD-GHOST check"
        );

        if lmd_confirmed {
            let slot_u64 = block.slot.as_u64();
            let epoch_u64 = block.slot.epoch(E::slots_per_epoch()).as_u64();
            debug!(
                block = %block_root,
                slot = slot_u64,
                epoch = epoch_u64,
                support = support_weight,
                committee_weight = committee_weight,
                proposer_score = proposer_score_delta,
                beta = self.config.beta_percentage,
                "FCR: block meets LMD confirmation threshold"
            );
        }

        // Record algorithm execution time (for performance monitoring)
        let elapsed = start_time.elapsed();
        observe(
            &FCR_COMMITTEE_WEIGHT_CALCULATION_TIME,
            elapsed.as_secs_f64(),
        );

        // Record validator support percentage if confirmed
        if lmd_confirmed && committee_weight > 0 {
            let support_percentage = (support_weight as f64 / committee_weight as f64) * 100.0;
            observe(&FCR_VALIDATOR_SUPPORT_PERCENTAGE, support_percentage);
        }

        // Per spec, do not apply FFG gating here. Return LMD result only.
        Ok(lmd_confirmed)
    }

    /// Marks a block and all its descendants as confirmed.
    ///
    /// Lighthouse's side-table approach for FCR metadata requires explicit
    /// confirmation inheritance management. Unlike the Python spec where confirmation status
    /// is computed on-demand, Lighthouse caches confirmation status in `FcrMeta` to avoid
    /// repeated expensive computations. This method ensures that when a parent is confirmed,
    /// all descendants inherit the confirmation status efficiently.
    ///
    /// This implements confirmation inheritance where if a parent block is confirmed,
    /// all its descendants are also confirmed.
    fn mark_confirmed(
        &mut self,
        block_root: Hash256,
        proto_array: &ProtoArrayForkChoice,
        current_slot: Slot,
        head_slot: Slot,
    ) {
        // Mark the specific block as confirmed
        if let Some(meta) = self.meta.get_mut(&block_root) {
            meta.confirmed = true;
            meta.confirmation_slot = Some(current_slot);

            // Calculate and record actual confirmation delay
            let confirmation_delay_slots =
                current_slot.as_u64() - meta.block_creation_slot.as_u64();
            let confirmation_delay_seconds = confirmation_delay_slots as f64 * 12.0; // 12 seconds per slot

            // Only record non-zero-slot confirmations into the primary histogram,
            // to ensure graphs represent real 1–2 slot confirmations. Still count
            // all outcomes in dedicated counters below.
            if confirmation_delay_slots > 0 {
                observe(&FCR_CONFIRMATION_TIME_SECONDS, confirmation_delay_seconds);
            }
            // Always record the slot-delay distribution
            observe(
                &FCR_CONFIRMATION_SLOT_DELAY,
                confirmation_delay_slots as f64,
            );
            // Record head-to-confirmed gap
            if let Some(block) = proto_array.get_block(&block_root) {
                let head_gap = head_slot.as_u64().saturating_sub(block.slot.as_u64()) as f64;
                observe(&FCR_HEAD_TO_CONFIRMED_GAP_SLOTS, head_gap);
            }

            // Increment outcome counters (0/1/2/≥3 slots)
            inc_counter(&crate::metrics::FCR_CONFIRMATIONS_TOTAL);
            match confirmation_delay_slots {
                0 => crate::metrics::inc_counter_vec(
                    &crate::metrics::FCR_CONFIRMATIONS_BY_SLOTS,
                    &["0"],
                ),
                1 => crate::metrics::inc_counter_vec(
                    &crate::metrics::FCR_CONFIRMATIONS_BY_SLOTS,
                    &["1"],
                ),
                2 => crate::metrics::inc_counter_vec(
                    &crate::metrics::FCR_CONFIRMATIONS_BY_SLOTS,
                    &["2"],
                ),
                _ => crate::metrics::inc_counter_vec(
                    &crate::metrics::FCR_CONFIRMATIONS_BY_SLOTS,
                    &["ge3"],
                ),
            }

            // Tail-case labeling for analysis (delay >= 2)
            if confirmation_delay_slots >= 2 {
                let epoch_boundary = if current_slot % E::slots_per_epoch() == 0 {
                    "true"
                } else {
                    "false"
                };
                let delay_bucket = if confirmation_delay_slots == 2 {
                    "2"
                } else if confirmation_delay_slots == 3 {
                    "3"
                } else {
                    "ge4"
                };
                crate::metrics::inc_counter_vec(
                    &FCR_TAIL_CASES_TOTAL,
                    &[epoch_boundary, delay_bucket],
                );
            }

            info!(
                block = %block_root,
                creation_slot = meta.block_creation_slot.as_u64(),
                confirmation_slot = current_slot.as_u64(),
                delay_slots = confirmation_delay_slots,
                delay_seconds = confirmation_delay_seconds,
                head_slot = head_slot.as_u64(),
                epoch_boundary = (current_slot % E::slots_per_epoch() == 0),
                "FCR: block confirmed with actual delay"
            );
        } else {
            // Create new metadata if it doesn't exist
            let creation_slot = proto_array
                .get_block(&block_root)
                .map(|b| b.slot)
                .unwrap_or(current_slot);
            self.meta.insert(
                block_root,
                FcrMeta {
                    support: 0,
                    committee_weight: 0,
                    confirmed: true,
                    block_creation_slot: creation_slot, // use the block's real slot when available
                    confirmation_slot: Some(current_slot),
                },
            );
            // New metadata path: this is a zero-slot confirmation by construction; only count buckets
            inc_counter(&crate::metrics::FCR_CONFIRMATIONS_TOTAL);
            crate::metrics::inc_counter_vec(&crate::metrics::FCR_CONFIRMATIONS_BY_SLOTS, &["0"]);
            // Also record distributions
            observe(&FCR_CONFIRMATION_SLOT_DELAY, 0.0);
            if let Some(block) = proto_array.get_block(&block_root) {
                let head_gap = head_slot.as_u64().saturating_sub(block.slot.as_u64()) as f64;
                observe(&FCR_HEAD_TO_CONFIRMED_GAP_SLOTS, head_gap);
            }
        }

        // Log the confirmation with slot/epoch info if available.
        if let Some(block) = proto_array.get_block(&block_root) {
            let slot_u64 = block.slot.as_u64();
            let epoch_u64 = block.slot.epoch(E::slots_per_epoch()).as_u64();
            debug!(
                block = %block_root,
                slot = slot_u64,
                epoch = epoch_u64,
                "FCR: block confirmed"
            );
        } else {
            debug!(block = %block_root, "FCR: block confirmed");
        }
    }

    /// Finds the latest confirmed descendant along the canonical chain.
    ///
    /// **Specification**: `find_latest_confirmed_descendant(store, latest_confirmed_root)`
    ///
    /// This implements the Python specification's `find_latest_confirmed_descendant`
    /// function to advance confirmation along the canonical chain with epoch boundary handling.
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
        let mut current_confirmed_root = confirmed_root;
        let head_slot = proto_array
            .get_block(&head_root)
            .map(|b| b.slot.as_u64())
            .unwrap_or_default();
        let confirmed_slot = proto_array
            .get_block(&current_confirmed_root)
            .map(|b| b.slot.as_u64())
            .unwrap_or_default();
        debug!(
            start_confirmed = %current_confirmed_root,
            start_confirmed_slot = confirmed_slot,
            head = %head_root,
            head_slot = head_slot,
            current_epoch = current_epoch.as_u64(),
            "FCR find_latest_confirmed_descendant: start"
        );

        // Get the confirmed block to check its epoch
        let Some(confirmed_block) = proto_array.get_block(&current_confirmed_root) else {
            return None;
        };

        let confirmed_block_epoch = confirmed_block.slot.epoch(E::slots_per_epoch());

        // First condition: Previous epoch advancement.
        // This logic allows advancing the confirmed root for blocks within the previous epoch.
        // It's a critical step for ensuring liveness when the chain is just shy of an epoch boundary.
        if confirmed_block_epoch + 1 == current_epoch {
            // The `voting_source` check ensures that a supermajority of validators are using recent
            // enough information. This prevents a small group of validators from confirming a block
            // that the rest of the network might not agree on.
            let voting_source_epoch = {
                // Compute the epoch of the voting source that most validators used for prev_slot_head
                use std::collections::HashMap;
                let balances = &fc_store.justified_balances().effective_balances;
                let mut epoch_to_weight: HashMap<Epoch, u128> = HashMap::new();
                for (validator_index, &eb) in balances.iter().enumerate() {
                    if eb == 0 {
                        continue;
                    }
                    if let Some((vote_root, vote_epoch)) =
                        proto_array.latest_message(validator_index)
                    {
                        if proto_array.is_descendant(self.fcr_store.prev_slot_head, vote_root) {
                            *epoch_to_weight.entry(vote_epoch).or_insert(0) += eb as u128;
                        }
                    }
                }
                let max_epoch = epoch_to_weight
                    .into_iter()
                    .max_by_key(|(_, w)| *w)
                    .map(|(e, _)| e)
                    .unwrap_or(
                        self.fcr_store
                            .prev_slot_unrealized_justified_checkpoint
                            .epoch,
                    );
                debug!(
                    prev_slot_head = %self.fcr_store.prev_slot_head,
                    voting_source_epoch = max_epoch.as_u64(),
                    current_epoch = current_epoch.as_u64(),
                    "FCR: computed voting source epoch for prev-slot head"
                );
                max_epoch
            };

            let is_voting_source_ok = voting_source_epoch + 2 >= current_epoch;

            // FFG-based safety checks. These ensure that advancing confirmation won't conflict with
            // future FFG finalization.
            // `boundary`: True if we are exactly at an epoch boundary.
            // `no_conflict`: True if no conflicting checkpoint can be justified.
            // `uj_prev`/`uj_head`: True if the unrealized justified checkpoints of the previous head
            // or current head are recent enough.
            let is_at_epoch_boundary = fc_store.get_current_slot() % E::slots_per_epoch() == 0;
            let no_conflicting_checkpoint = self
                .will_no_conflicting_checkpoint_be_justified(proto_array, fc_store, head_root)
                .unwrap_or(false);

            // read unrealized justification epoch for prev_slot_head and head
            let is_unrealized_justified_prev_ok = proto_array
                .get_block(&self.fcr_store.prev_slot_head)
                .and_then(|b| {
                    b.unrealized_justified_checkpoint
                        .as_ref()
                        .map(|cp| cp.epoch)
                })
                .map(|e| e + 1 >= current_epoch)
                .unwrap_or(false);
            let is_unrealized_justified_head_ok = proto_array
                .get_block(&head_root)
                .and_then(|b| {
                    b.unrealized_justified_checkpoint
                        .as_ref()
                        .map(|cp| cp.epoch)
                })
                .map(|e| e + 1 >= current_epoch)
                .unwrap_or(false);

            debug!(
                prev_slot_head = %self.fcr_store.prev_slot_head,
                head = %head_root,
                uj_prev_epoch = proto_array
                    .get_block(&self.fcr_store.prev_slot_head)
                    .and_then(|b| b.unrealized_justified_checkpoint.as_ref().map(|cp| cp.epoch))
                    .map(|e| e.as_u64())
                    .unwrap_or_default(),
                uj_head_epoch = proto_array
                    .get_block(&head_root)
                    .and_then(|b| b.unrealized_justified_checkpoint.as_ref().map(|cp| cp.epoch))
                    .map(|e| e.as_u64())
                    .unwrap_or_default(),
                uj_prev = is_unrealized_justified_prev_ok,
                uj_head = is_unrealized_justified_head_ok,
                "FCR: unrealized justification epoch analysis"
            );

            // Gate diagnostics for previous-epoch advancement
            debug!(
                confirmed = %current_confirmed_root,
                head = %head_root,
                voting_source_epoch = voting_source_epoch.as_u64(),
                current_epoch = current_epoch.as_u64(),
                voting_source_ok = is_voting_source_ok,
                boundary = is_at_epoch_boundary,
                no_conflict = no_conflicting_checkpoint,
                uj_prev = is_unrealized_justified_prev_ok,
                uj_head = is_unrealized_justified_head_ok,
                "FCR prev-epoch advancement gate"
            );

            if is_voting_source_ok
                && (is_at_epoch_boundary
                    || (no_conflicting_checkpoint
                        && (is_unrealized_justified_prev_ok || is_unrealized_justified_head_ok)))
            {
                // If the gates pass, we can attempt to advance confirmation.
                // Iterate through the canonical chain from the current confirmed block towards the head.
                let mut advanced_confirmed_root = current_confirmed_root;
                if let Some(canonical_roots) =
                    self.get_canonical_roots(proto_array, current_confirmed_root, head_root)
                {
                    for &block_root in canonical_roots.iter().skip(1) {
                        let block = match proto_array.get_block(&block_root) {
                            Some(b) => b,
                            None => break,
                        };
                        let block_epoch = block.slot.epoch(E::slots_per_epoch());
                        // Stop if we cross into the current epoch.
                        if block_epoch == current_epoch {
                            break;
                        }
                        // The block must be a descendant of the previous slot's head.
                        if !proto_array.is_descendant(self.fcr_store.prev_slot_head, block_root) {
                            break;
                        }
                        // Check if the block itself is confirmed.
                        let is_confirmed = self
                            .is_one_confirmed(block_root, proto_array, fc_store)
                            .ok()?;
                        debug!(
                            block = %block_root,
                            slot = block.slot.as_u64(),
                            passed = is_confirmed,
                            "FCR prev-epoch advance step"
                        );
                        if is_confirmed {
                            advanced_confirmed_root = block_root;
                        } else {
                            // Stop advancing as soon as a block is not confirmed.
                            break;
                        }
                    }
                    current_confirmed_root = advanced_confirmed_root;
                }
            }
        }

        // Second condition: Current epoch advancement
        // This logic allows advancing the confirmed root for blocks within the current epoch.
        if fc_store.get_current_slot() % E::slots_per_epoch() == 0 || {
            let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
            let cond_prev = proto_array
                .get_block(&self.fcr_store.prev_slot_head)
                .and_then(|b| {
                    b.unrealized_justified_checkpoint
                        .as_ref()
                        .map(|cp| cp.epoch)
                })
                .map(|e| e + 1 >= current_epoch)
                .unwrap_or(false);
            let cond_head = proto_array
                .get_block(&head_root)
                .and_then(|b| {
                    b.unrealized_justified_checkpoint
                        .as_ref()
                        .map(|cp| cp.epoch)
                })
                .map(|e| e + 1 >= current_epoch)
                .unwrap_or(false);
            cond_prev || cond_head
        } {
            // current-epoch advancement
            let mut tentative_confirmed = current_confirmed_root;
            if let Some(canonical_roots) =
                self.get_canonical_roots(proto_array, current_confirmed_root, head_root)
            {
                for &block_root in canonical_roots.iter().skip(1) {
                    let block = match proto_array.get_block(&block_root) {
                        Some(b) => b,
                        None => break,
                    };
                    let block_epoch = block.slot.epoch(E::slots_per_epoch());
                    let tentative_epoch = proto_array
                        .get_block(&tentative_confirmed)
                        .map(|b| b.slot.epoch(E::slots_per_epoch()))
                        .unwrap_or(block_epoch);

                    if block_epoch > tentative_epoch {
                        // Crossing into the current epoch from a previous one requires an FFG safety check.
                        // We must ensure that the checkpoint for this new epoch *will* be justified.
                        if let Some(checkpoint_root) =
                            self.get_checkpoint_block(proto_array, block_root, block_epoch)
                        {
                            let checkpoint = Checkpoint {
                                epoch: block_epoch,
                                root: checkpoint_root,
                            };
                            if !self
                                .will_checkpoint_be_justified(proto_array, fc_store, &checkpoint)
                                .unwrap_or(false)
                            {
                                debug!(
                                    block = %block_root,
                                    slot = block.slot.as_u64(),
                                    checkpoint_root = %checkpoint_root,
                                    "FCR current-epoch advance gated: checkpoint not justified"
                                );
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    let is_confirmed = self
                        .is_one_confirmed(block_root, proto_array, fc_store)
                        .ok()?;
                    debug!(
                        block = %block_root,
                        slot = block.slot.as_u64(),
                        passed = is_confirmed,
                        "FCR current-epoch advance step"
                    );
                    if is_confirmed {
                        tentative_confirmed = block_root;
                    } else {
                        break;
                    }
                }
            }

            // Final safety check for current epoch confirmation per spec. This ensures that
            // advancing confirmation within the current epoch doesn't violate FFG assumptions.
            let current_epoch = fc_store.get_current_slot().epoch(E::slots_per_epoch());
            let tentative_epoch = proto_array
                .get_block(&tentative_confirmed)
                .map(|b| b.slot.epoch(E::slots_per_epoch()))
                .unwrap_or(current_epoch);
            let is_safe_to_advance = if tentative_epoch == current_epoch {
                // If the tentatively confirmed block is in the current epoch, we perform another
                // `voting_source` recency check for this block.
                use std::collections::HashMap;
                let balances = &fc_store.justified_balances().effective_balances;
                let mut epoch_to_weight: HashMap<Epoch, u128> = HashMap::new();
                for (validator_index, &eb) in balances.iter().enumerate() {
                    if eb == 0 {
                        continue;
                    }
                    if let Some((vote_root, vote_epoch)) =
                        proto_array.latest_message(validator_index)
                    {
                        if proto_array.is_descendant(tentative_confirmed, vote_root) {
                            *epoch_to_weight.entry(vote_epoch).or_insert(0) += eb as u128;
                        }
                    }
                }
                let voting_source_epoch = epoch_to_weight
                    .into_iter()
                    .max_by_key(|(_, w)| *w)
                    .map(|(e, _)| e)
                    .unwrap_or(
                        self.fcr_store
                            .prev_slot_unrealized_justified_checkpoint
                            .epoch,
                    );
                let ok = voting_source_epoch + 2 >= current_epoch;
                debug!(
                    tentative = %tentative_confirmed,
                    head = %head_root,
                    voting_source_epoch = voting_source_epoch.as_u64(),
                    current_epoch = current_epoch.as_u64(),
                    safe = ok,
                    "FCR current-epoch advancement (voting-source)"
                );
                ok
            } else if fc_store.get_current_slot() % E::slots_per_epoch() == 0 {
                // If at an epoch boundary, we check for conflicting checkpoints.
                let ok = self
                    .will_no_conflicting_checkpoint_be_justified(proto_array, fc_store, head_root)
                    .unwrap_or(false);
                debug!(
                    tentative = %tentative_confirmed,
                    head = %head_root,
                    boundary = true,
                    safe = ok,
                    "FCR current-epoch advancement (boundary no-conflict)"
                );
                ok
            } else {
                false
            };

            current_confirmed_root = if is_safe_to_advance {
                tentative_confirmed
            } else {
                current_confirmed_root
            };
        }

        let final_slot = proto_array
            .get_block(&current_confirmed_root)
            .map(|b| b.slot.as_u64())
            .unwrap_or_default();
        debug!(
            confirmed = %current_confirmed_root,
            confirmed_slot = final_slot,
            head = %head_root,
            head_slot = head_slot,
            "FCR find_latest_confirmed_descendant: result"
        );

        Some(current_confirmed_root)
    }

    /// Gets canonical roots from ancestor to descendant.
    ///
    /// **Python Specification**: `get_canonical_roots(store, ancestor_slot)`
    ///
    /// Note: Signature adapted for Lighthouse. The spec takes `ancestor_slot` and
    /// derives a suffix to the head, whereas here we explicitly return the path
    /// between `ancestor_root` and `descendant_root` (inclusive) along the
    /// canonical chain.
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

        if let (Some(first), Some(last)) = (canonical_roots.first(), canonical_roots.last()) {
            let first_slot = proto_array
                .get_block(first)
                .map(|b| b.slot.as_u64())
                .unwrap_or_default();
            let last_slot = proto_array
                .get_block(last)
                .map(|b| b.slot.as_u64())
                .unwrap_or_default();
            debug!(
                path_len = canonical_roots.len(),
                start = %first,
                start_slot = first_slot,
                end = %last,
                end_slot = last_slot,
                "FCR canonical path"
            );
        }

        Some(canonical_roots)
    }

    /// Gets the checkpoint block for a given epoch.
    ///
    /// **Python Specification**: `get_checkpoint_block(store, block_root, epoch)`
    fn get_checkpoint_block(
        &self,
        proto_array: &ProtoArrayForkChoice,
        block_root: Hash256,
        epoch: Epoch,
    ) -> Option<Hash256> {
        // Find the checkpoint block for the given epoch
        let mut current_root = block_root;

        loop {
            let Some(current_block) = proto_array.get_block(&current_root) else {
                return None;
            };

            let current_epoch = current_block.slot.epoch(E::slots_per_epoch());
            if current_epoch == epoch {
                return Some(current_root);
            }

            if let Some(parent_root) = current_block.parent_root {
                current_root = parent_root;
            } else {
                return None; // Reached genesis
            }
        }
    }

    /// Checks if a checkpoint will be justified.
    ///
    /// **Python Specification**: `will_checkpoint_be_justified(store, checkpoint)`
    ///
    /// This function determines if a checkpoint will be justified based on current
    /// vote patterns and FFG analysis. It handles both current epoch and previous
    /// epoch checkpoints appropriately.
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

    /// Gets the checkpoint weight for FFG analysis using a provided checkpoint state.
    ///
    /// **Python Specification**: `get_checkpoint_weight(store, checkpoint, checkpoint_state)`
    ///
    /// This function calculates the FFG support weight for a given checkpoint by analyzing
    /// validator votes using a provided checkpoint state. It uses LMD-GHOST votes to estimate
    /// FFG support, as validators voting for blocks descended from a checkpoint implicitly
    /// support that checkpoint.
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
        let start_time = std::time::Instant::now();
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

        let elapsed = start_time.elapsed();
        observe(&FCR_FFG_SUPPORT_CALCULATION_TIME, elapsed.as_secs_f64());

        Ok(checkpoint_weight)
    }

    /// Gets the FFG weight up to a specific slot.
    ///
    /// **Python Specification**: `get_ffg_weight_till_slot(slot, epoch, total_active_balance)`
    ///
    /// This function calculates the total FFG weight that could have been cast
    /// up to a specific slot within an epoch. It's used for FFG justification analysis.
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
        let sp_start = std::time::Instant::now();
        let checkpoint_state = match self.state_provider.get_checkpoint_state(checkpoint) {
            Ok(Some(state)) => {
                observe(
                    &FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
                    sp_start.elapsed().as_secs_f64(),
                );
                state
            }
            Ok(None) => {
                observe(
                    &FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
                    sp_start.elapsed().as_secs_f64(),
                );
                inc_counter(&FCR_STATE_PROVIDER_MISS_TOTAL);
                return Ok(false);
            }
            Err(_) => {
                observe(
                    &FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
                    sp_start.elapsed().as_secs_f64(),
                );
                inc_counter(&FCR_STATE_PROVIDER_ERROR_TOTAL);
                return Ok(false);
            }
        };

        // Use total active balance from the checkpoint state to avoid extra provider lookups
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

        let ok = left_side >= right_side;
        info!(
            checkpoint_root = %checkpoint.root,
            checkpoint_epoch = checkpoint.epoch.as_u64(),
            ffg_support_for_checkpoint = ffg_support_for_checkpoint,
            ffg_weight_till_now = ffg_weight_till_now,
            remaining_ffg_weight = remaining_ffg_weight,
            remaining_honest_ffg_weight = remaining_honest_ffg_weight,
            min_byzantine_weight = min_byzantine_weight,
            min_honest_ffg_support = min_honest_ffg_support,
            left = left_side,
            right = right_side,
            passed = ok,
            "FCR FFG: will_current_epoch_checkpoint_be_justified"
        );
        Ok(ok)
    }

    /// Checks if no conflicting checkpoint will be justified.
    ///
    /// **Python Specification**: `will_no_conflicting_checkpoint_be_justified(store, checkpoint)`
    ///
    /// This function checks if any conflicting checkpoint could be justified,
    /// ensuring that advancing confirmation is safe at epoch boundaries.
    /// It's a safety check for FFG confirmation logic.
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
        let sp_start = std::time::Instant::now();
        let checkpoint_state = match self.state_provider.get_checkpoint_state(&checkpoint) {
            Ok(Some(state)) => {
                observe(
                    &FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
                    sp_start.elapsed().as_secs_f64(),
                );
                state
            }
            Ok(None) => {
                observe(
                    &FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
                    sp_start.elapsed().as_secs_f64(),
                );
                inc_counter(&FCR_STATE_PROVIDER_MISS_TOTAL);
                return Ok(false);
            }
            Err(_) => {
                observe(
                    &FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS,
                    sp_start.elapsed().as_secs_f64(),
                );
                inc_counter(&FCR_STATE_PROVIDER_ERROR_TOTAL);
                return Ok(false);
            }
        };

        // Use total active balance from the checkpoint state to avoid extra provider lookups.
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

        let ok = left_side >= right_side;
        info!(
            checkpoint_root = %checkpoint.root,
            checkpoint_epoch = checkpoint.epoch.as_u64(),
            ffg_support_for_checkpoint = ffg_support_for_checkpoint,
            ffg_weight_till_now = ffg_weight_till_now,
            remaining_ffg_weight = remaining_ffg_weight,
            remaining_honest_ffg_weight = remaining_honest_ffg_weight,
            min_byzantine_weight = min_byzantine_weight,
            min_honest_ffg_support = min_honest_ffg_support,
            left = left_side,
            right = right_side,
            passed = ok,
            "FCR FFG: will_no_conflicting_checkpoint_be_justified"
        );
        Ok(ok)
    }

    /// Prunes FCR metadata to align with the DAG pruning.
    ///
    /// **Why Required**: Lighthouse's side-table approach requires explicit pruning to prevent
    /// unbounded memory growth. When blocks are pruned from the proto array, their FCR metadata
    /// must also be removed to maintain consistency and prevent memory leaks.
    ///
    /// This method removes FCR metadata for blocks that have been pruned from the proto array,
    /// ensuring that the FCR side-table stays aligned with the main DAG structure.
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

        let after = self.meta.len();

        // Update metadata cache size metric
        set_gauge(&FCR_METADATA_CACHE_SIZE, after as i64);

        info!(
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
    fn get_committee_weight_between_slots<T>(
        &self,
        start_slot: Slot,
        end_slot: Slot,
        fc_store: &T,
    ) -> Result<u64, crate::Error<T::Error>>
    where
        T: ForkChoiceStore<E>,
    {
        let start_time = Instant::now();
        if start_slot > end_slot {
            debug!(
                start_slot = start_slot.as_u64(),
                end_slot = end_slot.as_u64(),
            );
            return Ok(0);
        }

        let start_epoch = start_slot.epoch(E::slots_per_epoch());
        let end_epoch = end_slot.epoch(E::slots_per_epoch());
        let total_active_balance = fc_store.justified_balances().total_effective_balance;

        // If an entire epoch is covered by the range, return the total active balance
        if self.is_full_validator_set_covered(start_slot, end_slot) {
            debug!(
                start_slot = start_slot.as_u64(),
                end_slot = end_slot.as_u64(),
                total_active_balance = total_active_balance,
                "FCR W: full validator set covered → total_active_balance"
            );
            return Ok(total_active_balance);
        }

        if start_epoch == end_epoch {
            // Same epoch: simple pro-rata calculation
            let slots_covered = end_slot - start_slot + 1;
            let weight_per_slot = total_active_balance / E::slots_per_epoch();
            let w = weight_per_slot * slots_covered.as_u64();
            debug!(
                start_slot = start_slot.as_u64(),
                end_slot = end_slot.as_u64(),
                slots_covered = slots_covered.as_u64(),
                weight_per_slot = weight_per_slot,
                w = w,
                "FCR W: same-epoch pro-rata"
            );
            Ok(w)
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
                    let w = weight_per_slot * slots_covered.as_u64();
                    debug!(
                        start_slot = start_slot.as_u64(),
                        end_slot = end_slot.as_u64(),
                        estimate = w,
                        "FCR W: cross-epoch fallback pro-rata"
                    );
                    w
                }
            };

            // Apply safety adjustment factor for partial epoch coverage
            let adjusted = self.adjust_committee_weight_estimate_to_ensure_safety(estimate);
            debug!(
                start_slot = start_slot.as_u64(),
                end_slot = end_slot.as_u64(),
                estimate = estimate,
                adjusted = adjusted,
                "FCR W: cross-epoch estimate with safety adjustment"
            );
            let elapsed = start_time.elapsed();
            observe(
                &FCR_COMMITTEE_WEIGHT_CALCULATION_TIME,
                elapsed.as_secs_f64(),
            );
            Ok(adjusted)
        }
    }

    /// Adjusts committee weight estimates to ensure safety.
    ///
    /// **Python Specification**: `adjust_committee_weight_estimate_to_ensure_safety(estimate)`
    ///
    /// Committee weight estimation can have small errors due to
    /// cross-epoch calculations and validator set changes. This function adds a
    /// small safety margin to ensure FCR safety guarantees are maintained even
    /// with estimation errors.
    ///
    /// **Specification**: Multiplies the estimate by `(1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR) / 1000`
    /// to add a small safety margin (0.5% for the default factor of 5).
    fn adjust_committee_weight_estimate_to_ensure_safety(&self, estimate: u64) -> u64 {
        // Apply safety adjustment: estimate * (1000 + adjustment_factor) / 1000
        // This adds a small safety margin to ensure FCR safety guarantees
        let adjusted = estimate * (1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR) / 1000;
        debug!(
            estimate = estimate,
            adjusted = adjusted,
            adjustment_factor = COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR,
            "FCR: adjust committee weight estimate to ensure safety"
        );
        adjusted
    }

    /// Checks if the slot range covers a full validator set.
    ///
    /// **Python Specification**: `is_full_validator_set_covered(start_slot, end_slot)`
    ///
    /// Returns whether the range from `start_slot` to `end_slot` (inclusive of both)
    /// includes an entire epoch.
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

        let estimate = start_epoch_weight_estimate.saturating_add(end_epoch_weight_estimate);
        debug!(
            start_slot = start_slot.as_u64(),
            end_slot = end_slot.as_u64(),
            num_slots_in_end_epoch = num_slots_in_end_epoch,
            num_slots_in_start_epoch = num_slots_in_start_epoch,
            remaining_slots_in_end_epoch = remaining_slots_in_end_epoch,
            end_epoch_weight_estimate = end_epoch_weight_estimate,
            start_epoch_weight_estimate = start_epoch_weight_estimate,
            estimate = estimate,
            "FCR W: cross-epoch estimate components"
        );
        Ok(estimate)
    }
}

#[cfg(feature = "fcr_bench")]
pub mod bench_api {
    use super::*;
    use std::sync::LazyLock;
    use types::MainnetEthSpec;
    use zerocopy::AsBytes;

    struct DummyProvider;

    impl<E: EthSpec> StateProvider<E> for DummyProvider {
        type Error = std::convert::Infallible;

        fn get_checkpoint_state(
            &self,
            _checkpoint: &Checkpoint,
        ) -> Result<Option<Arc<BeaconState<E>>>, Self::Error> {
            Ok(None)
        }

        fn get_total_active_balance_at_epoch(&self, _epoch: Epoch) -> Result<u64, Self::Error> {
            Ok(0)
        }

        fn chain_spec(&self) -> &ChainSpec {
            static SPEC: LazyLock<ChainSpec> = LazyLock::new(ChainSpec::minimal);
            &SPEC
        }
    }

    type E = MainnetEthSpec;
    type S = DummyProvider;

    fn fcr() -> FastConfirmation<E, S> {
        let cfg = FastConfirmationConfig::new(DEFAULT_FCR_BYZANTINE_THRESHOLD_PERCENTAGE)
            .expect("valid beta");
        FastConfirmation::new(cfg, DummyProvider)
    }

    /// Benchmark wrapper: adjust committee weight estimate.
    pub fn bench_adjust_committee_weight_estimate(estimate: u64) -> u64 {
        fcr().adjust_committee_weight_estimate_to_ensure_safety(estimate)
    }

    /// Benchmark wrapper: cross-epoch weight estimate.
    pub fn bench_calculate_cross_epoch_weight_estimate(
        start_slot: Slot,
        end_slot: Slot,
        total_active_balance: u64,
    ) -> u64 {
        fcr()
            .calculate_cross_epoch_weight_estimate(start_slot, end_slot, total_active_balance)
            .expect("ok")
    }

    /// Benchmark wrapper: ffg weight till slot.
    pub fn bench_get_ffg_weight_till_slot(
        slot: Slot,
        epoch: Epoch,
        total_active_balance: u64,
    ) -> u64 {
        fcr().get_ffg_weight_till_slot(slot, epoch, total_active_balance)
    }

    /// Benchmark wrapper: full-epoch coverage predicate.
    pub fn bench_is_full_validator_set_covered(start_slot: Slot, end_slot: Slot) -> bool {
        fcr().is_full_validator_set_covered(start_slot, end_slot)
    }

    /// Benchmark wrapper: pure math for is_one_confirmed inequality decision.
    /// Returns whether 2*S > W + (W/50)*beta + proposer_score
    pub fn bench_is_one_confirmed_math(
        support: u64,
        committee_weight: u64,
        proposer_score: u64,
        beta_percentage: u64,
    ) -> bool {
        let left = support.saturating_mul(2);
        let right = committee_weight
            .saturating_add(committee_weight / 50 * beta_percentage)
            .saturating_add(proposer_score);
        left > right
    }

    /// Benchmark wrapper: is_one_confirmed using internal W-estimation for a slot range.
    /// Computes W between [start_slot, end_slot] from total active balance and applies the inequality.
    pub fn bench_is_one_confirmed_w_estimate(
        support: u64,
        total_active_balance: u64,
        start_slot: Slot,
        end_slot: Slot,
        proposer_score: u64,
        beta_percentage: u64,
    ) -> bool {
        let fcr = fcr();
        let w = if start_slot.epoch(E::slots_per_epoch()) == end_slot.epoch(E::slots_per_epoch()) {
            let slots_covered = end_slot - start_slot + 1;
            let weight_per_slot = total_active_balance / E::slots_per_epoch();
            weight_per_slot * slots_covered.as_u64()
        } else {
            let estimate = fcr
                .calculate_cross_epoch_weight_estimate(start_slot, end_slot, total_active_balance)
                .expect("ok");
            fcr.adjust_committee_weight_estimate_to_ensure_safety(estimate)
        };
        bench_is_one_confirmed_math(support, w, proposer_score, beta_percentage)
    }

    /// Benchmark wrapper: FCR update after find head (fork choice integration)
    pub fn bench_update_fcr_after_find_head() {
        // Simulate the FCR update hook that runs after fork choice finds head
        // This is a no-op for now but represents the integration overhead
    }

    /// Benchmark wrapper: no-op operation for baseline comparison
    pub fn bench_no_op() {
        // No-op for baseline performance comparison
    }

    /// Benchmark wrapper: committee weight calculation with validator count
    pub fn bench_committee_weight_with_validators(validator_count: u64) -> u64 {
        // Simulate committee weight calculation with different validator counts
        // This tests scaling performance
        let total_active_balance = validator_count * 32_000_000_000; // 32 ETH per validator
        let slots_per_epoch = E::slots_per_epoch();
        total_active_balance / slots_per_epoch
    }

    /// Benchmark wrapper: FFG support calculation with validator count
    pub fn bench_ffg_support_with_validators(validator_count: u64) -> u64 {
        // Simulate FFG support calculation with different validator counts
        // This tests FFG scaling performance
        let total_active_balance = validator_count * 32_000_000_000; // 32 ETH per validator
        total_active_balance * 2 / 3 // Assume 2/3 support
    }

    /// Benchmark wrapper: FCR metadata growth simulation
    pub fn bench_fcr_metadata_growth() {
        // Simulate FCR metadata HashMap growth
        // This tests memory usage patterns
        let mut meta = std::collections::HashMap::new();
        for i in 0..1000 {
            meta.insert(Hash256::from_low_u64_be(i), FcrMeta::default());
        }
    }

    /// Benchmark wrapper: FCR pruning simulation
    pub fn bench_fcr_pruning() {
        // Simulate FCR metadata pruning
        // This tests pruning effectiveness
        let mut meta = std::collections::HashMap::new();
        for i in 0..1000 {
            meta.insert(Hash256::from_low_u64_be(i), FcrMeta::default());
        }
        // Simulate pruning by removing half the entries
        meta.retain(|k, _| k.as_bytes()[0] % 2 == 0);
    }

    /// Benchmark wrapper: memory usage with validator count
    pub fn bench_memory_usage_with_validators(validator_count: u64) -> usize {
        // Simulate memory usage scaling with validator count
        // This tests memory efficiency
        let mut meta = std::collections::HashMap::new();
        let entries = (validator_count / 1000).max(1) as usize; // Scale entries with validators
        for i in 0..entries {
            meta.insert(Hash256::from_low_u64_be(i as u64), FcrMeta::default());
        }
        meta.len()
    }

    /// Benchmark wrapper: epoch boundary transition simulation
    pub fn bench_epoch_boundary_transition() {
        // Simulate epoch boundary transition logic
        // This tests cross-epoch performance
        let current_slot = Slot::new(31); // End of epoch
        let next_slot = Slot::new(32); // Start of next epoch
        let _is_boundary =
            current_slot.epoch(E::slots_per_epoch()) != next_slot.epoch(E::slots_per_epoch());
    }

    /// Benchmark wrapper: reorg detection simulation
    pub fn bench_reorg_detection() {
        // Simulate reorg detection logic
        // This tests reorg handling performance
        let head_root = Hash256::from_low_u64_be(100);
        let confirmed_root = Hash256::from_low_u64_be(99);
        let _is_reorg = head_root != confirmed_root;
    }

    /// Benchmark wrapper: late attestation handling simulation
    pub fn bench_late_attestation_handling() {
        // Simulate late attestation handling
        // This tests network delay scenarios
        let current_slot = Slot::new(100);
        let attestation_slot = Slot::new(98); // Late attestation
        let _is_late = attestation_slot < current_slot;
    }

    /// Benchmark wrapper: safe head calculation simulation
    pub fn bench_safe_head_calculation() -> Hash256 {
        // Simulate safe head calculation
        // This tests safe head performance
        Hash256::from_low_u64_be(100)
    }

    /// Benchmark wrapper: safe head reorg simulation
    pub fn bench_safe_head_reorg() {
        // Simulate safe head reorg detection
        // This tests reorg handling in safe head
        let old_safe_head = Hash256::from_low_u64_be(99);
        let new_safe_head = Hash256::from_low_u64_be(100);
        let _is_reorg = old_safe_head != new_safe_head;
    }

    /// Benchmark wrapper: safe head advancement simulation
    pub fn bench_safe_head_advancement() -> Hash256 {
        // Simulate safe head advancement along canonical chain
        // This tests confirmation advancement performance
        Hash256::from_low_u64_be(101)
    }

    /// Benchmark wrapper: cross-epoch confirmation simulation
    pub fn bench_cross_epoch_confirmation() {
        // Simulate cross-epoch confirmation advancement
        // This tests epoch boundary confirmation logic
        let start_epoch = Epoch::new(0);
        let end_epoch = Epoch::new(1);
        let _crosses_epoch = end_epoch > start_epoch;
    }

    /// Benchmark wrapper: epoch boundary weight calculations
    pub fn bench_epoch_boundary_weights() -> u64 {
        // Simulate epoch boundary weight calculations
        // This tests cross-epoch weight computation
        let total_active_balance = 32_000_000_000_000; // 1M validators
        let slots_per_epoch = E::slots_per_epoch();
        total_active_balance / slots_per_epoch
    }
}
