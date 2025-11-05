pub use metrics::*;
use std::sync::LazyLock;
use types::EthSpec;

use crate::{ForkChoice, ForkChoiceStore, StateProvider};

/// Gauge for the number of attestations currently queued for processing.
///
/// This metric is crucial for monitoring the load on the fork choice mechanism. A persistently
/// high number of queued attestations might indicate a performance bottleneck or an issue with
/// attestation processing.
pub static FORK_CHOICE_QUEUED_ATTESTATIONS: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fork_choice_queued_attestations",
        "Current count of queued attestations",
    )
});

/// Gauge for the total number of nodes in the `ProtoArray` fork choice DAG.
///
/// This metric reflects the size of the block DAG being maintained by the fork choice. An
/// unusually large number of nodes could indicate a fragmented network or a chain that is
/// not finalizing.
pub static FORK_CHOICE_NODES: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge("fork_choice_nodes", "Current count of proto array nodes")
});

/// Gauge for the number of indices in the `ProtoArray` fork choice DAG.
///
/// This metric provides another view on the size and complexity of the fork choice DAG. It
/// should be monitored alongside `FORK_CHOICE_NODES` to get a complete picture of the DAG's health.
pub static FORK_CHOICE_INDICES: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fork_choice_indices",
        "Current count of proto array indices",
    )
});

/// Counter for the total number of attestations dequeued for processing.
///
/// This metric tracks the throughput of attestation processing. It's useful for understanding
/// how many attestations are being processed over time.
pub static FORK_CHOICE_DEQUEUED_ATTESTATIONS: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fork_choice_dequeued_attestations_total",
        "Total count of dequeued attestations",
    )
});

/// Histogram for the duration of `on_block` processing.
///
/// This metric is essential for identifying performance bottlenecks in block processing. High
/// latency in `on_block` could delay the propagation of new blocks and affect the timeliness
/// of attestations.
pub static FORK_CHOICE_ON_BLOCK_TIMES: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "beacon_fork_choice_process_block_seconds",
        "The duration in seconds of on_block runs",
    )
});

/// Histogram for the duration of `on_attestation` processing.
///
/// This metric helps monitor the performance of attestation processing. High latency could
/// indicate a problem with the fork choice mechanism or the underlying hardware.
pub static FORK_CHOICE_ON_ATTESTATION_TIMES: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "beacon_fork_choice_process_attestation_seconds",
        "The duration in seconds of on_attestation runs",
    )
});

/// Histogram for the duration of `on_attester_slashing` processing.
///
/// While less frequent than block or attestation processing, monitoring the performance of
/// `on_attester_slashing` is important for ensuring that slashings are processed efficiently.
pub static FORK_CHOICE_ON_ATTESTER_SLASHING_TIMES: LazyLock<Result<Histogram>> =
    LazyLock::new(|| {
        try_create_histogram(
            "beacon_fork_choice_on_attester_slashing_seconds",
            "The duration in seconds on on_attester_slashing runs",
        )
    });

// --- Fast Confirmation Rule (FCR) Metrics ---

/// Gauge for the current slot number of the FCR safe head.
///
/// The "safe head" is the latest block that is considered safe by the Fast Confirmation Rule.
/// Monitoring its slot number provides a view into how quickly the FCR is advancing the
/// confirmed chain.
pub static FCR_SAFE_HEAD_SLOT_NUMBER: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fcr_safe_head_slot_number",
        "Current slot number of the fast confirmed safe head",
    )
});

/// Counter for the total number of safe head reorgs detected.
///
/// A "safe head reorg" occurs when the FCR-confirmed safe head changes to a block that is not
/// a descendant of the previous safe head. This is a critical safety metric. While occasional
/// small reorgs might be acceptable, a high frequency or large depth of reorgs could
/// indicate a network issue or a problem with the FCR implementation.
pub static FCR_SAFE_HEAD_REORG_COUNT: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_safe_head_reorg_count_total",
        "Total number of safe head reorgs detected",
    )
});

/// Histogram for the distance of safe head reorgs in blocks.
///
/// This metric measures the number of blocks that were "reorged out" during a safe head reorg.
/// It provides insight into the severity of reorgs.
pub static FCR_SAFE_HEAD_REORG_DISTANCE: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_safe_head_reorg_distance",
        "Distance of safe head reorgs in blocks (1, 2, 4, 8, 16, 32, 64)",
    )
});

/// Histogram for the depth of safe head reorgs in blocks.
///
/// This metric measures how deep in the chain a safe head reorg occurred. Deeper reorgs are
/// more serious as they indicate a more significant chain reorganization.
pub static FCR_SAFE_HEAD_REORG_DEPTH: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_safe_head_reorg_depth",
        "Depth of safe head reorgs in blocks (1, 2, 4, 8, 16, 32)",
    )
});

/// Counter for the total number of times a previously confirmed block was reorged out.
///
/// This is a critical safety metric. A non-zero value indicates that a block that was
/// previously considered "confirmed" by FCR has been removed from the canonical chain.
/// This should be an extremely rare event in a healthy network.
pub static FCR_CONFIRMED_REORG_COUNT: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_confirmed_reorg_count_total",
        "Total number of times a previously confirmed block was reorged out",
    )
});

/// Histogram for the slot distance between old and new confirmed roots during a reorg.
///
/// When a confirmed reorg occurs, this metric measures the slot difference between the old
/// (reorged out) confirmed block and the new confirmed block. It helps quantify the
/// magnitude of the reorg.
pub static FCR_CONFIRMED_REORG_SLOTS: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_confirmed_reorg_slots",
        "Slot distance between old and new confirmed roots when a reorg occurs",
    )
});

/// Counter for the total number of times the confirmed root moved to an earlier slot.
///
/// A "rollback" is a situation where the FCR-confirmed root moves to a block at an earlier
/// slot than the previous confirmed root. This can happen during reorgs and is a key
/// indicator of chain instability.
pub static FCR_CONFIRMED_ROOT_ROLLBACK_COUNT: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_confirmed_root_rollback_count_total",
        "Total number of times the confirmed root moved to an earlier slot",
    )
});

/// Histogram for the number of slots decreased when the confirmed root rolled back.
///
/// This metric quantifies the magnitude of a confirmed root rollback, providing insight into
/// the severity of the chain instability that caused it.
pub static FCR_CONFIRMED_ROOT_ROLLBACK_SLOTS: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_confirmed_root_rollback_slots",
        "Slots decreased when the confirmed root rolled back",
    )
});

/// Histogram for the distribution of slot delay between block creation and confirmation.
///
/// This is a key performance metric for FCR. It measures the time (in slots) from when a
/// block is created to when it is confirmed by FCR. The goal of FCR is to achieve a delay
/// of 1-2 slots.
pub static FCR_CONFIRMATION_SLOT_DELAY: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    // Focused buckets for 0–4 and some headroom
    let buckets = vec![0.0, 1.0, 2.0, 3.0, 4.0, 8.0];
    try_create_histogram_with_buckets(
        "fcr_confirmation_slot_delay",
        "Slots between block slot and confirmation slot",
        Ok(buckets),
    )
});

/// Histogram for the distribution of the gap between the head and the confirmed block at the time of confirmation.
///
/// This metric measures `head_slot - confirmed_block_slot` at the moment a block is confirmed.
/// A large gap can indicate that the confirmed chain is lagging significantly behind the head
/// of the chain, which could be a sign of network latency or other issues.
pub static FCR_HEAD_TO_CONFIRMED_GAP_SLOTS: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    let buckets = vec![0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0];
    try_create_histogram_with_buckets(
        "fcr_head_to_confirmed_gap_slots",
        "Gap in slots between current head and confirmed block at confirmation time",
        Ok(buckets),
    )
});

/// Counter for FCR restarts, labeled by reason.
///
/// FCR might need to "restart" its confirmation process from a safe point (like the last
/// finalized block) if it detects a potential safety violation. This metric tracks how often
/// these restarts occur and why (e.g., due to a reorg or a stale state).
pub static FCR_RESTARTS_TOTAL: LazyLock<Result<IntCounterVec>> = LazyLock::new(|| {
    try_create_int_counter_vec(
        "fcr_restarts_total",
        "Total number of confirmation rule restarts, labeled by reason",
        &["reason"],
    )
});

/// Counter for tail-case confirmations, labeled by epoch-boundary and delay bucket.
///
/// A "tail-case" is a confirmation that takes longer than expected (e.g., 2 or more slots).
/// This metric helps diagnose the conditions under which these delays occur, such as at
/// epoch boundaries.
pub static FCR_TAIL_CASES_TOTAL: LazyLock<Result<IntCounterVec>> = LazyLock::new(|| {
    try_create_int_counter_vec(
        "fcr_tail_cases_total",
        "Tail confirmations (delay >=2 slots) labeled by epoch_boundary and delay bucket",
        &["epoch_boundary", "delay_bucket"],
    )
});

/// Gauge indicating the sync state of the node.
///
/// This metric is 1 if `head_slot == current_slot`, and 0 otherwise. It's a simple way to
/// check if the node is in sync with the network, which is a prerequisite for FCR to
/// operate correctly.
pub static FCR_IN_SYNC: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fcr_in_sync_state",
        "1 if head_slot == current_slot at measurement time, else 0",
    )
});

/// Histogram for the time taken for block confirmation in seconds.
///
/// This is the wall-clock time equivalent of `FCR_CONFIRMATION_SLOT_DELAY`. It provides a more
/// intuitive measure of confirmation latency.
pub static FCR_CONFIRMATION_TIME_SECONDS: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    // Buckets tailored for FCR confirmation delay (seconds), covering 0–120s.
    let buckets = vec![
        0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
        30.0, 45.0, 60.0, 90.0, 120.0,
    ];
    try_create_histogram_with_buckets(
        "fcr_confirmation_time_seconds",
        "Time taken for block confirmation in seconds",
        Ok(buckets),
    )
});

/// Histogram for the percentage of validator support for confirmed blocks.
///
/// This metric measures the percentage of the total committee weight that voted for a block
/// at the time of its confirmation. It provides insight into how much agreement there was
/// in the network when the block was confirmed.
pub static FCR_VALIDATOR_SUPPORT_PERCENTAGE: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_validator_support_percentage",
        "Percentage of validator support for confirmed blocks",
    )
});

/// Gauge for the current Byzantine threshold percentage used by FCR.
///
/// This metric simply exposes the configured Byzantine threshold (`β`). It's useful for
/// verifying that the node is running with the intended FCR configuration.
pub static FCR_BYZANTINE_THRESHOLD_PERCENTAGE: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fcr_byzantine_threshold_percentage",
        "Current Byzantine threshold percentage used by FCR",
    )
});

/// Histogram for the time taken for committee weight calculations.
///
/// This is a performance metric for the FCR algorithm itself. High latency in this calculation
/// could indicate a performance issue in the FCR implementation.
pub static FCR_COMMITTEE_WEIGHT_CALCULATION_TIME: LazyLock<Result<Histogram>> =
    LazyLock::new(|| {
        try_create_histogram(
            "fcr_committee_weight_calculation_seconds",
            "Time taken for committee weight calculations",
        )
    });

/// Histogram for the time taken for FFG support calculations.
///
/// This is another performance metric for the FCR algorithm, specifically for the parts of
/// the algorithm that interact with the FFG justification mechanism.
pub static FCR_FFG_SUPPORT_CALCULATION_TIME: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_ffg_support_calculation_seconds",
        "Time taken for FFG support calculations",
    )
});

/// Performance and availability metrics for the `StateProvider`.
///
/// The `StateProvider` is a crucial dependency for FCR, as it provides access to historical
/// beacon states. These metrics monitor its performance and reliability.
pub static FCR_STATE_PROVIDER_GET_CHECKPOINT_STATE_SECONDS: LazyLock<Result<Histogram>> =
    LazyLock::new(|| {
        try_create_histogram(
            "fcr_state_provider_get_checkpoint_state_seconds",
            "Latency of StateProvider::get_checkpoint_state calls",
        )
    });

/// Counter for the total number of times the checkpoint state was unavailable.
///
/// A "miss" indicates that the `StateProvider` returned `None` for a requested state. A high
/// miss rate could indicate a problem with the state management or persistence layer.
pub static FCR_STATE_PROVIDER_MISS_TOTAL: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_state_provider_checkpoint_state_miss_total",
        "Total number of times checkpoint state was unavailable (None)",
    )
});

/// Counter for the total number of errors returned by the `StateProvider`.
///
/// This metric tracks errors returned by `get_checkpoint_state`. A non-zero value indicates
/// a problem with the `StateProvider` implementation or its underlying data source.
pub static FCR_STATE_PROVIDER_ERROR_TOTAL: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_state_provider_checkpoint_state_error_total",
        "Total number of errors returned by StateProvider::get_checkpoint_state",
    )
});

/// Gauge for the current size of the FCR metadata cache.
///
/// FCR maintains a cache of metadata for each block. This metric tracks the size of that
/// cache, which is useful for monitoring memory usage and detecting potential memory leaks.
pub static FCR_METADATA_CACHE_SIZE: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fcr_metadata_cache_size",
        "Current size of FCR metadata cache",
    )
});

/// Counter for the total number of epoch boundary transitions processed.
///
/// FCR has special logic for handling epoch boundaries. This metric tracks how many of
/// these transitions have been processed.
pub static FCR_EPOCH_BOUNDARY_TRANSITIONS: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_epoch_boundary_transitions_total",
        "Total number of epoch boundary transitions processed",
    )
});

/// Counter for the total number of FCR confirmations.
///
/// This is a simple counter for the total number of blocks confirmed by FCR, regardless of
/// the confirmation delay.
pub static FCR_CONFIRMATIONS_TOTAL: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_confirmations_total",
        "Total number of FCR confirmations (all delays)",
    )
});

/// Counter for FCR confirmations, bucketed by delay in slots.
///
/// This metric provides a breakdown of confirmation delays, allowing for a more granular
/// analysis of FCR performance. The labels `0`, `1`, `2`, `ge3` correspond to confirmations
/// that occurred in the same slot, 1 slot after, 2 slots after, and 3 or more slots after
/// the block was created, respectively.
pub static FCR_CONFIRMATIONS_BY_SLOTS: LazyLock<Result<IntCounterVec>> = LazyLock::new(|| {
    try_create_int_counter_vec(
        "fcr_confirmations_by_slots_total",
        "FCR confirmations bucketed by delay slots (labels: slots=0,1,2,ge3)",
        &["slots"],
    )
});

/// Counter for the total number of late attestations detected.
///
/// A "late" attestation is one that is processed in a later slot than the one it attests to.
/// While some amount of late attestations is normal due to network latency, a high number
/// could indicate network problems.
pub static FCR_LATE_ATTESTATION_COUNT: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_late_attestation_count_total",
        "Total number of late attestations detected",
    )
});

/// Update the global metrics `DEFAULT_REGISTRY` with info from the fork choice.
pub fn scrape_for_metrics<T: ForkChoiceStore<E>, E: EthSpec, S: StateProvider<E>>(
    fork_choice: &ForkChoice<T, E, S>,
) {
    set_gauge(
        &FORK_CHOICE_QUEUED_ATTESTATIONS,
        fork_choice.queued_attestations().len() as i64,
    );
    set_gauge(
        &FORK_CHOICE_NODES,
        fork_choice.proto_array().core_proto_array().nodes.len() as i64,
    );
    set_gauge(
        &FORK_CHOICE_INDICES,
        fork_choice.proto_array().core_proto_array().indices.len() as i64,
    );

    // Update FCR-specific metrics if FCR is enabled
    if fork_choice.is_fast_confirmation_enabled() {
        if let Some(fcr) = fork_choice.fast_confirmation() {
            // Update safe head slot number
            if let Some(safe_head) = fork_choice.get_fast_confirmed_head() {
                if let Some(block) = fork_choice.get_block(&safe_head) {
                    set_gauge(&FCR_SAFE_HEAD_SLOT_NUMBER, block.slot.as_u64() as i64);
                }
            } else {
                // Set to 0 if no safe head
                set_gauge(&FCR_SAFE_HEAD_SLOT_NUMBER, 0);
            }

            // Update metadata cache size
            set_gauge(&FCR_METADATA_CACHE_SIZE, fcr.metadata_cache_size() as i64);

            // Update Byzantine threshold (this should already be set during FCR creation)
            set_gauge(
                &FCR_BYZANTINE_THRESHOLD_PERCENTAGE,
                fcr.beta_percentage() as i64,
            );
        }
    } else {
        // FCR is disabled, set metrics to 0
        set_gauge(&FCR_SAFE_HEAD_SLOT_NUMBER, 0);
        set_gauge(&FCR_METADATA_CACHE_SIZE, 0);
        set_gauge(&FCR_BYZANTINE_THRESHOLD_PERCENTAGE, 0);
    }
}
