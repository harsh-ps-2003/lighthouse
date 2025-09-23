pub use metrics::*;
use std::sync::LazyLock;
use types::EthSpec;

use crate::{ForkChoice, ForkChoiceStore, StateProvider};

pub static FORK_CHOICE_QUEUED_ATTESTATIONS: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fork_choice_queued_attestations",
        "Current count of queued attestations",
    )
});
pub static FORK_CHOICE_NODES: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge("fork_choice_nodes", "Current count of proto array nodes")
});
pub static FORK_CHOICE_INDICES: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fork_choice_indices",
        "Current count of proto array indices",
    )
});
pub static FORK_CHOICE_DEQUEUED_ATTESTATIONS: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fork_choice_dequeued_attestations_total",
        "Total count of dequeued attestations",
    )
});
pub static FORK_CHOICE_ON_BLOCK_TIMES: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "beacon_fork_choice_process_block_seconds",
        "The duration in seconds of on_block runs",
    )
});
pub static FORK_CHOICE_ON_ATTESTATION_TIMES: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "beacon_fork_choice_process_attestation_seconds",
        "The duration in seconds of on_attestation runs",
    )
});
pub static FORK_CHOICE_ON_ATTESTER_SLASHING_TIMES: LazyLock<Result<Histogram>> =
    LazyLock::new(|| {
        try_create_histogram(
            "beacon_fork_choice_on_attester_slashing_seconds",
            "The duration in seconds on on_attester_slashing runs",
        )
    });

pub static FCR_SAFE_HEAD_SLOT_NUMBER: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fcr_safe_head_slot_number",
        "Current slot number of the fast confirmed safe head",
    )
});

pub static FCR_SAFE_HEAD_REORG_COUNT: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_safe_head_reorg_count_total",
        "Total number of safe head reorgs detected",
    )
});

pub static FCR_SAFE_HEAD_REORG_DISTANCE: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_safe_head_reorg_distance",
        "Distance of safe head reorgs in blocks (1, 2, 4, 8, 16, 32, 64)",
    )
});

pub static FCR_SAFE_HEAD_REORG_DEPTH: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_safe_head_reorg_depth",
        "Depth of safe head reorgs in blocks (1, 2, 4, 8, 16, 32)",
    )
});

pub static FCR_CONFIRMED_REORG_COUNT: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_confirmed_reorg_count_total",
        "Total number of times a previously confirmed block was reorged out",
    )
});

pub static FCR_CONFIRMED_REORG_SLOTS: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_confirmed_reorg_slots",
        "Slot distance between old and new confirmed roots when a reorg occurs",
    )
});

pub static FCR_CONFIRMED_ROOT_ROLLBACK_COUNT: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_confirmed_root_rollback_count_total",
        "Total number of times the confirmed root moved to an earlier slot",
    )
});

pub static FCR_CONFIRMED_ROOT_ROLLBACK_SLOTS: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_confirmed_root_rollback_slots",
        "Slots decreased when the confirmed root rolled back",
    )
});

// Distribution of slot delay between block creation and confirmation (in slots)
pub static FCR_CONFIRMATION_SLOT_DELAY: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    // Focused buckets for 0–4 and some headroom
    let buckets = vec![0.0, 1.0, 2.0, 3.0, 4.0, 8.0];
    try_create_histogram_with_buckets(
        "fcr_confirmation_slot_delay",
        "Slots between block slot and confirmation slot",
        Ok(buckets),
    )
});

// Distribution of head-to-confirmed gap at confirmation time (head_slot - confirmed_block_slot)
pub static FCR_HEAD_TO_CONFIRMED_GAP_SLOTS: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    let buckets = vec![0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0];
    try_create_histogram_with_buckets(
        "fcr_head_to_confirmed_gap_slots",
        "Gap in slots between current head and confirmed block at confirmation time",
        Ok(buckets),
    )
});

// Restarts of confirmation rule with reason labels (stale|reorg)
pub static FCR_RESTARTS_TOTAL: LazyLock<Result<IntCounterVec>> = LazyLock::new(|| {
    try_create_int_counter_vec(
        "fcr_restarts_total",
        "Total number of confirmation rule restarts, labeled by reason",
        &["reason"],
    )
});

// Tail-case confirmations labeled by epoch-boundary and delay bucket
pub static FCR_TAIL_CASES_TOTAL: LazyLock<Result<IntCounterVec>> = LazyLock::new(|| {
    try_create_int_counter_vec(
        "fcr_tail_cases_total",
        "Tail confirmations (delay >=2 slots) labeled by epoch_boundary and delay bucket",
        &["epoch_boundary", "delay_bucket"],
    )
});

pub static FCR_CONFIRMATION_TIME_SECONDS: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    // Buckets tailored for FCR confirmation delay (seconds), covering 0–120s.
    let buckets = vec![
        0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0,
        10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
        30.0, 45.0, 60.0, 90.0, 120.0,
    ];
    try_create_histogram_with_buckets(
        "fcr_confirmation_time_seconds",
        "Time taken for block confirmation in seconds",
        Ok(buckets),
    )
});

pub static FCR_VALIDATOR_SUPPORT_PERCENTAGE: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_validator_support_percentage",
        "Percentage of validator support for confirmed blocks",
    )
});

pub static FCR_BYZANTINE_THRESHOLD_PERCENTAGE: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fcr_byzantine_threshold_percentage",
        "Current Byzantine threshold percentage used by FCR",
    )
});

pub static FCR_COMMITTEE_WEIGHT_CALCULATION_TIME: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_committee_weight_calculation_seconds",
        "Time taken for committee weight calculations",
    )
});

pub static FCR_FFG_SUPPORT_CALCULATION_TIME: LazyLock<Result<Histogram>> = LazyLock::new(|| {
    try_create_histogram(
        "fcr_ffg_support_calculation_seconds",
        "Time taken for FFG support calculations",
    )
});

pub static FCR_METADATA_CACHE_SIZE: LazyLock<Result<IntGauge>> = LazyLock::new(|| {
    try_create_int_gauge(
        "fcr_metadata_cache_size",
        "Current size of FCR metadata cache",
    )
});

pub static FCR_EPOCH_BOUNDARY_TRANSITIONS: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_epoch_boundary_transitions_total",
        "Total number of epoch boundary transitions processed",
    )
});

// Confirmation outcome metrics
pub static FCR_CONFIRMATIONS_TOTAL: LazyLock<Result<IntCounter>> = LazyLock::new(|| {
    try_create_int_counter(
        "fcr_confirmations_total",
        "Total number of FCR confirmations (all delays)",
    )
});

pub static FCR_CONFIRMATIONS_BY_SLOTS: LazyLock<Result<IntCounterVec>> = LazyLock::new(|| {
    try_create_int_counter_vec(
        "fcr_confirmations_by_slots_total",
        "FCR confirmations bucketed by delay slots (labels: slots=0,1,2,ge3)",
        &["slots"],
    )
});

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
            set_gauge(&FCR_BYZANTINE_THRESHOLD_PERCENTAGE, fcr.beta_percentage() as i64);
        }
    } else {
        // FCR is disabled, set metrics to 0
        set_gauge(&FCR_SAFE_HEAD_SLOT_NUMBER, 0);
        set_gauge(&FCR_METADATA_CACHE_SIZE, 0);
        set_gauge(&FCR_BYZANTINE_THRESHOLD_PERCENTAGE, 0);
    }
}
