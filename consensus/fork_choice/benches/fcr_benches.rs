use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use fork_choice::fast_confirmation::bench_api as fcb;
use types::{Epoch, Slot};

fn bench_adjust(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/adjust_committee_weight");
    for &estimate in &[1_000_000u64, 1_000_000_000u64, 10_000_000_000u64] {
        g.bench_with_input(BenchmarkId::new("adjust", estimate), &estimate, |b, &e| {
            b.iter(|| fcb::bench_adjust_committee_weight_estimate(e))
        });
    }
}

fn bench_cross_epoch_estimate(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/cross_epoch_weight_estimate");
    let tab = 32_000_000_000u64; // synthetic total active balance (gwei)
    // Ranges: same-epoch, boundary, multi-slot cross-epoch
    let ranges = vec![
        (Slot::new(0), Slot::new(3)),            // small same-epoch
        (Slot::new(31), Slot::new(32)),          // boundary 31->32
        (Slot::new(28), Slot::new(40)),          // cross-epoch
    ];
    for (start, end) in ranges {
        let name = format!("{}-{}", start.as_u64(), end.as_u64());
        g.bench_function(BenchmarkId::new("estimate", name), |b| {
            b.iter(|| fcb::bench_calculate_cross_epoch_weight_estimate(start, end, tab))
        });
    }
}

fn bench_ffg_weight(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/ffg_weight_till_slot");
    let tab = 32_000_000_000u64;
    for slot in [0u64, 1, 15, 31, 32, 40].into_iter().map(Slot::new) {
        g.bench_with_input(BenchmarkId::new("ffg_weight", slot.as_u64()), &slot, |b, &s| {
            b.iter(|| fcb::bench_get_ffg_weight_till_slot(s, Epoch::new(0), tab))
        });
    }
}

fn bench_full_validator_set_covered(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/full_validator_set_covered");
    let cases = vec![
        (Slot::new(0), Slot::new(31)),
        (Slot::new(0), Slot::new(32)),
        (Slot::new(1), Slot::new(40)),
    ];
    for (start, end) in cases {
        let name = format!("{}-{}", start.as_u64(), end.as_u64());
        g.bench_function(BenchmarkId::new("covered", name), |b| {
            b.iter(|| fcb::bench_is_full_validator_set_covered(start, end))
        });
    }
}

fn bench_is_one_confirmed_math(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/is_one_confirmed_math");
    // (S, W, proposer_score, beta%)
    let cases: &[(u64, u64, u64, u64)] = &[
        (10_000_000_000, 15_000_000_000, 0, 25),
        (18_000_000_000, 20_000_000_000, 100_000, 25),
        (30_000_000_000, 32_000_000_000, 500_000, 33),
        (6_000_000_000, 10_000_000_000, 0, 10),
    ];
    for &(s, w, pb, beta) in cases {
        let name = format!("S{}_W{}_pb{}_b{}", s, w, pb, beta);
        g.bench_function(BenchmarkId::new("ineq", name), |b| {
            b.iter(|| fcb::bench_is_one_confirmed_math(s, w, pb, beta))
        });
    }
}

fn bench_is_one_confirmed_w_estimate(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/is_one_confirmed_w_estimate");
    let tab = 32_000_000_000u64;
    let ranges = vec![
        (Slot::new(0), Slot::new(3)),
        (Slot::new(31), Slot::new(32)),
        (Slot::new(28), Slot::new(40)),
    ];
    // Fixed support values spanning below/above threshold in different ranges.
    let supports = [8_000_000_000u64, 16_000_000_000u64, 24_000_000_000u64];
    let proposer = 200_000u64;
    let betas = [10u64, 25u64, 33u64];
    for (start, end) in &ranges {
        for &s in &supports {
            for &beta in &betas {
                let name = format!(
                    "{}-{}_S{}_b{}",
                    start.as_u64(),
                    end.as_u64(),
                    s,
                    beta
                );
                g.bench_function(BenchmarkId::new("ineq_w_est", name), |b| {
                    b.iter(|| fcb::bench_is_one_confirmed_w_estimate(s, tab, *start, *end, proposer, beta))
                });
            }
        }
    }
}

// Fork choice integration overhead benchmarks
fn bench_fork_choice_overhead(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/fork_choice_overhead");
    
    // Test FCR overhead on get_head() operations
    g.bench_function("get_head_with_fcr", |b| {
        b.iter(|| {
            // This would test the actual get_head() with FCR enabled
            // For now, we'll benchmark the FCR update hook
            fcb::bench_update_fcr_after_find_head()
        });
    });
    
    g.bench_function("get_head_without_fcr", |b| {
        b.iter(|| {
            // This would test get_head() without FCR
            // For now, we'll benchmark a no-op
            fcb::bench_no_op()
        });
    });
}

// Validator scaling benchmarks
fn bench_validator_scaling(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/validator_scaling");
    
    // Test with different validator counts: 100K, 500K, 1M+
    let validator_counts = [100_000u64, 500_000u64, 1_000_000u64];
    
    for &count in &validator_counts {
        g.bench_with_input(BenchmarkId::new("committee_weight", count), &count, |b, &count| {
            b.iter(|| fcb::bench_committee_weight_with_validators(count))
        });
        
        g.bench_with_input(BenchmarkId::new("ffg_support", count), &count, |b, &count| {
            b.iter(|| fcb::bench_ffg_support_with_validators(count))
        });
    }
}

// Memory usage benchmarks
fn bench_memory_usage(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/memory_usage");
    
    // Test FCR metadata HashMap growth
    g.bench_function("metadata_growth", |b| {
        b.iter(|| fcb::bench_fcr_metadata_growth())
    });
    
    // Test pruning effectiveness
    g.bench_function("pruning_effectiveness", |b| {
        b.iter(|| fcb::bench_fcr_pruning())
    });
    
    // Test memory usage vs validator count
    let validator_counts = [100_000u64, 500_000u64, 1_000_000u64];
    for &count in &validator_counts {
        g.bench_with_input(BenchmarkId::new("memory_vs_validators", count), &count, |b, &count| {
            b.iter(|| fcb::bench_memory_usage_with_validators(count))
        });
    }
}

// Production scenario benchmarks
fn bench_production_scenarios(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/production_scenarios");
    
    // Test epoch boundary transitions
    g.bench_function("epoch_boundary_transition", |b| {
        b.iter(|| fcb::bench_epoch_boundary_transition())
    });
    
    // Test reorg scenarios
    g.bench_function("reorg_detection", |b| {
        b.iter(|| fcb::bench_reorg_detection())
    });
    
    // Test late attestation handling
    g.bench_function("late_attestation_handling", |b| {
        b.iter(|| fcb::bench_late_attestation_handling())
    });
}

// Safe head calculation performance (like Prysm)
fn bench_safe_head_performance(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/safe_head_performance");
    
    // Test safe head calculation
    g.bench_function("safe_head_calculation", |b| {
        b.iter(|| fcb::bench_safe_head_calculation())
    });
    
    // Test safe head reorg detection
    g.bench_function("safe_head_reorg", |b| {
        b.iter(|| fcb::bench_safe_head_reorg())
    });
    
    // Test safe head advancement
    g.bench_function("safe_head_advancement", |b| {
        b.iter(|| fcb::bench_safe_head_advancement())
    });
}

// Cross-epoch performance benchmarks
fn bench_cross_epoch_performance(c: &mut Criterion) {
    let mut g = c.benchmark_group("fcr/cross_epoch_performance");
    
    // Test cross-epoch confirmation advancement
    g.bench_function("cross_epoch_confirmation", |b| {
        b.iter(|| fcb::bench_cross_epoch_confirmation())
    });
    
    // Test epoch boundary weight calculations
    g.bench_function("epoch_boundary_weights", |b| {
        b.iter(|| fcb::bench_epoch_boundary_weights())
    });
}

criterion_group!(
    benches,
    bench_adjust,
    bench_cross_epoch_estimate,
    bench_ffg_weight,
    bench_full_validator_set_covered,
    bench_is_one_confirmed_math,
    bench_is_one_confirmed_w_estimate,
    bench_fork_choice_overhead,
    bench_validator_scaling,
    bench_memory_usage,
    bench_production_scenarios,
    bench_safe_head_performance,
    bench_cross_epoch_performance
);

criterion_main!(benches);