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

criterion_group!(benches, bench_adjust, bench_cross_epoch_estimate, bench_ffg_weight, bench_full_validator_set_covered);
criterion_main!(benches);


