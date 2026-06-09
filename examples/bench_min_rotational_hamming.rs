//! Micro-benchmark for `min_rotational_hamming_distance`.
//!
//! Mirrors the three scenarios in the Scala twin's ComparingBench:
//!   * rotation  — best case, distance 0 (`a` vs a rotation of `a`)
//!   * oneOff    — mid case, distance 1
//!   * reversed  — worst case, distance close to `n`
//!
//! Run with: `cargo run --release --example bench_min_rotational_hamming`.
//!
//! Note: this example needs Rust 1.66+ for `std::hint::black_box`. The
//! library itself keeps its 1.63 MSRV; examples are dev-only targets and
//! are never built by crate consumers.
#![allow(clippy::incompatible_msrv)]

use std::hint::black_box;
use std::time::Instant;

use ring_seq::AsCircular;

fn rotated(a: &[u32], k: usize) -> Vec<u32> {
    let n = a.len();
    (0..n).map(|i| a[(i + k) % n]).collect()
}

fn one_off(a: &[u32]) -> Vec<u32> {
    let mut v = a.to_vec();
    if !v.is_empty() {
        v[0] = v[0].wrapping_add(1);
    }
    v
}

fn reversed(a: &[u32]) -> Vec<u32> {
    let mut v = a.to_vec();
    v.reverse();
    v
}

fn bench_case(label: &str, n: usize, iters: u64, a: &[u32], b: &[u32]) {
    // warm-up
    for _ in 0..(iters / 10).max(1) {
        black_box(a.circular().min_rotational_hamming_distance(black_box(b)));
    }
    let start = Instant::now();
    let mut acc: usize = 0;
    for _ in 0..iters {
        acc = acc.wrapping_add(black_box(
            a.circular().min_rotational_hamming_distance(black_box(b)),
        ));
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "{:<10} n={:<5} iters={:<10} total={:>10.3?} ns/op={:>12.0} (acc={})",
        label, n, iters, elapsed, ns_per_op, acc
    );
}

fn iters_for(n: usize) -> u64 {
    match n {
        16 => 1_000_000,
        256 => 10_000,
        4096 => 50,
        _ => 1_000,
    }
}

fn main() {
    for &n in &[16usize, 256, 4096] {
        let base: Vec<u32> = (0..n as u32).collect();
        let rot = rotated(&base, n / 3);
        let off = one_off(&base);
        let rev = reversed(&base);
        let iters = iters_for(n);

        bench_case("rotation", n, iters, &base, &rot);
        bench_case("oneOff  ", n, iters, &base, &off);
        bench_case("reversed", n, iters, &base, &rev);
        println!();
    }
}
