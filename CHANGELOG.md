# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI verifies the crate compiles for `wasm32-unknown-unknown` both with
  and without the `alloc` feature. The library was already wasm-portable
  by construction (no_std, zero deps, zero unsafe, no I/O); the CI guard
  makes the support claim binding.

## [0.3.0] - 2026-05-17

Full structural redesign. The `RingSeq` trait is gone; every circular
operation now lives on a new [`Circular`] wrapper reached via
[`AsCircular::circular`]. See [ADR 0001](docs/adr/0001-circular-wrapper-and-no-std.md).

### Added

- New [`Circular<'a, T>`] wrapper carrying `(slice, offset, reflected)`.
  Reach it through `slice.circular()`. All operations on it are lazy and
  allocation-free.
- New [`AsCircular`] trait with a single method `circular()`, implemented
  for `[T]` (covers `Vec<T>`, arrays, and `Box<[T]>` via deref).
- New element iterator [`CircularIter`] and view iterators [`Rotations`],
  [`Reflections`], [`Reversions`], [`RotationsAndReflections`], and
  [`Windows`]. Rotations-family iterators yield `Circular` views;
  `Windows` (also returned by `chunks`) yields `CircularIter`s.
- New [`Enumerate`] iterator yielding `(&T, ring_index)` pairs.
- `#![no_std]`. New `alloc` feature (default-on) gates the methods that
  return owned collections (`canonical`, `bracelet`, `symmetry_indices`,
  `reflectional_symmetry_axes`, `to_vec`) and the `canonical_index`
  implementation (Booth's algorithm allocates internally). Pure `core`
  is enough for the rest.

### Removed (breaking)

- The `RingSeq` trait and all its methods. Replaced by methods on
  [`Circular`].
- The `_o` suffix and `circular_` prefix on method names: `apply_o`,
  `slice_o`, `circular_windows`, `circular_chunks`, `circular_enumerate`
  become `apply`, `slice`, `windows`, `chunks`, `enumerate` on the
  wrapper.
- `SlidingO` (the old windows/chunks iterator). Replaced by `Windows`.
- The `span(pred, from)` convenience: use `take_while` and `drop_while`
  on the wrapper directly.

### Changed (breaking)

- Methods that previously returned `Vec<T>` (rotations, slicing, etc.)
  now return either a `Circular` view or an iterator. To get a `Vec`,
  call `.iter().cloned().collect()` or `.to_vec()` on the result.
- `windows(size)` now uses an implicit step of 1; the previous
  `circular_windows(size, step)` step parameter is no longer exposed.
- `chunks(size)` continues to partition the ring into
  `ceil(n / size)` non-overlapping chunks (matches v0.2 semantics).

### Migration

```rust
// v0.2
use ring_seq::RingSeq;
let v: Vec<i32> = ring.rotate_right(2);
let canon: Vec<i32> = ring.canonical();
for r in ring.rotations() { /* r is Vec<i32> */ }

// v0.3
use ring_seq::AsCircular;
let v: Vec<i32> = ring.circular().rotate_right(2).to_vec();
let canon: Vec<i32> = ring.circular().canonical();
for r in ring.circular().rotations() { /* r is Circular<i32> */ }
```

## [0.2.0] - 2026-04-20

### Changed

- **Breaking**: `circular_chunks` now partitions the sequence into
  `ceil(n / size)` non-overlapping chunks of exactly `size` elements each,
  with the last chunk wrapping across the seam when `size` does not divide
  `n`. Previously it delegated to `circular_windows(size, size)` and produced
  `n` overlapping windows.
- **Breaking**: `min_rotational_hamming_distance` trait bound relaxed from
  `T: PartialEq + Clone` to `T: PartialEq` (no rotation materialization
  needed).

### Performance

- `min_rotational_hamming_distance` now short-circuits the inner loop once
  the running count reaches the best-so-far, and skips remaining rotations
  after a perfect match.

## [0.1.3] - 2026-04-18

- Initial published releases.

[Unreleased]: https://github.com/scala-tessella/ring-seq-rs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/scala-tessella/ring-seq-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/scala-tessella/ring-seq-rs/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/scala-tessella/ring-seq-rs/releases/tag/v0.1.3
