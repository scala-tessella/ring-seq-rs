# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - Unreleased

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

## [0.1.3] - 2026-01-24

- Initial published releases.

[Unreleased]: https://github.com/scala-tessella/ring-seq-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/scala-tessella/ring-seq-rs/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/scala-tessella/ring-seq-rs/releases/tag/v0.1.3
