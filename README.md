# ring-seq

Circular (ring) sequence operations for Rust slices.

[![Crates.io](https://img.shields.io/crates/v/ring-seq.svg)](https://crates.io/crates/ring-seq)
[![docs.rs](https://docs.rs/ring-seq/badge.svg)](https://docs.rs/ring-seq)
[![License](https://img.shields.io/crates/l/ring-seq.svg)](https://github.com/scala-tessella/ring-seq-rs#license)

Treat any `[T]`, `Vec<T>`, array, or `Box<[T]>` as a **circular** sequence
— the element after the last wraps back to the first.

- Zero dependencies. Zero `unsafe`. `#![no_std]` with a default `alloc`
  feature.
- All transforms return lazy views or iterators; allocation is opt-in.
- Single entry point: `slice.circular()` gives you a `Circular<T>`
  wrapper that hosts every operation.

## Quick start

```toml
[dependencies]
ring-seq = "0.3"
```

```rust
use ring_seq::AsCircular;

let r = [10, 20, 30].circular();

// Indexing wraps in both directions
assert_eq!(*r.apply(4), 20);
assert_eq!(*r.apply(-1), 30);

// Reindexed views — no allocation
let rotated: Vec<_> = r.rotate_right(1).iter().copied().collect();
assert_eq!(rotated, [30, 10, 20]);

// Comparison up to rotation / reflection
assert!(r.is_rotation_of(&[20, 30, 10]));
assert!(r.is_reflection_of(&[10, 30, 20]));

// Canonical (necklace) form — O(n) minimal rotation
assert_eq!(r.canonical(), [10, 20, 30]);

// Lazy iterators of views compose naturally
let firsts: Vec<i32> = r.rotations().map(|v| *v.apply(0)).collect();
assert_eq!(firsts, [10, 20, 30]);

// Symmetry detection
assert_eq!([0, 1, 0, 1].circular().rotational_symmetry(), 2);
```

## Operations on `Circular<T>`

### Indexing & iteration

| Method | Returns | Description |
|---|---|---|
| `apply(i)` / `r[i]` | `&T` | Element at circular index (panics if empty) |
| `get(i)` | `Option<&T>` | Non-panicking `apply` |
| `index_from(i)` | `usize` | Normalize a circular index to `[0, len)` |
| `iter()` | `CircularIter` | Walk the view's elements (lazy, double-ended) |

`Circular` also implements `IntoIterator`, so `for x in ring.circular()`
walks the view directly.

### Reindexed views (lazy)

| Method | Returns | Description |
|---|---|---|
| `start_at(i)` | `Circular` | View starts at circular index `i` |
| `rotate_left(step)` | `Circular` | Shift left by `step` (negative = right) |
| `rotate_right(step)` | `Circular` | Shift right by `step` (negative = left) |
| `reflect_at(i)` | `Circular` | Reflect around index `i` |

### Bounded iteration (lazy)

| Method | Returns | Description |
|---|---|---|
| `slice(from, to)` | `CircularIter` | `max(to - from, 0)` elements, wrapping |
| `take_while(pred, from)` | `impl Iterator` | Prefix satisfying `pred` (≤ one lap) |
| `drop_while(pred, from)` | `impl Iterator` | Remainder after the prefix |
| `enumerate(from)` | `Enumerate` | `(&T, ring_index)` pairs |

### Iterators of views (lazy)

| Method | Yields | Description |
|---|---|---|
| `rotations()` | `Circular` × `n` | Every rotation |
| `reflections()` | `Circular` × 2 | Original + `reflect_at(0)` |
| `reversions()` | `Circular` × 2 | Original + reverse |
| `rotations_and_reflections()` | `Circular` × `2n` | All dihedral variants |
| `windows(size)` | `CircularIter` × `n` | Sliding windows, step 1 |
| `chunks(size)` | `CircularIter` × `ceil(n/size)` | Non-overlapping chunks |

### Comparison

| Method | Description |
|---|---|
| `is_rotation_of(other)` | Same elements, possibly rotated? |
| `is_reflection_of(other)` | Equals `self` or `self.reflect_at(0)` |
| `is_reversion_of(other)` | Equals `self` or its reverse |
| `is_rotation_or_reflection_of(other)` | Any dihedral variant |
| `rotation_offset(other)` | `Some(k)` where `self.start_at(k) == other` |
| `hamming_distance(other)` | Positional mismatches |
| `min_rotational_hamming_distance(other)` | Minimum over all rotations |
| `contains_slice(needle)` | Does `needle` appear circularly? |
| `index_of_slice(needle, from)` | First view position where `needle` matches |

The rotation/needle searches and symmetry counts are intentionally
simple quadratic scans (`O(n·m)` / `O(n²)`); `canonical_index` is `O(n)`.

### Necklace & symmetry

| Method | Returns | Notes |
|---|---|---|
| `canonical_index()` | `usize` | Index of lex-smallest rotation (*O(n)* time, *O(1)* space) |
| `canonical()` | `Vec<T>` | Lex-smallest rotation (`alloc`) |
| `bracelet()` | `Vec<T>` | Lex-smallest under rotation + reflection (`alloc`) |
| `rotational_symmetry()` | `usize` | Order of rotational symmetry |
| `symmetry()` | `usize` | Number of reflectional axes |
| `symmetry_indices()` | `Vec<usize>` | Shifts where the view equals its reverse (`alloc`) |
| `reflectional_symmetry_axes()` | `Vec<(AxisLocation, AxisLocation)>` | Full axis geometry (`alloc`) |

### Materialization

| Method | Returns | Notes |
|---|---|---|
| `to_vec()` | `Vec<T>` | Materialize the view (`alloc`, `T: Clone`) |

## `no_std`

```toml
[dependencies]
ring-seq = { version = "0.3", default-features = false }
```

Disabling the default `alloc` feature drops only the methods that return
owned collections (`canonical`, `bracelet`, `symmetry_indices`,
`reflectional_symmetry_axes`, `to_vec`). Everything else — the `Circular`
wrapper, every reindexed-view method, every iterator, and every scalar
query including `canonical_index` — depends only on `core`.

CI verifies the crate compiles for `wasm32-unknown-unknown` both with and
without `alloc`.

## Use cases

- **Bioinformatics** — circular DNA/RNA sequence alignment and comparison
- **Graphics** — polygon vertex manipulation, closed curve operations
- **Procedural generation** — tile rings, symmetry-aware pattern generation
- **Music theory** — pitch-class sets, chord inversions
- **Combinatorics** — necklace/bracelet enumeration, Burnside's lemma
- **Embedded / robotics** — circular sensor arrays, rotary encoder positions

## Minimum Rust version

1.63

## Other languages

The same library, adapted for the specific idiom, is available also for:

- Python — [ring-seq-py](https://github.com/scala-tessella/ring-seq-py)
- Scala — [ring-seq](https://github.com/scala-tessella/ring-seq)

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
