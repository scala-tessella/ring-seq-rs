# ring-seq

Circular (ring) sequence operations for Rust slices.

[![Crates.io](https://img.shields.io/crates/v/ring-seq.svg)](https://crates.io/crates/ring-seq)
[![docs.rs](https://docs.rs/ring-seq/badge.svg)](https://docs.rs/ring-seq)
[![License](https://img.shields.io/crates/l/ring-seq.svg)](https://github.com/scala-tessella/ring-seq-rs#license)

`ring-seq` extends `[T]` — and by `Deref` coercion `Vec<T>`, arrays, and
`Box<[T]>` — with operations that treat the sequence as **circular**: the
element after the last wraps back to the first.

Zero dependencies. Zero unsafe. `#[must_use]` on every method that returns a
value.

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
ring-seq = "0.2"
```

Then import the trait and call methods on any slice:

```rust
use ring_seq::RingSeq;

// Indexing wraps around
assert_eq!(*[10, 20, 30].apply_o(4), 20);

// Rotation produces a new Vec
assert_eq!([0, 1, 2].rotate_right(1), [2, 0, 1]);

// Comparison up to rotation
assert!([0, 1, 2].is_rotation_of(&[2, 0, 1]));

// Canonical (necklace) form for deduplication
assert_eq!([2, 0, 1].canonical(), [0, 1, 2]);

// Symmetry detection
assert_eq!([0, 1, 0, 1].rotational_symmetry(), 2);
```

## Operations

### Indexing

| Method | Description |
|---|---|
| `index_from(i)` | Normalize a circular index to `[0, len)` |
| `apply_o(i)` | Element at circular index (panics if empty) |

### Transforming

| Method | Description |
|---|---|
| `rotate_right(step)` | Rotate right by `step` (negative = left) |
| `rotate_left(step)` | Rotate left by `step` (negative = right) |
| `start_at(i)` | Rotate so index `i` is first |
| `reflect_at(i)` | Reflect and rotate so index `i` is first |

### Slicing

| Method | Description |
|---|---|
| `slice_o(from, to)` | Circular interval (can exceed ring length) |
| `contains_slice(s)` | Does the ring contain `s` circularly? |
| `index_of_slice(s, from)` | First circular position of `s` |
| `last_index_of_slice(s, end)` | Last circular position of `s` |
| `segment_length(pred, from)` | Length of prefix satisfying `pred` |
| `take_while(pred, from)` | Prefix satisfying `pred` |
| `drop_while(pred, from)` | Remainder after prefix |
| `span(pred, from)` | `(take_while, drop_while)` in one pass |

### Iterating

| Method | Description |
|---|---|
| `rotations()` | All `n` rotations (lazy) |
| `reflections()` | Original + reflection (lazy) |
| `reversions()` | Original + reversal (lazy) |
| `rotations_and_reflections()` | All `2n` variants (lazy) |
| `circular_windows(size, step)` | Sliding windows wrapping around |
| `circular_chunks(size)` | Fixed-size circular groups |
| `circular_enumerate(from)` | Elements paired with circular indices |

### Comparing

| Method | Description |
|---|---|
| `is_rotation_of(that)` | Same elements, possibly rotated? |
| `is_reflection_of(that)` | Same elements, possibly reflected? |
| `is_reversion_of(that)` | Same elements, possibly reversed? |
| `is_rotation_or_reflection_of(that)` | Either of the above? |
| `rotation_offset(that)` | `Some(k)` where `start_at(k) == that` |
| `hamming_distance(that)` | Positional mismatches |
| `min_rotational_hamming_distance(that)` | Minimum over all rotations |

### Necklace

| Method | Description |
|---|---|
| `canonical_index()` | Index of lex-smallest rotation (Booth's *O(n)*) |
| `canonical()` | Lex-smallest rotation (necklace form) |
| `bracelet()` | Lex-smallest under rotation + reflection |

### Symmetry

| Method | Description |
|---|---|
| `rotational_symmetry()` | Order of rotational symmetry |
| `symmetry_indices()` | Shift values for reflectional symmetry |
| `reflectional_symmetry_axes()` | Full axis geometry (`Vertex` / `Edge`) |
| `symmetry()` | Number of reflectional symmetry axes |

## Naming convention

Every method on `RingSeq` is circular by definition, so most use plain
Rust-idiomatic names. A few carry a distinguishing name to avoid shadowing
standard-library methods:

| This crate | Standard library |
|---|---|
| `apply_o` | `[]` indexing |
| `slice_o` | `[a..b]` slicing |
| `circular_windows` | `[T]::windows` |
| `circular_chunks` | `[T]::chunks` |
| `circular_enumerate` | `Iterator::enumerate` |

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
