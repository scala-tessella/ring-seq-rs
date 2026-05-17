# ADR 0001 — Circular wrapper API and `no_std` support

- **Status:** Accepted
- **Date:** 2026-05-17
- **Target release:** v0.3.0

## Context

v0.2.0 exposes a single `RingSeq` trait directly on `[T]`. Its transform
methods (`rotate_right`, `start_at`, `reflect_at`, `slice_o`, `take_while`,
`drop_while`, `canonical`, `bracelet`, …) all return owned `Vec<T>`. The
iterator-yielding methods (`Rotations`, `Reflections`, …) yield `Vec<T>` per
step.

Three problems follow from this shape:

1. **Forced allocation.** Every transform allocates, even when the caller only
   needs a few elements (`take(3)`, `find`, `eq`, `any`). Iterators of
   iterators are effectively `Vec<Vec<T>>`, paying twice.
2. **No `no_std` story.** The trait can't be used without `alloc`, and the
   most useful primitives (lazy index walks, scalar queries) don't actually
   need it.
3. **Awkward names.** Because methods live directly on `[T]` alongside its
   native methods, every operation needs a disambiguating suffix or prefix:
   `apply_o`, `slice_o`, `circular_windows`, `circular_chunks`,
   `circular_enumerate`. The suffixes carry no semantic content; they exist
   only to avoid collisions.

A two-prototype spike (`examples/wrapper_prototype.rs` and
`examples/wrapper_prototype_unified.rs`) explored a wrapper-based redesign and
validated both the ergonomics of chained operations
(`ring.circular().rotations().map(|r| r.canonical_index())`) and the
algebraic identities the unified `(offset, reflected)` model relies on.

The crate is pre-1.0 with no known external users, so backwards compatibility
is not a constraint.

## Decision

Replace the `RingSeq` trait with a wrapper type reached via an extension
method on slices.

### Surface

```rust
pub struct Circular<'a, T> {
    ring: &'a [T],
    offset: usize,
    reflected: bool,
}

pub trait AsCircular<T> {
    fn circular(&self) -> Circular<'_, T>;
}

impl<T> AsCircular<T> for [T] { /* ... */ }
```

`Circular<'a, T>` is the single home for every circular operation:

- **Reindexed views** (return `Circular<'a, T>`): `rotate_right`,
  `rotate_left`, `start_at`, `reflect_at`, `slice`.
- **Element iteration**: `iter() -> CircularIter<'a, T>` yielding `&'a T`.
- **Iterators of views** (return iterators yielding `Circular<'a, T>`):
  `rotations`, `reflections`, `reversions`, `rotations_and_reflections`.
- **Sliding iterators** (return iterators yielding `CircularIter<'a, T>`):
  `windows`, `chunks`. See "Alternatives considered" for why these yield a
  bounded element iterator rather than a `Circular`.
- **Predicate-bounded**: `take_while`, `drop_while`, `enumerate`.
- **Scalar queries**: `apply`, `is_rotation_of`, `is_reflection_of`,
  `canonical_index`, `bracelet_index`.
- **Variable-length results** (allocate, gated by `feature = "alloc"`):
  `symmetry_indices`, `reflectional_symmetry_axes`, `canonical`, `bracelet`.

### Naming

Inside `Circular`, names drop the disambiguators they needed on `[T]`. The
wrapper itself is the context: `slice`, `windows`, `chunks`, `enumerate`,
`apply` — no `_o`, no `circular_` prefix.

### Implementation invariants

- `Circular` is value-typed and trivially copyable. `Clone` and `Copy` are
  hand-implemented (not `#[derive]`d) so they do not pick up a phantom
  `T: Clone` bound. The derive trap was caught during prototyping when
  iterator `next` methods could not move `self.base` through `&mut self`.
- A single `map_index(pos) -> usize` is the only place that translates a
  view position to a slice index. Every other operation routes through it,
  ensuring that the `(offset, reflected)` algebra stays consistent.

### `no_std`

The crate becomes `#![no_std]`. A default-on `alloc` feature gates anything
that returns an owned collection: doc-tests, the variable-length algorithms
listed above, and a `to_vec()` convenience method on `CircularIter`. The
core wrapper and its element-yielding iterators depend only on `core`.

### Compatibility

None preserved. v0.3.0 is fully breaking. The mechanical migration for any
old call site is:

```rust
//   slice.foo(x)                       // v0.2
slice.circular().foo(x).iter().cloned().collect::<Vec<_>>() // v0.3
```

For most call sites the lazy form (`.iter()` then a real combinator) is
preferable and the `.collect()` falls away.

## Consequences

### Positive

- **Allocation becomes a caller choice.** Predicates and short-circuiting
  combinators no longer pay for a full `Vec`.
- **`no_std + alloc` falls out for free.** The split between core ops and
  collection-returning ops aligns naturally with the feature boundary.
- **API names are shorter and self-consistent.** No more `_o` suffixes.
- **Operations chain.** `rotate_right(k).reflect_at(0)` returns a `Circular`
  that other methods accept; iterators of rotations yield `Circular` views
  rather than opaque `Vec`s.
- **One canonical type, one entry point.** Easier to document, easier to
  discover via rustdoc.

### Negative

- **One extra method call at every call site** (`.circular()`).
- **Fully breaking release.** Acceptable here (pre-1.0, no users) but worth
  noting in the changelog.
- **Two iterator categories.** `CircularIter<&T>` for elements, and the
  per-operation iterators (`Rotations` etc.) that yield `Circular` views.
  Manageable, but more surface to learn than a single iterator type.
- **Algorithmic implementations carry over unchanged.** v0.2.0 already
  uses Booth's O(n) for `canonical_index`; v0.3.0 keeps that implementation
  and simply relocates it onto `Circular`. The redesign is structural, not
  algorithmic.

## Alternatives considered

- **Add `_iter` suffixed methods on the existing `RingSeq` trait.** Smaller
  diff, but the end state is worse: every operation exists in two forms
  (`rotate_right` and `rotate_right_iter`, `circular_windows` and
  `circular_windows_iter`), the trait surface doubles, and the disambiguator
  suffixes stay forever.
- **Keep `RingSeq` as thin wrappers over `Circular`.** Considered to
  preserve a Vec-returning convenience layer. Rejected because no users
  exist to preserve, and the trait's naming awkwardness is precisely what
  the redesign is meant to remove.
- **Split `Circular` and `CircularView` (with offset).** This was the v1
  prototype shape. Collapsed into a single unified type in the v2 prototype:
  `Circular` is just `CircularView { offset: 0 }`, so two types are one too
  many.
- **Separate `ReflectedView` type instead of a `reflected: bool` flag.**
  Doubles the surface area for no semantic gain; the prototype's two extra
  branches in `map_index` are cheaper and keep the API uniform.
- **`windows` / `chunks` yielding `Circular<'a, T>` of window length.**
  Considered during initial design and noted as needing a pressure test.
  During implementation we ruled it out: it would require adding a `len`
  field to `Circular` and propagating it through every existing method
  (`apply`, `iter`, `start_at`, `reflect_at`, `map_index`), with the
  reflected-view math gaining an extra `% len` step. The benefit — "every
  yielded item supports every wrapper method" — is mostly theoretical for
  windows, since a window is conceptually a sub-range, not a smaller ring.
  Calling `canonical_index` on a 3-element window of a 7-element ring is
  not meaningfully different from calling it on a `Vec<T>` of length 3.
  We chose `CircularIter` instead: simpler, allocation-free, and the rare
  caller who needs `Circular`-style methods on a window can `.collect()`
  to a Vec and call `.circular()` on it.

