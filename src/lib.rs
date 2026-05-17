//! Circular (ring) sequence operations for Rust slices.
//!
//! Reach the circular API by calling [`AsCircular::circular`] on any slice,
//! [`Vec<T>`](alloc::vec::Vec), array, or [`Box<[T]>`](alloc::boxed::Box).
//! The returned [`Circular`] wrapper is the single home for every circular
//! operation; every transform it offers is lazy and allocation-free.
//!
//! # Quick start
//!
//! ```
//! use ring_seq::AsCircular;
//!
//! let r = [10, 20, 30].circular();
//!
//! // Indexing wraps in both directions
//! assert_eq!(*r.apply(4), 20);
//! assert_eq!(*r.apply(-1), 30);
//!
//! // Reindexed views — no allocation
//! let rotated: Vec<_> = r.rotate_right(1).iter().copied().collect();
//! assert_eq!(rotated, vec![30, 10, 20]);
//!
//! // Comparisons up to rotation/reflection
//! assert!(r.is_rotation_of(&[20, 30, 10]));
//! assert!(r.is_reflection_of(&[10, 30, 20]));
//!
//! // Canonical (necklace) form — uses Booth's O(n)
//! assert_eq!(r.canonical(), vec![10, 20, 30]);
//!
//! // Lazy iterators of views
//! let firsts: Vec<i32> = r.rotations().map(|v| *v.apply(0)).collect();
//! assert_eq!(firsts, vec![10, 20, 30]);
//! ```
//!
//! # `no_std`
//!
//! The crate is `#![no_std]`. The default `alloc` feature enables the
//! methods that return owned collections ([`Circular::to_vec`],
//! [`Circular::canonical`], [`Circular::bracelet`],
//! [`Circular::symmetry_indices`],
//! [`Circular::reflectional_symmetry_axes`]) and the implementation of
//! [`Circular::canonical_index`] (Booth's algorithm allocates internally).
//! With `--no-default-features` the wrapper and its element/view iterators
//! remain available and depend only on `core`.

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

mod circular;

pub use circular::{
    AsCircular, Circular, CircularIter, Enumerate, Reflections, Reversions, Rotations,
    RotationsAndReflections, Windows,
};

// ============================================================================
// AxisLocation
// ============================================================================

/// A location on a circular sequence where a reflectional-symmetry axis
/// passes.
///
/// # Variants
///
/// * [`Vertex`](AxisLocation::Vertex) — the axis passes through the
///   element at the given index.
/// * [`Edge`](AxisLocation::Edge) — the axis passes between two
///   consecutive elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisLocation {
    /// The axis passes through the element at this index.
    Vertex(usize),
    /// The axis passes between the elements at these two consecutive indices.
    /// The invariant `j == (i + 1) % n` is maintained by
    /// [`AxisLocation::edge`].
    Edge(usize, usize),
}

impl AxisLocation {
    /// Constructs an [`Edge`](AxisLocation::Edge) between consecutive
    /// elements of a ring of size `n`, starting at index `i`.
    ///
    /// The second index is computed as `(i + 1) % n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AxisLocation;
    ///
    /// assert_eq!(AxisLocation::edge(2, 3), AxisLocation::Edge(2, 0));
    /// assert_eq!(AxisLocation::edge(0, 4), AxisLocation::Edge(0, 1));
    /// ```
    #[must_use]
    pub fn edge(i: usize, n: usize) -> Self {
        assert!(n > 0, "ring size must be positive");
        let ii = i % n;
        AxisLocation::Edge(ii, (ii + 1) % n)
    }
}

#[cfg(test)]
mod axis_tests {
    use super::*;

    #[test]
    fn edge_constructor() {
        assert_eq!(AxisLocation::edge(2, 3), AxisLocation::Edge(2, 0));
        assert_eq!(AxisLocation::edge(0, 4), AxisLocation::Edge(0, 1));
        assert_eq!(AxisLocation::edge(3, 4), AxisLocation::Edge(3, 0));
    }

    #[test]
    #[should_panic(expected = "ring size must be positive")]
    fn edge_zero_size_panics() {
        let _ = AxisLocation::edge(0, 0);
    }
}
