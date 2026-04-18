//! Circular (ring) sequence operations for Rust slices.
//!
//! `ring-seq` extends `[T]` (and, through [`Deref`] coercion, [`Vec<T>`],
//! arrays, and [`Box<[T]>`](Box)) with operations that treat the sequence as
//! **circular** — the element after the last wraps back to the first.
//!
//! # Quick start
//!
//! ```
//! use ring_seq::RingSeq;
//!
//! // Indexing wraps around
//! assert_eq!(*[10, 20, 30].apply_o(4), 20);
//!
//! // Rotation produces a new Vec
//! assert_eq!([0, 1, 2].rotate_right(1), vec![2, 0, 1]);
//!
//! // Comparison up to rotation
//! assert!([0, 1, 2].is_rotation_of(&[2, 0, 1]));
//!
//! // Canonical (necklace) form for deduplication
//! assert_eq!([2, 0, 1].canonical(), vec![0, 1, 2]);
//!
//! // Symmetry detection
//! assert_eq!([0, 1, 0, 1].rotational_symmetry(), 2);
//! ```
//!
//! # Naming convention
//!
//! Most methods use plain, Rust-idiomatic names (`take_while`, `span`,
//! `contains_slice`, etc.) since every method on [`RingSeq`] is circular by
//! definition.
//!
//! A few methods carry a distinguishing name to avoid confusion with
//! standard-library counterparts:
//!
//! * `apply_o` — circular element access (the `_o` signals a circular index).
//! * `slice_o` — circular slicing (`slice` is a fundamental Rust type).
//! * `circular_windows`, `circular_chunks`, `circular_enumerate` — circular
//!   variants of [`windows`](slice::windows), [`chunks`](slice::chunks), and
//!   [`enumerate`](Iterator::enumerate).
//!
//! # Interaction with `[T]::rotate_left` / `[T]::rotate_right`
//!
//! The standard library provides in-place `rotate_left(&mut self, mid: usize)`
//! and `rotate_right(&mut self, mid: usize)` on mutable slices. This crate's
//! methods have the same names but differ in signature: they take `&self` and
//! an `isize` step (which may be negative), and return a new `Vec<T>`. The
//! compiler resolves calls unambiguously based on mutability and argument type.
//! If you need to force the circular variant on a `&mut` slice, call
//! `.as_ref()` or reborrow as `&*slice` first.

mod iterators;

pub use iterators::{Reflections, Reversions, Rotations, RotationsAndReflections, SlidingO};

// ============================================================================
// AxisLocation
// ============================================================================

/// A location on a circular sequence where a reflectional-symmetry axis passes.
///
/// # Variants
///
/// * [`Vertex`](AxisLocation::Vertex) — the axis passes through the element at
///   the given index.
/// * [`Edge`](AxisLocation::Edge) — the axis passes between two consecutive
///   elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisLocation {
    /// The axis passes through the element at this index.
    Vertex(usize),
    /// The axis passes between the elements at these two consecutive indices.
    /// The invariant `j == (i + 1) % n` is maintained by [`AxisLocation::edge`].
    Edge(usize, usize),
}

impl AxisLocation {
    /// Constructs an [`Edge`](AxisLocation::Edge) between consecutive elements
    /// of a ring of size `n`, starting at index `i`.
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

// ============================================================================
// RingSeq trait
// ============================================================================

/// Circular (ring) sequence operations on `[T]`.
///
/// Import this trait to gain circular methods on slices, [`Vec`]s, and arrays:
///
/// ```
/// use ring_seq::RingSeq;
/// ```
pub trait RingSeq<T> {
    // ── Indexing ────────────────────────────────────────────────────────

    /// Normalizes a circular index to `[0, len)`.
    ///
    /// Uses Euclidean remainder so that negative indices wrap correctly.
    ///
    /// # Panics
    ///
    /// Panics if the slice is empty (division by zero).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([10, 20, 30].index_from(4), 1);
    /// assert_eq!([10, 20, 30].index_from(-1), 2);
    /// ```
    #[must_use]
    fn index_from(&self, i: isize) -> usize;

    /// Returns a reference to the element at circular index `i`.
    ///
    /// # Panics
    ///
    /// Panics if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!(*[10, 20, 30].apply_o(3), 10);
    /// assert_eq!(*[10, 20, 30].apply_o(-1), 30);
    /// ```
    #[must_use]
    fn apply_o(&self, i: isize) -> &T;

    // ── Transforming ───────────────────────────────────────────────────

    /// Rotates the sequence to the right by `step` positions, returning a new
    /// `Vec`.
    ///
    /// A negative step rotates to the left.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2].rotate_right(1), vec![2, 0, 1]);
    /// assert_eq!([0, 1, 2].rotate_right(-1), vec![1, 2, 0]);
    /// ```
    #[must_use]
    fn rotate_right(&self, step: isize) -> Vec<T>
    where
        T: Clone;

    /// Rotates the sequence to the left by `step` positions, returning a new
    /// `Vec`.
    ///
    /// A negative step rotates to the right.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2].rotate_left(1), vec![1, 2, 0]);
    /// ```
    #[must_use]
    fn rotate_left(&self, step: isize) -> Vec<T>
    where
        T: Clone;

    /// Rotates the sequence so that circular index `i` becomes the first
    /// element.
    ///
    /// Equivalent to [`rotate_left`](RingSeq::rotate_left).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2].start_at(1), vec![1, 2, 0]);
    /// ```
    #[must_use]
    fn start_at(&self, i: isize) -> Vec<T>
    where
        T: Clone;

    /// Reflects (reverses) the sequence and rotates so that circular index `i`
    /// is the first element.
    ///
    /// Equivalent to `start_at(i + 1)` followed by `reverse`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2].reflect_at(0), vec![0, 2, 1]);
    /// ```
    #[must_use]
    fn reflect_at(&self, i: isize) -> Vec<T>
    where
        T: Clone;

    // ── Slicing ────────────────────────────────────────────────────────

    /// Length of the longest prefix of elements, starting from circular index
    /// `from`, that satisfy `pred`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2].segment_length(|x| x % 2 == 0, 2), 2);
    /// ```
    #[must_use]
    fn segment_length(&self, pred: impl Fn(&T) -> bool, from: isize) -> usize;

    /// Takes the longest prefix of elements, starting from circular index
    /// `from`, that satisfy `pred`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2, 3, 4].take_while(|&x| x < 3, 1), vec![1, 2]);
    /// ```
    #[must_use]
    fn take_while(&self, pred: impl Fn(&T) -> bool, from: isize) -> Vec<T>
    where
        T: Clone;

    /// Drops the longest prefix of elements, starting from circular index
    /// `from`, that satisfy `pred`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2, 3, 4].drop_while(|&x| x < 3, 1), vec![3, 4, 0]);
    /// ```
    #[must_use]
    fn drop_while(&self, pred: impl Fn(&T) -> bool, from: isize) -> Vec<T>
    where
        T: Clone;

    /// Splits the circular sequence at the first element (starting from
    /// circular index `from`) that does **not** satisfy `pred`.
    ///
    /// Returns `(take_while(pred, from), drop_while(pred, from))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// let (a, b) = [0, 1, 2, 3, 4].span(|&x| x < 3, 1);
    /// assert_eq!(a, vec![1, 2]);
    /// assert_eq!(b, vec![3, 4, 0]);
    /// ```
    #[must_use]
    fn span(&self, pred: impl Fn(&T) -> bool, from: isize) -> (Vec<T>, Vec<T>)
    where
        T: Clone;

    /// Selects a circular interval of elements from index `from` (inclusive)
    /// to index `to` (exclusive).
    ///
    /// The resulting slice can be **longer** than the ring when `to - from`
    /// exceeds the ring length, because the ring repeats circularly.
    ///
    /// Returns an empty `Vec` when `from >= to` or the ring is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2].slice_o(-1, 4), vec![2, 0, 1, 2, 0]);
    /// assert_eq!([0, 1, 2].slice_o(1, 3), vec![1, 2]);
    /// ```
    #[must_use]
    fn slice_o(&self, from: isize, to: isize) -> Vec<T>
    where
        T: Clone;

    /// Tests whether this ring contains `slice` as a contiguous circular
    /// subsequence.
    ///
    /// The slice may wrap around the ring boundary and may even be longer
    /// than the ring (repeating elements are matched cyclically).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert!([0, 1, 2].contains_slice(&[2, 0, 1, 2, 0]));
    /// assert!(![0, 1, 2].contains_slice(&[1, 0]));
    /// ```
    #[must_use]
    fn contains_slice(&self, slice: &[T]) -> bool
    where
        T: PartialEq;

    /// Finds the first circular index at or after `from` where `slice` appears
    /// as a contiguous subsequence.
    ///
    /// Returns `None` if not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2].index_of_slice(&[2, 0, 1, 2, 0], 0), Some(2));
    /// ```
    #[must_use]
    fn index_of_slice(&self, slice: &[T], from: isize) -> Option<usize>
    where
        T: PartialEq;

    /// Finds the last circular index at or before `end` where `slice` appears
    /// as a contiguous subsequence.
    ///
    /// Returns `None` if not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!(
    ///     [0, 1, 2, 0, 1, 2].last_index_of_slice(&[2, 0], -1),
    ///     Some(5),
    /// );
    /// ```
    #[must_use]
    fn last_index_of_slice(&self, slice: &[T], end: isize) -> Option<usize>
    where
        T: PartialEq;

    // ── Iterating ──────────────────────────────────────────────────────

    /// Sliding windows of `size` elements, advancing by `step` each time,
    /// wrapping around the ring boundary.
    ///
    /// # Panics
    ///
    /// Panics if `size` or `step` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// let windows: Vec<_> = [0, 1, 2].circular_windows(2, 1).collect();
    /// assert_eq!(windows, vec![vec![0, 1], vec![1, 2], vec![2, 0]]);
    /// ```
    #[must_use]
    fn circular_windows(&self, size: usize, step: usize) -> SlidingO<T>
    where
        T: Clone;

    /// Fixed-size circular groups (like [`circular_windows`](RingSeq::circular_windows) with
    /// `step == size`).
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// let groups: Vec<_> = [0, 1, 2, 3, 4].circular_chunks(2).collect();
    /// assert_eq!(
    ///     groups,
    ///     vec![vec![0, 1], vec![2, 3], vec![4, 0], vec![1, 2], vec![3, 4]],
    /// );
    /// ```
    #[must_use]
    fn circular_chunks(&self, size: usize) -> SlidingO<T>
    where
        T: Clone;

    /// Pairs each element with its original (circular) index, starting from
    /// circular index `from`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!(
    ///     ['a', 'b', 'c'].circular_enumerate(1),
    ///     vec![('b', 1), ('c', 2), ('a', 0)],
    /// );
    /// ```
    #[must_use]
    fn circular_enumerate(&self, from: isize) -> Vec<(T, usize)>
    where
        T: Clone;

    /// All rotations of this ring.
    ///
    /// Yields `n` items for a non-empty ring (starting from the identity
    /// rotation), or a single empty `Vec` for an empty ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// let rots: Vec<_> = [0, 1, 2].rotations().collect();
    /// assert_eq!(rots, vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]]);
    /// ```
    #[must_use]
    fn rotations(&self) -> Rotations<'_, T>;

    /// The original ring and its reflection at index 0.
    ///
    /// Yields 2 items for a non-empty ring, or a single empty `Vec` for an
    /// empty ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// let refs: Vec<_> = [0, 1, 2].reflections().collect();
    /// assert_eq!(refs, vec![vec![0, 1, 2], vec![0, 2, 1]]);
    /// ```
    #[must_use]
    fn reflections(&self) -> Reflections<'_, T>;

    /// The original ring and its reversal.
    ///
    /// Yields 2 items for a non-empty ring, or a single empty `Vec` for an
    /// empty ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// let revs: Vec<_> = [0, 1, 2].reversions().collect();
    /// assert_eq!(revs, vec![vec![0, 1, 2], vec![2, 1, 0]]);
    /// ```
    #[must_use]
    fn reversions(&self) -> Reversions<'_, T>;

    /// All rotations of the original ring followed by all rotations of its
    /// reflection.
    ///
    /// Yields `2n` items for a non-empty ring, or a single empty `Vec` for an
    /// empty ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// let all: Vec<_> = [0, 1, 2].rotations_and_reflections().collect();
    /// assert_eq!(all, vec![
    ///     vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1],
    ///     vec![0, 2, 1], vec![2, 1, 0], vec![1, 0, 2],
    /// ]);
    /// ```
    #[must_use]
    fn rotations_and_reflections(&self) -> RotationsAndReflections<'_, T>
    where
        T: Clone;

    // ── Comparing ──────────────────────────────────────────────────────

    /// Tests whether this ring is a rotation of `that`.
    ///
    /// Two sequences are rotations of each other iff they have the same length
    /// and one appears as a contiguous substring inside the other repeated
    /// twice.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert!([0, 1, 2].is_rotation_of(&[1, 2, 0]));
    /// assert!(![0, 1, 2].is_rotation_of(&[0, 2, 1]));
    /// ```
    #[must_use]
    fn is_rotation_of(&self, that: &[T]) -> bool
    where
        T: PartialEq;

    /// Tests whether this ring is the reflection at index 0 of `that` (or
    /// identical to it).
    ///
    /// This checks a single specific reflection axis.  For rotation-insensitive
    /// comparison, see [`is_rotation_or_reflection_of`](RingSeq::is_rotation_or_reflection_of).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert!([0, 1, 2].is_reflection_of(&[0, 2, 1]));
    /// ```
    #[must_use]
    fn is_reflection_of(&self, that: &[T]) -> bool
    where
        T: PartialEq + Clone;

    /// Tests whether this ring is the reversal of `that` (or identical to it).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert!([0, 1, 2].is_reversion_of(&[2, 1, 0]));
    /// ```
    #[must_use]
    fn is_reversion_of(&self, that: &[T]) -> bool
    where
        T: PartialEq;

    /// Tests whether this ring is a rotation **or** a reflection of `that`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert!([0, 1, 2].is_rotation_or_reflection_of(&[2, 0, 1]));
    /// assert!([0, 1, 2].is_rotation_or_reflection_of(&[0, 2, 1]));
    /// ```
    #[must_use]
    fn is_rotation_or_reflection_of(&self, that: &[T]) -> bool
    where
        T: PartialEq + Clone;

    /// Finds the rotation offset that aligns this ring with `that`.
    ///
    /// Returns `Some(k)` such that `self.start_at(k) == that`, or `None` if no
    /// rotation matches (or sizes differ).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2].rotation_offset(&[2, 0, 1]), Some(2));
    /// assert_eq!([0, 1, 2].rotation_offset(&[0, 2, 1]), None);
    /// ```
    #[must_use]
    fn rotation_offset(&self, that: &[T]) -> Option<usize>
    where
        T: PartialEq;

    /// The number of positions at which corresponding elements differ.
    ///
    /// # Panics
    ///
    /// Panics if the two slices have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([1, 0, 1, 1].hamming_distance(&[1, 1, 0, 1]), 2);
    /// ```
    #[must_use]
    fn hamming_distance(&self, that: &[T]) -> usize
    where
        T: PartialEq;

    /// The minimum Hamming distance over all rotations of this ring.
    ///
    /// Returns `0` iff `that` is a rotation of `self`.
    ///
    /// # Panics
    ///
    /// Panics if the two slices have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([1, 0, 1, 1].min_rotational_hamming_distance(&[1, 1, 0, 1]), 0);
    /// ```
    #[must_use]
    fn min_rotational_hamming_distance(&self, that: &[T]) -> usize
    where
        T: PartialEq + Clone;

    // ── Necklace ───────────────────────────────────────────────────────

    /// The starting index of the lexicographically smallest rotation
    /// ([Booth's algorithm](https://en.wikipedia.org/wiki/Lexicographically_least_circular_substring),
    /// *O(n)*).
    ///
    /// Returns `0` for empty or single-element sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([2, 0, 1].canonical_index(), 1);
    /// ```
    #[must_use]
    fn canonical_index(&self) -> usize
    where
        T: Ord;

    /// The lexicographically smallest rotation of this ring (necklace canonical
    /// form).
    ///
    /// Two rings are rotations of each other iff their canonical forms are
    /// equal — useful for hashing and deduplicating equivalent rings.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([2, 0, 1].canonical(), vec![0, 1, 2]);
    /// ```
    #[must_use]
    fn canonical(&self) -> Vec<T>
    where
        T: Clone + Ord;

    /// The lexicographically smallest representative under both rotation and
    /// reflection (bracelet canonical form).
    ///
    /// Two rings belong to the same bracelet equivalence class iff their
    /// bracelet forms are equal — useful when mirror images are considered
    /// identical.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([2, 1, 0].bracelet(), vec![0, 1, 2]);
    /// assert_eq!([0, 1, 2].bracelet(), vec![0, 1, 2]);
    /// ```
    #[must_use]
    fn bracelet(&self) -> Vec<T>
    where
        T: Clone + Ord;

    // ── Symmetry ───────────────────────────────────────────────────────

    /// The order of rotational symmetry: the number of distinct rotations that
    /// map the ring onto itself.
    ///
    /// Returns 1 for sequences with no rotational symmetry (only the identity),
    /// and `n` when all elements are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([0, 1, 2, 0, 1, 2].rotational_symmetry(), 2);
    /// assert_eq!([0, 1, 2].rotational_symmetry(), 1);
    /// assert_eq!([5, 5, 5].rotational_symmetry(), 3);
    /// ```
    #[must_use]
    fn rotational_symmetry(&self) -> usize
    where
        T: PartialEq;

    /// Indices of elements where a reflectional-symmetry axis passes nearby.
    ///
    /// More precisely, the "shift" values for which the ring equals its
    /// reversal rotated by that shift.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!(
    ///     [2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2].symmetry_indices(),
    ///     vec![0, 3, 6, 9],
    /// );
    /// ```
    #[must_use]
    fn symmetry_indices(&self) -> Vec<usize>
    where
        T: PartialEq;

    /// Axes of reflectional symmetry, expressed as pairs of
    /// [`AxisLocation`]s where each axis intersects the ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::{AxisLocation, RingSeq};
    ///
    /// let axes = [0, 1, 0].reflectional_symmetry_axes();
    /// assert_eq!(axes.len(), 1);
    /// assert_eq!(axes[0], (AxisLocation::Vertex(1), AxisLocation::Edge(2, 0)));
    /// ```
    #[must_use]
    fn reflectional_symmetry_axes(&self) -> Vec<(AxisLocation, AxisLocation)>
    where
        T: PartialEq;

    /// The number of reflectional symmetries (the number of symmetry axes).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::RingSeq;
    ///
    /// assert_eq!([2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2].symmetry(), 4);
    /// assert_eq!([0, 1, 2].symmetry(), 0);
    /// ```
    #[must_use]
    fn symmetry(&self) -> usize
    where
        T: PartialEq;
}

// ============================================================================
// Implementation
// ============================================================================

impl<T> RingSeq<T> for [T] {
    // ── Indexing ────────────────────────────────────────────────────────

    #[inline]
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn index_from(&self, i: isize) -> usize {
        i.rem_euclid(self.len() as isize) as usize
    }

    #[inline]
    fn apply_o(&self, i: isize) -> &T {
        &self[self.index_from(i)]
    }

    // ── Transforming ───────────────────────────────────────────────────

    fn rotate_right(&self, step: isize) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_empty() {
            return vec![];
        }
        let n = self.len();
        let j = n - self.index_from(step);
        let mut v = Vec::with_capacity(n);
        v.extend_from_slice(&self[j..]);
        v.extend_from_slice(&self[..j]);
        v
    }

    #[inline]
    fn rotate_left(&self, step: isize) -> Vec<T>
    where
        T: Clone,
    {
        self.rotate_right(-step)
    }

    #[inline]
    fn start_at(&self, i: isize) -> Vec<T>
    where
        T: Clone,
    {
        self.rotate_left(i)
    }

    fn reflect_at(&self, i: isize) -> Vec<T>
    where
        T: Clone,
    {
        let mut v = self.start_at(i + 1);
        v.reverse();
        v
    }

    // ── Slicing ────────────────────────────────────────────────────────

    fn segment_length(&self, pred: impl Fn(&T) -> bool, from: isize) -> usize {
        if self.is_empty() {
            return 0;
        }
        let n = self.len();
        let start = self.index_from(from);
        let mut count = 0;
        for k in 0..n {
            if pred(&self[(start + k) % n]) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    fn take_while(&self, pred: impl Fn(&T) -> bool, from: isize) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_empty() {
            return vec![];
        }
        let n = self.len();
        let start = self.index_from(from);
        let mut result = Vec::new();
        for k in 0..n {
            let elem = &self[(start + k) % n];
            if pred(elem) {
                result.push(elem.clone());
            } else {
                break;
            }
        }
        result
    }

    fn drop_while(&self, pred: impl Fn(&T) -> bool, from: isize) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_empty() {
            return vec![];
        }
        let n = self.len();
        let start = self.index_from(from);
        let prefix_len = self.segment_length(&pred, from);
        let mut result = Vec::with_capacity(n - prefix_len);
        for k in prefix_len..n {
            result.push(self[(start + k) % n].clone());
        }
        result
    }

    fn span(&self, pred: impl Fn(&T) -> bool, from: isize) -> (Vec<T>, Vec<T>)
    where
        T: Clone,
    {
        if self.is_empty() {
            return (vec![], vec![]);
        }
        let n = self.len();
        let start = self.index_from(from);
        let prefix_len = self.segment_length(&pred, from);
        let mut taken = Vec::with_capacity(prefix_len);
        for k in 0..prefix_len {
            taken.push(self[(start + k) % n].clone());
        }
        let mut dropped = Vec::with_capacity(n - prefix_len);
        for k in prefix_len..n {
            dropped.push(self[(start + k) % n].clone());
        }
        (taken, dropped)
    }

    fn slice_o(&self, from: isize, to: isize) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_empty() || from >= to {
            return vec![];
        }
        #[allow(clippy::cast_sign_loss)] // `to - from` is positive due to guard above
        let length = (to - from) as usize;
        let start = self.index_from(from);
        self[start..]
            .iter()
            .chain(self.iter().cycle())
            .take(length)
            .cloned()
            .collect()
    }

    fn contains_slice(&self, slice: &[T]) -> bool
    where
        T: PartialEq,
    {
        if slice.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        let n = self.len();
        let m = slice.len();
        (0..n).any(|start| (0..m).all(|j| self[(start + j) % n] == slice[j]))
    }

    fn index_of_slice(&self, slice: &[T], from: isize) -> Option<usize>
    where
        T: PartialEq,
    {
        if slice.is_empty() {
            return if self.is_empty() {
                Some(0)
            } else {
                Some(self.index_from(from))
            };
        }
        if self.is_empty() {
            return None;
        }
        let n = self.len();
        let m = slice.len();
        let start = self.index_from(from);
        for k in 0..n {
            let i = (start + k) % n;
            if (0..m).all(|j| self[(i + j) % n] == slice[j]) {
                return Some(i);
            }
        }
        None
    }

    fn last_index_of_slice(&self, slice: &[T], end: isize) -> Option<usize>
    where
        T: PartialEq,
    {
        if slice.is_empty() {
            return if self.is_empty() {
                Some(0)
            } else {
                Some(self.index_from(end))
            };
        }
        if self.is_empty() {
            return None;
        }
        let n = self.len();
        let m = slice.len();
        let end_idx = self.index_from(end);
        for k in 0..n {
            let i = (end_idx + n - k) % n;
            if (0..m).all(|j| self[(i + j) % n] == slice[j]) {
                return Some(i);
            }
        }
        None
    }

    // ── Iterating ──────────────────────────────────────────────────────

    fn circular_windows(&self, size: usize, step: usize) -> SlidingO<T>
    where
        T: Clone,
    {
        assert!(size > 0, "window size must be positive");
        assert!(step > 0, "step must be positive");
        if self.is_empty() {
            return SlidingO {
                data: vec![],
                window_size: size,
                step,
                pos: 0,
            };
        }
        let total_len = step * (self.len() - 1) + size;
        #[allow(clippy::cast_possible_wrap)]
        SlidingO {
            data: self.slice_o(0, total_len as isize),
            window_size: size,
            step,
            pos: 0,
        }
    }

    fn circular_chunks(&self, size: usize) -> SlidingO<T>
    where
        T: Clone,
    {
        assert!(size > 0, "group size must be positive");
        if self.is_empty() {
            return SlidingO {
                data: vec![],
                window_size: size,
                step: size,
                pos: 0,
            };
        }
        self.circular_windows(size, size)
    }

    fn circular_enumerate(&self, from: isize) -> Vec<(T, usize)>
    where
        T: Clone,
    {
        let n = self.len();
        if n == 0 {
            return vec![];
        }
        let start = self.index_from(from);
        (0..n)
            .map(|k| {
                let idx = (start + k) % n;
                (self[idx].clone(), idx)
            })
            .collect()
    }

    fn rotations(&self) -> Rotations<'_, T> {
        Rotations {
            ring: self,
            index: 0,
            total: if self.is_empty() { 1 } else { self.len() },
        }
    }

    fn reflections(&self) -> Reflections<'_, T> {
        Reflections {
            ring: self,
            state: 0,
        }
    }

    fn reversions(&self) -> Reversions<'_, T> {
        Reversions {
            ring: self,
            state: 0,
        }
    }

    fn rotations_and_reflections(&self) -> RotationsAndReflections<'_, T>
    where
        T: Clone,
    {
        let reflected = self.reflect_at(0);
        RotationsAndReflections {
            ring: self,
            reflected,
            index: 0,
            total: if self.is_empty() {
                1
            } else {
                self.len() * 2
            },
        }
    }

    // ── Comparing ──────────────────────────────────────────────────────

    fn is_rotation_of(&self, that: &[T]) -> bool
    where
        T: PartialEq,
    {
        if self.len() != that.len() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        contains_as_rotation(self, that)
    }

    fn is_reflection_of(&self, that: &[T]) -> bool
    where
        T: PartialEq + Clone,
    {
        if self.len() != that.len() {
            return false;
        }
        self == that || self.reflect_at(0) == that
    }

    fn is_reversion_of(&self, that: &[T]) -> bool
    where
        T: PartialEq,
    {
        if self.len() != that.len() {
            return false;
        }
        self == that || self.iter().rev().eq(that.iter())
    }

    fn is_rotation_or_reflection_of(&self, that: &[T]) -> bool
    where
        T: PartialEq + Clone,
    {
        if self.len() != that.len() {
            return false;
        }
        contains_as_rotation(self, that)
            || contains_as_rotation(&self.reflect_at(0), that)
    }

    fn rotation_offset(&self, that: &[T]) -> Option<usize>
    where
        T: PartialEq,
    {
        if self.len() != that.len() {
            return None;
        }
        if self.is_empty() {
            return Some(0);
        }
        let n = self.len();
        (0..n).find(|&i| (0..n).all(|j| self[(i + j) % n] == that[j]))
    }

    fn hamming_distance(&self, that: &[T]) -> usize
    where
        T: PartialEq,
    {
        assert_eq!(
            self.len(),
            that.len(),
            "sequences must have the same size"
        );
        self.iter()
            .zip(that.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    fn min_rotational_hamming_distance(&self, that: &[T]) -> usize
    where
        T: PartialEq + Clone,
    {
        assert_eq!(
            self.len(),
            that.len(),
            "sequences must have the same size"
        );
        if self.is_empty() {
            return 0;
        }
        let n = self.len();
        (0..n)
            .map(|rot| {
                (0..n)
                    .filter(|&j| self[(rot + j) % n] != that[j])
                    .count()
            })
            .min()
            .unwrap()
    }

    // ── Necklace ───────────────────────────────────────────────────────

    fn canonical_index(&self) -> usize
    where
        T: Ord,
    {
        if self.len() <= 1 {
            0
        } else {
            booth_least_rotation(self)
        }
    }

    fn canonical(&self) -> Vec<T>
    where
        T: Clone + Ord,
    {
        #[allow(clippy::cast_possible_wrap)]
        self.start_at(self.canonical_index() as isize)
    }

    fn bracelet(&self) -> Vec<T>
    where
        T: Clone + Ord,
    {
        let a = self.canonical();
        let b = self.reflect_at(0).canonical();
        if a <= b {
            a
        } else {
            b
        }
    }

    // ── Symmetry ───────────────────────────────────────────────────────

    fn rotational_symmetry(&self) -> usize
    where
        T: PartialEq,
    {
        let n = self.len();
        if n < 2 {
            return 1;
        }
        let smallest_period = (1..=n).find(|&shift| {
            n % shift == 0 && (0..n - shift).all(|i| self[i] == self[i + shift])
        });
        n / smallest_period.unwrap_or(n)
    }

    fn symmetry_indices(&self) -> Vec<usize>
    where
        T: PartialEq,
    {
        let n = self.len();
        if n == 0 {
            return vec![];
        }
        let reversed: Vec<&T> = self.iter().rev().collect();
        (0..n)
            .filter(|&shift| {
                (0..n).all(|i| self[i] == *reversed[(i + shift) % n])
            })
            .collect()
    }

    fn reflectional_symmetry_axes(&self) -> Vec<(AxisLocation, AxisLocation)>
    where
        T: PartialEq,
    {
        let n = self.len();
        if n == 0 {
            return vec![];
        }

        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
        self.symmetry_indices()
            .into_iter()
            .map(|shift| {
                let raw_k = (n as isize - 1 - shift as isize) % n as isize;
                let k = if raw_k < 0 {
                    (raw_k + n as isize) as usize
                } else {
                    raw_k as usize
                };

                if n % 2 != 0 {
                    // Odd n: one vertex fixed point, opposite edge
                    let v = (k * (n + 1) / 2) % n;
                    let opp = (v + n / 2) % n;
                    (AxisLocation::Vertex(v), AxisLocation::edge(opp, n))
                } else if k % 2 == 0 {
                    // Even n, even k: two vertex fixed points
                    let v1 = k / 2;
                    let v2 = (v1 + n / 2) % n;
                    (AxisLocation::Vertex(v1), AxisLocation::Vertex(v2))
                } else {
                    // Even n, odd k: two edge midpoints
                    let e1 = (k - 1) / 2;
                    let e2 = (e1 + n / 2) % n;
                    (AxisLocation::edge(e1, n), AxisLocation::edge(e2, n))
                }
            })
            .collect()
    }

    fn symmetry(&self) -> usize
    where
        T: PartialEq,
    {
        self.symmetry_indices().len()
    }
}

// ============================================================================
// Private helpers
// ============================================================================

/// Tests whether `that` appears as a rotation of `ring`, using the
/// "concatenate and search" technique: any rotation of `ring` is a contiguous
/// substring of `ring ++ ring[..n-1]`.
fn contains_as_rotation<T: PartialEq>(ring: &[T], that: &[T]) -> bool {
    if ring.is_empty() {
        return true;
    }
    let n = ring.len();
    // Check if `that` appears as a contiguous substring in ring++ring[..n-1]
    // without allocating, by using circular indexing.
    let doubled_len = 2 * n - 1;
    (0..n).any(|start| {
        (0..n).all(|j| {
            let idx = (start + j) % doubled_len;
            let elem = if idx < n { &ring[idx] } else { &ring[idx - n] };
            *elem == that[j]
        })
    })
}

/// Booth's O(n) algorithm for finding the starting index of the
/// lexicographically smallest rotation.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap, clippy::many_single_char_names)]
fn booth_least_rotation<T: Ord>(s: &[T]) -> usize {
    let n = s.len();
    let len = 2 * n;
    let mut f: Vec<isize> = vec![-1; len];
    let mut k: usize = 0;
    let at = |idx: usize| &s[idx % n];

    for j in 1..len {
        let sj = at(j);
        let mut i = f[j - k - 1];
        while i != -1 && at(k + i as usize + 1) != sj {
            if sj < at(k + i as usize + 1) {
                k = j - i as usize - 1;
            }
            i = f[i as usize];
        }
        if i == -1 && at(k) != sj {
            if sj < at(k) {
                k = j;
            }
            f[j - k] = -1;
        } else {
            f[j - k] = i + 1;
        }
    }
    k
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Indexing ────────────────────────────────────────────────────────

    #[test]
    fn index_from_positive() {
        assert_eq!([10, 20, 30].index_from(0), 0);
        assert_eq!([10, 20, 30].index_from(2), 2);
        assert_eq!([10, 20, 30].index_from(3), 0);
        assert_eq!([10, 20, 30].index_from(7), 1);
    }

    #[test]
    fn index_from_negative() {
        assert_eq!([10, 20, 30].index_from(-1), 2);
        assert_eq!([10, 20, 30].index_from(-3), 0);
        assert_eq!([10, 20, 30].index_from(-4), 2);
    }

    #[test]
    #[should_panic]
    fn index_from_empty_panics() {
        let empty: &[i32] = &[];
        let _ = empty.index_from(0);
    }

    #[test]
    fn apply_o_wraps() {
        assert_eq!(*[10, 20, 30].apply_o(0), 10);
        assert_eq!(*[10, 20, 30].apply_o(3), 10);
        assert_eq!(*[10, 20, 30].apply_o(-1), 30);
    }

    #[test]
    #[should_panic]
    fn apply_o_empty_panics() {
        let empty: &[i32] = &[];
        let _ = empty.apply_o(0);
    }

    #[test]
    fn apply_o_single_element() {
        assert_eq!(*[42].apply_o(0), 42);
        assert_eq!(*[42].apply_o(100), 42);
        assert_eq!(*[42].apply_o(-99), 42);
    }

    // ── Transforming ───────────────────────────────────────────────────

    #[test]
    fn rotate_right_basic() {
        assert_eq!([0, 1, 2].rotate_right(1), vec![2, 0, 1]);
        assert_eq!([0, 1, 2].rotate_right(2), vec![1, 2, 0]);
        assert_eq!([0, 1, 2].rotate_right(3), vec![0, 1, 2]);
    }

    #[test]
    fn rotate_right_negative() {
        assert_eq!([0, 1, 2].rotate_right(-1), vec![1, 2, 0]);
    }

    #[test]
    fn rotate_right_empty() {
        let empty: &[i32] = &[];
        assert_eq!(empty.rotate_right(5), Vec::<i32>::new());
    }

    #[test]
    fn rotate_left_basic() {
        assert_eq!([0, 1, 2].rotate_left(1), vec![1, 2, 0]);
    }

    #[test]
    fn start_at_basic() {
        assert_eq!([0, 1, 2].start_at(1), vec![1, 2, 0]);
        assert_eq!([0, 1, 2].start_at(-1), vec![2, 0, 1]);
    }

    #[test]
    fn reflect_at_basic() {
        assert_eq!([0, 1, 2].reflect_at(0), vec![0, 2, 1]);
        assert_eq!([0, 1, 2].reflect_at(1), vec![1, 0, 2]);
    }

    #[test]
    fn reflect_at_empty() {
        let empty: &[i32] = &[];
        assert_eq!(empty.reflect_at(0), Vec::<i32>::new());
    }

    #[test]
    fn reflect_at_single() {
        assert_eq!([42].reflect_at(0), vec![42]);
    }

    // ── Slicing ────────────────────────────────────────────────────────

    #[test]
    fn segment_length_basic() {
        assert_eq!([0, 1, 2].segment_length(|x| x % 2 == 0, 2), 2);
        assert_eq!([0, 1, 2].segment_length(|_| true, 0), 3);
        assert_eq!([0, 1, 2].segment_length(|_| false, 0), 0);
    }

    #[test]
    fn segment_length_empty() {
        let empty: &[i32] = &[];
        assert_eq!(empty.segment_length(|_| true, 0), 0);
    }

    #[test]
    fn take_while_basic() {
        assert_eq!(
            [0, 1, 2, 3, 4].take_while(|&x| x < 3, 1),
            vec![1, 2]
        );
    }

    #[test]
    fn drop_while_basic() {
        assert_eq!(
            [0, 1, 2, 3, 4].drop_while(|&x| x < 3, 1),
            vec![3, 4, 0]
        );
    }

    #[test]
    fn span_basic() {
        let (a, b) = [0, 1, 2, 3, 4].span(|&x| x < 3, 1);
        assert_eq!(a, vec![1, 2]);
        assert_eq!(b, vec![3, 4, 0]);
    }

    #[test]
    fn span_all_pass() {
        let (a, b) = [1, 2, 3].span(|_| true, 0);
        assert_eq!(a, vec![1, 2, 3]);
        assert!(b.is_empty());
    }

    #[test]
    fn span_none_pass() {
        let (a, b) = [1, 2, 3].span(|_| false, 0);
        assert!(a.is_empty());
        assert_eq!(b, vec![1, 2, 3]);
    }

    #[test]
    fn slice_o_basic() {
        assert_eq!([0, 1, 2].slice_o(1, 3), vec![1, 2]);
    }

    #[test]
    fn slice_o_wrapping() {
        assert_eq!([0, 1, 2].slice_o(-1, 4), vec![2, 0, 1, 2, 0]);
    }

    #[test]
    fn slice_o_empty_result() {
        assert_eq!([0, 1, 2].slice_o(3, 3), Vec::<i32>::new());
        assert_eq!([0, 1, 2].slice_o(5, 2), Vec::<i32>::new());
    }

    #[test]
    fn slice_o_empty_ring() {
        let empty: &[i32] = &[];
        assert_eq!(empty.slice_o(0, 5), Vec::<i32>::new());
    }

    #[test]
    fn contains_slice_basic() {
        assert!([0, 1, 2].contains_slice(&[2, 0]));
        assert!([0, 1, 2].contains_slice(&[2, 0, 1, 2, 0]));
        assert!(![0, 1, 2].contains_slice(&[1, 0]));
    }

    #[test]
    fn contains_slice_empty_slice() {
        assert!([0, 1, 2].contains_slice(&[]));
    }

    #[test]
    fn contains_slice_empty_ring() {
        let empty: &[i32] = &[];
        assert!(!empty.contains_slice(&[1]));
        assert!(empty.contains_slice(&[]));
    }

    #[test]
    fn index_of_slice_basic() {
        assert_eq!([0, 1, 2].index_of_slice(&[2, 0, 1, 2, 0], 0), Some(2));
        assert_eq!([0, 1, 2].index_of_slice(&[0, 1], 0), Some(0));
        assert_eq!([0, 1, 2].index_of_slice(&[9], 0), None);
    }

    #[test]
    fn index_of_slice_with_from() {
        assert_eq!([0, 1, 2, 0, 1, 2].index_of_slice(&[0, 1], 1), Some(3));
    }

    #[test]
    fn last_index_of_slice_basic() {
        assert_eq!(
            [0, 1, 2, 0, 1, 2].last_index_of_slice(&[2, 0], -1),
            Some(5)
        );
    }

    // ── Iterating ──────────────────────────────────────────────────────

    #[test]
    fn circular_windows_basic() {
        let windows: Vec<_> = [0, 1, 2].circular_windows(2, 1).collect();
        assert_eq!(windows, vec![vec![0, 1], vec![1, 2], vec![2, 0]]);
    }

    #[test]
    fn circular_windows_empty() {
        let windows: Vec<Vec<i32>> = ([] as [i32; 0]).circular_windows(2, 1).collect();
        assert!(windows.is_empty());
    }

    #[test]
    fn circular_windows_exact_size() {
        let iter = [0, 1, 2].circular_windows(2, 1);
        assert_eq!(iter.len(), 3);
    }

    #[test]
    #[should_panic]
    fn circular_windows_zero_size_panics() {
        let _ = [0, 1, 2].circular_windows(0, 1);
    }

    #[test]
    #[should_panic]
    fn circular_windows_zero_step_panics() {
        let _ = [0, 1, 2].circular_windows(1, 0);
    }

    #[test]
    fn circular_chunks_basic() {
        let groups: Vec<_> = [0, 1, 2, 3, 4].circular_chunks(2).collect();
        assert_eq!(
            groups,
            vec![vec![0, 1], vec![2, 3], vec![4, 0], vec![1, 2], vec![3, 4]]
        );
    }

    #[test]
    fn circular_chunks_evenly_divisible() {
        let groups: Vec<_> = [0, 1, 2, 3].circular_chunks(2).collect();
        assert_eq!(groups, vec![vec![0, 1], vec![2, 3], vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn circular_enumerate_basic() {
        assert_eq!(
            ['a', 'b', 'c'].circular_enumerate(1),
            vec![('b', 1), ('c', 2), ('a', 0)]
        );
    }

    #[test]
    fn circular_enumerate_empty() {
        let empty: &[i32] = &[];
        assert!(empty.circular_enumerate(0).is_empty());
    }

    #[test]
    fn rotations_basic() {
        let rots: Vec<_> = [0, 1, 2].rotations().collect();
        assert_eq!(rots, vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]]);
    }

    #[test]
    fn rotations_empty() {
        let empty: &[i32] = &[];
        let rots: Vec<_> = empty.rotations().collect();
        assert_eq!(rots, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn rotations_exact_size() {
        assert_eq!([0, 1, 2].rotations().len(), 3);
        let empty: &[i32] = &[];
        assert_eq!(empty.rotations().len(), 1);
    }

    #[test]
    fn reflections_basic() {
        let refs: Vec<_> = [0, 1, 2].reflections().collect();
        assert_eq!(refs, vec![vec![0, 1, 2], vec![0, 2, 1]]);
    }

    #[test]
    fn reflections_empty() {
        let empty: &[i32] = &[];
        let refs: Vec<_> = empty.reflections().collect();
        assert_eq!(refs, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn reversions_basic() {
        let revs: Vec<_> = [0, 1, 2].reversions().collect();
        assert_eq!(revs, vec![vec![0, 1, 2], vec![2, 1, 0]]);
    }

    #[test]
    fn rotations_and_reflections_basic() {
        let all: Vec<_> = [0, 1, 2].rotations_and_reflections().collect();
        assert_eq!(
            all,
            vec![
                vec![0, 1, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![0, 2, 1],
                vec![2, 1, 0],
                vec![1, 0, 2],
            ]
        );
    }

    #[test]
    fn rotations_and_reflections_empty() {
        let empty: &[i32] = &[];
        let all: Vec<_> = empty.rotations_and_reflections().collect();
        assert_eq!(all, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn rotations_and_reflections_exact_size() {
        assert_eq!([0, 1, 2].rotations_and_reflections().len(), 6);
    }

    // ── Comparing ──────────────────────────────────────────────────────

    #[test]
    fn is_rotation_of_true() {
        assert!([0, 1, 2].is_rotation_of(&[1, 2, 0]));
        assert!([0, 1, 2].is_rotation_of(&[0, 1, 2]));
    }

    #[test]
    fn is_rotation_of_false() {
        assert!(![0, 1, 2].is_rotation_of(&[0, 2, 1]));
        assert!(![0, 1, 2].is_rotation_of(&[0, 1]));
    }

    #[test]
    fn is_rotation_of_empty() {
        let empty: &[i32] = &[];
        assert!(empty.is_rotation_of(&[]));
    }

    #[test]
    fn is_reflection_of_true() {
        assert!([0, 1, 2].is_reflection_of(&[0, 2, 1]));
        assert!([0, 1, 2].is_reflection_of(&[0, 1, 2])); // identity
    }

    #[test]
    fn is_reflection_of_false() {
        assert!(![0, 1, 2].is_reflection_of(&[1, 0, 2]));
    }

    #[test]
    fn is_reversion_of_true() {
        assert!([0, 1, 2].is_reversion_of(&[2, 1, 0]));
        assert!([0, 1, 2].is_reversion_of(&[0, 1, 2])); // identity
    }

    #[test]
    fn is_reversion_of_false() {
        assert!(![0, 1, 2].is_reversion_of(&[0, 2, 1]));
    }

    #[test]
    fn is_rotation_or_reflection_of_true() {
        assert!([0, 1, 2].is_rotation_or_reflection_of(&[2, 0, 1])); // rotation
        assert!([0, 1, 2].is_rotation_or_reflection_of(&[0, 2, 1])); // reflection
        assert!([0, 1, 2].is_rotation_or_reflection_of(&[1, 0, 2])); // rotated reflection
    }

    #[test]
    fn align_to_found() {
        assert_eq!([0, 1, 2].rotation_offset(&[2, 0, 1]), Some(2));
        assert_eq!([0, 1, 2].rotation_offset(&[0, 1, 2]), Some(0));
    }

    #[test]
    fn align_to_not_found() {
        assert_eq!([0, 1, 2].rotation_offset(&[0, 2, 1]), None);
    }

    #[test]
    fn align_to_different_sizes() {
        assert_eq!([0, 1, 2].rotation_offset(&[0, 1]), None);
    }

    #[test]
    fn align_to_empty() {
        let empty: &[i32] = &[];
        assert_eq!(empty.rotation_offset(&[]), Some(0));
    }

    #[test]
    fn hamming_distance_basic() {
        assert_eq!([1, 0, 1, 1].hamming_distance(&[1, 1, 0, 1]), 2);
        assert_eq!([1, 2, 3].hamming_distance(&[1, 2, 3]), 0);
    }

    #[test]
    #[should_panic(expected = "sequences must have the same size")]
    fn hamming_distance_different_sizes_panics() {
        let _ = [1, 2, 3].hamming_distance(&[1, 2]);
    }

    #[test]
    fn min_rotational_hamming_distance_basic() {
        assert_eq!(
            [1, 0, 1, 1].min_rotational_hamming_distance(&[1, 1, 0, 1]),
            0
        );
    }

    #[test]
    fn min_rotational_hamming_distance_nonzero() {
        assert_eq!(
            [0, 0, 0, 1].min_rotational_hamming_distance(&[1, 1, 0, 0]),
            1
        );
    }

    #[test]
    fn min_rotational_hamming_distance_empty() {
        let empty: &[i32] = &[];
        assert_eq!(empty.min_rotational_hamming_distance(&[]), 0);
    }

    // ── Necklace ───────────────────────────────────────────────────────

    #[test]
    fn canonical_index_basic() {
        assert_eq!([2, 0, 1].canonical_index(), 1);
        assert_eq!([0, 1, 2].canonical_index(), 0);
    }

    #[test]
    fn canonical_index_empty_and_single() {
        let empty: &[i32] = &[];
        assert_eq!(empty.canonical_index(), 0);
        assert_eq!([42].canonical_index(), 0);
    }

    #[test]
    fn canonical_basic() {
        assert_eq!([2, 0, 1].canonical(), vec![0, 1, 2]);
        assert_eq!([0, 1, 2].canonical(), vec![0, 1, 2]);
    }

    #[test]
    fn canonical_all_equal() {
        assert_eq!([5, 5, 5].canonical(), vec![5, 5, 5]);
    }

    #[test]
    fn canonical_rotations_are_equal() {
        let ring = [3, 1, 4, 1, 5];
        let canon = ring.canonical();
        for rot in ring.rotations() {
            assert_eq!(rot.canonical(), canon);
        }
    }

    #[test]
    fn bracelet_basic() {
        assert_eq!([2, 1, 0].bracelet(), vec![0, 1, 2]);
        assert_eq!([0, 1, 2].bracelet(), vec![0, 1, 2]);
    }

    #[test]
    fn bracelet_reflection_equivalence() {
        let a = [0, 1, 2];
        let b = [0, 2, 1];
        assert_eq!(a.bracelet(), b.bracelet());
    }

    // ── Symmetry ───────────────────────────────────────────────────────

    #[test]
    fn rotational_symmetry_basic() {
        assert_eq!([0, 1, 2, 0, 1, 2].rotational_symmetry(), 2);
        assert_eq!([0, 1, 2].rotational_symmetry(), 1);
        assert_eq!([5, 5, 5].rotational_symmetry(), 3);
    }

    #[test]
    fn rotational_symmetry_small() {
        let empty: &[i32] = &[];
        assert_eq!(empty.rotational_symmetry(), 1);
        assert_eq!([42].rotational_symmetry(), 1);
        assert_eq!([1, 1].rotational_symmetry(), 2);
        assert_eq!([1, 2].rotational_symmetry(), 1);
    }

    #[test]
    fn symmetry_indices_basic() {
        assert_eq!(
            [2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2].symmetry_indices(),
            vec![0, 3, 6, 9]
        );
    }

    #[test]
    fn symmetry_indices_none() {
        assert!([0, 1, 2].symmetry_indices().is_empty());
    }

    #[test]
    fn symmetry_indices_palindrome() {
        assert_eq!([0, 1, 0].symmetry_indices(), vec![0]);
    }

    #[test]
    fn symmetry_basic() {
        assert_eq!([2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2].symmetry(), 4);
        assert_eq!([0, 1, 2].symmetry(), 0);
    }

    #[test]
    fn symmetry_empty() {
        let empty: &[i32] = &[];
        assert_eq!(empty.symmetry(), 0);
    }

    #[test]
    fn reflectional_symmetry_axes_odd() {
        let axes = [0, 1, 0].reflectional_symmetry_axes();
        assert_eq!(axes.len(), 1);
        assert_eq!(axes[0], (AxisLocation::Vertex(1), AxisLocation::Edge(2, 0)));
    }

    #[test]
    fn reflectional_symmetry_axes_even_vertices() {
        // [0, 1, 1, 0] is symmetric — axis through vertices 0 and 2
        let axes = [0, 1, 1, 0].reflectional_symmetry_axes();
        assert!(!axes.is_empty());
    }

    #[test]
    fn reflectional_symmetry_axes_empty() {
        let empty: &[i32] = &[];
        assert!(empty.reflectional_symmetry_axes().is_empty());
    }

    // ── AxisLocation ───────────────────────────────────────────────────

    #[test]
    fn axis_location_edge_constructor() {
        assert_eq!(AxisLocation::edge(2, 3), AxisLocation::Edge(2, 0));
        assert_eq!(AxisLocation::edge(0, 4), AxisLocation::Edge(0, 1));
        assert_eq!(AxisLocation::edge(3, 4), AxisLocation::Edge(3, 0));
    }

    #[test]
    #[should_panic(expected = "ring size must be positive")]
    fn axis_location_edge_zero_size_panics() {
        let _ = AxisLocation::edge(0, 0);
    }

    // ── Vec and array interop ──────────────────────────────────────────

    #[test]
    fn works_on_vec() {
        let v = vec![0, 1, 2];
        assert_eq!(v.rotate_right(1), vec![2, 0, 1]);
        assert!(v.is_rotation_of(&[1, 2, 0]));
    }

    #[test]
    fn works_on_boxed_slice() {
        let b: Box<[i32]> = vec![0, 1, 2].into_boxed_slice();
        assert_eq!(b.rotate_right(1), vec![2, 0, 1]);
    }

    #[test]
    fn chained_operations() {
        // rotate_right returns Vec<T>, which derefs to [T], so we can chain
        let result = [0, 1, 2].rotate_right(1).reflect_at(0);
        // rotate_right(1) -> [2, 0, 1]
        // reflect_at(0) of [2, 0, 1] -> start_at(1).reverse() = [0, 1, 2].reverse() = [2, 1, 0]
        assert_eq!(result, vec![2, 1, 0]);
    }

    // ── Booth's algorithm ──────────────────────────────────────────────

    #[test]
    fn booth_all_same() {
        assert_eq!(booth_least_rotation(&[5, 5, 5, 5]), 0);
    }

    #[test]
    fn booth_sorted() {
        assert_eq!(booth_least_rotation(&[0, 1, 2, 3]), 0);
    }

    #[test]
    fn booth_rotated() {
        assert_eq!(booth_least_rotation(&[2, 3, 0, 1]), 2);
    }

    #[test]
    fn booth_two_elements() {
        assert_eq!(booth_least_rotation(&[1, 0]), 1);
    }

    // ── Property-style tests ───────────────────────────────────────────

    #[test]
    fn rotate_right_then_left_is_identity() {
        let ring = [3, 1, 4, 1, 5, 9];
        for step in -10..=10 {
            assert_eq!(
                ring.rotate_right(step).rotate_left(step),
                ring.to_vec(),
                "failed for step = {step}"
            );
        }
    }

    #[test]
    fn start_at_then_apply_o_recovers_element() {
        let ring = [10, 20, 30, 40, 50];
        for i in -10..10 {
            let rotated = ring.start_at(i);
            assert_eq!(rotated[0], *ring.apply_o(i));
        }
    }

    #[test]
    fn all_rotations_have_same_canonical() {
        let ring = [5, 3, 1, 4, 2];
        let canon = ring.canonical();
        for rot in ring.rotations() {
            assert_eq!(rot.canonical(), canon);
        }
    }

    #[test]
    fn all_rotations_and_reflections_have_same_bracelet() {
        let ring = [5, 3, 1, 4, 2];
        let brace = ring.bracelet();
        for variant in ring.rotations_and_reflections() {
            assert_eq!(variant.bracelet(), brace);
        }
    }

    #[test]
    fn is_rotation_of_all_rotations() {
        let ring = [1, 2, 3, 4];
        for rot in ring.rotations() {
            assert!(ring.is_rotation_of(&rot));
        }
    }

    #[test]
    fn reflect_at_is_involution() {
        let ring = [0, 1, 2, 3];
        // Applying reflect_at(0) twice should return to the original
        let once = ring.reflect_at(0);
        let twice = once.reflect_at(0);
        assert_eq!(twice, ring.to_vec());
    }

    #[test]
    fn symmetry_count_divides_twice_size_for_nonempty() {
        // For a non-empty ring of size n, the number of reflectional
        // symmetries divides n.
        for ring in [
            vec![0, 1, 2],
            vec![1, 1, 1],
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 3],
            vec![2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2],
        ] {
            let n = ring.len();
            let s = ring.symmetry();
            if s > 0 {
                assert_eq!(
                    n % s,
                    0,
                    "symmetry {s} does not divide ring size {n} for {ring:?}"
                );
            }
        }
    }
}
