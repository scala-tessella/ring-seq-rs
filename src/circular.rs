//! The [`Circular`] wrapper and its element iterator.
//!
//! Reach the wrapper through [`AsCircular::circular`] on any slice (or
//! anything that derefs to one — [`Vec<T>`](alloc::vec::Vec), arrays,
//! [`Box<[T]>`](alloc::boxed::Box)). The wrapper is the single home for
//! every circular operation; all transforms it offers are lazy and
//! allocation-free.

use core::iter::FusedIterator;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use crate::AxisLocation;

/// A borrowed view of a slice as a circular sequence.
///
/// Carries a `(slice, offset, reflected)` triple. Every operation routes
/// through a single index-mapping function, so the algebra stays consistent
/// however the view was constructed.
///
/// The type is `Copy` and trivially cheap to pass by value.
///
/// # Examples
///
/// ```
/// use ring_seq::AsCircular;
///
/// let ring = [10, 20, 30];
/// let c = ring.circular();
/// assert_eq!(c.len(), 3);
/// assert_eq!(*c.apply(4), 20); // wraps
/// ```
pub struct Circular<'a, T> {
    ring: &'a [T],
    offset: usize,
    reflected: bool,
}

// Hand-implemented so they don't pick up a phantom `T: Clone` bound from
// the derive macro. The wrapper is value-typed and trivially copyable for
// every `T`.
impl<T> Clone for Circular<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Circular<'_, T> {}

impl<T: core::fmt::Debug> core::fmt::Debug for Circular<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Circular")
            .field("ring", &self.ring)
            .field("offset", &self.offset)
            .field("reflected", &self.reflected)
            .finish()
    }
}

impl<'a, T> Circular<'a, T> {
    /// Constructs a fresh view over `ring` with zero offset and no reflection.
    #[inline]
    pub(crate) fn new(ring: &'a [T]) -> Self {
        Circular {
            ring,
            offset: 0,
            reflected: false,
        }
    }

    /// Number of elements in the underlying ring.
    #[inline]
    #[must_use]
    pub fn len(self) -> usize {
        self.ring.len()
    }

    /// Returns `true` if the ring has no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.ring.is_empty()
    }

    /// Single source of truth: map a position within this view to an index
    /// into the underlying slice.
    ///
    /// Caller must ensure `len() > 0`.
    #[inline]
    fn map_index(self, pos: usize) -> usize {
        let n = self.ring.len();
        debug_assert!(n > 0);
        let p = pos % n;
        if self.reflected {
            (self.offset + n - p) % n
        } else {
            (self.offset + p) % n
        }
    }

    /// Returns the element at the (possibly negative, possibly out-of-bounds)
    /// circular index `i`.
    ///
    /// # Panics
    ///
    /// Panics if the ring is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [10, 20, 30].circular();
    /// assert_eq!(*r.apply(0), 10);
    /// assert_eq!(*r.apply(3), 10);    // wraps forward
    /// assert_eq!(*r.apply(-1), 30);   // wraps backward
    /// ```
    #[inline]
    #[must_use]
    pub fn apply(self, i: isize) -> &'a T {
        let n = self.ring.len();
        assert!(n > 0, "cannot index into an empty ring");
        let pos = i.rem_euclid(n as isize) as usize;
        &self.ring[self.map_index(pos)]
    }

    /// Returns an iterator yielding the elements of this view in order.
    ///
    /// The iterator is `ExactSizeIterator` and `FusedIterator`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3].circular();
    /// let collected: Vec<_> = r.iter().copied().collect();
    /// assert_eq!(collected, vec![1, 2, 3]);
    /// ```
    #[inline]
    #[must_use]
    pub fn iter(self) -> CircularIter<'a, T> {
        CircularIter {
            view: self,
            pos: 0,
            remaining: self.ring.len(),
        }
    }

    // -----------------------------------------------------------------------
    // Reindexed views — return a new Circular, do not allocate
    // -----------------------------------------------------------------------

    /// Returns a view whose position 0 is the element at circular index `i`.
    ///
    /// Equivalent to [`rotate_left(i)`](Circular::rotate_left). All other
    /// reindexed-view methods are defined in terms of this one.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [10, 20, 30, 40].circular();
    /// let v: Vec<_> = r.start_at(2).iter().copied().collect();
    /// assert_eq!(v, vec![30, 40, 10, 20]);
    /// ```
    #[must_use]
    pub fn start_at(self, i: isize) -> Circular<'a, T> {
        let n = self.ring.len();
        if n == 0 {
            return self;
        }
        let i_pos = i.rem_euclid(n as isize) as usize;
        let new_offset = if self.reflected {
            (self.offset + n - i_pos) % n
        } else {
            (self.offset + i_pos) % n
        };
        Circular {
            ring: self.ring,
            offset: new_offset,
            reflected: self.reflected,
        }
    }

    /// Shifts the view left by `step` positions. Position 0 of the result
    /// is what was at position `step` of `self`.
    ///
    /// `step` may be negative (delegates to [`rotate_right`](Self::rotate_right))
    /// and may exceed `len` (wraps).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3, 4].circular();
    /// let v: Vec<_> = r.rotate_left(1).iter().copied().collect();
    /// assert_eq!(v, vec![2, 3, 4, 1]);
    /// ```
    #[inline]
    #[must_use]
    pub fn rotate_left(self, step: isize) -> Circular<'a, T> {
        self.start_at(step)
    }

    /// Shifts the view right by `step` positions. Position 0 of the result
    /// is what was at position `-step` of `self`.
    ///
    /// `step` may be negative (delegates to [`rotate_left`](Self::rotate_left))
    /// and may exceed `len` (wraps).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3, 4].circular();
    /// let v: Vec<_> = r.rotate_right(1).iter().copied().collect();
    /// assert_eq!(v, vec![4, 1, 2, 3]);
    /// ```
    #[inline]
    #[must_use]
    pub fn rotate_right(self, step: isize) -> Circular<'a, T> {
        self.start_at(-step)
    }

    /// Reflects the view around position `i`.
    ///
    /// Position 0 of the result is the element at position `i`; subsequent
    /// positions walk backward from there. Flips the `reflected` bit of the
    /// view (a second reflection at the same index recovers the original).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = ['a', 'b', 'c', 'd'].circular();
    /// let v: Vec<_> = r.reflect_at(0).iter().copied().collect();
    /// assert_eq!(v, vec!['a', 'd', 'c', 'b']);
    /// ```
    #[must_use]
    pub fn reflect_at(self, i: isize) -> Circular<'a, T> {
        let n = self.ring.len();
        if n == 0 {
            return self;
        }
        let i_pos = i.rem_euclid(n as isize) as usize;
        let new_offset = if self.reflected {
            (self.offset + n - i_pos) % n
        } else {
            (self.offset + i_pos) % n
        };
        Circular {
            ring: self.ring,
            offset: new_offset,
            reflected: !self.reflected,
        }
    }

    // -----------------------------------------------------------------------
    // Bounded iteration — return an iterator, do not allocate
    // -----------------------------------------------------------------------

    /// Returns an iterator over the circular range `[from, to)`.
    ///
    /// The range is taken in the view's forward direction starting at the
    /// element with circular index `from`. Yields `max(to - from, 0)`
    /// elements, wrapping around the ring as many times as needed; an
    /// empty range produces an empty iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [0, 1, 2, 3, 4].circular();
    /// let v: Vec<_> = r.slice(2, 7).copied().collect();
    /// assert_eq!(v, vec![2, 3, 4, 0, 1]);
    /// ```
    #[must_use]
    pub fn slice(self, from: isize, to: isize) -> CircularIter<'a, T> {
        if self.ring.is_empty() || to <= from {
            return CircularIter {
                view: self,
                pos: 0,
                remaining: 0,
            };
        }
        let count = (to - from) as usize;
        let started = self.start_at(from);
        CircularIter {
            view: started,
            pos: 0,
            remaining: count,
        }
    }

    /// Returns an iterator yielding elements from circular index `from`
    /// while `pred` holds, stopping at the first element that fails (or
    /// after one full lap).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3, 4, 5].circular();
    /// let v: Vec<_> = r.take_while(|&x| x < 4, 0).copied().collect();
    /// assert_eq!(v, vec![1, 2, 3]);
    /// ```
    pub fn take_while<P>(self, mut pred: P, from: isize) -> impl Iterator<Item = &'a T>
    where
        P: FnMut(&T) -> bool,
    {
        self.start_at(from).iter().take_while(move |x| pred(*x))
    }

    /// Returns an iterator that skips elements from circular index `from`
    /// while `pred` holds, then yields the rest of the lap (one full
    /// rotation maximum).
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3, 4, 5].circular();
    /// let v: Vec<_> = r.drop_while(|&x| x < 4, 0).copied().collect();
    /// assert_eq!(v, vec![4, 5]);
    /// ```
    pub fn drop_while<P>(self, mut pred: P, from: isize) -> impl Iterator<Item = &'a T>
    where
        P: FnMut(&T) -> bool,
    {
        self.start_at(from).iter().skip_while(move |x| pred(*x))
    }

    /// Returns an iterator yielding `(element, ring_index)` pairs starting
    /// at circular index `from` and going around once.
    ///
    /// Unlike [`Iterator::enumerate`], the second element of each pair is
    /// the element's index in the underlying ring (already wrapped to
    /// `[0, len)`), not the iteration count.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = ['a', 'b', 'c'].circular();
    /// let v: Vec<_> = r.enumerate(1).map(|(&c, i)| (c, i)).collect();
    /// assert_eq!(v, vec![('b', 1), ('c', 2), ('a', 0)]);
    /// ```
    #[must_use]
    pub fn enumerate(self, from: isize) -> Enumerate<'a, T> {
        let started = if self.ring.is_empty() {
            self
        } else {
            self.start_at(from)
        };
        Enumerate {
            view: started,
            pos: 0,
            remaining: started.ring.len(),
        }
    }

    // -----------------------------------------------------------------------
    // Iterators of views — yield Circular<'a, T>
    // -----------------------------------------------------------------------

    /// Returns an iterator yielding each rotation of this view as a fresh
    /// [`Circular`].
    ///
    /// For a non-empty ring of length `n` the iterator yields `n` items;
    /// for an empty ring it yields one (empty) item. The `reflected` bit
    /// of `self` is preserved by every yielded rotation.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3].circular();
    /// let starts: Vec<i32> = r.rotations().map(|v| *v.apply(0)).collect();
    /// assert_eq!(starts, vec![1, 2, 3]);
    /// ```
    #[must_use]
    pub fn rotations(self) -> Rotations<'a, T> {
        let total = if self.ring.is_empty() {
            1
        } else {
            self.ring.len()
        };
        Rotations {
            base: self,
            index: 0,
            total,
        }
    }

    /// Returns an iterator yielding the view and its reflection at
    /// position 0.
    ///
    /// For a non-empty ring the iterator yields two items; for an empty
    /// ring it yields one. The first item is always `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3, 4].circular();
    /// let count = r.reflections().count();
    /// assert_eq!(count, 2);
    /// ```
    #[must_use]
    pub fn reflections(self) -> Reflections<'a, T> {
        Reflections {
            base: self,
            state: 0,
        }
    }

    /// Returns an iterator yielding the view and its reverse.
    ///
    /// "Reverse" here means walking the ring in the opposite direction
    /// starting from position `len - 1` — equivalent to
    /// `reflect_at(-1)`.
    ///
    /// For a non-empty ring the iterator yields two items; for an empty
    /// ring it yields one.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3].circular();
    /// let last: Vec<_> = r.reversions()
    ///     .map(|v| v.iter().copied().collect::<Vec<_>>())
    ///     .collect();
    /// assert_eq!(last, vec![vec![1, 2, 3], vec![3, 2, 1]]);
    /// ```
    #[must_use]
    pub fn reversions(self) -> Reversions<'a, T> {
        Reversions {
            base: self,
            state: 0,
        }
    }

    /// Returns an iterator yielding every rotation of this view followed
    /// by every rotation of its reflection at position 0.
    ///
    /// For a non-empty ring of length `n` the iterator yields `2n` items;
    /// for an empty ring it yields one.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3].circular();
    /// assert_eq!(r.rotations_and_reflections().count(), 6);
    /// ```
    #[must_use]
    pub fn rotations_and_reflections(self) -> RotationsAndReflections<'a, T> {
        let n = self.ring.len();
        let total = if n == 0 { 1 } else { 2 * n };
        let reflected = if n == 0 { self } else { self.reflect_at(0) };
        RotationsAndReflections {
            base: self,
            reflected,
            index: 0,
            total,
            n,
        }
    }

    /// Returns an iterator yielding every circular window of length `size`
    /// (step 1) as a [`CircularIter`].
    ///
    /// For a non-empty ring of length `n` the iterator yields `n` windows;
    /// for an empty ring it yields no windows.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3, 4].circular();
    /// let windows: Vec<Vec<i32>> = r
    ///     .windows(2)
    ///     .map(|w| w.copied().collect())
    ///     .collect();
    /// assert_eq!(windows, vec![vec![1, 2], vec![2, 3], vec![3, 4], vec![4, 1]]);
    /// ```
    #[must_use]
    pub fn windows(self, size: usize) -> Windows<'a, T> {
        assert!(size > 0, "window size must be positive");
        let total = if self.ring.is_empty() {
            0
        } else {
            self.ring.len()
        };
        Windows {
            base: self,
            size,
            step: 1,
            index: 0,
            total,
        }
    }

    // -----------------------------------------------------------------------
    // Indexing
    // -----------------------------------------------------------------------

    /// Normalizes a circular index to `[0, len)`.
    ///
    /// Uses Euclidean remainder so that negative indices wrap correctly.
    ///
    /// # Panics
    ///
    /// Panics if the ring is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [10, 20, 30].circular();
    /// assert_eq!(r.index_from(0), 0);
    /// assert_eq!(r.index_from(4), 1);
    /// assert_eq!(r.index_from(-1), 2);
    /// ```
    #[must_use]
    pub fn index_from(self, i: isize) -> usize {
        let n = self.ring.len();
        assert!(n > 0, "cannot normalize against an empty ring");
        let pos = i.rem_euclid(n as isize) as usize;
        self.map_index(pos)
    }

    // -----------------------------------------------------------------------
    // Comparison — alloc-free
    // -----------------------------------------------------------------------

    /// Returns `true` if `other` is some rotation of this view.
    #[must_use]
    pub fn is_rotation_of(self, other: &[T]) -> bool
    where
        T: PartialEq,
    {
        if self.len() != other.len() {
            return false;
        }
        if self.ring.is_empty() {
            return true;
        }
        self.rotations().any(|rot| rot.iter().eq(other.iter()))
    }

    /// Returns `true` if `other` equals this view or its reflection at
    /// position 0.
    #[must_use]
    pub fn is_reflection_of(self, other: &[T]) -> bool
    where
        T: PartialEq,
    {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter()) || self.reflect_at(0).iter().eq(other.iter())
    }

    /// Returns `true` if `other` equals this view or its reverse.
    #[must_use]
    pub fn is_reversion_of(self, other: &[T]) -> bool
    where
        T: PartialEq,
    {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter()) || self.reflect_at(-1).iter().eq(other.iter())
    }

    /// Returns `true` if `other` is any rotation of this view or of its
    /// reflection at position 0.
    #[must_use]
    pub fn is_rotation_or_reflection_of(self, other: &[T]) -> bool
    where
        T: PartialEq,
    {
        if self.len() != other.len() {
            return false;
        }
        self.rotations_and_reflections()
            .any(|v| v.iter().eq(other.iter()))
    }

    /// Returns the starting offset at which `other` matches this view as a
    /// rotation, or `None` if no such offset exists.
    #[must_use]
    pub fn rotation_offset(self, other: &[T]) -> Option<usize>
    where
        T: PartialEq,
    {
        if self.len() != other.len() {
            return None;
        }
        if self.ring.is_empty() {
            return Some(0);
        }
        let n = self.ring.len();
        (0..n).find(|&i| self.start_at(i as isize).iter().eq(other.iter()))
    }

    /// Counts positions where this view and `other` differ.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    #[must_use]
    pub fn hamming_distance(self, other: &[T]) -> usize
    where
        T: PartialEq,
    {
        assert_eq!(self.len(), other.len(), "sequences must have the same size");
        self.iter()
            .zip(other.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Minimum Hamming distance over all rotations of this view against
    /// `other`.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    #[must_use]
    pub fn min_rotational_hamming_distance(self, other: &[T]) -> usize
    where
        T: PartialEq,
    {
        assert_eq!(self.len(), other.len(), "sequences must have the same size");
        let n = self.ring.len();
        if n == 0 {
            return 0;
        }
        let mut best = usize::MAX;
        for rot in self.rotations() {
            let count = rot.iter().zip(other.iter()).filter(|(a, b)| a != b).count();
            if count < best {
                best = count;
            }
            if best == 0 {
                break;
            }
        }
        best
    }

    /// Returns `true` if `needle` appears as a contiguous (possibly
    /// wrapping) substring of this view.
    ///
    /// An empty `needle` is always contained.
    #[must_use]
    pub fn contains_slice(self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        if needle.is_empty() {
            return true;
        }
        if self.ring.is_empty() {
            return false;
        }
        let n = self.ring.len();
        let m = needle.len();
        (0..n).any(|start| (0..m).all(|j| *self.apply((start + j) as isize) == needle[j]))
    }

    /// Returns the starting circular index at which `needle` appears, or
    /// `None` if it does not. Searches forward from `from`.
    #[must_use]
    pub fn index_of_slice(self, needle: &[T], from: isize) -> Option<usize>
    where
        T: PartialEq,
    {
        if needle.is_empty() {
            return if self.ring.is_empty() {
                Some(0)
            } else {
                Some(self.index_from(from))
            };
        }
        if self.ring.is_empty() {
            return None;
        }
        let n = self.ring.len();
        let m = needle.len();
        let start = self.index_from(from);
        (0..n)
            .map(|k| (start + k) % n)
            .find(|&i| (0..m).all(|j| *self.apply((i + j) as isize) == needle[j]))
    }

    // -----------------------------------------------------------------------
    // Symmetry — alloc-free counts
    // -----------------------------------------------------------------------

    /// Returns the rotational symmetry order: `n / smallest_period`, where
    /// the smallest period is the smallest divisor `d` of `n` such that
    /// `apply(i) == apply(i + d)` for all `i`. Empty and single-element
    /// rings have rotational symmetry 1.
    #[must_use]
    pub fn rotational_symmetry(self) -> usize
    where
        T: PartialEq,
    {
        let n = self.ring.len();
        if n < 2 {
            return 1;
        }
        let smallest = (1..=n).find(|&d| {
            n % d == 0
                && (0..n - d).all(|i| *self.apply(i as isize) == *self.apply((i + d) as isize))
        });
        n / smallest.unwrap_or(n)
    }

    /// Returns an iterator over the shifts at which this view equals its
    /// own reverse rotated by that shift — i.e., the count of reflectional
    /// symmetry axes (without allocating the indices themselves).
    ///
    /// Use [`symmetry_indices`](Self::symmetry_indices) (alloc-gated) if
    /// you need the actual indices.
    #[must_use]
    pub fn symmetry(self) -> usize
    where
        T: PartialEq,
    {
        let n = self.ring.len();
        if n == 0 {
            return 0;
        }
        let reversed = self.reflect_at(-1);
        (0..n)
            .filter(|&shift| {
                (0..n).all(|i| *self.apply(i as isize) == *reversed.apply((i + shift) as isize))
            })
            .count()
    }

    /// Returns an iterator yielding `ceil(len / size)` non-overlapping
    /// circular chunks of length `size` as [`CircularIter`]s.
    ///
    /// When `size` does not divide `len`, the final chunk wraps across
    /// the seam so every chunk has exactly `size` elements. An empty ring
    /// yields no chunks.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_seq::AsCircular;
    ///
    /// let r = [1, 2, 3, 4, 5].circular();
    /// let chunks: Vec<Vec<i32>> = r
    ///     .chunks(2)
    ///     .map(|c| c.copied().collect())
    ///     .collect();
    /// // ceil(5/2) = 3 chunks; the last one wraps
    /// assert_eq!(chunks, vec![vec![1, 2], vec![3, 4], vec![5, 1]]);
    /// ```
    #[must_use]
    pub fn chunks(self, size: usize) -> Windows<'a, T> {
        assert!(size > 0, "chunk size must be positive");
        let n = self.ring.len();
        let total = if n == 0 { 0 } else { (n + size - 1) / size };
        Windows {
            base: self,
            size,
            step: size,
            index: 0,
            total,
        }
    }
}

// ---------------------------------------------------------------------------
// CircularIter
// ---------------------------------------------------------------------------

/// Iterator over the elements of a [`Circular`] view.
///
/// Created by [`Circular::iter`]. Yields `&'a T` and is
/// [`ExactSizeIterator`] + [`FusedIterator`].
#[derive(Debug)]
pub struct CircularIter<'a, T> {
    view: Circular<'a, T>,
    pos: usize,
    remaining: usize,
}

impl<T> Clone for CircularIter<'_, T> {
    fn clone(&self) -> Self {
        CircularIter {
            view: self.view,
            pos: self.pos,
            remaining: self.remaining,
        }
    }
}

impl<'a, T> Iterator for CircularIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.remaining == 0 {
            return None;
        }
        let item = &self.view.ring[self.view.map_index(self.pos)];
        self.pos += 1;
        self.remaining -= 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for CircularIter<'_, T> {}
impl<T> FusedIterator for CircularIter<'_, T> {}

// ---------------------------------------------------------------------------
// Enumerate — yields (&T, ring_index)
// ---------------------------------------------------------------------------

/// Iterator over `(element, ring_index)` pairs of a [`Circular`] view.
///
/// Created by [`Circular::enumerate`]. Yields `(&'a T, usize)` where the
/// `usize` is the position in the underlying ring (already wrapped to
/// `[0, len)`).
#[derive(Debug)]
pub struct Enumerate<'a, T> {
    view: Circular<'a, T>,
    pos: usize,
    remaining: usize,
}

impl<T> Clone for Enumerate<'_, T> {
    fn clone(&self) -> Self {
        Enumerate {
            view: self.view,
            pos: self.pos,
            remaining: self.remaining,
        }
    }
}

impl<'a, T> Iterator for Enumerate<'a, T> {
    type Item = (&'a T, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let idx = self.view.map_index(self.pos);
        let item = &self.view.ring[idx];
        self.pos += 1;
        self.remaining -= 1;
        Some((item, idx))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for Enumerate<'_, T> {}
impl<T> FusedIterator for Enumerate<'_, T> {}

// ---------------------------------------------------------------------------
// Rotations — yields each rotation as a Circular view
// ---------------------------------------------------------------------------

/// Iterator over the rotations of a [`Circular`] view.
///
/// Created by [`Circular::rotations`]. Yields one [`Circular`] per
/// starting position; preserves the `reflected` bit of the source view.
#[derive(Debug)]
pub struct Rotations<'a, T> {
    base: Circular<'a, T>,
    index: usize,
    total: usize,
}

impl<T> Clone for Rotations<'_, T> {
    fn clone(&self) -> Self {
        Rotations {
            base: self.base,
            index: self.index,
            total: self.total,
        }
    }
}

impl<'a, T> Iterator for Rotations<'a, T> {
    type Item = Circular<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        let item = if self.base.ring.is_empty() {
            self.base
        } else {
            self.base.start_at(self.index as isize)
        };
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.total - self.index;
        (r, Some(r))
    }
}

impl<T> ExactSizeIterator for Rotations<'_, T> {}
impl<T> FusedIterator for Rotations<'_, T> {}

// ---------------------------------------------------------------------------
// Reflections — yields the view and its reflection at position 0
// ---------------------------------------------------------------------------

/// Iterator yielding the source [`Circular`] and its [`reflect_at(0)`].
///
/// Created by [`Circular::reflections`].
///
/// [`reflect_at(0)`]: Circular::reflect_at
#[derive(Debug)]
pub struct Reflections<'a, T> {
    base: Circular<'a, T>,
    state: u8,
}

impl<T> Clone for Reflections<'_, T> {
    fn clone(&self) -> Self {
        Reflections {
            base: self.base,
            state: self.state,
        }
    }
}

impl<'a, T> Iterator for Reflections<'a, T> {
    type Item = Circular<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            0 => {
                self.state = if self.base.ring.is_empty() { 2 } else { 1 };
                Some(self.base)
            }
            1 => {
                self.state = 2;
                Some(self.base.reflect_at(0))
            }
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = match self.state {
            0 => {
                if self.base.ring.is_empty() {
                    1
                } else {
                    2
                }
            }
            1 => 1,
            _ => 0,
        };
        (r, Some(r))
    }
}

impl<T> ExactSizeIterator for Reflections<'_, T> {}
impl<T> FusedIterator for Reflections<'_, T> {}

// ---------------------------------------------------------------------------
// Reversions — yields the view and its reverse
// ---------------------------------------------------------------------------

/// Iterator yielding the source [`Circular`] and its reverse
/// (`reflect_at(-1)`).
///
/// Created by [`Circular::reversions`].
#[derive(Debug)]
pub struct Reversions<'a, T> {
    base: Circular<'a, T>,
    state: u8,
}

impl<T> Clone for Reversions<'_, T> {
    fn clone(&self) -> Self {
        Reversions {
            base: self.base,
            state: self.state,
        }
    }
}

impl<'a, T> Iterator for Reversions<'a, T> {
    type Item = Circular<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            0 => {
                self.state = if self.base.ring.is_empty() { 2 } else { 1 };
                Some(self.base)
            }
            1 => {
                self.state = 2;
                Some(self.base.reflect_at(-1))
            }
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = match self.state {
            0 => {
                if self.base.ring.is_empty() {
                    1
                } else {
                    2
                }
            }
            1 => 1,
            _ => 0,
        };
        (r, Some(r))
    }
}

impl<T> ExactSizeIterator for Reversions<'_, T> {}
impl<T> FusedIterator for Reversions<'_, T> {}

// ---------------------------------------------------------------------------
// RotationsAndReflections — yields 2n views
// ---------------------------------------------------------------------------

/// Iterator yielding every rotation of the source followed by every
/// rotation of its reflection at position 0.
///
/// Created by [`Circular::rotations_and_reflections`].
#[derive(Debug)]
pub struct RotationsAndReflections<'a, T> {
    base: Circular<'a, T>,
    reflected: Circular<'a, T>,
    index: usize,
    total: usize,
    n: usize,
}

impl<T> Clone for RotationsAndReflections<'_, T> {
    fn clone(&self) -> Self {
        RotationsAndReflections {
            base: self.base,
            reflected: self.reflected,
            index: self.index,
            total: self.total,
            n: self.n,
        }
    }
}

impl<'a, T> Iterator for RotationsAndReflections<'a, T> {
    type Item = Circular<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        if self.n == 0 {
            self.index += 1;
            return Some(self.base);
        }
        let item = if self.index < self.n {
            self.base.start_at(self.index as isize)
        } else {
            self.reflected.start_at((self.index - self.n) as isize)
        };
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.total - self.index;
        (r, Some(r))
    }
}

impl<T> ExactSizeIterator for RotationsAndReflections<'_, T> {}
impl<T> FusedIterator for RotationsAndReflections<'_, T> {}

// ---------------------------------------------------------------------------
// Windows / Chunks — yield CircularIter<'a, T>
// ---------------------------------------------------------------------------

/// Iterator yielding circular windows (or chunks) of a [`Circular`] view
/// as [`CircularIter`]s.
///
/// Created by [`Circular::windows`] (step 1) and [`Circular::chunks`]
/// (step equal to the chunk size).
#[derive(Debug)]
pub struct Windows<'a, T> {
    base: Circular<'a, T>,
    size: usize,
    step: usize,
    index: usize,
    total: usize,
}

impl<T> Clone for Windows<'_, T> {
    fn clone(&self) -> Self {
        Windows {
            base: self.base,
            size: self.size,
            step: self.step,
            index: self.index,
            total: self.total,
        }
    }
}

impl<'a, T> Iterator for Windows<'a, T> {
    type Item = CircularIter<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        let start = (self.index * self.step) as isize;
        let view = self.base.start_at(start);
        let iter = CircularIter {
            view,
            pos: 0,
            remaining: self.size,
        };
        self.index += 1;
        Some(iter)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.total - self.index;
        (r, Some(r))
    }
}

impl<T> ExactSizeIterator for Windows<'_, T> {}
impl<T> FusedIterator for Windows<'_, T> {}

// ---------------------------------------------------------------------------
// AsCircular
// ---------------------------------------------------------------------------

/// Extension trait providing [`circular`](AsCircular::circular) on slices.
///
/// Implemented for `[T]`; reaches `Vec<T>`, arrays, and `Box<[T]>` via
/// deref coercion.
///
/// # Examples
///
/// ```
/// use ring_seq::AsCircular;
///
/// let v = vec![1, 2, 3];
/// let r = v.circular();
/// assert_eq!(*r.apply(5), 3);
/// ```
pub trait AsCircular<T> {
    /// Returns a [`Circular`] view of this slice.
    fn circular(&self) -> Circular<'_, T>;
}

impl<T> AsCircular<T> for [T] {
    #[inline]
    fn circular(&self) -> Circular<'_, T> {
        Circular::new(self)
    }
}

// ---------------------------------------------------------------------------
// Alloc-gated section: methods returning owned collections, plus Booth's
// algorithm for canonical-index search.
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<'a, T> Circular<'a, T> {
    /// Materializes this view as a `Vec<T>`. Requires `T: Clone`.
    #[must_use]
    pub fn to_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Returns the starting offset of the lexicographically smallest
    /// rotation of this view (Booth's O(n) algorithm).
    ///
    /// Single-element and empty views return `0`.
    #[must_use]
    pub fn canonical_index(self) -> usize
    where
        T: Ord,
    {
        if self.ring.len() <= 1 {
            0
        } else {
            booth_least_rotation(self)
        }
    }

    /// Returns the canonical (lexicographically smallest) rotation of
    /// this view as a `Vec<T>`. Requires `T: Clone + Ord`.
    #[must_use]
    pub fn canonical(self) -> Vec<T>
    where
        T: Clone + Ord,
    {
        let idx = self.canonical_index();
        self.start_at(idx as isize).to_vec()
    }

    /// Returns the bracelet form: the lexicographically smallest among
    /// `canonical()` and `reflect_at(0).canonical()`. Requires
    /// `T: Clone + Ord`.
    #[must_use]
    pub fn bracelet(self) -> Vec<T>
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

    /// Returns the shifts `k` in `[0, n)` for which this view equals its
    /// own reverse rotated by `k`. Each shift corresponds to one axis of
    /// reflectional symmetry.
    #[must_use]
    pub fn symmetry_indices(self) -> Vec<usize>
    where
        T: PartialEq,
    {
        let n = self.ring.len();
        if n == 0 {
            return Vec::new();
        }
        let reversed = self.reflect_at(-1);
        (0..n)
            .filter(|&shift| {
                (0..n).all(|i| *self.apply(i as isize) == *reversed.apply((i + shift) as isize))
            })
            .collect()
    }

    /// Returns the axes of reflectional symmetry as
    /// `(AxisLocation, AxisLocation)` pairs (each axis hits the ring in
    /// two locations).
    #[must_use]
    pub fn reflectional_symmetry_axes(self) -> Vec<(AxisLocation, AxisLocation)>
    where
        T: PartialEq,
    {
        let n = self.ring.len();
        if n == 0 {
            return Vec::new();
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
                    let v = (k * (n + 1) / 2) % n;
                    let opp = (v + n / 2) % n;
                    (AxisLocation::Vertex(v), AxisLocation::edge(opp, n))
                } else if k % 2 == 0 {
                    let v1 = k / 2;
                    let v2 = (v1 + n / 2) % n;
                    (AxisLocation::Vertex(v1), AxisLocation::Vertex(v2))
                } else {
                    let e1 = (k - 1) / 2;
                    let e2 = (e1 + n / 2) % n;
                    (AxisLocation::edge(e1, n), AxisLocation::edge(e2, n))
                }
            })
            .collect()
    }
}

/// Booth's O(n) algorithm for finding the starting offset of the
/// lexicographically smallest rotation of a `Circular` view.
#[cfg(feature = "alloc")]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]
fn booth_least_rotation<T: Ord>(c: Circular<'_, T>) -> usize {
    let n = c.ring.len();
    let len = 2 * n;
    let mut f: Vec<isize> = alloc::vec![-1; len];
    let mut k: usize = 0;
    let at = |idx: usize| &c.ring[c.map_index(idx % n)];

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

// ---------------------------------------------------------------------------
// Tests — pure core, no alloc needed
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ring_is_empty() {
        let empty: [i32; 0] = [];
        let c = empty.circular();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.iter().count(), 0);
    }

    #[test]
    fn len_and_is_empty() {
        let r = [1, 2, 3].circular();
        assert_eq!(r.len(), 3);
        assert!(!r.is_empty());
    }

    #[test]
    fn apply_positive() {
        let r = [10, 20, 30].circular();
        assert_eq!(*r.apply(0), 10);
        assert_eq!(*r.apply(1), 20);
        assert_eq!(*r.apply(2), 30);
    }

    #[test]
    fn apply_wraps_forward() {
        let r = [10, 20, 30].circular();
        assert_eq!(*r.apply(3), 10);
        assert_eq!(*r.apply(7), 20);
    }

    #[test]
    fn apply_wraps_backward() {
        let r = [10, 20, 30].circular();
        assert_eq!(*r.apply(-1), 30);
        assert_eq!(*r.apply(-3), 10);
        assert_eq!(*r.apply(-4), 30);
    }

    #[test]
    #[should_panic]
    fn apply_panics_on_empty() {
        let empty: [i32; 0] = [];
        let _ = empty.circular().apply(0);
    }

    #[test]
    fn iter_yields_elements_in_order() {
        let r = [1, 2, 3].circular();
        let mut it = r.iter();
        assert_eq!(it.next(), Some(&1));
        assert_eq!(it.next(), Some(&2));
        assert_eq!(it.next(), Some(&3));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn iter_size_hint_and_exact_size() {
        let r = [1, 2, 3, 4].circular();
        let mut it = r.iter();
        assert_eq!(it.len(), 4);
        assert_eq!(it.size_hint(), (4, Some(4)));
        it.next();
        assert_eq!(it.len(), 3);
    }

    #[test]
    fn iter_is_fused() {
        let r = [1, 2].circular();
        let mut it = r.iter();
        it.next();
        it.next();
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None); // stays None
    }

    #[test]
    fn copy_semantics() {
        let r = [1, 2, 3].circular();
        let r2 = r; // copy
        assert_eq!(r.len(), 3);
        assert_eq!(r2.len(), 3);
    }

    // T does not need to be Clone for Circular to be Copy.
    #[test]
    fn copy_works_without_t_clone() {
        #[allow(dead_code)]
        struct NoClone(i32);
        let arr = [NoClone(1), NoClone(2)];
        let r = arr.circular();
        let r2 = r; // would fail to compile if Copy carried a T: Clone bound
        assert_eq!(r.len(), 2);
        assert_eq!(r2.len(), 2);
    }

    // ── Lazy view operations ────────────────────────────────────────────

    fn into_array<const N: usize>(c: Circular<'_, i32>) -> [i32; N] {
        let mut out = [0; N];
        for (slot, &x) in out.iter_mut().zip(c.iter()) {
            *slot = x;
        }
        out
    }

    #[test]
    fn start_at_basic() {
        let r = [10, 20, 30, 40].circular();
        assert_eq!(into_array::<4>(r.start_at(0)), [10, 20, 30, 40]);
        assert_eq!(into_array::<4>(r.start_at(2)), [30, 40, 10, 20]);
        assert_eq!(into_array::<4>(r.start_at(-1)), [40, 10, 20, 30]);
        assert_eq!(into_array::<4>(r.start_at(5)), [20, 30, 40, 10]);
    }

    #[test]
    fn start_at_empty() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().start_at(3).len(), 0);
    }

    #[test]
    fn rotate_right_basic() {
        let r = [1, 2, 3, 4].circular();
        assert_eq!(into_array::<4>(r.rotate_right(0)), [1, 2, 3, 4]);
        assert_eq!(into_array::<4>(r.rotate_right(1)), [4, 1, 2, 3]);
        assert_eq!(into_array::<4>(r.rotate_right(-1)), [2, 3, 4, 1]);
        assert_eq!(into_array::<4>(r.rotate_right(5)), [4, 1, 2, 3]);
    }

    #[test]
    fn rotate_left_basic() {
        let r = [1, 2, 3, 4].circular();
        assert_eq!(into_array::<4>(r.rotate_left(1)), [2, 3, 4, 1]);
        assert_eq!(into_array::<4>(r.rotate_left(-1)), [4, 1, 2, 3]);
    }

    #[test]
    fn rotate_right_then_left_is_identity() {
        let ring = [1, 2, 3, 4, 5, 6, 7];
        let r = ring.circular();
        for step in -10..=10 {
            assert_eq!(
                into_array::<7>(r.rotate_right(step).rotate_left(step)),
                ring,
                "failed for step {}",
                step,
            );
        }
    }

    #[test]
    fn reflect_at_basic() {
        let r = [10, 20, 30, 40].circular();
        // reflect_at(0): pos 0 = ring[0]=10, pos 1 = ring[3]=40, pos 2 = ring[2]=30, pos 3 = ring[1]=20
        assert_eq!(into_array::<4>(r.reflect_at(0)), [10, 40, 30, 20]);
        // reflect_at(2): pos 0 = ring[2]=30, pos 1 = ring[1]=20, pos 2 = ring[0]=10, pos 3 = ring[3]=40
        assert_eq!(into_array::<4>(r.reflect_at(2)), [30, 20, 10, 40]);
    }

    #[test]
    fn reflect_at_is_involution() {
        let ring = [1, 2, 3, 4, 5, 6, 7];
        let r = ring.circular();
        for i in -3..10 {
            assert_eq!(
                into_array::<7>(r.reflect_at(i).reflect_at(i)),
                ring,
                "failed for i {}",
                i
            );
        }
    }

    #[test]
    fn rotate_then_reflect_commutes() {
        // rotate_right(k).reflect_at(0) == reflect_at(0).rotate_left(k)
        let ring = [1, 2, 3, 4, 5];
        let r = ring.circular();
        for k in -7..=7 {
            assert_eq!(
                into_array::<5>(r.rotate_right(k).reflect_at(0)),
                into_array::<5>(r.reflect_at(0).rotate_left(k)),
                "failed for k {}",
                k,
            );
        }
    }

    // ── slice ──────────────────────────────────────────────────────────

    #[test]
    fn slice_within_one_lap() {
        let r = [0, 1, 2, 3, 4].circular();
        let v: [i32; 3] = {
            let mut out = [0; 3];
            for (s, &x) in out.iter_mut().zip(r.slice(1, 4)) {
                *s = x;
            }
            out
        };
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn slice_wraps() {
        let r = [0, 1, 2, 3, 4].circular();
        let mut out = [0; 5];
        for (s, &x) in out.iter_mut().zip(r.slice(2, 7)) {
            *s = x;
        }
        assert_eq!(out, [2, 3, 4, 0, 1]);
    }

    #[test]
    fn slice_negative_from() {
        let r = [0, 1, 2, 3, 4].circular();
        let mut out = [0; 3];
        for (s, &x) in out.iter_mut().zip(r.slice(-2, 1)) {
            *s = x;
        }
        assert_eq!(out, [3, 4, 0]);
    }

    #[test]
    fn slice_empty_range() {
        let r = [0, 1, 2, 3, 4].circular();
        assert_eq!(r.slice(3, 3).count(), 0);
        assert_eq!(r.slice(5, 2).count(), 0);
    }

    #[test]
    fn slice_empty_ring() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().slice(0, 4).count(), 0);
    }

    // ── take_while / drop_while ────────────────────────────────────────

    #[test]
    fn take_while_basic() {
        let r = [1, 2, 3, 4, 5].circular();
        let mut out = [0; 3];
        for (s, &x) in out.iter_mut().zip(r.take_while(|&x| x < 4, 0)) {
            *s = x;
        }
        assert_eq!(out, [1, 2, 3]);
    }

    #[test]
    fn take_while_stops_after_one_lap() {
        // If the predicate is always true, take_while still terminates at n elements.
        let r = [1, 1, 1].circular();
        assert_eq!(r.take_while(|_| true, 0).count(), 3);
    }

    #[test]
    fn drop_while_basic() {
        let r = [1, 2, 3, 4, 5].circular();
        let mut out = [0; 2];
        for (s, &x) in out.iter_mut().zip(r.drop_while(|&x| x < 4, 0)) {
            *s = x;
        }
        assert_eq!(out, [4, 5]);
    }

    #[test]
    fn drop_while_all_dropped() {
        let r = [1, 2, 3].circular();
        assert_eq!(r.drop_while(|_| true, 0).count(), 0);
    }

    // ── enumerate ──────────────────────────────────────────────────────

    #[test]
    fn enumerate_yields_ring_indices() {
        let r = [10, 20, 30].circular();
        let mut out: [(i32, usize); 3] = [(0, 0); 3];
        for (s, (&x, i)) in out.iter_mut().zip(r.enumerate(1)) {
            *s = (x, i);
        }
        assert_eq!(out, [(20, 1), (30, 2), (10, 0)]);
    }

    #[test]
    fn enumerate_empty() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().enumerate(0).count(), 0);
    }

    #[test]
    fn enumerate_is_exact_size() {
        let r = [10, 20, 30, 40].circular();
        let it = r.enumerate(2);
        assert_eq!(it.len(), 4);
    }

    // ── Iterators of views ─────────────────────────────────────────────

    #[test]
    fn rotations_yields_n_views() {
        let ring = [1, 2, 3, 4];
        let r = ring.circular();
        let firsts: [i32; 4] = {
            let mut out = [0; 4];
            for (s, v) in out.iter_mut().zip(r.rotations()) {
                *s = *v.apply(0);
            }
            out
        };
        assert_eq!(firsts, [1, 2, 3, 4]);
    }

    #[test]
    fn rotations_empty_yields_one() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().rotations().count(), 1);
    }

    #[test]
    fn rotations_exact_size() {
        let r = [1, 2, 3, 4, 5].circular();
        let it = r.rotations();
        assert_eq!(it.len(), 5);
    }

    #[test]
    fn rotations_preserves_reflected() {
        // Rotations of the reflected view should each be reflected too.
        let r = [1, 2, 3].circular().reflect_at(0);
        let firsts: [i32; 3] = {
            let mut out = [0; 3];
            for (s, v) in out.iter_mut().zip(r.rotations()) {
                *s = *v.apply(0);
            }
            out
        };
        // Reflected of [1,2,3] at 0 = [1, 3, 2]. Its rotations start at 1, 3, 2.
        assert_eq!(firsts, [1, 3, 2]);
    }

    #[test]
    fn reflections_yields_two() {
        let r = [1, 2, 3, 4].circular();
        let count = r.reflections().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn reflections_second_is_reflect_at_zero() {
        let r = [1, 2, 3, 4].circular();
        let mut it = r.reflections();
        let first = it.next().unwrap();
        let second = it.next().unwrap();
        let first_v: [i32; 4] = into_array(first);
        let second_v: [i32; 4] = into_array(second);
        assert_eq!(first_v, [1, 2, 3, 4]);
        assert_eq!(second_v, [1, 4, 3, 2]);
    }

    #[test]
    fn reflections_empty_yields_one() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().reflections().count(), 1);
    }

    #[test]
    fn reversions_yields_two() {
        let r = [1, 2, 3].circular();
        assert_eq!(r.reversions().count(), 2);
    }

    #[test]
    fn reversions_second_is_reverse() {
        let r = [1, 2, 3, 4].circular();
        let mut it = r.reversions();
        let first = it.next().unwrap();
        let second = it.next().unwrap();
        let first_v: [i32; 4] = into_array(first);
        let second_v: [i32; 4] = into_array(second);
        assert_eq!(first_v, [1, 2, 3, 4]);
        assert_eq!(second_v, [4, 3, 2, 1]);
    }

    #[test]
    fn rotations_and_reflections_yields_2n() {
        let r = [1, 2, 3, 4].circular();
        assert_eq!(r.rotations_and_reflections().count(), 8);
    }

    #[test]
    fn rotations_and_reflections_distinct_views() {
        // Distinct shapes for an aperiodic ring.
        let r = [1, 2, 3, 4].circular();
        let all: [[i32; 4]; 8] = {
            let mut out = [[0; 4]; 8];
            for (s, v) in out.iter_mut().zip(r.rotations_and_reflections()) {
                *s = into_array(v);
            }
            out
        };
        assert_eq!(all[0], [1, 2, 3, 4]);
        assert_eq!(all[1], [2, 3, 4, 1]);
        assert_eq!(all[2], [3, 4, 1, 2]);
        assert_eq!(all[3], [4, 1, 2, 3]);
        // Reflected half: rotations of [1, 4, 3, 2] (reflect_at(0) of [1,2,3,4])
        assert_eq!(all[4], [1, 4, 3, 2]);
        assert_eq!(all[5], [4, 3, 2, 1]);
        assert_eq!(all[6], [3, 2, 1, 4]);
        assert_eq!(all[7], [2, 1, 4, 3]);
    }

    #[test]
    fn rotations_and_reflections_empty_yields_one() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().rotations_and_reflections().count(), 1);
    }

    // ── Windows / Chunks ───────────────────────────────────────────────

    fn collect_iter<const N: usize>(mut iter: CircularIter<'_, i32>) -> [i32; N] {
        let mut out = [0; N];
        for s in out.iter_mut() {
            *s = *iter.next().unwrap();
        }
        out
    }

    #[test]
    fn windows_basic() {
        let r = [1, 2, 3, 4].circular();
        let mut it = r.windows(2);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [1, 2]);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [2, 3]);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [3, 4]);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [4, 1]); // wraps
        assert!(it.next().is_none());
    }

    #[test]
    fn windows_count_equals_ring_len() {
        let r = [1, 2, 3, 4, 5].circular();
        assert_eq!(r.windows(2).count(), 5);
        assert_eq!(r.windows(3).count(), 5);
        assert_eq!(r.windows(5).count(), 5);
    }

    #[test]
    fn windows_size_greater_than_ring() {
        let r = [1, 2, 3].circular();
        // Each window of size 5 wraps multiple times.
        let mut it = r.windows(5);
        assert_eq!(collect_iter::<5>(it.next().unwrap()), [1, 2, 3, 1, 2]);
        assert_eq!(collect_iter::<5>(it.next().unwrap()), [2, 3, 1, 2, 3]);
        assert_eq!(collect_iter::<5>(it.next().unwrap()), [3, 1, 2, 3, 1]);
        assert!(it.next().is_none());
    }

    #[test]
    fn windows_empty_ring() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().windows(3).count(), 0);
    }

    #[test]
    #[should_panic]
    fn windows_zero_size_panics() {
        let _ = [1, 2, 3].circular().windows(0);
    }

    #[test]
    fn chunks_evenly_divisible() {
        let r = [1, 2, 3, 4].circular();
        let mut it = r.chunks(2);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [1, 2]);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [3, 4]);
        assert!(it.next().is_none());
    }

    #[test]
    fn chunks_wrap_at_seam() {
        // ceil(5/2) = 3 chunks; last one wraps
        let r = [1, 2, 3, 4, 5].circular();
        let mut it = r.chunks(2);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [1, 2]);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [3, 4]);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [5, 1]);
        assert!(it.next().is_none());
    }

    #[test]
    fn chunks_size_one() {
        let r = [1, 2, 3].circular();
        assert_eq!(r.chunks(1).count(), 3);
    }

    #[test]
    fn chunks_size_greater_than_ring() {
        let r = [1, 2, 3].circular();
        // ceil(3/5) = 1 chunk
        let mut it = r.chunks(5);
        assert_eq!(collect_iter::<5>(it.next().unwrap()), [1, 2, 3, 1, 2]);
        assert!(it.next().is_none());
    }

    #[test]
    fn chunks_empty_ring() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().chunks(2).count(), 0);
    }

    #[test]
    #[should_panic]
    fn chunks_zero_size_panics() {
        let _ = [1, 2, 3].circular().chunks(0);
    }

    // ── Chained operations ─────────────────────────────────────────────

    #[test]
    fn chained_rotations_first_element() {
        let r = [3, 1, 2].circular();
        let firsts: [i32; 3] = {
            let mut out = [0; 3];
            for (s, v) in out.iter_mut().zip(r.rotations()) {
                *s = *v.apply(0);
            }
            out
        };
        assert_eq!(firsts, [3, 1, 2]);
    }

    #[test]
    fn chained_windows_count_satisfying() {
        let r = [1, 2, 3, 4, 5].circular();
        // Count windows whose first element is even
        let n = r
            .windows(2)
            .filter(|w| {
                let mut clone = w.clone();
                clone.next().map_or(false, |x| x % 2 == 0)
            })
            .count();
        assert_eq!(n, 2); // windows starting at 2 and 4
    }

    // ── Index normalization ────────────────────────────────────────────

    #[test]
    fn index_from_basic() {
        let r = [10, 20, 30].circular();
        assert_eq!(r.index_from(0), 0);
        assert_eq!(r.index_from(2), 2);
        assert_eq!(r.index_from(3), 0);
        assert_eq!(r.index_from(7), 1);
        assert_eq!(r.index_from(-1), 2);
        assert_eq!(r.index_from(-4), 2);
    }

    #[test]
    #[should_panic]
    fn index_from_empty_panics() {
        let empty: [i32; 0] = [];
        let _ = empty.circular().index_from(0);
    }

    // ── Comparison predicates ──────────────────────────────────────────

    #[test]
    fn is_rotation_of_true() {
        let a = [1, 2, 3, 4];
        let b = [3, 4, 1, 2];
        assert!(a.circular().is_rotation_of(&b));
    }

    #[test]
    fn is_rotation_of_false() {
        let a = [1, 2, 3, 4];
        let b = [4, 3, 2, 1];
        assert!(!a.circular().is_rotation_of(&b));
    }

    #[test]
    fn is_rotation_of_empty() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        assert!(a.circular().is_rotation_of(&b));
    }

    #[test]
    fn is_rotation_of_length_mismatch() {
        assert!(![1, 2].circular().is_rotation_of(&[1, 2, 3]));
    }

    #[test]
    fn is_reflection_of_basic() {
        let a = [1, 2, 3, 4];
        assert!(a.circular().is_reflection_of(&[1, 4, 3, 2]));
        assert!(a.circular().is_reflection_of(&[1, 2, 3, 4]));
        assert!(!a.circular().is_reflection_of(&[2, 1, 4, 3]));
    }

    #[test]
    fn is_reversion_of_basic() {
        let a = [1, 2, 3, 4];
        assert!(a.circular().is_reversion_of(&[1, 2, 3, 4]));
        assert!(a.circular().is_reversion_of(&[4, 3, 2, 1]));
        assert!(!a.circular().is_reversion_of(&[3, 2, 1, 4]));
    }

    #[test]
    fn is_rotation_or_reflection_of_basic() {
        let a = [1, 2, 3, 4];
        assert!(a.circular().is_rotation_or_reflection_of(&[3, 4, 1, 2])); // rotation
        assert!(a.circular().is_rotation_or_reflection_of(&[2, 1, 4, 3])); // rotation of reflection
        assert!(!a.circular().is_rotation_or_reflection_of(&[1, 3, 2, 4])); // neither
    }

    #[test]
    fn rotation_offset_basic() {
        let a = [10, 20, 30, 40];
        assert_eq!(a.circular().rotation_offset(&[30, 40, 10, 20]), Some(2));
        assert_eq!(a.circular().rotation_offset(&[10, 20, 30, 40]), Some(0));
        assert_eq!(a.circular().rotation_offset(&[1, 2, 3, 4]), None);
    }

    #[test]
    fn rotation_offset_length_mismatch() {
        assert_eq!([1, 2, 3].circular().rotation_offset(&[1, 2]), None);
    }

    // ── Distance ───────────────────────────────────────────────────────

    #[test]
    fn hamming_distance_basic() {
        let a = [1, 2, 3, 4];
        let b = [1, 2, 0, 4];
        assert_eq!(a.circular().hamming_distance(&b), 1);
        assert_eq!(a.circular().hamming_distance(&a), 0);
    }

    #[test]
    #[should_panic]
    fn hamming_distance_length_mismatch_panics() {
        let _ = [1, 2].circular().hamming_distance(&[1, 2, 3]);
    }

    #[test]
    fn min_rotational_hamming_distance_basic() {
        let a = [1, 2, 3, 4];
        let b = [3, 4, 1, 2]; // rotation of a
        assert_eq!(a.circular().min_rotational_hamming_distance(&b), 0);
        assert_eq!(a.circular().min_rotational_hamming_distance(&a), 0);
        let c = [3, 4, 1, 5]; // 1 off from a rotation
        assert_eq!(a.circular().min_rotational_hamming_distance(&c), 1);
    }

    // ── contains_slice / index_of_slice ────────────────────────────────

    #[test]
    fn contains_slice_basic() {
        let r = [1, 2, 3, 4, 5].circular();
        assert!(r.contains_slice(&[2, 3, 4]));
        assert!(r.contains_slice(&[5, 1, 2])); // wraps
        assert!(!r.contains_slice(&[2, 4]));
    }

    #[test]
    fn contains_slice_empty() {
        assert!([1, 2, 3].circular().contains_slice(&[]));
        let empty: [i32; 0] = [];
        assert!(!empty.circular().contains_slice(&[1]));
        assert!(empty.circular().contains_slice(&[]));
    }

    #[test]
    fn index_of_slice_basic() {
        let r = [1, 2, 3, 4, 5].circular();
        assert_eq!(r.index_of_slice(&[3, 4], 0), Some(2));
        assert_eq!(r.index_of_slice(&[5, 1], 0), Some(4)); // wraps
        assert_eq!(r.index_of_slice(&[9], 0), None);
    }

    // ── Symmetry counts ────────────────────────────────────────────────

    #[test]
    fn rotational_symmetry_basic() {
        assert_eq!([1, 2, 3, 4].circular().rotational_symmetry(), 1);
        assert_eq!([1, 2, 1, 2].circular().rotational_symmetry(), 2);
        assert_eq!([1, 1, 1, 1].circular().rotational_symmetry(), 4);
    }

    #[test]
    fn rotational_symmetry_short() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().rotational_symmetry(), 1);
        assert_eq!([5].circular().rotational_symmetry(), 1);
    }

    #[test]
    fn symmetry_palindrome() {
        // Palindrome has at least one reflectional symmetry axis.
        assert!([1, 2, 3, 2, 1].circular().symmetry() >= 1);
    }

    #[test]
    fn symmetry_empty() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().symmetry(), 0);
    }
}

#[cfg(all(test, feature = "alloc"))]
mod alloc_tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // ── Booth / canonical ──────────────────────────────────────────────

    #[test]
    fn canonical_index_basic() {
        assert_eq!([3, 1, 2].circular().canonical_index(), 1);
        assert_eq!([1, 2, 3].circular().canonical_index(), 0);
        assert_eq!([2, 3, 0, 1].circular().canonical_index(), 2);
    }

    #[test]
    fn canonical_index_short() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().canonical_index(), 0);
        assert_eq!([7].circular().canonical_index(), 0);
    }

    #[test]
    fn canonical_basic() {
        assert_eq!([3, 1, 2].circular().canonical(), vec![1, 2, 3]);
        assert_eq!([1, 2, 3].circular().canonical(), vec![1, 2, 3]);
    }

    #[test]
    fn canonical_all_equal() {
        assert_eq!([5, 5, 5].circular().canonical(), vec![5, 5, 5]);
    }

    #[test]
    fn canonical_rotations_share_canonical() {
        let a = [3, 1, 2];
        let b = [1, 2, 3];
        let c = [2, 3, 1];
        assert_eq!(a.circular().canonical(), b.circular().canonical());
        assert_eq!(b.circular().canonical(), c.circular().canonical());
    }

    #[test]
    fn bracelet_basic() {
        // Bracelet treats sequence and its reflection as equivalent.
        let a = [3, 1, 2];
        let b = [3, 2, 1]; // reflection of a at index 0 = [3, 2, 1]
        assert_eq!(a.circular().bracelet(), b.circular().bracelet());
    }

    // ── Symmetry indices / axes ────────────────────────────────────────

    #[test]
    fn symmetry_indices_palindrome() {
        let r = [1, 2, 3, 2, 1].circular();
        assert!(!r.symmetry_indices().is_empty());
    }

    #[test]
    fn symmetry_indices_none() {
        let r = [1, 2, 3, 4].circular();
        // Asymmetric ring has no reflectional symmetries
        assert_eq!(r.symmetry_indices(), Vec::<usize>::new());
    }

    #[test]
    fn symmetry_indices_empty() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().symmetry_indices(), Vec::<usize>::new());
    }

    #[test]
    fn reflectional_symmetry_axes_count_matches_symmetry() {
        let r = [1, 2, 1, 2].circular();
        let axes = r.reflectional_symmetry_axes();
        assert_eq!(axes.len(), r.symmetry());
    }

    // ── to_vec ─────────────────────────────────────────────────────────

    #[test]
    fn to_vec_basic() {
        let r = [10, 20, 30].circular();
        assert_eq!(r.to_vec(), vec![10, 20, 30]);
        assert_eq!(r.rotate_right(1).to_vec(), vec![30, 10, 20]);
        assert_eq!(r.reflect_at(0).to_vec(), vec![10, 30, 20]);
    }

    #[test]
    fn to_vec_empty() {
        let empty: [i32; 0] = [];
        assert_eq!(empty.circular().to_vec(), Vec::<i32>::new());
    }

    // ── Chained: prototype's headline example ──────────────────────────

    #[test]
    fn rotations_map_canonical_index() {
        let r = [3, 1, 2, 3, 1, 2].circular();
        let indices: Vec<usize> = r.rotations().map(|v| v.canonical_index()).collect();
        assert_eq!(indices, vec![1, 0, 2, 1, 0, 2]);
    }
}
