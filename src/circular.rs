//! The [`Circular`] wrapper and its element iterator.
//!
//! Reach the wrapper through [`AsCircular::circular`] on any slice (or
//! anything that derefs to one — [`Vec<T>`](alloc::vec::Vec), arrays,
//! [`Box<[T]>`](alloc::boxed::Box)). The wrapper is the single home for
//! every circular operation; all transforms it offers are lazy and
//! allocation-free.

use core::iter::FusedIterator;

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
        Circular { ring, offset: 0, reflected: false }
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
        CircularIter { view: self, pos: 0, remaining: self.ring.len() }
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
        Circular { ring: self.ring, offset: new_offset, reflected: self.reflected }
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
            return CircularIter { view: self, pos: 0, remaining: 0 };
        }
        let count = (to - from) as usize;
        let started = self.start_at(from);
        CircularIter { view: started, pos: 0, remaining: count }
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
        let started = if self.ring.is_empty() { self } else { self.start_at(from) };
        Enumerate { view: started, pos: 0, remaining: started.ring.len() }
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
        CircularIter { view: self.view, pos: self.pos, remaining: self.remaining }
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
        Enumerate { view: self.view, pos: self.pos, remaining: self.remaining }
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
            assert_eq!(into_array::<7>(r.reflect_at(i).reflect_at(i)), ring, "failed for i {}", i);
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
            for (s, &x) in out.iter_mut().zip(r.slice(1, 4)) { *s = x; }
            out
        };
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn slice_wraps() {
        let r = [0, 1, 2, 3, 4].circular();
        let mut out = [0; 5];
        for (s, &x) in out.iter_mut().zip(r.slice(2, 7)) { *s = x; }
        assert_eq!(out, [2, 3, 4, 0, 1]);
    }

    #[test]
    fn slice_negative_from() {
        let r = [0, 1, 2, 3, 4].circular();
        let mut out = [0; 3];
        for (s, &x) in out.iter_mut().zip(r.slice(-2, 1)) { *s = x; }
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
        for (s, &x) in out.iter_mut().zip(r.take_while(|&x| x < 4, 0)) { *s = x; }
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
        for (s, &x) in out.iter_mut().zip(r.drop_while(|&x| x < 4, 0)) { *s = x; }
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
}
