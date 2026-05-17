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
}
