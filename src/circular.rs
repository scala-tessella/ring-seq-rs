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
        let total = if self.ring.is_empty() { 1 } else { self.ring.len() };
        Rotations { base: self, index: 0, total }
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
        Reflections { base: self, state: 0 }
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
        Reversions { base: self, state: 0 }
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
        RotationsAndReflections { base: self, reflected, index: 0, total, n }
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
        let total = if self.ring.is_empty() { 0 } else { self.ring.len() };
        Windows { base: self, size, step: 1, index: 0, total }
    }

    /// Returns an iterator yielding non-overlapping circular chunks of
    /// length `size` as [`CircularIter`]s.
    ///
    /// For a non-empty ring of length `n` the iterator yields `n` chunks
    /// (step equal to `size`, wrapping); for an empty ring it yields none.
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
    /// let chunks: Vec<Vec<i32>> = r
    ///     .chunks(2)
    ///     .map(|c| c.copied().collect())
    ///     .collect();
    /// // Four starting positions (0, 2, 4≡0, 6≡2), each yielding 2 elements
    /// assert_eq!(chunks.len(), 4);
    /// assert_eq!(chunks[0], vec![1, 2]);
    /// assert_eq!(chunks[1], vec![3, 4]);
    /// ```
    #[must_use]
    pub fn chunks(self, size: usize) -> Windows<'a, T> {
        assert!(size > 0, "chunk size must be positive");
        let total = if self.ring.is_empty() { 0 } else { self.ring.len() };
        Windows { base: self, size, step: size, index: 0, total }
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
        Rotations { base: self.base, index: self.index, total: self.total }
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
        Reflections { base: self.base, state: self.state }
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
        Reversions { base: self.base, state: self.state }
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
        let iter = CircularIter { view, pos: 0, remaining: self.size };
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

    // ── Iterators of views ─────────────────────────────────────────────

    #[test]
    fn rotations_yields_n_views() {
        let ring = [1, 2, 3, 4];
        let r = ring.circular();
        let firsts: [i32; 4] = {
            let mut out = [0; 4];
            for (s, v) in out.iter_mut().zip(r.rotations()) { *s = *v.apply(0); }
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
    fn chunks_basic() {
        let r = [1, 2, 3, 4].circular();
        let mut it = r.chunks(2);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [1, 2]);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [3, 4]);
        // Step is `size`, so positions 0, 2, 4≡0, 6≡2 → still 4 items
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [1, 2]);
        assert_eq!(collect_iter::<2>(it.next().unwrap()), [3, 4]);
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
        let n = r.windows(2).filter(|w| {
            let mut clone = w.clone();
            clone.next().map_or(false, |x| x % 2 == 0)
        }).count();
        assert_eq!(n, 2); // windows starting at 2 and 4
    }
}
