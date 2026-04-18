//! Iterator types returned by [`RingSeq`](crate::RingSeq) methods.

/// Sliding windows over a circular sequence.
///
/// Created by [`RingSeq::sliding_o`](crate::RingSeq::sliding_o) and
/// [`RingSeq::grouped_o`](crate::RingSeq::grouped_o).
#[derive(Debug, Clone)]
pub struct SlidingO<T> {
    pub(crate) data: Vec<T>,
    pub(crate) window_size: usize,
    pub(crate) step: usize,
    pub(crate) pos: usize,
}

impl<T: Clone> Iterator for SlidingO<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.window_size > self.data.len() {
            return None;
        }
        let window = self.data[self.pos..self.pos + self.window_size].to_vec();
        self.pos += self.step;
        Some(window)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.pos + self.window_size > self.data.len() {
            0
        } else {
            (self.data.len() - self.pos - self.window_size) / self.step + 1
        };
        (remaining, Some(remaining))
    }
}

impl<T: Clone> ExactSizeIterator for SlidingO<T> {}

// ---------------------------------------------------------------------------
// Helpers shared by the iterators below
// ---------------------------------------------------------------------------

fn rotate_clone<T: Clone>(ring: &[T], i: usize) -> Vec<T> {
    let n = ring.len();
    if n == 0 {
        return vec![];
    }
    let i = i % n;
    let mut v = Vec::with_capacity(n);
    v.extend_from_slice(&ring[i..]);
    v.extend_from_slice(&ring[..i]);
    v
}

fn reflect_at_zero<T: Clone>(ring: &[T]) -> Vec<T> {
    if ring.is_empty() {
        return vec![];
    }
    // reflect_at(0) = start_at(1).reverse()
    let mut v = Vec::with_capacity(ring.len());
    v.extend_from_slice(&ring[1..]);
    v.extend_from_slice(&ring[..1]);
    v.reverse();
    v
}

// ---------------------------------------------------------------------------
// Rotations
// ---------------------------------------------------------------------------

/// All rotations of a circular sequence.
///
/// Created by [`RingSeq::rotations`](crate::RingSeq::rotations).
/// Yields `n` items for a non-empty ring (one per starting position),
/// or a single empty `Vec` for an empty ring.
#[derive(Debug, Clone)]
pub struct Rotations<'a, T> {
    pub(crate) ring: &'a [T],
    pub(crate) index: usize,
    pub(crate) total: usize,
}

impl<T: Clone> Iterator for Rotations<'_, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        let result = rotate_clone(self.ring, self.index);
        self.index += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.total - self.index;
        (r, Some(r))
    }
}

impl<T: Clone> ExactSizeIterator for Rotations<'_, T> {}

// ---------------------------------------------------------------------------
// Reflections
// ---------------------------------------------------------------------------

/// The two orientations of a circular sequence: the original and its
/// reflection at index 0.
///
/// Created by [`RingSeq::reflections`](crate::RingSeq::reflections).
/// Yields 2 items for a non-empty ring, or a single empty `Vec` for an
/// empty ring.
#[derive(Debug, Clone)]
pub struct Reflections<'a, T> {
    pub(crate) ring: &'a [T],
    pub(crate) state: u8,
}

impl<T: Clone> Iterator for Reflections<'_, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            0 => {
                self.state = if self.ring.is_empty() { 2 } else { 1 };
                Some(self.ring.to_vec())
            }
            1 => {
                self.state = 2;
                Some(reflect_at_zero(self.ring))
            }
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = match self.state {
            0 => {
                if self.ring.is_empty() {
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

impl<T: Clone> ExactSizeIterator for Reflections<'_, T> {}

// ---------------------------------------------------------------------------
// Reversions
// ---------------------------------------------------------------------------

/// The two orientations of a circular sequence: the original and its
/// reversal.
///
/// Created by [`RingSeq::reversions`](crate::RingSeq::reversions).
/// Yields 2 items for a non-empty ring, or a single empty `Vec` for an
/// empty ring.
#[derive(Debug, Clone)]
pub struct Reversions<'a, T> {
    pub(crate) ring: &'a [T],
    pub(crate) state: u8,
}

impl<T: Clone> Iterator for Reversions<'_, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            0 => {
                self.state = if self.ring.is_empty() { 2 } else { 1 };
                Some(self.ring.to_vec())
            }
            1 => {
                self.state = 2;
                let mut v = self.ring.to_vec();
                v.reverse();
                Some(v)
            }
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = match self.state {
            0 => {
                if self.ring.is_empty() {
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

impl<T: Clone> ExactSizeIterator for Reversions<'_, T> {}

// ---------------------------------------------------------------------------
// RotationsAndReflections
// ---------------------------------------------------------------------------

/// All rotations of the original sequence followed by all rotations of its
/// reflection.
///
/// Created by
/// [`RingSeq::rotations_and_reflections`](crate::RingSeq::rotations_and_reflections).
/// Yields `2n` items for a non-empty ring, or a single empty `Vec` for an
/// empty ring.
#[derive(Debug, Clone)]
pub struct RotationsAndReflections<'a, T> {
    pub(crate) ring: &'a [T],
    pub(crate) reflected: Vec<T>,
    pub(crate) index: usize,
    pub(crate) total: usize,
}

impl<T: Clone> Iterator for RotationsAndReflections<'_, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        let n = self.ring.len();
        if n == 0 {
            // Empty ring: yield one empty vec
            self.index += 1;
            return Some(vec![]);
        }
        let (source, rot) = if self.index < n {
            (self.ring, self.index)
        } else {
            (self.reflected.as_slice(), self.index - n)
        };
        self.index += 1;
        Some(rotate_clone(source, rot))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.total - self.index;
        (r, Some(r))
    }
}

impl<T: Clone> ExactSizeIterator for RotationsAndReflections<'_, T> {}
