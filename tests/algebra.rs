//! Exhaustive property tests for the `(offset, reflected)` view algebra.
//!
//! Every ring of length 0..=4 over the alphabet {0, 1, 2}, and of length
//! 5..=6 over {0, 1}, is checked under every reachable view transform —
//! all offsets, reflected or not. The view state space is exactly
//! `(offset, reflected)`, so this enumeration covers it completely. Each
//! wrapper operation is compared against a naive reference computed on
//! the materialized view.
//!
//! Exhaustive small-case enumeration is preferred here over a
//! property-testing crate: it is deterministic, adds no dependencies (the
//! crate has none), and small rings with small alphabets already exercise
//! every frame-mapping and periodicity case.

use ring_seq::{AsCircular, Circular};

// ── naive references ────────────────────────────────────────────────────

/// Euclidean modulus of `i` into `[0, n)`.
fn m(i: isize, n: usize) -> usize {
    i.rem_euclid(n as isize) as usize
}

/// `v` rotated left by `k`.
fn rot(v: &[i32], k: usize) -> Vec<i32> {
    let n = v.len();
    (0..n).map(|i| v[(i + k) % n]).collect()
}

/// `v` reflected at position 0: `[v[0], v[n-1], ..., v[1]]`.
fn refl0(v: &[i32]) -> Vec<i32> {
    let n = v.len();
    (0..n).map(|p| v[(n - p) % n]).collect()
}

fn reversed(v: &[i32]) -> Vec<i32> {
    let mut out = v.to_vec();
    out.reverse();
    out
}

fn naive_index_of(exp: &[i32], needle: &[i32], from: usize) -> Option<usize> {
    let n = exp.len();
    if n == 0 {
        return if needle.is_empty() { Some(0) } else { None };
    }
    if needle.is_empty() {
        return Some(from % n);
    }
    (0..n).map(|k| (from + k) % n).find(|&s| {
        needle
            .iter()
            .enumerate()
            .all(|(j, &x)| exp[(s + j) % n] == x)
    })
}

fn to_vec(view: Circular<'_, i32>) -> Vec<i32> {
    view.iter().copied().collect()
}

// ── ring enumeration ────────────────────────────────────────────────────

fn enumerate_rings(n: usize, alphabet: usize) -> Vec<Vec<i32>> {
    let count = alphabet.pow(n as u32);
    (0..count)
        .map(|mut code| {
            (0..n)
                .map(|_| {
                    let d = (code % alphabet) as i32;
                    code /= alphabet;
                    d
                })
                .collect()
        })
        .collect()
}

fn all_rings() -> Vec<Vec<i32>> {
    let mut rings: Vec<Vec<i32>> = (0..=4).flat_map(|n| enumerate_rings(n, 3)).collect();
    rings.extend((5..=6).flat_map(|n| enumerate_rings(n, 2)));
    rings
}

// ── per-view checks ─────────────────────────────────────────────────────

fn check_iteration(v: Circular<'_, i32>, exp: &[i32]) {
    let n = exp.len();
    assert_eq!(to_vec(v), exp);
    assert_eq!(v.len(), n);
    assert_eq!(v.is_empty(), n == 0);
    let back: Vec<i32> = v.iter().rev().copied().collect();
    assert_eq!(back, reversed(exp));
    assert_eq!(v.iter().count(), n);
    assert_eq!(v.iter().last(), exp.last());
    for k in 0..n + 2 {
        assert_eq!(v.iter().nth(k), exp.get(k));
    }
    // IntoIterator, by value and by reference
    assert!(v.into_iter().eq(exp.iter()));
    assert!((&v).into_iter().eq(exp.iter()));
}

fn check_indexing(v: Circular<'_, i32>, exp: &[i32], ring: &[i32]) {
    let n = exp.len();
    if n == 0 {
        assert_eq!(v.get(0), None);
        return;
    }
    for i in -(2 * n as isize + 1)..=(2 * n as isize + 1) {
        let want = exp[m(i, n)];
        assert_eq!(*v.apply(i), want);
        assert_eq!(v.get(i), Some(&want));
        assert_eq!(v[i], want);
        // index_from must point at the same element in the underlying ring
        assert_eq!(ring[v.index_from(i)], want);
    }
}

fn check_transforms(v: Circular<'_, i32>, exp: &[i32]) {
    let n = exp.len();
    if n == 0 {
        assert!(to_vec(v.start_at(3)).is_empty());
        assert!(to_vec(v.reflect_at(3)).is_empty());
        return;
    }
    for k in 0..n {
        assert_eq!(to_vec(v.start_at(k as isize)), rot(exp, k));
        // reflect_at(k): position p holds exp[(k - p) mod n]
        let refl: Vec<i32> = (0..n).map(|p| exp[m(k as isize - p as isize, n)]).collect();
        assert_eq!(to_vec(v.reflect_at(k as isize)), refl);
        // involution
        assert_eq!(to_vec(v.reflect_at(k as isize).reflect_at(k as isize)), exp);
    }
    assert_eq!(to_vec(v.rotate_left(1)), rot(exp, 1));
    assert_eq!(to_vec(v.rotate_right(1)), rot(exp, n - 1));
}

fn check_view_iterators(v: Circular<'_, i32>, exp: &[i32]) {
    let n = exp.len();
    let rots: Vec<Vec<i32>> = v.rotations().map(to_vec).collect();
    if n == 0 {
        assert_eq!(rots, vec![Vec::<i32>::new()]);
    } else {
        assert_eq!(rots.len(), n);
        for (k, r) in rots.iter().enumerate() {
            assert_eq!(*r, rot(exp, k));
        }
    }

    let refls: Vec<Vec<i32>> = v.reflections().map(to_vec).collect();
    if n == 0 {
        assert_eq!(refls.len(), 1);
    } else {
        assert_eq!(refls, vec![exp.to_vec(), refl0(exp)]);
    }

    let revs: Vec<Vec<i32>> = v.reversions().map(to_vec).collect();
    if n == 0 {
        assert_eq!(revs.len(), 1);
    } else {
        assert_eq!(revs, vec![exp.to_vec(), reversed(exp)]);
    }

    let rr: Vec<Vec<i32>> = v.rotations_and_reflections().map(to_vec).collect();
    if n == 0 {
        assert_eq!(rr.len(), 1);
    } else {
        assert_eq!(rr.len(), 2 * n);
        let refl = refl0(exp);
        for (k, item) in rr.iter().enumerate() {
            let want = if k < n {
                rot(exp, k)
            } else {
                rot(&refl, k - n)
            };
            assert_eq!(*item, want);
        }
    }
}

fn check_bounded_iteration(v: Circular<'_, i32>, exp: &[i32]) {
    let n = exp.len();
    for from in -(n as isize + 1)..=(n as isize + 1) {
        for extra in 0..=(2 * n as isize + 1) {
            let to = from + extra;
            let got: Vec<i32> = v.slice(from, to).copied().collect();
            let want: Vec<i32> = if n == 0 {
                Vec::new()
            } else {
                (from..to).map(|i| exp[m(i, n)]).collect()
            };
            assert_eq!(got, want);
        }
    }

    for size in 1..=n + 2 {
        let ws: Vec<Vec<i32>> = v.windows(size).map(|w| w.copied().collect()).collect();
        if n == 0 {
            assert!(ws.is_empty());
        } else {
            assert_eq!(ws.len(), n);
            for (s, w) in ws.iter().enumerate() {
                let want: Vec<i32> = (0..size).map(|j| exp[(s + j) % n]).collect();
                assert_eq!(*w, want);
            }
        }
        let cs: Vec<Vec<i32>> = v.chunks(size).map(|c| c.copied().collect()).collect();
        if n == 0 {
            assert!(cs.is_empty());
        } else {
            assert_eq!(cs.len(), (n + size - 1) / size);
            for (idx, c) in cs.iter().enumerate() {
                let want: Vec<i32> = (0..size).map(|j| exp[(idx * size + j) % n]).collect();
                assert_eq!(*c, want);
            }
        }
    }

    for from in 0..n {
        let shifted = rot(exp, from);
        let tw: Vec<i32> = v.take_while(|&x| x < 2, from as isize).copied().collect();
        let want_tw: Vec<i32> = shifted.iter().copied().take_while(|&x| x < 2).collect();
        assert_eq!(tw, want_tw);
        let dw: Vec<i32> = v.drop_while(|&x| x < 2, from as isize).copied().collect();
        let want_dw: Vec<i32> = shifted.iter().copied().skip_while(|&x| x < 2).collect();
        assert_eq!(dw, want_dw);

        let pairs: Vec<(i32, usize)> = v.enumerate(from as isize).map(|(&x, i)| (x, i)).collect();
        assert_eq!(pairs.len(), n);
        for (j, &(x, _)) in pairs.iter().enumerate() {
            assert_eq!(x, shifted[j]);
        }
    }
}

fn check_search(v: Circular<'_, i32>, exp: &[i32]) {
    let n = exp.len();
    for from in 0..n.max(1) {
        for start in 0..n {
            for len in 0..=n {
                let needle: Vec<i32> = (0..len).map(|j| exp[(start + j) % n]).collect();
                assert!(v.contains_slice(&needle));
                assert_eq!(
                    v.index_of_slice(&needle, from as isize),
                    naive_index_of(exp, &needle, from),
                );
            }
        }
    }
    // A needle longer than the ring matches by wrapping.
    if n > 0 {
        let wrap: Vec<i32> = (0..n + 1).map(|j| exp[j % n]).collect();
        assert!(v.contains_slice(&wrap));
        assert_eq!(v.index_of_slice(&wrap, 0), Some(0));
    }
    // 9 is outside the alphabet.
    assert!(!v.contains_slice(&[9]));
    assert_eq!(v.index_of_slice(&[9], 0), None);
    assert!(v.contains_slice(&[]));
}

fn check_necklace_and_symmetry(v: Circular<'_, i32>, exp: &[i32]) {
    let n = exp.len();
    if n == 0 {
        assert_eq!(v.canonical_index(), 0);
        assert_eq!(v.rotational_symmetry(), 1);
        assert_eq!(v.symmetry(), 0);
    } else {
        let canon = (0..n).map(|k| rot(exp, k)).min().unwrap();
        let canon_idx = (0..n).find(|&k| rot(exp, k) == canon).unwrap();
        assert_eq!(v.canonical_index(), canon_idx);

        let rot_sym = (0..n).filter(|&k| rot(exp, k) == exp).count();
        assert_eq!(v.rotational_symmetry(), rot_sym);

        // An axis at shift k: exp[i] == exp[(k - i) mod n] for all i.
        let axes = (0..n)
            .filter(|&k| (0..n).all(|i| exp[i] == exp[m(k as isize - i as isize, n)]))
            .count();
        assert_eq!(v.symmetry(), axes);
    }
    check_alloc_ops(v, exp);
}

/// Sort key for an axis location: vertices before edges, then by index.
#[cfg(feature = "alloc")]
type LocKey = (u8, usize, usize);

#[cfg(feature = "alloc")]
fn loc_key(loc: &ring_seq::AxisLocation, n: usize) -> LocKey {
    match *loc {
        ring_seq::AxisLocation::Vertex(v) => {
            assert!(v < n);
            (0, v, v)
        }
        ring_seq::AxisLocation::Edge(i, j) => {
            // Documented invariant: an edge joins consecutive indices.
            assert_eq!(j, (i + 1) % n);
            (1, i, j)
        }
    }
}

/// Independent axis geometry: the reflection `i -> (k - i) mod n` fixes
/// the vertices with `2i = k (mod n)` and the midpoints of the edges
/// `(j, j+1)` with `2j + 1 = k (mod n)`. Every axis crosses the ring in
/// exactly two such locations.
#[cfg(feature = "alloc")]
fn naive_axes(exp: &[i32]) -> Vec<(LocKey, LocKey)> {
    let n = exp.len();
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    for k in 0..n {
        if (0..n).all(|i| exp[i] == exp[m(k as isize - i as isize, n)]) {
            let mut locs = Vec::new();
            for i in 0..n {
                if (2 * i) % n == k {
                    locs.push((0u8, i, i));
                }
                if (2 * i + 1) % n == k {
                    locs.push((1u8, i, (i + 1) % n));
                }
            }
            assert_eq!(locs.len(), 2, "an axis must cross the ring twice");
            locs.sort_unstable();
            out.push((locs[0], locs[1]));
        }
    }
    out.sort_unstable();
    out
}

#[cfg(feature = "alloc")]
fn check_alloc_ops(v: Circular<'_, i32>, exp: &[i32]) {
    let n = exp.len();
    assert_eq!(v.to_vec(), exp);
    assert_eq!(v.symmetry_indices().len(), v.symmetry());

    // Axis locations must match the fixed points of each symmetric
    // reflection, as an unordered set of unordered pairs.
    let mut got: Vec<_> = v
        .reflectional_symmetry_axes()
        .iter()
        .map(|(a, b)| {
            let mut pair = [loc_key(a, n), loc_key(b, n)];
            pair.sort_unstable();
            (pair[0], pair[1])
        })
        .collect();
    got.sort_unstable();
    assert_eq!(got, naive_axes(exp));
    assert_eq!(got.len(), v.symmetry());

    if n > 0 {
        let canon = (0..n).map(|k| rot(exp, k)).min().unwrap();
        assert_eq!(v.canonical(), canon);
        let refl = refl0(exp);
        let bracelet = (0..n)
            .map(|k| rot(exp, k))
            .chain((0..n).map(|k| rot(&refl, k)))
            .min()
            .unwrap();
        assert_eq!(v.bracelet(), bracelet);
    }
}

#[cfg(not(feature = "alloc"))]
fn check_alloc_ops(_v: Circular<'_, i32>, _exp: &[i32]) {}

fn check_comparisons(v: Circular<'_, i32>, exp: &[i32]) {
    let n = exp.len();
    if n == 0 {
        assert!(v.is_rotation_of(&[]));
        assert_eq!(v.rotation_offset(&[]), Some(0));
        return;
    }
    let other = rot(exp, 1);
    assert!(v.is_rotation_of(&other));
    assert_eq!(
        v.rotation_offset(&other),
        (0..n).find(|&k| rot(exp, k) == other),
    );
    assert!(v.is_reflection_of(&refl0(exp)));
    assert!(v.is_reversion_of(&reversed(exp)));
    assert!(v.is_rotation_or_reflection_of(&rot(&refl0(exp), 1)));

    assert_eq!(v.hamming_distance(exp), 0);
    let mut mutated = exp.to_vec();
    mutated[0] += 10; // 10+ is outside the alphabet
    assert!(!v.is_rotation_of(&mutated));
    assert_eq!(v.hamming_distance(&mutated), 1);
    let naive_min = (0..n)
        .map(|k| {
            rot(exp, k)
                .iter()
                .zip(mutated.iter())
                .filter(|(a, b)| a != b)
                .count()
        })
        .min()
        .unwrap();
    assert_eq!(v.min_rotational_hamming_distance(&mutated), naive_min);
}

// ── driver ──────────────────────────────────────────────────────────────

fn check_ring(ring: &[i32]) {
    let n = ring.len();
    for offset in 0..n.max(1) {
        for reflected in [false, true] {
            let mut view = ring.circular().start_at(offset as isize);
            let mut exp = rot(ring, offset);
            if reflected {
                view = view.reflect_at(0);
                exp = refl0(&exp);
            }
            check_iteration(view, &exp);
            check_indexing(view, &exp, ring);
            check_transforms(view, &exp);
            check_view_iterators(view, &exp);
            check_bounded_iteration(view, &exp);
            check_search(view, &exp);
            check_necklace_and_symmetry(view, &exp);
            check_comparisons(view, &exp);
        }
    }
}

#[test]
fn exhaustive_view_algebra() {
    let rings = all_rings();
    assert_eq!(rings.len(), 1 + 3 + 9 + 27 + 81 + 32 + 64);
    for ring in &rings {
        check_ring(ring);
    }
}
