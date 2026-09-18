"""
Exact test for whether K disks of equal radius R have a common point:
true iff the smallest enclosing circle of their centers has radius <= R.

The smallest enclosing circle of a finite point set is always determined
by at most 3 of the points (classical fact): either two points (as a
diameter), or three points (as a circumcircle). We use this directly
since our candidate sets are always small (<=6 points here), rather than
a general-purpose algorithm like Welzl's -- both are exact; this one is
simplest to verify by hand.
"""
import itertools
import numpy as np


def circle_from_2(p, q):
    c = (p + q) / 2.0
    r = np.linalg.norm(p - q) / 2.0
    return c, r


def circle_from_3(p, q, r):
    ax, ay = p; bx, by = q; cx, cy = r
    d = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-12:
        return None  # collinear, no finite circumcircle
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / d
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / d
    center = np.array([ux, uy])
    radius = np.linalg.norm(center - p)
    return center, radius


def min_enclosing_circle_radius(points, tol=1e-9):
    """Exact minimum enclosing circle radius for a small point set,
    via the classical fact that it is determined by <=3 of the points."""
    pts = [np.asarray(p, dtype=float) for p in points]
    n = len(pts)
    if n == 1:
        return 0.0
    best_r = None
    # candidates from pairs (as diameters)
    for i, j in itertools.combinations(range(n), 2):
        c, r = circle_from_2(pts[i], pts[j])
        if all(np.linalg.norm(c - pts[k]) <= r + tol for k in range(n)):
            if best_r is None or r < best_r:
                best_r = r
    # candidates from triples (as circumcircles)
    for i, j, k in itertools.combinations(range(n), 3):
        res = circle_from_3(pts[i], pts[j], pts[k])
        if res is None:
            continue
        c, r = res
        if all(np.linalg.norm(c - pts[m]) <= r + tol for m in range(n)):
            if best_r is None or r < best_r:
                best_r = r
    return best_r


def disks_intersect_exact(idxs, sensors, R):
    pts = [sensors[i] for i in idxs]
    r_min = min_enclosing_circle_radius(pts)
    return r_min <= R + 1e-9


if __name__ == "__main__":
    # quick self-tests
    import numpy.testing as npt
    # two points distance 2 apart, R=1.0 -> circle of radius 1 exactly touches
    assert disks_intersect_exact([0, 1], np.array([[0, 0], [2, 0]]), 1.0)
    assert not disks_intersect_exact([0, 1], np.array([[0, 0], [2.01, 0]]), 1.0)
    # equilateral triangle side s, circumradius = s/sqrt(3)
    s = 1.0
    pts = np.array([[0, 0], [s, 0], [s/2, s*np.sqrt(3)/2]])
    circumradius = s / np.sqrt(3)
    assert disks_intersect_exact([0, 1, 2], pts, circumradius + 1e-6)
    assert not disks_intersect_exact([0, 1, 2], pts, circumradius - 1e-3)
    print("self-tests passed")
