"""Sufficient bounds for continuum k-coverage on a piecewise-flat mesh.

The caller must supply genuine upper bounds on each contact-to-vertex distance
in the intended metric. This module checks the finite algebraic inequality; it
does not verify that a numerical geodesic solver supplied upper bounds.
See docs/mesh-coverage-bound.md for the complete proof and scope.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


@dataclass(frozen=True)
class MeshCoverageBound:
    """Passing faces satisfy the sufficient bound, subject to the input contract.

    An unresolved face may already be covered; failure is not a hole certificate.
    The target is the union of the supplied triangular faces.
    """

    k: int
    open_sets: bool
    passing_faces: tuple[int, ...]
    unresolved_faces: tuple[int, ...]

    @property
    def all_faces_pass(self) -> bool:
        return not self.unresolved_faces


def _finite_array(values, label):
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _squared_norm(vector):
    return sum((x * x for x in vector), Fraction(0))


def certify_mesh_k_coverage(
    vertices, faces, contact_distance_upper_bounds, radius, *, k=1, open_sets=True
):
    """Check a sufficient whole-face k-coverage inequality exactly.

    vertices: finite binary64 coordinates, shape (V, 2) or (V, 3).
    faces: nonempty integer vertex-index array, shape (F, 3), no degenerate faces.
    contact_distance_upper_bounds: finite nonnegative array of shape (C, V).
      Entry [i, v] must bound contact i's true distance to vertex v from above.
      Face segments must be admissible paths of their Euclidean lengths in that
      metric, as for the intrinsic metric of the represented piecewise-flat mesh.
    radius: common finite nonnegative contact radius; 1 <= k <= C.

    For each face let m be the largest vertex kth-distance upper bound and let
    D be its largest edge length. The face passes iff m + D/sqrt(3) < radius
    (open sets), or <= radius (closed sets). Comparisons use exact Fractions of
    the binary64 inputs, avoiding rounded square roots and scale tolerances.
    Passing all faces implies coverage of their union under the input contract.
    """
    xyz = _finite_array(vertices, "vertices")
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3) or not len(xyz):
        raise ValueError("vertices must have shape (V, 2) or (V, 3), V > 0")
    triangles = np.asarray(faces)
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)):
        raise ValueError("faces must be a nonempty integer array of shape (F, 3)")
    if np.any(triangles < 0) or np.any(triangles >= len(xyz)):
        raise ValueError("face indices are outside the vertex array")
    upper = _finite_array(contact_distance_upper_bounds, "distance upper bounds")
    if upper.ndim != 2 or upper.shape[1] != len(xyz) or not len(upper):
        raise ValueError("distance upper bounds must have shape (C, V), C > 0")
    if np.any(upper < 0):
        raise ValueError("distance upper bounds must be nonnegative")
    if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 1 <= k <= len(upper)):
        raise ValueError("k must be an integer between 1 and the number of contacts")
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if not isinstance(open_sets, (bool, np.bool_)):
        raise ValueError("open_sets must be boolean")
    qradius = Fraction(radius)
    points = [tuple(Fraction(float(x)) for x in row) for row in xyz]
    kth_upper = [sorted(Fraction(float(x)) for x in upper[:, v])[k - 1]
                 for v in range(len(xyz))]
    passed, unresolved = [], []
    for index, face in enumerate(triangles):
        a, b, c = (points[int(v)] for v in face)
        ab = tuple(y - x for x, y in zip(a, b))
        ac = tuple(y - x for x, y in zip(a, c))
        bc = tuple(y - x for x, y in zip(b, c))
        ab2, ac2, bc2 = map(_squared_norm, (ab, ac, bc))
        dot = sum((x * y for x, y in zip(ab, ac)), Fraction(0))
        if ab2 * ac2 - dot * dot <= 0:
            raise ValueError(f"face {index} is degenerate")
        diameter2 = max(ab2, ac2, bc2)
        margin = qradius - max(kth_upper[int(v)] for v in face)
        if open_sets:
            ok = margin > 0 and diameter2 < 3 * margin * margin
        else:
            ok = margin >= 0 and diameter2 <= 3 * margin * margin
        (passed if ok else unresolved).append(index)
    return MeshCoverageBound(int(k), bool(open_sets), tuple(passed), tuple(unresolved))
