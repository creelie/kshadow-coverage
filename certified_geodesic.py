"""Outward bounds on intrinsic mesh distances from explicit edge paths.

Edge lengths are rounded upwards to multiples of a dyadic quantum using
integer square roots. Integer Dijkstra then has no accumulated float error.
The bound concerns the supplied piecewise-flat mesh, not anatomical fidelity.
"""
from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
import numpy as np


@dataclass(frozen=True)
class CertifiedPathBounds:
    upper: np.ndarray
    quantum: float
    vertices: int
    edges: int
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    search_limit: float | None


def _outward_float(units, quantum):
    exact = units * quantum
    try:
        result = float(exact)
    except OverflowError as exc:
        raise ValueError("path bound exceeds finite binary64 range") from exc
    if not math.isfinite(result):
        raise ValueError("path bound exceeds finite binary64 range")
    if Fraction(result) < exact:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise ValueError("path bound exceeds finite binary64 range")
    return result


def certified_graph_vertex_bounds(vertices, faces, sources, *, targets=None,
                                  quantum=2.0**-20, search_limit=None,
                                  progress=None):
    """Return valid upper bounds for contact-to-vertex intrinsic distances.

    Sources are mesh vertex indices. Every edge is a permitted surface path.
    A truncated search fills unreached targets with a conservative connected
    graph bound; it never treats an unreached target as covered. The graph
    must be connected. ``quantum`` is a finite positive power of two in the
    coordinate units. Target faces are validated by the coverage certificate.
    """
    xyz = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces)
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3) or len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("vertices must be finite with shape (V,2) or (V,3)")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or not np.issubdtype(tri.dtype, np.integer):
        raise ValueError("faces must be a nonempty integer array with shape (F,3)")
    if np.any(tri < 0) or np.any(tri >= len(xyz)) or np.any(np.diff(np.sort(tri, axis=1), axis=1) == 0):
        raise ValueError("invalid or repeated face indices")
    def indices(values, label):
        out = np.asarray(values)
        if out.ndim != 1 or not len(out) or not np.issubdtype(out.dtype, np.integer):
            raise ValueError(f"{label} must be a nonempty integer vector")
        if np.any(out < 0) or np.any(out >= len(xyz)):
            raise ValueError(f"{label} outside vertex range")
        return tuple(int(x) for x in out)
    source_ids = indices(sources, "sources")
    target_ids = tuple(range(len(xyz))) if targets is None else indices(targets, "targets")
    quantum = float(quantum)
    if not math.isfinite(quantum) or quantum <= 0:
        raise ValueError("quantum must be finite and positive")
    q = Fraction(quantum)
    if q.numerator & (q.numerator-1) or q.denominator & (q.denominator-1):
        raise ValueError("quantum must be a power of two")
    if search_limit is not None:
        search_limit = float(search_limit)
        if not math.isfinite(search_limit) or search_limit < 0:
            raise ValueError("search_limit must be finite and nonnegative")
    ratios = [[float(x).as_integer_ratio() for x in row] for row in xyz]
    denominator = max(d for row in ratios for _, d in row)
    coords = [tuple(n*(denominator//d) for n, d in row) for row in ratios]
    del ratios
    edges = np.unique(np.sort(np.vstack([tri[:, [0,1]], tri[:, [1,2]], tri[:, [2,0]]]), axis=1), axis=0)
    graph = [[] for _ in xyz]
    divisor = (denominator*q.numerator)**2
    multiplier = q.denominator**2
    maximum = 0
    for i0, j0 in edges:
        i, j = int(i0), int(j0)
        squared = sum((a-b)**2 for a,b in zip(coords[i], coords[j])) * multiplier
        weight = math.isqrt(squared//divisor)
        if weight*weight*divisor < squared:
            weight += 1
        graph[i].append((j, weight)); graph[j].append((i, weight))
        maximum = max(maximum, weight)
    seen = {0}; stack = [0]
    while stack:
        for v, _ in graph[stack.pop()]:
            if v not in seen:
                seen.add(v); stack.append(v)
    if len(seen) != len(xyz):
        raise ValueError("mesh edge graph must be connected")
    # Every connected graph admits a simple path of at most V-1 edges.
    fallback = (len(xyz)-1)*maximum
    limit = fallback if search_limit is None else min(fallback, int(Fraction(search_limit)//q))
    upper = np.empty((len(source_ids), len(target_ids)), dtype=float)
    for row, source in enumerate(source_ids):
        distance = [None]*len(xyz); distance[source] = 0
        heap = [(0, source)]
        while heap:
            cost, u = heapq.heappop(heap)
            if distance[u] != cost:
                continue
            for v, weight in graph[u]:
                candidate = cost+weight
                if candidate <= limit and (distance[v] is None or candidate < distance[v]):
                    distance[v] = candidate
                    heapq.heappush(heap, (candidate, v))
        upper[row] = [_outward_float(fallback if distance[v] is None else distance[v], q)
                      for v in target_ids]
        if progress is not None:
            progress(row+1, len(source_ids))
    return CertifiedPathBounds(upper, quantum, len(xyz), len(edges), source_ids,
                               target_ids, search_limit)
