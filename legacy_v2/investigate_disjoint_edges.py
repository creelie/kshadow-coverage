"""
Reproduces the disjoint-edge investigation reported in Section 8.1 of
the manuscript.

A meta-triangle of Delta_2(N) is three N-edges whose vertex union is a
face of N. This union falls into one of four patterns depending on how
many of the three pairs of edges share a sensor: all three pairs share
a vertex, exactly two do, exactly one does, or none do.

This script:

  1. Classifies every true meta-triangle of Field A by this pattern,
     using the already-verified brute-force search as ground truth.

  2. Demonstrates that the disjoint-edge case (zero shared pairs) can
     be found efficiently by enumerating N's own size-six faces
     directly, rather than searching Delta_2(N)'s much larger
     meta-graph, and verifies this gives an exact match against the
     brute-force count.

  3. Reports honestly that this case is a small fraction of the total
     (about 3 per cent on Field A), so it does not by itself resolve
     Field E's incompleteness.

Usage:
    python3 investigate_disjoint_edges.py
"""
import itertools
import json
from collections import defaultdict

import numpy as np
import networkx as nx

from ondemand_pipeline import build_nerve_2skeleton, NerveOracle


def classify_true_triangles(sensors, R, skel):
    """Brute-force ground truth: every meta-triangle of Delta_2(N),
    classified by how many of its three edge-pairs share a sensor."""
    edges = skel["edges"]
    oracle = NerveOracle(sensors, R, skel)
    M = len(edges)

    meta_edges = []
    for i in range(M):
        for j in range(i + 1, M):
            u = set(edges[i]) | set(edges[j])
            if oracle.is_face(tuple(sorted(u))):
                meta_edges.append((i, j))
    meta_adj = {i: set() for i in range(M)}
    for i, j in meta_edges:
        meta_adj[i].add(j)
        meta_adj[j].add(i)

    true_triangles = set()
    for i in range(M):
        for j in meta_adj[i]:
            if j <= i:
                continue
            for k in (meta_adj[i] & meta_adj[j]):
                if k <= j:
                    continue
                u = set(edges[i]) | set(edges[j]) | set(edges[k])
                if oracle.is_face(tuple(sorted(u))):
                    true_triangles.add((i, j, k))

    pattern_counts = defaultdict(int)
    disjoint_triples = set()
    for (i, j, k) in true_triangles:
        ei, ej, ek = set(edges[i]), set(edges[j]), set(edges[k])
        sharing = sum([bool(ei & ej), bool(ei & ek), bool(ej & ek)])
        pattern_counts[sharing] += 1
        if sharing == 0:
            disjoint_triples.add((i, j, k))

    return true_triangles, pattern_counts, disjoint_triples


def find_disjoint_triangles_via_faces(sensors, R, skel):
    """The efficient route: enumerate N's size-six faces directly
    (bounded by N's own clique structure), then for each one find
    which N-edges lie inside it and check for perfect matchings of
    size three covering all six vertices."""
    edges = skel["edges"]
    oracle = NerveOracle(sensors, R, skel)
    edge_index = {tuple(sorted(e)): idx for idx, e in enumerate(edges)}

    G = nx.Graph()
    G.add_nodes_from(range(skel["N"]))
    G.add_edges_from(edges)
    cliques = list(nx.find_cliques(G))

    faces6 = set()
    for c in cliques:
        c = sorted(c)
        if len(c) < 6:
            continue
        for sub in itertools.combinations(c, 6):
            if sub in faces6:
                continue
            if oracle.is_face(sub):
                faces6.add(sub)

    found = set()
    for face in faces6:
        face_set = set(face)
        local_edges = [e for e in edges
                        if e[0] in face_set and e[1] in face_set]
        for triple in itertools.combinations(local_edges, 3):
            covered = set()
            ok = True
            for e in triple:
                if e[0] in covered or e[1] in covered:
                    ok = False
                    break
                covered.add(e[0])
                covered.add(e[1])
            if ok and covered == face_set:
                idxs = tuple(sorted(edge_index[tuple(sorted(e))]
                                      for e in triple))
                found.add(idxs)

    return found, len(faces6)


def main():
    sensors = np.load("fields/A_ring_grid_seed3/sensors.npy")
    meta = json.load(open("fields/A_ring_grid_seed3/meta.json"))
    R = meta["R"]
    skel = build_nerve_2skeleton(sensors, R)

    print("Classifying all true meta-triangles of Field A by "
          "vertex-sharing pattern (brute-force ground truth)...")
    true_triangles, pattern_counts, true_disjoint = \
        classify_true_triangles(sensors, R, skel)

    print(f"\nTotal meta-triangles: {len(true_triangles)}")
    print("Breakdown by number of edge-pairs sharing a vertex:")
    for pattern in (3, 2, 1, 0):
        count = pattern_counts.get(pattern, 0)
        pct = 100 * count / len(true_triangles)
        label = {3: "all three pairs share a vertex",
                  2: "exactly two pairs share a vertex",
                  1: "exactly one pair shares a vertex",
                  0: "no pair shares a vertex (fully disjoint)"}[pattern]
        print(f"  pattern {pattern}: {count:6d} ({pct:5.1f}%)  {label}")

    print()
    print("Finding the disjoint-edge case via the efficient route "
          "(N's own size-six faces)...")
    found_disjoint, n_faces6 = find_disjoint_triangles_via_faces(
        sensors, R, skel)

    print(f"Size-six faces of N found: {n_faces6}")
    print(f"Disjoint-edge meta-triangles found this way: "
          f"{len(found_disjoint)}")
    print(f"Disjoint-edge meta-triangles in ground truth: "
          f"{len(true_disjoint)}")
    print(f"Exact match: {found_disjoint == true_disjoint}")
    print()
    print(f"This case accounts for {len(true_disjoint)} of "
          f"{len(true_triangles)} total meta-triangles on Field A "
          f"({100*len(true_disjoint)/len(true_triangles):.1f}%). "
          f"The remaining patterns, where at least one pair of edges "
          f"shares a vertex, were not found to have an equivalent "
          f"efficient search in this investigation.")


if __name__ == "__main__":
    main()
