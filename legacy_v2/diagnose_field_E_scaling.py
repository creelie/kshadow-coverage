"""
Diagnostic script for Field E (dense cluster + sparse halo).

Reproduces the analysis behind Sections 6.7 and 8.1 of the manuscript
in full, in three parts:

  Part 1: the original obstruction. An initial implementation that
  enumerates, for each maximal clique of N, every face up to dimension
  7, has cost growing as C(clique_size, 8); Field E's 41-sensor clique
  alone makes this approximately 9.55 x 10^7 candidate subsets, which
  did not complete.

  Part 2: the fix. Replacing clique-based face enumeration with an
  on-demand oracle (ondemand_pipeline.py) that tests face membership
  directly against N's 2-skeleton removes this obstruction entirely.
  This part verifies the oracle reproduces Field A's published numbers
  exactly and resolves Field E's nerve construction in about a second.

  Part 3: the deeper finding the fix revealed. With the clique
  obstruction removed, Delta_2(N) for Field E still does not complete,
  because Delta_2(N) itself has on the order of 837,000 vertices, a
  genuine property of the object rather than an artifact of how it was
  computed. This part counts Field E's exact meta-edge total and
  reports the resulting search-space estimate.

Usage:
    python3 diagnose_field_E_scaling.py
"""
import itertools
import json
import resource
import time
from math import comb

import numpy as np
import networkx as nx

from exact_geometry import disks_intersect_exact
from ondemand_pipeline import (build_nerve_2skeleton, build_delta2_ondemand,
                                NerveOracle)


def part1_original_obstruction():
    print("=" * 70)
    print("PART 1: the original obstruction (clique-based enumeration)")
    print("=" * 70)
    sensors = np.load("fields/E_two_cluster/sensors.npy")
    meta = json.load(open("fields/E_two_cluster/meta.json"))
    R = meta["R"]
    N = len(sensors)

    print(f"Field E: n={N} sensors, R={R}")
    print(f"  {meta['description']}")
    print()

    t0 = time.time()
    pairs = [(i, j) for i, j in itertools.combinations(range(N), 2)
             if np.hypot(*(sensors[i] - sensors[j])) <= 2 * R]
    edges = [e for e in pairs if disks_intersect_exact(e, sensors, R)]
    print(f"Edges: {len(edges)} (of {len(pairs)} candidate pairs), "
          f"{time.time()-t0:.2f}s")

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    maximal_cliques = list(nx.find_cliques(G))
    sizes = sorted((len(c) for c in maximal_cliques), reverse=True)

    print(f"Maximal cliques: {len(maximal_cliques)}")
    print(f"Largest clique: {sizes[0]} sensors")
    print()

    largest = sizes[0]
    n_subsets = comb(largest, 8)
    print(f"Combinatorial cost of the original clique-based enumeration "
          f"strategy for the largest clique:")
    print(f"  C({largest}, 8) = {n_subsets:,} candidate subsets from "
          f"this one clique alone, before the other "
          f"{len(maximal_cliques)-1} maximal cliques are even touched.")
    print(f"This is the obstruction reported in earlier diagnosis: "
          f"the strategy never completed.")
    print()
    return sizes[0]


def part2_the_fix():
    print("=" * 70)
    print("PART 2: the fix (on-demand oracle, no clique enumeration)")
    print("=" * 70)

    # Verify the fix reproduces Field A exactly first.
    sensors = np.load("fields/A_ring_grid_seed3/sensors.npy")
    meta = json.load(open("fields/A_ring_grid_seed3/meta.json"))
    R = meta["R"]
    t0 = time.time()
    skel = build_nerve_2skeleton(sensors, R)
    result = build_delta2_ondemand(sensors, R, skel, progress_every=None)
    elapsed = time.time() - t0
    print(f"Field A verification (must match published values exactly):")
    print(f"  M={result['M']}, meta_edges={result['meta_edges']}, "
          f"rank_d1={result['rank_d1']}, rank_d2={result['rank_d2']}, "
          f"b0={result['b0']}, b1={result['b1']}")
    print(f"  expected: M=493, meta_edges=5124, rank_d1=492, "
          f"rank_d2=4624, b0=1, b1=8")
    assert (result["M"], result["meta_edges"], result["rank_d1"],
            result["rank_d2"], result["b0"], result["b1"]) == \
           (493, 5124, 492, 4624, 1, 8), "Oracle does not match Field A!"
    print(f"  MATCH CONFIRMED. Time: {elapsed:.2f}s")
    print()

    # Now show the clique obstruction is gone for Field E's nerve.
    sensors = np.load("fields/E_two_cluster/sensors.npy")
    meta = json.load(open("fields/E_two_cluster/meta.json"))
    R = meta["R"]
    t0 = time.time()
    skel = build_nerve_2skeleton(sensors, R)
    elapsed = time.time() - t0
    print(f"Field E nerve 2-skeleton via the oracle approach:")
    print(f"  {len(skel['edges'])} edges, {len(skel['triangles'])} "
          f"triangles, largest clique {skel['largest_clique']} "
          f"(not used in this construction)")
    print(f"  Time: {elapsed:.2f}s -- no dependence on clique size.")
    print()
    return skel


def part3_the_deeper_finding(skel):
    print("=" * 70)
    print("PART 3: what the fix revealed (the true size of Delta_2(N))")
    print("=" * 70)

    sensors = np.load("fields/E_two_cluster/sensors.npy")
    meta = json.load(open("fields/E_two_cluster/meta.json"))
    R = meta["R"]
    edges = skel["edges"]
    M = len(edges)
    oracle = NerveOracle(sensors, R, skel)

    print(f"Counting Field E's exact meta-edge total "
          f"(vertices of Delta_2(N))...")
    t0 = time.time()
    meta_edge_count = 0
    for i in range(M):
        for j in range(i + 1, M):
            union = set(edges[i]) | set(edges[j])
            if oracle.is_face(union):
                meta_edge_count += 1
    elapsed = time.time() - t0
    total_candidates = M * (M - 1) // 2

    print(f"  Exact meta-edges: {meta_edge_count:,} out of "
          f"{total_candidates:,} candidate pairs "
          f"({100*meta_edge_count/total_candidates:.1f}%)")
    print(f"  Time: {elapsed:.2f}s")
    print()
    print(f"Compare: Field A's Delta_2(N) has 493 vertices. "
          f"Field E's has {meta_edge_count:,} -- a factor of "
          f"{meta_edge_count/493:.0f}x larger.")
    print()
    print("This is an intrinsic property of Delta_2(N) for a densely")
    print("clustered field, not a consequence of how it is computed:")
    print("over half of all candidate N-edge pairs union into a face")
    print("of N, because the dense cluster makes most pairs of edges")
    print("drawn from it mutually close enough to share a common point.")


if __name__ == "__main__":
    largest_clique = part1_original_obstruction()
    skel = part2_the_fix()
    part3_the_deeper_finding(skel)
