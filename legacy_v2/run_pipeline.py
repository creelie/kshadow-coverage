"""
Run the full corrected pipeline (nerve -> Delta_1 check -> Delta_2 ->
independent ground truth) on a single field directory produced by
generate_fields.py. This is the same logic as build_complex2.py,
check_k1.py, delta2.py, and ground_truth_clean.py in the v1 archive,
refactored to take a field directory as an argument so it can be run
unchanged across all five v2 validation fields without duplicating
the exact-geometry test or the GF(2) homology routine.

Usage:
    python3 run_pipeline.py fields/A_ring_grid_seed3
"""
import sys
import os
import json
import itertools
import numpy as np
import networkx as nx
from scipy import ndimage

from exact_geometry import disks_intersect_exact


def build_nerve(sensors, R, max_clique_for_full_enum=20):
    """Build the nerve N. For maximal cliques up to size
    max_clique_for_full_enum, every face up to dimension 7 (8-element
    subsets) is enumerated and Helly-verified, exactly as in the v1
    pipeline. Cliques LARGER than this threshold are reported but their
    higher faces (dimension >= 7) are not exhaustively enumerated, since
    the subset count grows as C(clique_size, 8) and becomes impractical
    well before clique_size = 41 (the largest clique found in Field E:
    C(41,8) is approximately 9.5 x 10^7 candidate subsets from that one
    clique alone). This is a genuine, reported scaling limitation of the
    brute-force face-enumeration strategy used here, not a limitation of
    Theorem 3 itself, which places no bound on simplex dimension. Fields
    whose nerve contains a clique above this threshold are flagged in
    the returned dict via 'oversized_cliques' rather than silently
    truncated or allowed to exhaust memory."""
    N = len(sensors)
    pairs = [(i, j) for i, j in itertools.combinations(range(N), 2)
             if np.hypot(*(sensors[i] - sensors[j])) <= 2 * R]
    edges = [e for e in pairs if disks_intersect_exact(e, sensors, R)]

    adj = {i: set() for i in range(N)}
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)

    triangle_candidates = []
    for i in range(N):
        for j in adj[i]:
            if j <= i:
                continue
            for k in (adj[i] & adj[j]):
                if k > j:
                    triangle_candidates.append((i, j, k))
    triangles = [t for t in triangle_candidates
                 if disks_intersect_exact(t, sensors, R)]
    triangle_set = set(triangles)

    def is_face_by_helly(vertex_set):
        for trio in itertools.combinations(sorted(vertex_set), 3):
            if trio not in triangle_set:
                return False
        return True

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    maximal_cliques = list(nx.find_cliques(G))

    oversized = [c for c in maximal_cliques
                 if len(c) > max_clique_for_full_enum]
    safe_cliques = [c for c in maximal_cliques
                     if len(c) <= max_clique_for_full_enum]

    all_faces = set()
    for c in safe_cliques:
        c = sorted(c)
        cap = min(len(c), 8)
        for r in range(1, cap + 1):
            for sub in itertools.combinations(c, r):
                if r <= 2:
                    all_faces.add(sub)
                elif r == 3:
                    if sub in triangle_set:
                        all_faces.add(sub)
                else:
                    if is_face_by_helly(sub):
                        all_faces.add(sub)
    # for oversized cliques, still record their vertices, edges, and
    # triangles (already captured above, since those come from the
    # exhaustive pairwise/triple-wise scan over ALL sensors, not from
    # the per-clique subset enumeration), so Delta_1(N) statistics for
    # dimension <= 2 remain complete and exact even when higher-face
    # enumeration is capped.

    faces_by_dim = {}
    for f in all_faces:
        faces_by_dim.setdefault(len(f), []).append(f)

    import random
    random.seed(0)
    size4 = faces_by_dim.get(4, [])
    sample = random.sample(size4, min(30, len(size4)))
    bad = sum(1 for f in sample if not disks_intersect_exact(f, sensors, R))

    largest_clique = max((len(c) for c in maximal_cliques), default=0)

    return {
        "N": N,
        "edges": edges,
        "triangles": triangles,
        "all_faces": all_faces,
        "faces_by_dim": {d: len(v) for d, v in faces_by_dim.items()},
        "maximal_cliques": len(maximal_cliques),
        "largest_clique": largest_clique,
        "oversized_cliques": [len(c) for c in oversized],
        "crosscheck_sample": len(sample),
        "crosscheck_failures": bad,
    }


def gf2_rank_sparse_columns(rows, columns):
    """Rank over GF(2) of a matrix given as a list of columns, each column
    a set of row indices with a 1 entry (all other entries 0). This is
    the natural sparse representation for simplicial boundary matrices,
    where d1 has exactly 2 nonzeros per column and d2 has exactly 3, no
    matter how large the complex is. Standard sparse column-reduction:
    maintain a pivot row for each column processed so far, encoded
    as a dict from pivot row -> the (sparse, frozenset) column that
    produced it; reduce each new column against existing pivots using
    symmetric difference (GF(2) addition), exactly as dense row-reduction
    does, but touching only nonzero entries.

    This never allocates a rows x cols dense array, so the memory cost
    is proportional to the number of nonzero entries actually present,
    not to rows x cols. For d1 (2 nonzeros/column) and d2 (3 nonzeros/
    column) this is the difference between a megabyte-scale computation
    and the multi-gigabyte dense array that caused the out-of-memory
    failure on fields C and E described in the main text.
    """
    pivot_of_row = {}  # row index -> column (as frozenset) with that pivot
    rank = 0
    for col in columns:
        cur = set(col)
        while cur:
            piv = min(cur)
            if piv in pivot_of_row:
                cur ^= pivot_of_row[piv]
            else:
                pivot_of_row[piv] = cur
                rank += 1
                break
    return rank


def gf2_rank(Mat):
    """Dense fallback, kept only for the small case (k=1, i.e. N and
    Delta_1(N) itself) where the original v1 dense routine is fast and
    the matrices are small enough that sparsity gives no benefit. Not
    used for Delta_2(N) onward; see gf2_rank_sparse_columns."""
    Mat = Mat.copy().astype(np.uint8) % 2
    rows, cols = Mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if Mat[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        Mat[[rank, pivot]] = Mat[[pivot, rank]]
        for r in range(rows):
            if r != rank and Mat[r, col] == 1:
                Mat[r] = (Mat[r] + Mat[rank]) % 2
        rank += 1
        if rank == rows:
            break
    return rank


def betti_of_complex(M_vertices, edges_1simplex, triangles_2simplex):
    """Compute b0, b1 of a complex given as (vertex count, list of
    1-simplices as index pairs, list of 2-simplices as index triples),
    using the sparse-column GF(2) rank routine throughout. This is
    mathematically identical to the dense v1 routine (both compute the
    rank of the same boundary matrices over the same field) but does not
    require forming a dense M_vertices x len(edges) or len(edges) x
    len(triangles) array, which is what made the original dense approach
    fail on fields with a denser nerve (Fields C and E; see main text)."""
    e_index = {e: k for k, e in enumerate(edges_1simplex)}

    d1_columns = [frozenset((i, j)) for (i, j) in edges_1simplex]

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    d2_columns = []
    for (i, j, k) in triangles_2simplex:
        col = set()
        for (a, b) in itertools.combinations((i, j, k), 2):
            col.add(e_index[ekey(a, b)])
        d2_columns.append(frozenset(col))

    rank_d1 = gf2_rank_sparse_columns(M_vertices, d1_columns)
    rank_d2 = gf2_rank_sparse_columns(len(edges_1simplex), d2_columns)

    b0 = M_vertices - rank_d1
    b1 = len(edges_1simplex) - rank_d1 - rank_d2
    return int(b0), int(b1), int(rank_d1), int(rank_d2)


def build_delta2(sensors, R, edges, all_faces, triangles):
    triangles_set = set(triangles)

    def is_face(vertex_set):
        vertex_set = tuple(sorted(set(vertex_set)))
        if len(vertex_set) <= 2:
            return True
        if vertex_set in all_faces:
            return True
        if len(vertex_set) == 3:
            return vertex_set in triangles_set
        return disks_intersect_exact(vertex_set, sensors, R)

    M = len(edges)
    meta_edges = []
    for i in range(M):
        for j in range(i + 1, M):
            union = set(edges[i]) | set(edges[j])
            if is_face(union):
                meta_edges.append((i, j))

    meta_adj = {i: set() for i in range(M)}
    for (i, j) in meta_edges:
        meta_adj[i].add(j)
        meta_adj[j].add(i)

    meta_triangles = []
    for i in range(M):
        for j in meta_adj[i]:
            if j <= i:
                continue
            for k in (meta_adj[i] & meta_adj[j]):
                if k > j:
                    union = set(edges[i]) | set(edges[j]) | set(edges[k])
                    if is_face(union):
                        meta_triangles.append((i, j, k))

    b0, b1, r1, r2 = betti_of_complex(M, meta_edges, meta_triangles)
    return {
        "M": M,
        "meta_edges": len(meta_edges),
        "meta_triangles": len(meta_triangles),
        "rank_d1": r1,
        "rank_d2": r2,
        "b0": b0,
        "b1": b1,
    }


def ground_truth_at_spacing(sensors, R, grid_spacing, k_values,
                             window=None, pad=3.0):
    """Sample the multiplicity field on a window.

    IMPORTANT: the v1 script (ground_truth_clean.py) hardcoded the
    sampling window to xs = linspace(-3, 15, 700), i.e. pad=3 around a
    DESIGN extent of [0, 12] for that specific field's fence geometry
    (centre (6,6), outer fence radius 7.4). That fixed window does not
    coincide with pad=3 applied to the field's ACTUAL sensor coordinate
    extent, which is [-1.4, 13.4] for that field (the fence ring points
    extend slightly past the nominal [0,12] box). The two windows differ
    by about 1.4 units on each side, which is enough to change which
    side of a tiny marginal hole near the field boundary gets counted as
    enclosed vs. border-touching -- the same class of resolution
    sensitivity the paper's Section 6.1 discusses for the nerve
    construction itself, now showing up in the independent ground-truth
    sampler too.

    To keep v2 honest and reproducible, every field below is given an
    explicit design window at generation time (see generate_fields.py),
    and that window is used here directly rather than re-derived from
    the realised sensor coordinates. If no window is supplied, we fall
    back to pad applied to the realised sensor extent, but this is
    flagged in the result as a fallback so it is never silently
    conflated with a field's canonical window.
    """
    if window is not None:
        xmin, xmax, ymin, ymax = window
        used_fallback = False
    else:
        xmin, xmax = sensors[:, 0].min() - pad, sensors[:, 0].max() + pad
        ymin, ymax = sensors[:, 1].min() - pad, sensors[:, 1].max() + pad
        used_fallback = True
    nx = max(int(round((xmax - xmin) / grid_spacing)), 50)
    ny = max(int(round((ymax - ymin) / grid_spacing)), 50)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    gx, gy = np.meshgrid(xs, ys)

    mult = np.zeros_like(gx, dtype=int)
    for (cx, cy) in sensors:
        mult += ((gx - cx) ** 2 + (gy - cy) ** 2 <= R ** 2).astype(int)

    def region_topology(region):
        lab_fg, n_fg = ndimage.label(region)
        lab_bg, n_bg = ndimage.label(~region)
        border_labels = (set(lab_bg[0, :]) | set(lab_bg[-1, :]) |
                          set(lab_bg[:, 0]) | set(lab_bg[:, -1]))
        border_labels.discard(0)
        n_holes = sum(1 for lbl in range(1, n_bg + 1)
                      if lbl not in border_labels)
        return int(n_fg), int(n_holes)

    out = {}
    for k in k_values:
        region = mult >= k
        b0, b1 = region_topology(region)
        out[k] = {"b0": b0, "b1": b1}
    return out


def ground_truth_with_convergence(sensors, R, k_values=(1, 2, 3),
                                   spacings=(18.0 / 350, 18.0 / 700, 18.0 / 1400),
                                   window=None, pad=3.0):
    """Run the ground-truth sampler at three different absolute grid
    resolutions (half, baseline, and double the v1 resolution) and report
    whether the resulting Betti numbers are stable. A field whose
    raster-based Betti numbers depend on which of these three resolutions
    is used is, by definition, exactly the kind of marginal-overlap case
    Section 6.1 of the paper is about, and is flagged rather than
    silently reported at only one resolution."""
    runs = {}
    for spacing in spacings:
        runs[spacing] = ground_truth_at_spacing(sensors, R, spacing,
                                                  k_values, window=window,
                                                  pad=pad)
    converged = {}
    flags = {}
    for k in k_values:
        vals = [(runs[s][k]["b0"], runs[s][k]["b1"]) for s in spacings]
        stable = len(set(vals)) == 1
        converged[k] = runs[spacings[1]][k]  # baseline (v1-matching) value
        flags[k] = {"stable_across_resolutions": stable,
                    "values_at_each_resolution": vals}
    return converged, flags


def ground_truth(sensors, R, k_values=(1, 2, 3), grid_spacing=18.0 / 700,
                  pad=3.0):
    """Single-resolution convenience wrapper, kept for direct comparability
    with the v1 ground_truth_clean.py at its original effective spacing."""
    return ground_truth_at_spacing(sensors, R, grid_spacing, k_values, pad)


def run_field(field_dir):
    sensors = np.load(os.path.join(field_dir, "sensors.npy"))
    meta = json.load(open(os.path.join(field_dir, "meta.json")))
    R = meta["R"]

    print(f"\n{'='*60}")
    print(f"Field: {field_dir}")
    print(f"  {meta['description']}")
    print(f"  n_sensors = {meta['n_sensors']}, R = {R}")
    print(f"{'='*60}")

    nerve = build_nerve(sensors, R)
    print(f"Nerve N: {nerve['N']} vertices, {len(nerve['edges'])} edges, "
          f"{len(nerve['triangles'])} triangles")
    print(f"  maximal cliques: {nerve['maximal_cliques']}, "
          f"largest = {nerve['largest_clique']}")
    print(f"  faces by dimension: {nerve['faces_by_dim']}")
    print(f"  crosscheck: {nerve['crosscheck_sample']-nerve['crosscheck_failures']}"
          f"/{nerve['crosscheck_sample']} size-4 faces confirmed")
    if nerve["oversized_cliques"]:
        print(f"  WARNING: {len(nerve['oversized_cliques'])} maximal clique(s) "
              f"exceed the full-enumeration threshold "
              f"(sizes: {sorted(nerve['oversized_cliques'], reverse=True)}). "
              f"Faces of dimension >= 7 inside these cliques are NOT "
              f"exhaustively enumerated; b1(N) and b1(Delta_2(N)) below "
              f"are reported on the resulting INCOMPLETE face set and "
              f"are not directly comparable to ground truth.")

    # Delta_1(N) = N betti numbers
    b0_1, b1_1, _, _ = betti_of_complex(
        nerve["N"], nerve["edges"], nerve["triangles"])
    print(f"Delta_1(N) = N:  b0 = {b0_1}, b1 = {b1_1}")

    delta2 = build_delta2(sensors, R, nerve["edges"], nerve["all_faces"],
                           nerve["triangles"])
    print(f"Delta_2(N): {delta2['M']} vertices, {delta2['meta_edges']} "
          f"meta-edges, {delta2['meta_triangles']} meta-triangles")
    print(f"  rank d1 = {delta2['rank_d1']}, rank d2 = {delta2['rank_d2']}")
    print(f"  b0 = {delta2['b0']}, b1 = {delta2['b1']}")

    window = tuple(meta["window"])
    gt, gt_flags = ground_truth_with_convergence(sensors, R, k_values=(1, 2, 3),
                                                  window=window)
    print(f"Ground truth k=1: b0={gt[1]['b0']}, b1={gt[1]['b1']}  "
          f"(stable across resolution: {gt_flags[1]['stable_across_resolutions']})")
    print(f"Ground truth k=2: b0={gt[2]['b0']}, b1={gt[2]['b1']}  "
          f"(stable across resolution: {gt_flags[2]['stable_across_resolutions']})")
    print(f"Ground truth k=3: b0={gt[3]['b0']}, b1={gt[3]['b1']}  "
          f"(stable across resolution: {gt_flags[3]['stable_across_resolutions']})")
    for k in (1, 2, 3):
        if not gt_flags[k]["stable_across_resolutions"]:
            print(f"  NOTE k={k}: values at half/baseline/double resolution = "
                  f"{gt_flags[k]['values_at_each_resolution']}")

    match_k1 = (b0_1, b1_1) == (gt[1]["b0"], gt[1]["b1"])
    match_k2 = (delta2["b0"], delta2["b1"]) == (gt[2]["b0"], gt[2]["b1"])
    print(f"k=1 match: {match_k1}")
    print(f"k=2 match: {match_k2}")

    result = {
        "field_dir": field_dir,
        "meta": meta,
        "nerve": {
            "vertices": nerve["N"],
            "edges": len(nerve["edges"]),
            "triangles": len(nerve["triangles"]),
            "maximal_cliques": nerve["maximal_cliques"],
            "largest_clique": nerve["largest_clique"],
            "oversized_cliques": nerve["oversized_cliques"],
            "faces_by_dim": nerve["faces_by_dim"],
            "crosscheck_sample": nerve["crosscheck_sample"],
            "crosscheck_failures": nerve["crosscheck_failures"],
        },
        "delta1": {"b0": b0_1, "b1": b1_1},
        "delta2": {k: v for k, v in delta2.items()},
        "ground_truth": gt,
        "ground_truth_resolution_stable": {k: gt_flags[k]["stable_across_resolutions"]
                                            for k in (1, 2, 3)},
        "ground_truth_resolution_detail": {k: gt_flags[k]["values_at_each_resolution"]
                                            for k in (1, 2, 3)},
        "match_k1": match_k1,
        "match_k2": match_k2,
    }

    out_path = os.path.join(field_dir, "result.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")
    return result


if __name__ == "__main__":
    field_dir = sys.argv[1] if len(sys.argv) > 1 else "fields/A_ring_grid_seed3"
    run_field(field_dir)
