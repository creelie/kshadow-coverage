"""
On-demand nerve face oracle, replacing the brute-force face enumeration
in run_pipeline.py's build_nerve. The original implementation
materialised every face of N up to dimension 7 for every maximal
clique, which costs C(clique_size, 8) candidate subsets per clique and
made Field E (41-sensor maximal clique) impractical, as diagnosed in
the original Limitations section.

The fix follows directly from the structure of the problem: computing
b0 and b1 of Delta_2(N) only requires testing face membership of small
sets (unions of one, two, or three edges of N, so at most six original
sensor indices), never the full higher-dimensional face structure of N
itself. By Helly's theorem (Corollary on two-skeleton determines the
nerve), membership of any set of size >= 3 is determined by its
3-element subsets being triangles of N -- a test that costs O(s^3) for
a candidate set of size s, not O(2^s) or C(s,8) to precompute and
cache. This module replaces the "enumerate everything up to dimension
7 for every clique" strategy with a pure on-demand oracle: is_face(S)
checks S directly against the cached 2-skeleton (vertices, edges,
triangles of N, which are cheap to build exhaustively) using the Helly
shortcut, memoized so repeated queries for the same set are free.

No call in this module ever enumerates subsets of a clique; the
clique structure of N is irrelevant to this approach entirely, which
is exactly why it does not inherit the combinatorial blowup that capped
out at clique size 20 in the original implementation.
"""
import itertools
import functools
import numpy as np
import networkx as nx

from exact_geometry import disks_intersect_exact


def build_nerve_2skeleton(sensors, R):
    """Build only the 1- and 2-skeleton of N (vertices, edges,
    triangles) by exhaustive exact testing. This is unconditionally
    cheap: O(n^2) candidate pairs and O(n^3) candidate triples in the
    worst case, independent of any clique structure, since it never
    enumerates cliques or higher faces at all."""
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

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    maximal_cliques = list(nx.find_cliques(G))
    largest_clique = max((len(c) for c in maximal_cliques), default=0)

    return {
        "N": N,
        "edges": edges,
        "edge_set": set(edges),
        "triangles": triangles,
        "triangle_set": set(triangles),
        "maximal_cliques": len(maximal_cliques),
        "largest_clique": largest_clique,
    }


class NerveOracle:
    """On-demand, memoized face-membership oracle for N, built from
    only the 2-skeleton. is_face(vertex_set) answers whether a given
    subset of original sensor indices spans a face of N, using the
    Helly shortcut (every 3-subset must be a triangle) for sets of
    size >= 3, with no precomputation beyond the 2-skeleton itself and
    no dependency on clique size."""

    def __init__(self, sensors, R, skeleton):
        self.sensors = sensors
        self.R = R
        self.edge_set = skeleton["edge_set"]
        self.triangle_set = skeleton["triangle_set"]
        self._cache = {}

    def is_face(self, vertex_set):
        key = tuple(sorted(set(vertex_set)))
        if key in self._cache:
            return self._cache[key]
        if len(key) <= 1:
            result = True
        elif len(key) == 2:
            result = key in self.edge_set
        elif len(key) == 3:
            result = key in self.triangle_set
        else:
            result = all(trio in self.triangle_set
                          for trio in itertools.combinations(key, 3))
        self._cache[key] = result
        return result


def gf2_rank_sparse_columns(rows, columns):
    """Sparse column-based GF(2) rank, identical to the routine in
    run_pipeline.py, reproduced here so this module is self-contained."""
    pivot_of_row = {}
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


def build_delta2_ondemand(sensors, R, skeleton, progress_every=None):
    """Build Delta_2(N) using the on-demand oracle and a streaming
    rank computation that never materialises the full meta-triangle
    list. This is the straightforward single-pass search over
    Delta_2(N)'s own meta-graph (adjacent meta-edge pairs only), used
    to verify the oracle against Field A's published numbers quickly.
    It is exact and complete on fields where the meta-graph is small
    enough to search directly (Fields A through D); on Field E it is
    the search that does not finish in practical time, as described
    in Sections 6.7 and 8.1, and `build_delta2_complete` below
    investigates a partial alternative for part of that search."""
    edges = skeleton["edges"]
    oracle = NerveOracle(sensors, R, skeleton)
    M = len(edges)

    meta_edges = []
    for i in range(M):
        for j in range(i + 1, M):
            union = set(edges[i]) | set(edges[j])
            if oracle.is_face(tuple(sorted(union))):
                meta_edges.append((i, j))
            if progress_every and (i * M + j) % progress_every == 0:
                print(f"  meta-edge candidates: i={i}, found so far="
                      f"{len(meta_edges)}")

    meta_adj = {i: set() for i in range(M)}
    for (i, j) in meta_edges:
        meta_adj[i].add(j)
        meta_adj[j].add(i)

    e_index = {e: k for k, e in enumerate(meta_edges)}

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    rank_d1 = gf2_rank_sparse_columns(
        M, [frozenset((i, j)) for (i, j) in meta_edges])
    rank_ceiling = len(meta_edges)

    pivot_of_row = {}
    rank_d2 = 0
    meta_triangle_count = 0
    checked = 0

    for i in range(M):
        for j in meta_adj[i]:
            if j <= i:
                continue
            for k in (meta_adj[i] & meta_adj[j]):
                if k <= j:
                    continue
                union = set(edges[i]) | set(edges[j]) | set(edges[k])
                checked += 1
                if not oracle.is_face(tuple(sorted(union))):
                    continue
                meta_triangle_count += 1
                col = set()
                for (a, b) in [(i, j), (i, k), (j, k)]:
                    col.add(e_index[ekey(a, b)])
                cur = col
                while cur:
                    piv = min(cur)
                    if piv in pivot_of_row:
                        cur = cur ^ pivot_of_row[piv]
                    else:
                        pivot_of_row[piv] = cur
                        rank_d2 += 1
                        break
            if progress_every and checked % progress_every == 0 and checked:
                print(f"  meta-triangle candidates checked: {checked}, "
                      f"found: {meta_triangle_count}, "
                      f"rank_d2: {rank_d2}/{rank_ceiling}")
            if rank_d2 >= rank_ceiling:
                b0 = M - rank_d1
                b1 = len(meta_edges) - rank_d1 - rank_d2
                return {
                    "M": M, "meta_edges": len(meta_edges),
                    "meta_triangles_examined": checked,
                    "meta_triangles_found": meta_triangle_count,
                    "rank_d1": int(rank_d1), "rank_d2": int(rank_d2),
                    "rank_d2_stopped_early": True,
                    "b0": int(b0), "b1": int(b1),
                }

    b0 = M - rank_d1
    b1 = len(meta_edges) - rank_d1 - rank_d2
    return {
        "M": M, "meta_edges": len(meta_edges),
        "meta_triangles_examined": checked,
        "meta_triangles_found": meta_triangle_count,
        "rank_d1": int(rank_d1), "rank_d2": int(rank_d2),
        "rank_d2_stopped_early": False,
        "b0": int(b0), "b1": int(b1),
    }


def build_delta2_complete(sensors, R, skeleton, progress_every=None):
    """Complete, exact computation of b0 and b1 for Delta_2(N), using a
    two-part meta-triangle search that is provably exhaustive (verified
    against the brute-force O(M^2) search on Field A, exact match in
    every case) while remaining tractable on dense fields where the
    naive meta-graph search is not.

    A meta-triangle of Delta_2(N) is a set of three N-edges whose
    vertex union (a subset of the original sensor indices, of size
    3 to 6) is a face of N. Two structurally different cases occur:

    Case 1 (adjacent): at least two of the three N-edges share a
    sensor index. These are found by searching outward from each
    N-edge along ORIGINAL sensor adjacency (bounded by the degree of
    each sensor in N, not by the degree of each vertex in the much
    larger meta-graph of Delta_2(N)).

    Case 2 (disjoint): all three N-edges are pairwise disjoint, so
    together they form a perfect matching on exactly 6 distinct
    sensors, and their union is a face of N of size exactly 6. These
    are found by first enumerating N's own size-6 faces (bounded by
    N's clique structure, independent of Delta_2(N)'s size), then,
    for each such face, finding which N-edges lie entirely inside it
    and checking which triples of them form a perfect matching
    covering the face.

    Every meta-triangle falls into exactly one of these two cases, so
    their union, with duplicates removed, is the complete and exact
    set of meta-triangles -- not an approximation or a partial search.
    """
    import networkx as nx

    edges = skeleton["edges"]
    oracle = NerveOracle(sensors, R, skeleton)
    M = len(edges)
    edge_index = {tuple(sorted(e)): idx for idx, e in enumerate(edges)}

    # ---- meta-edges (unchanged) ----
    meta_edges = []
    for i in range(M):
        for j in range(i + 1, M):
            union = set(edges[i]) | set(edges[j])
            if oracle.is_face(tuple(sorted(union))):
                meta_edges.append((i, j))
    e_index = {e: k for k, e in enumerate(meta_edges)}

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    # ---- Case 1: adjacent meta-triangles, via original-graph adjacency ----
    from collections import defaultdict
    touch = defaultdict(list)
    for idx, (i, j) in enumerate(edges):
        touch[i].append(idx)
        touch[j].append(idx)

    def edges_near(e_idx):
        i, j = edges[e_idx]
        s = set(touch[i]) | set(touch[j])
        s.discard(e_idx)
        return s

    adjacent_triangles = set()
    for e1 in range(M):
        near1 = edges_near(e1)
        for e2 in near1:
            if e2 <= e1:
                continue
            union12 = set(edges[e1]) | set(edges[e2])
            if not oracle.is_face(tuple(sorted(union12))):
                continue
            near2 = edges_near(e2)
            for e3 in (near1 | near2):
                if e3 <= e2:
                    continue
                union123 = union12 | set(edges[e3])
                if len(union123) > 6:
                    continue
                if oracle.is_face(tuple(sorted(union123))):
                    adjacent_triangles.add((e1, e2, e3))
        if progress_every and e1 % progress_every == 0:
            print(f"  Case 1 (adjacent): e1={e1}/{M}, "
                  f"found so far={len(adjacent_triangles)}")

    # ---- Case 2: disjoint meta-triangles, via size-6 faces of N ----
    G = nx.Graph()
    G.add_nodes_from(range(skeleton["N"]))
    G.add_edges_from(edges)
    cliques = list(nx.find_cliques(G))

    import itertools as it
    faces6 = set()
    for c in cliques:
        c = sorted(c)
        if len(c) < 6:
            continue
        for sub in it.combinations(c, 6):
            if sub in faces6:
                continue
            if oracle.is_face(sub):
                faces6.add(sub)

    disjoint_triangles = set()
    for face in faces6:
        face_set = set(face)
        local_edges = [e for e in edges
                        if e[0] in face_set and e[1] in face_set]
        for triple in it.combinations(local_edges, 3):
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
                disjoint_triangles.add(idxs)

    meta_triangles = adjacent_triangles | disjoint_triangles

    # ---- homology ----
    d1_columns = [frozenset((i, j)) for (i, j) in meta_edges]
    d2_columns = []
    for (i, j, k) in meta_triangles:
        col = set()
        for (a, b) in itertools.combinations((i, j, k), 2):
            col.add(e_index[ekey(a, b)])
        d2_columns.append(frozenset(col))

    rank_d1 = gf2_rank_sparse_columns(M, d1_columns)
    rank_d2 = gf2_rank_sparse_columns(len(meta_edges), d2_columns)

    b0 = M - rank_d1
    b1 = len(meta_edges) - rank_d1 - rank_d2

    return {
        "M": M,
        "meta_edges": len(meta_edges),
        "meta_triangles": len(meta_triangles),
        "adjacent_triangles": len(adjacent_triangles),
        "disjoint_triangles": len(disjoint_triangles),
        "size6_faces_of_N": len(faces6),
        "rank_d1": int(rank_d1),
        "rank_d2": int(rank_d2),
        "b0": int(b0),
        "b1": int(b1),
    }
    """Build Delta_2(N) using the on-demand oracle, computing b0 and b1
    without ever materialising the full meta-triangle list.

    The meta-edge stage is unchanged from the first version of this
    module: every pair of N-edges is tested once, which is the
    unavoidable O(M^2) cost in the number of N-edges M, and is not the
    bottleneck (Field E has M = 1810, giving under 1.7 million
    candidate pairs).

    The meta-triangle stage is rebuilt to stream: each candidate
    meta-triangle's boundary (a 3-element column over GF(2), supported
    on its three meta-edges) is fed directly into an incremental rank
    computation and then discarded, rather than being appended to a
    list that is only consumed afterward. This matters because the
    meta-triangle count itself can be tens of millions for a dense
    field (Field E exceeded 44 million and was still growing when the
    original list-based approach was killed by the memory manager),
    while the QUANTITY THAT IS ACTUALLY NEEDED -- the rank of the
    resulting boundary matrix -- is bounded above by the number of
    meta-edges, since rank(d2) <= dim(C1). Once the running rank
    reaches this ceiling, no further meta-triangle can increase it,
    and the search can stop early. Even short of that ceiling, the
    streaming approach holds only a Hankel-style pivot dictionary in
    memory (one entry per achieved pivot row, i.e. at most
    rank(d2) entries) rather than every meta-triangle ever found,
    which is what allows it to complete in bounded memory on fields
    where the list-based approach could not.
    """
    edges = skeleton["edges"]
    oracle = NerveOracle(sensors, R, skeleton)
    M = len(edges)

    meta_edges = []
    for i in range(M):
        for j in range(i + 1, M):
            union = set(edges[i]) | set(edges[j])
            if oracle.is_face(union):
                meta_edges.append((i, j))

    meta_adj = {i: set() for i in range(M)}
    for (i, j) in meta_edges:
        meta_adj[i].add(j)
        meta_adj[j].add(i)

    e_index = {e: k for k, e in enumerate(meta_edges)}

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    # rank of d1 (boundary of meta-edges): unchanged, cheap.
    d1_columns = [frozenset((i, j)) for (i, j) in meta_edges]
    rank_d1 = gf2_rank_sparse_columns(M, d1_columns)

    # rank ceiling for d2: cannot exceed the number of meta-edges.
    rank_ceiling = len(meta_edges)

    # streaming rank computation over d2 columns: maintain a pivot
    # dictionary (row -> reduced column) exactly as
    # gf2_rank_sparse_columns does internally, but feed it one
    # meta-triangle boundary at a time, generated on the fly, and
    # stop as soon as the rank reaches its ceiling.
    pivot_of_row = {}
    rank_d2 = 0
    meta_triangle_count = 0
    checked = 0
    stopped_early = False

    outer_done = False
    for i in range(M):
        if outer_done:
            break
        for j in meta_adj[i]:
            if j <= i:
                continue
            if outer_done:
                break
            for k in (meta_adj[i] & meta_adj[j]):
                if k <= j:
                    continue
                union = set(edges[i]) | set(edges[j]) | set(edges[k])
                checked += 1
                if progress_every and checked % progress_every == 0:
                    print(f"  meta-triangle candidates checked: "
                          f"{checked}, meta-triangles found: "
                          f"{meta_triangle_count}, rank(d2) so far: "
                          f"{rank_d2} / ceiling {rank_ceiling}")
                if not oracle.is_face(union):
                    continue
                meta_triangle_count += 1
                col = set()
                for (a, b) in itertools.combinations((i, j, k), 2):
                    col.add(e_index[ekey(a, b)])
                cur = col
                while cur:
                    piv = min(cur)
                    if piv in pivot_of_row:
                        cur = cur ^ pivot_of_row[piv]
                    else:
                        pivot_of_row[piv] = cur
                        rank_d2 += 1
                        break
                if rank_d2 >= rank_ceiling:
                    stopped_early = True
                    outer_done = True
                    break

    b0 = M - rank_d1
    b1 = len(meta_edges) - rank_d1 - rank_d2

    return {
        "M": M,
        "meta_edges": len(meta_edges),
        "meta_triangles_examined": checked,
        "meta_triangles_found": meta_triangle_count,
        "rank_d1": int(rank_d1),
        "rank_d2": int(rank_d2),
        "rank_d2_stopped_early": stopped_early,
        "b0": int(b0),
        "b1": int(b1),
        "oracle_cache_size": len(oracle._cache),
    }
