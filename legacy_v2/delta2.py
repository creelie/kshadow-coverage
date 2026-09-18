"""
Build Delta_2(N): vertices = edges (1-faces) of N; a set of edges spans a
face of Delta_2(N) iff the union of their endpoints is itself a face of N
(checked via the same Helly shortcut used to build N, since N's "all_faces"
set is already closed under taking subsets and was itself Helly-verified).

Then compute b0, b1 of Delta_2(N) over GF(2), and compare to the ground
truth from check_multiplicity.py: the number of connected components of
the under-2-covered region {x : multiplicity(x) < 2} inside the test
region.
"""
import json
import itertools
import numpy as np
from exact_geometry import disks_intersect_exact

sensors = np.load("sensors.npy")
data = json.load(open("complex.json"))
R = data["R"]
edges = [tuple(e) for e in data["edges"]]
all_faces = set(tuple(f) for f in data["all_faces"])
triangles = set(tuple(t) for t in data["triangles"])

def is_face(vertex_set):
    vertex_set = tuple(sorted(set(vertex_set)))
    if len(vertex_set) <= 2:
        return True
    if vertex_set in all_faces:
        return True
    if len(vertex_set) == 3:
        return vertex_set in triangles
    # not found in cache: fall back to the exact geometric test directly
    # (no grid sampling involved -- exact smallest-enclosing-circle test)
    return disks_intersect_exact(vertex_set, sensors, R)

M = len(edges)
print(f"Delta_2(N) vertex count (= edges of N): {M}")

# meta-edges: pairs of N-edges whose union is a face of N
meta_edges = []
edge_arr = edges
for i in range(M):
    for j in range(i+1, M):
        union = set(edge_arr[i]) | set(edge_arr[j])
        if is_face(union):
            meta_edges.append((i, j))
print(f"Delta_2(N) 1-simplices (meta-edges): {len(meta_edges)}")

# meta-triangles: triples of N-edges, all 3 pairwise meta-edges present,
# AND the full 3-way union is itself a face of N.
meta_adj = {i: set() for i in range(M)}
for (i, j) in meta_edges:
    meta_adj[i].add(j); meta_adj[j].add(i)

meta_triangles = []
for i in range(M):
    for j in meta_adj[i]:
        if j <= i: continue
        for k in (meta_adj[i] & meta_adj[j]):
            if k > j:
                union = set(edge_arr[i]) | set(edge_arr[j]) | set(edge_arr[k])
                if is_face(union):
                    meta_triangles.append((i, j, k))
print(f"Delta_2(N) 2-simplices (meta-triangles): {len(meta_triangles)}")

json.dump({"meta_edges": meta_edges, "meta_triangles": meta_triangles, "M": M},
          open("delta2.json", "w"))

# ---- homology over GF(2) ----
def gf2_rank(Mat):
    Mat = Mat.copy().astype(np.uint8) % 2
    rows, cols = Mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if Mat[r, col] == 1:
                pivot = r; break
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

d1 = np.zeros((M, len(meta_edges)), dtype=np.uint8)
me_index = {e: k for k, e in enumerate(meta_edges)}
for k, (i, j) in enumerate(meta_edges):
    d1[i, k] = 1
    d1[j, k] = 1

def ekey(a, b):
    return (a, b) if a < b else (b, a)

d2 = np.zeros((len(meta_edges), len(meta_triangles)), dtype=np.uint8)
for t_idx, (i, j, k) in enumerate(meta_triangles):
    for (a, b) in itertools.combinations((i, j, k), 2):
        e = ekey(a, b)
        d2[me_index[e], t_idx] = 1

rank_d1 = gf2_rank(d1) if d1.size else 0
rank_d2 = gf2_rank(d2) if d2.size else 0

b0 = M - rank_d1
b1 = len(meta_edges) - rank_d1 - rank_d2

print(f"\nrank d1 = {rank_d1}, rank d2 = {rank_d2}")
print(f"Delta_2(N):  b0 = {b0}   b1 = {b1}")
print(f"\nb1 = {b1} should match the number of independent under-2-covered "
      f"components found by direct grid sampling.")

json.dump({"b0": int(b0), "b1": int(b1)}, open("delta2_betti.json", "w"))
