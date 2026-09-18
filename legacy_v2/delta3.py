import json, itertools, numpy as np, time
from exact_geometry import disks_intersect_exact

sensors = np.load("sensors.npy")
data = json.load(open("complex.json"))
R = data["R"]
triangles = [tuple(t) for t in data["triangles"]]
all_faces = set(tuple(f) for f in data["all_faces"])

def is_face(vs):
    vs = tuple(sorted(set(vs)))
    if len(vs) <= 3:
        return True
    if vs in all_faces:
        return True
    return disks_intersect_exact(vs, sensors, R)

M = len(triangles)
print(f"Delta_3(N) vertex count (= triangles of N): {M}")
t0 = time.time()
meta_edges = []
for i in range(M):
    si = set(triangles[i])
    for j in range(i+1, M):
        union = si | set(triangles[j])
        if is_face(union):
            meta_edges.append((i, j))
print(f"meta-edges: {len(meta_edges)}  ({time.time()-t0:.1f}s)")

meta_adj = {i: set() for i in range(M)}
for (i,j) in meta_edges:
    meta_adj[i].add(j); meta_adj[j].add(i)

t0 = time.time()
meta_tri = []
for i in range(M):
    for j in meta_adj[i]:
        if j <= i: continue
        for k in (meta_adj[i] & meta_adj[j]):
            if k > j:
                union = set(triangles[i]) | set(triangles[j]) | set(triangles[k])
                if is_face(union):
                    meta_tri.append((i,j,k))
print(f"meta-triangles: {len(meta_tri)}  ({time.time()-t0:.1f}s)")

def gf2_rank(Mat):
    Mat = Mat.copy().astype(np.uint8) % 2
    rows, cols = Mat.shape
    rank = 0
    for col in range(cols):
        piv=None
        for r in range(rank, rows):
            if Mat[r,col]==1: piv=r; break
        if piv is None: continue
        Mat[[rank,piv]] = Mat[[piv,rank]]
        for r in range(rows):
            if r!=rank and Mat[r,col]==1:
                Mat[r]=(Mat[r]+Mat[rank])%2
        rank+=1
        if rank==rows: break
    return rank

d1 = np.zeros((M, len(meta_edges)), dtype=np.uint8)
meidx = {e:k for k,e in enumerate(meta_edges)}
for k,(i,j) in enumerate(meta_edges):
    d1[i,k]=1; d1[j,k]=1

d2 = np.zeros((len(meta_edges), len(meta_tri)), dtype=np.uint8)
for t,(i,j,k) in enumerate(meta_tri):
    for a,b in itertools.combinations((i,j,k),2):
        e = (a,b) if a<b else (b,a)
        d2[meidx[e], t] = 1

r1 = gf2_rank(d1); r2 = gf2_rank(d2)
b0 = M - r1
b1 = len(meta_edges) - r1 - r2
print(f"Delta_3(N): b0={b0}  b1={b1}")
print("Ground truth for k=3 was b0=5, b1=60 -- ", "MATCH" if (b0,b1)==(5,60) else "MISMATCH")
