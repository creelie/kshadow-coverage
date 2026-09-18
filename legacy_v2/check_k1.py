import json
import numpy as np

data = json.load(open("complex.json"))
edges = [tuple(e) for e in data["edges"]]
triangles = [tuple(t) for t in data["triangles"]]
N = max(max(e) for e in edges) + 1

def gf2_rank(M):
    M = M.copy().astype(np.uint8) % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if M[r, col] == 1:
                pivot = r; break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(rows):
            if r != rank and M[r, col] == 1:
                M[r] = (M[r] + M[rank]) % 2
        rank += 1
        if rank == rows:
            break
    return rank

d1 = np.zeros((N, len(edges)), dtype=np.uint8)
eidx = {e: k for k, e in enumerate(edges)}
for k, (i, j) in enumerate(edges):
    d1[i, k] = 1; d1[j, k] = 1

import itertools
d2 = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
for t, (i, j, k) in enumerate(triangles):
    for (a, b) in itertools.combinations((i, j, k), 2):
        e = (a, b) if a < b else (b, a)
        d2[eidx[e], t] = 1

r1 = gf2_rank(d1)
r2 = gf2_rank(d2)
b0 = N - r1
b1 = len(edges) - r1 - r2
print(f"N itself (= Delta_1(N)):  b0 = {b0}   b1 = {b1}")
print("Ground truth for k=1 was b0=1, b1=1 -- ", "MATCH" if (b0,b1)==(1,1) else "MISMATCH")
