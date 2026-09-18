import itertools
import json
import numpy as np
import networkx as nx
from exact_geometry import disks_intersect_exact

rng = np.random.default_rng(3)

R = 1.55

def ring_sensors(n, cx, cy, radius_to_center):
    return [(cx + radius_to_center*np.cos(2*np.pi*k/n),
             cy + radius_to_center*np.sin(2*np.pi*k/n)) for k in range(n)]

fence_outer = ring_sensors(24, 6.0, 6.0, 7.4)
fence_inner = ring_sensors(22, 6.0, 6.0, 5.9)
fence = fence_outer + fence_inner

coords = [6.0 + 1.4*k for k in range(-3, 4)]
grid_pts = []
for x in coords:
    for y in coords:
        jx, jy = rng.uniform(-0.08, 0.08, 2)
        grid_pts.append((x + jx, y + jy))

grid_arr = np.array(grid_pts)
center = np.array([6.0, 6.0])
dists = np.hypot(grid_arr[:,0]-center[0], grid_arr[:,1]-center[1])
remove_idx = set(np.argsort(dists)[:4])
removed_pts = [grid_pts[i] for i in remove_idx]
grid_pts = [p for i, p in enumerate(grid_pts) if i not in remove_idx]
print(f"Removed {len(removed_pts)} interior sensors near centre")

sensors = np.array(fence + grid_pts)
N = len(sensors)
print(f"N sensors = {N}")

# ---- exact 1- and 2-skeleton ----
pairs = [(i,j) for i,j in itertools.combinations(range(N), 2)
         if np.hypot(*(sensors[i]-sensors[j])) <= 2*R]
edges = [e for e in pairs if disks_intersect_exact(e, sensors, R)]
print(f"edges = {len(edges)} (of {len(pairs)} candidate pairs)")

adj = {i: set() for i in range(N)}
for (i,j) in edges:
    adj[i].add(j); adj[j].add(i)

triangle_candidates = []
for i in range(N):
    for j in adj[i]:
        if j <= i: continue
        for k in (adj[i] & adj[j]):
            if k > j:
                triangle_candidates.append((i,j,k))
triangles = [t for t in triangle_candidates if disks_intersect_exact(t, sensors, R)]
print(f"triangles = {len(triangles)} (of {len(triangle_candidates)} candidates)")
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
print(f"maximal cliques: {len(maximal_cliques)}, largest = "
      f"{max(len(c) for c in maximal_cliques)}")

all_faces = set()
for c in maximal_cliques:
    c = sorted(c)
    cap = min(len(c), 8)
    for r in range(1, cap+1):
        for sub in itertools.combinations(c, r):
            if r <= 2:
                all_faces.add(sub)
            elif r == 3:
                if sub in triangle_set:
                    all_faces.add(sub)
            else:
                if is_face_by_helly(sub):
                    all_faces.add(sub)

faces_by_dim = {}
for f in all_faces:
    faces_by_dim.setdefault(len(f), []).append(f)
for d in sorted(faces_by_dim):
    print(f"  faces of size {d}: {len(faces_by_dim[d])}")

# Cross-check a sample of higher faces directly (no Helly shortcut) as a
# safety net against any remaining logic error.
import random
random.seed(0)
size4_faces = faces_by_dim.get(4, [])
sample = random.sample(size4_faces, min(30, len(size4_faces)))
bad = 0
for f in sample:
    if not disks_intersect_exact(f, sensors, R):
        bad += 1
        print(f"  MISMATCH: {f} flagged as face but exact test says no")
print(f"Cross-check of {len(sample)} size-4 faces against direct exact test: "
      f"{len(sample)-bad}/{len(sample)} confirmed")

np.save("sensors.npy", sensors)
json.dump({
    "edges": edges, "triangles": triangles,
    "all_faces": [list(f) for f in all_faces],
    "R": R, "fence_inner": fence_inner, "fence_outer": fence_outer,
}, open("complex.json", "w"))
print("saved.")
