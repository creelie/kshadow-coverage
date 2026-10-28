"""Audit the stored free-contact sequence with rigorous mesh path bounds.

The target is the exact list of faces selected by the existing patch script.
Patch selection is a definition of this benchmark, not a certified geodesic
35 mm disk on the continuous surface. Results concern this supplied mesh.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import time
import numpy as np
import run_optimal as R
from certified_geodesic import certified_graph_vertex_bounds

ROOT = Path(__file__).resolve().parent
start = time.time()
stored = json.loads((ROOT/'results/optimal.json').read_text())
sources = stored['families']['free']['centres']
targets = R.PVf
def progress(i,n):
    if i % 20 == 0 or i == n:
        print(f'certified paths {i}/{n}, elapsed {time.time()-start:.1f}s', flush=True)
paths = certified_graph_vertex_bounds(R.POS,R.TRI,sources,targets=targets,
                                      search_limit=16.,progress=progress)
local = {int(v):i for i,v in enumerate(targets)}
faces = np.array([[local[int(v)] for v in face] for face in R.TRI[R.PF]])
vertices = R.POS[targets]
# Exact squared face diameters for the represented binary64 coordinates.
point = [tuple(Fraction(float(x)) for x in row) for row in vertices]
diameters = []
for face in faces:
    a,b,c = [point[int(v)] for v in face]
    diameters.append(max(sum((x-y)**2 for x,y in zip(u,v))
                         for u,v in [(a,b),(a,c),(b,c)]))
records = []
for n in [64,66,80,96,112,128,160]:
    for k in [1,2]:
        kth = np.partition(paths.upper[:n],k-1,axis=0)[k-1]
        maximum = kth[faces].max(axis=1)
        for radius in [8.,9.,10.,12.,14.]:
            passing = []
            for j,(m,d2) in enumerate(zip(maximum,diameters)):
                margin = Fraction(radius)-Fraction(float(m))
                if margin>0 and d2<3*margin*margin:
                    passing.append(j)
            records.append(dict(contacts=n,k=k,radius_mm=radius,
                                passing_faces=len(passing),total_faces=len(faces),
                                all_faces_pass=len(passing)==len(faces)))
            if len(passing)==len(faces):
                print(f'ALL PASS: n={n} k={k} r={radius:g}',flush=True)
result = dict(method='upward dyadic edge lengths; integer Dijkstra; exact rational whole-face test',
              quantum_mm=paths.quantum,search_limit_mm=paths.search_limit,
              surface_sha256=hashlib.sha256((ROOT/'data/colin27_lh_pial.npz').read_bytes()).hexdigest(),
              source_results_sha256=hashlib.sha256((ROOT/'results/optimal.json').read_bytes()).hexdigest(),
              target_face_indices=R.PF.tolist(),contact_vertex_indices=sources,
              surface_vertices=len(R.POS),surface_faces=len(R.TRI),
              target_vertices=len(targets),target_faces=len(faces),records=records,
              scope='Continuous coverage of the represented piecewise-flat target faces, conditional on the radius-footprint model; no patient efficacy claim',
              elapsed_seconds=round(time.time()-start,3))
(ROOT/'results/certified_mesh.json').write_text(json.dumps(result,indent=2)+'\n')
print('Saved results/certified_mesh.json',flush=True)
