"""
run_optimal2.py -- the certificates of the designs built by run_optimal.py.

Reads results/optimal.json, adds

  * the exact crown floor: the covering radius of the set of every gyral
    vertex of the patch at once, computed by one multi-source Dijkstra.  No
    array confined to the crowns can do better than this, whatever its
    contact count, so it is a lower bound on the footprint radius such an
    array needs before it can cover the patch at all;
  * the k-fold covering radii of the greedy sets, k = 1, 2, 3;
  * the full certificate at a range of contact counts and radii: the disk
    test on every face of the nerve, the Betti numbers of Delta_k(N), the
    mesh reference, the certified level and the dropout margin.

Writes results/optimal.json back in place.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

import run_optimal as R          # mesh, patch, topology, certificate

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
out = json.load(open(RES / 'optimal.json'))

# ------------------------------------------------ the exact crown floor
gyral = np.flatnonzero((R.SULC < 0) & R.inpatch)
d_all = dijkstra(R.G, directed=False, indices=gyral, min_only=True)
floor = float(d_all[R.PVf].max())
arg = int(R.PVf[np.argmax(d_all[R.PVf])])
out['crown_floor'] = dict(covering_radius_mm=round(floor, 4),
                          n_crown_vertices=int(len(gyral)),
                          worst_vertex=arg,
                          worst_vertex_mni=R.POS[arg].round(2).tolist(),
                          worst_vertex_sulc=float(R.SULC[arg]))
print('crown floor: %.3f mm, attained at vertex %d, sulcal depth %.2f'
      % (floor, arg, R.SULC[arg]), flush=True)

# ------------------------------------------------ k-fold covering radii
for fam in ('free', 'crown'):
    centres = np.array(out['families'][fam]['centres'])
    dm = dijkstra(R.G, directed=False, indices=centres)[:, R.PVf]
    kfold = {}
    for n in sorted(set(list(range(8, len(centres) + 1, 8)) + [len(centres)])):
        sub = np.sort(dm[:n], axis=0)
        row = {}
        for k in (1, 2, 3):
            row[str(k)] = float(sub[k - 1].max()) if n >= k else None
        kfold[str(n)] = row
    out['families'][fam]['kfold_rho_mm'] = kfold
print('k-fold covering radii done', flush=True)

# ------------------------------------------------ certificates
PLAN = {
    'free':  [(20, 14.0), (24, 13.0), (24, 14.0), (28, 12.0), (28, 13.0),
              (32, 11.0), (32, 12.0), (38, 10.0), (38, 11.0),
              (48, 9.0), (48, 10.0), (64, 8.0), (64, 9.0),
              (80, 7.0), (80, 8.0), (96, 7.0), (112, 6.0), (112, 7.0),
              (128, 6.0), (160, 5.0), (160, 6.0)],
    'crown': [(32, 13.0), (48, 13.0), (64, 10.0), (64, 12.0), (64, 13.0)],
}
for fam, plan in PLAN.items():
    centres_all = np.array(out['families'][fam]['centres'])
    recs = []
    for n, r in plan:
        if n > len(centres_all):
            continue
        t0 = time.time()
        rec = R.certificate(centres_all[:n], r)
        rec['family'] = fam
        rec['rho_mm'] = out['families'][fam]['rho_mm'][n - 1]
        rec['seconds'] = round(time.time() - t0, 1)
        recs.append(rec)
        print('%-5s n=%3d r=%4.1f  rho=%5.2f  faces=%6d fail=%4d  D1=%s D2=%s  '
              'mesh1=(%d,%d)  k*=%d q=%d  uncovered=%.1f mm^2  [%ds]'
              % (fam, n, r, rec['rho_mm'], rec['faces'], rec['disk_test_failures'],
                 rec['shadow']['1'], rec['shadow']['2'],
                 rec['mesh']['1']['components'], rec['mesh']['1']['b1'],
                 rec['certified_level'], rec['dropout_margin'],
                 rec['uncovered_patch_area'], time.time() - t0), flush=True)
    out['families'][fam]['certificates'] = recs

json.dump(out, open(RES / 'optimal.json', 'w'), indent=1)
print('wrote results/optimal.json')
