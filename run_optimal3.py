"""
run_optimal3.py -- the two design curves.

Stage one and two of this experiment build the contact sets and certify a
sample of them.  This stage answers the two questions the paper reports.

  How few contacts.  At a fixed footprint radius, the smallest n for which the
  greedy set of n contacts leaves nothing of the patch uncovered and its
  covered region is a disk: b_0 = 1, b_1 = 0 from the nerve alone and the same
  from the mesh reference.  Scanned one contact at a time.

  How few for redundancy.  The same question for the doubly covered region,
  which is what the dropout guarantee needs and what a failed contact eats
  into.

Adds the uncovered area at k = 1, 2, 3 to every record, so that the coverage
claim and the topology claim are kept apart: a region can be a disk and still
miss part of the target, and the difference is a bay rather than a hole.

Writes results/optimal.json back in place, under 'curves'.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

import run_optimal as R

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
out = json.load(open(RES / 'optimal.json'))


def full(centres, r):
    """R.certificate plus the uncovered area of the patch at k = 1, 2, 3."""
    rec = R.certificate(centres, r)
    dist = dijkstra(R.G, directed=False, indices=centres, limit=r + 1e-9)
    Fm = (dist < r)[:, R.TRI].all(axis=2)
    cnt = Fm.sum(axis=0)
    unc = {}
    for k in (1, 2, 3):
        miss = R.PF[cnt[R.PF] < k]
        unc[str(k)] = float(R.AREA[miss].sum())
    rec['uncovered_area_mm2'] = unc
    return rec


curves = {}

# ---------------------------------------------------- how few contacts, k = 1
for r in (float(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1
                             else ['14', '12', '10', '8'])):
    centres_all = np.array(out['families']['free']['centres'])
    row = []
    best = None
    for n in range(8, len(centres_all) + 1, 2):
        rec = full(centres_all[:n], r)
        ok = (rec['uncovered_area_mm2']['1'] == 0.0
              and rec['shadow']['1'] == [1, 0]
              and rec['mesh']['1']['components'] == 1
              and rec['mesh']['1']['b1'] == 0)
        row.append(dict(n=n, r=r, uncovered=rec['uncovered_area_mm2']['1'],
                        shadow1=rec['shadow']['1'],
                        mesh1=[rec['mesh']['1']['components'], rec['mesh']['1']['b1']],
                        fail=rec['disk_test_failures'], faces=rec['faces'],
                        certified_level=rec['certified_level'],
                        dropout_margin=rec['dropout_margin'], ok=ok))
        print('r=%4.1f n=%3d  uncov=%6.2f  D1=%s mesh1=(%d,%d) fail=%4d  %s'
              % (r, n, rec['uncovered_area_mm2']['1'], rec['shadow']['1'],
                 rec['mesh']['1']['components'], rec['mesh']['1']['b1'],
                 rec['disk_test_failures'], 'OK' if ok else ''), flush=True)
        if ok and best is None:
            best = rec
            best['n_contacts'] = n
            break
    curves['k1_r%g' % r] = dict(radius_mm=r, scan=row, minimal=best)
    out['curves'] = curves
    json.dump(out, open(RES / 'optimal.json', 'w'), indent=1)

# ---------------------------------------------------- redundancy, k = 2
red = []
for n, r in ((96, 10.0), (112, 9.0), (128, 8.0)):
    centres_all = np.array(out['families']['free']['centres'])
    if n > len(centres_all):
        continue
    t0 = time.time()
    rec = full(centres_all[:n], r)
    red.append(rec)
    print('k2  n=%3d r=%4.1f  unc1=%5.2f unc2=%6.2f  D1=%s D2=%s  k*=%d q=%d  [%ds]'
          % (n, r, rec['uncovered_area_mm2']['1'], rec['uncovered_area_mm2']['2'],
             rec['shadow']['1'], rec['shadow']['2'], rec['certified_level'],
             rec['dropout_margin'], time.time() - t0), flush=True)
    out['curves'] = dict(curves, k2=red)
    json.dump(out, open(RES / 'optimal.json', 'w'), indent=1)
curves['k2'] = red

out['curves'] = curves
json.dump(out, open(RES / 'optimal.json', 'w'), indent=1)
print('wrote results/optimal.json')
