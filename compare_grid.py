"""Matched comparison: the published 8x8 grid against the greedy design on the
same target patch, same footprint radius."""
import json
import numpy as np
from scipy.sparse.csgraph import dijkstra
import run_optimal as R

RES = R.ROOT / 'results'
O = json.load(open(RES / 'optimal.json'))
C = json.load(open(RES / 'cortex.json'))
grid_idx = np.array(C['grids']['parietal_8x8']['contact_vertices'])

rows = []
for r in (8.0, 10.0, 12.0):
    for name, idx in (('grid_8x8', grid_idx),
                      ('greedy', np.array(O['families']['free']['centres'][:len(grid_idx)]))):
        rec = R.certificate(idx, r)
        dist = dijkstra(R.G, directed=False, indices=idx, limit=r + 1e-9)
        cnt = ((dist < r)[:, R.TRI].all(axis=2)).sum(axis=0)
        miss = R.PF[cnt[R.PF] < 1]
        rows.append(dict(array=name, n=len(idx), radius_mm=r,
                         uncovered_mm2=float(R.AREA[miss].sum()),
                         uncovered_fraction=float(R.AREA[miss].sum() / R.PATCH_AREA),
                         shadow1=rec['shadow']['1'],
                         mesh1=[rec['mesh']['1']['components'], rec['mesh']['1']['b1']],
                         disk_test_failures=rec['disk_test_failures'],
                         certified_level=rec['certified_level'],
                         dropout_margin=rec['dropout_margin']))
        print('%-9s n=%d r=%4.1f  unseen %7.1f mm^2 (%5.1f%%)  D1=%s mesh1=%s  k*=%d'
              % (name, len(idx), r, rows[-1]['uncovered_mm2'],
                 100 * rows[-1]['uncovered_fraction'], rec['shadow']['1'],
                 rows[-1]['mesh1'], rec['certified_level']), flush=True)

O['matched_comparison'] = rows
json.dump(O, open(RES / 'optimal.json', 'w'), indent=1)
print('wrote results/optimal.json')
