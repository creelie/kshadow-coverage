"""
run_packing.py -- the lower bound that makes the design optimal.

A set of points of the patch that are pairwise more than 2r apart in the
geodesic metric cannot be covered by fewer geodesic balls of radius r than it
has points, because no such ball holds two of them.  So the size of a
2r-separated set is a lower bound on the number of contacts any array of
footprint radius r needs, wherever its contacts are put and however they are
arranged.  The bound is computed here by inserting patch vertices greedily
under the separation constraint, each insertion costing one Dijkstra.

Combined with the constructions of run_optimal3.py this brackets the minimum
contact count from both sides.

Writes the 'packing' entry of results/optimal.json.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

import run_optimal as R

RES = Path(__file__).resolve().parent / 'results'
out = json.load(open(RES / 'optimal.json'))

RADII = [float(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1
                            else ['14', '12', '10', '8'])]

pack = {}
for r in RADII:
    sep = 2.0 * r
    t0 = time.time()
    # start at the point farthest from the patch centre, then insert any
    # vertex still at least `sep` from every chosen point
    chosen = [int(R.PVf[np.argmax(R.d_a[R.PVf])])]
    dmin = dijkstra(R.G, directed=False, indices=chosen[0])[R.PVf]
    while True:
        j = int(np.argmax(dmin))
        if dmin[j] < sep:
            break
        v = int(R.PVf[j])
        chosen.append(v)
        dmin = np.minimum(dmin, dijkstra(R.G, directed=False, indices=v)[R.PVf])
    pack[str(r)] = dict(separation_mm=sep, size=len(chosen),
                        vertices=chosen,
                        vertices_mni=R.POS[chosen].round(2).tolist(),
                        seconds=round(time.time() - t0, 1))
    print('r=%4.1f  separation %4.1f mm  packing %3d points  [%ds]'
          % (r, sep, len(chosen), time.time() - t0), flush=True)

out['packing'] = pack
json.dump(out, open(RES / 'optimal.json', 'w'), indent=1)
print('wrote results/optimal.json')
