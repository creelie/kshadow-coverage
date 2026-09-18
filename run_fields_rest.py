import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields_v2 import FIELDS
from kshadow import *
import networkx as nx
out = json.load(open('results/fields.json'))
which = sys.argv[1:]
for name in which:
    pts, R, desc, window = FIELDS[name]()
    n = len(pts)
    rec = out.get(name, {'n': n, 'R': R, 'desc': desc})
    t0 = time.time()
    N = nerve_of_disks(pts, R, max_size=4) if name.startswith('E') else nerve_of_disks(pts, R)
    rec['nerve_time'] = time.time() - t0
    sizes = {}
    for s in N: sizes[len(s)] = sizes.get(len(s), 0) + 1
    rec['face_counts'] = {str(k): v for k, v in sorted(sizes.items())}
    rec['complete'] = N.complete
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(tuple(s) for s in N if len(s) == 2)
    rec['max_clique'] = max(len(c) for c in nx.find_cliques(G))
    if 'shadow' not in rec:
        rec['shadow'] = {}
        kmax = 1 if name.startswith('E') else 3
        for k in range(1, kmax + 1):
            t0 = time.time()
            if name.startswith('E') and k == 2:
                v, e, t = shadow_2skeleton(N, k)
                b0, _ = betti_2complex(v, e, [])
                rec['shadow'][str(k)] = {'b0': b0, 'b1': None, 'V': len(v), 'E': len(e), 'T': None, 'time': time.time() - t0}
            else:
                b0, b1, sz = shadow_betti(N, k)
                rec['shadow'][str(k)] = {'b0': b0, 'b1': b1, 'V': sz[0], 'E': sz[1], 'T': sz[2], 'time': time.time() - t0}
            print(name, k, rec['shadow'][str(k)], flush=True)
    rec['raster'] = {}
    for res in (700, 1400):
        m, xs, ys = raster_multiplicity(pts, R, window, res)
        rec['raster'][str(res)] = {str(k): raster_betti(m >= k) for k in range(1, 4)}
    print(name, 'raster', rec['raster'], flush=True)
    if not name.startswith('E'):
        rec['dropout_margin'] = dropout_margin(N, n)
    out[name] = rec
    json.dump(out, open('results/fields.json', 'w'), indent=1, default=str)
    if not name.startswith('E') and '--bars' in os.environ.get('BARS', ''):
        t0 = time.time()
        bars = subdivision_persistence(N)
        rec['bars'] = {str(d): bars[d] for d in bars}
        rec['bars_time'] = time.time() - t0
        rec['bars_betti'] = {str(k): betti_from_bars(bars, k) for k in range(1, 8)}
        print(name, 'bars_betti', rec['bars_betti'], flush=True)
        json.dump(out, open('results/fields.json', 'w'), indent=1, default=str)
