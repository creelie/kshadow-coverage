"""Exact k-shadow analysis of the flat sensor fields of the earlier
preprint, with raster ground truth at two resolutions and the redundancy
barcode.  By default the fields A, B and D are processed, whose subdivision
barcodes fit in ordinary memory; C (barcode out of memory, see the paper) and
E (41-clique) are handled by run_fields_rest.py, which computes no barcode.
Field names may be given on the command line.  Writes results/fields.json,
merging into it if it exists."""
import json, os, time, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fields_v2 import FIELDS
from kshadow import (nerve_of_disks, shadow_betti, shadow_2skeleton,
                     subdivision_persistence, betti_from_bars,
                     raster_multiplicity, raster_betti, betti_2complex,
                     dropout_margin)

os.makedirs('results', exist_ok=True)
out = json.load(open('results/fields.json')) if os.path.exists('results/fields.json') else {}
which = sys.argv[1:] or ['A_ring_grid_seed3', 'B_ring_grid_seed7', 'D_hex_dead_zone']
for name in which:
    fn = FIELDS[name]
    pts, R, desc, window = fn()
    n = len(pts)
    rec = {'n': n, 'R': R, 'desc': desc}
    t0 = time.time()
    if name.startswith('E'):
        N = nerve_of_disks(pts, R, max_size=4)
    else:
        N = nerve_of_disks(pts, R)
    rec['nerve_time'] = time.time() - t0
    sizes = {}
    for s in N:
        sizes[len(s)] = sizes.get(len(s), 0) + 1
    rec['face_counts'] = {str(k): v for k, v in sorted(sizes.items())}
    rec['complete'] = N.complete
    # largest clique in 1-skeleton
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(n))
    G.add_edges_from(tuple(s) for s in N if len(s) == 2)
    rec['max_clique'] = max(len(c) for c in nx.find_cliques(G))
    rec['shadow'] = {}
    kmax = 3 if not name.startswith('E') else 2
    for k in range(1, kmax + 1):
        t0 = time.time()
        if name.startswith('E') and k == 2:
            v, e, t = shadow_2skeleton(N, k)
            # only H_0 of Delta_2 is computed for Field E (triangle
            # enumeration inside the 41-clique is infeasible)
            b0, _ = betti_2complex(v, e, [])
            rec['shadow'][str(k)] = {'b0': b0, 'b1': None,
                                     'V': len(v), 'E': len(e), 'T': None,
                                     'time': time.time() - t0}
        else:
            b0, b1, sz = shadow_betti(N, k)
            rec['shadow'][str(k)] = {'b0': b0, 'b1': b1, 'V': sz[0],
                                     'E': sz[1], 'T': sz[2],
                                     'time': time.time() - t0}
        print(name, k, rec['shadow'][str(k)], flush=True)
    rec['raster'] = {}
    for res in (700, 1400):
        m, xs, ys = raster_multiplicity(pts, R, window, res)
        rec['raster'][str(res)] = {str(k): raster_betti(m >= k)
                                   for k in range(1, 4)}
    if not name.startswith('E'):
        t0 = time.time()
        bars = subdivision_persistence(N)
        rec['bars'] = {str(d): bars[d] for d in bars}
        rec['bars_time'] = time.time() - t0
        rec['bars_betti'] = {str(k): betti_from_bars(bars, k)
                             for k in range(1, 6)}
        rec['dropout_margin'] = dropout_margin(N, n)
    print(name, json.dumps({k: v for k, v in rec.items() if k != 'bars'},
                           default=str), flush=True)
    out[name] = rec
    json.dump(out, open('results/fields.json', 'w'), indent=1, default=str)
