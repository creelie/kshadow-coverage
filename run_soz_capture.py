"""
run_soz_capture.py -- how often an array sees a seizure onset zone whole.

The design experiment of run_optimal.py measures coverage of a target patch
X as an area.  In the presurgical evaluation of focal epilepsy the question
behind that area is sharper: the onset zone is a region of unknown position
inside the hypothesised territory X, and the resection that aims at seizure
freedom can only be planned around what the implanted contacts saw.  This
script asks, for each array, where an onset zone of geodesic radius s can
sit inside X and still be

  seen whole       every point of it lies in some footprint (k = 1);
  seen twice       every point lies in two footprints, which is the same as
                   being seen whole after any one contact fails (k = 2);
  missed           no point of it lies in any footprint.

The onset zone is modelled as the geodesic disk B(x, s) about a mesh vertex
x, and its centre is drawn with probability proportional to vertex area over
the admissible centres, those with B(x, s) inside X.  With U_k the vertices
of patch triangles held by fewer than k footprints, B(x, s) is seen k times
exactly when x is at distance at least s from U_k, so each fraction is one
multi-source Dijkstra.  Footprints are the geodesic disks of run_optimal.py:
a triangle belongs to the footprint of a contact when all three of its
vertices lie within distance r of it.

Everything is computed on the colin27 template of run_optimal.py.  No patient
data, onset zones or outcomes are used; the uniform prior over centres is an
assumption of the model and not a clinical measurement.

Writes results/soz_capture.json.
"""
import json
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

import run_optimal as R

RES = R.RES
O = json.load(open(RES / 'optimal.json'))
C = json.load(open(RES / 'cortex.json'))

S_GRID = np.round(np.arange(0.0, 15.0 + 1e-9, 0.5), 2)

free = np.array(O['families']['free']['centres'])
crown = np.array(O['families']['crown']['centres'])
grid = np.array(C['grids']['parietal_8x8']['contact_vertices'])

# (name, label, contact vertices, footprint radius in mm)
DESIGNS = [
    ('grid64_r8', '8x8 grid, 64 contacts', grid, 8.0),
    ('crown64_r8', '64 crown contacts by covering radius', crown[:64], 8.0),
    ('free64_r8', '64 contacts by covering radius', free[:64], 8.0),
    ('free66_r8', '66 contacts by covering radius', free[:66], 8.0),
    ('grid64_r10', '8x8 grid, 64 contacts', grid, 10.0),
    ('free46_r10', '46 contacts by covering radius', free[:46], 10.0),
    ('free96_r10', '96 contacts by covering radius', free[:96], 10.0),
    ('free112_r9', '112 contacts by covering radius', free[:112], 9.0),
]

# ------------------------------------------------------------ the patch X
PV = R.PVf                                    # vertices of patch triangles
inX = np.zeros(R.NV, bool); inX[PV] = True
VAREA = np.zeros(R.NV)
for j in range(3):
    np.add.at(VAREA, R.TRI[R.PF, j], R.AREA[R.PF] / 3.0)
# distance from each patch vertex to the nearest vertex outside X
outside = np.flatnonzero(~inX)
_d_out = dijkstra(R.G, directed=False, indices=outside, min_only=True,
                  limit=S_GRID.max() + 1.0)
D_EDGE = _d_out[PV]


def dist_to(sources):
    """Geodesic distance from every patch vertex to a vertex set."""
    if len(sources) == 0:
        return np.full(len(PV), np.inf)
    d = dijkstra(R.G, directed=False, indices=np.asarray(sources), min_only=True)
    return d[PV]


def fractions(d_avoid, s_grid):
    """Area-weighted share of admissible centres x whose closed disk
    B(x, s) = {v : d(x, v) <= s} avoids a vertex set, given the distance
    d_avoid from each patch vertex to that set.  The disk avoids the set
    exactly when d(x, set) > s.  A centre is admissible when its disk lies
    inside X, d(x, outside X) > s."""
    out = []
    w = VAREA[PV]
    for s in s_grid:
        adm = D_EDGE > s
        ok = d_avoid[adm] > s
        out.append(round(float(w[adm][ok].sum() / w[adm].sum()), 5))
    return out


def run(name, label, centres, r):
    t0 = time.time()
    dist = dijkstra(R.G, directed=False, indices=centres, limit=r + 1e-9)
    Fm = (dist < r)[:, R.TRI].all(axis=2)
    cnt = Fm.sum(axis=0)
    rec = dict(label=label, n_contacts=int(len(centres)), radius_mm=r)
    # covering radii of X: distance from its farthest vertex to the nearest
    # and to the second nearest contact
    dfull = np.sort(dijkstra(R.G, directed=False, indices=centres, limit=80.0)[:, PV], axis=0)
    rec['rho1_mm'] = round(float(dfull[0].max()), 3)
    rec['rho2_mm'] = round(float(dfull[1].max()), 3)
    pf_cnt = cnt[R.PF]
    rec['unseen_mm2'] = {str(k): round(float(R.AREA[R.PF][pf_cnt < k].sum()), 2)
                         for k in (1, 2)}
    # seen whole k times: the disk avoids every vertex of a triangle of X
    # held by fewer than k footprints
    for k in (1, 2):
        bad = np.unique(R.TRI[R.PF[pf_cnt < k]].ravel())
        rec['whole_k%d' % k] = fractions(dist_to(bad), S_GRID)
    # missed: the disk avoids every vertex of a triangle of X that is seen
    seen = np.unique(R.TRI[R.PF[pf_cnt >= 1]].ravel())
    d_seen = dist_to(seen)
    rec['missed'] = fractions(d_seen, S_GRID)
    # the largest onset zone that can hide entirely inside X, exactly: the
    # largest min(d(x, seen), d(x, outside X)) over patch vertices x
    rec['largest_hidden_s_mm'] = round(float(np.minimum(d_seen, D_EDGE).max()), 2)
    rec['seconds'] = round(time.time() - t0, 1)
    print('%-11s n=%3d r=%4.1f  unseen %7.1f mm^2  whole(s=5) %.3f  '
          'twice(s=5) %.3f  missed(s=5) %.3f  hidden<=%.1f mm  [%.0f s]'
          % (name, len(centres), r, rec['unseen_mm2']['1'],
             rec['whole_k1'][10], rec['whole_k2'][10], rec['missed'][10],
             rec['largest_hidden_s_mm'], rec['seconds']), flush=True)
    return rec


# ------------------------------------------------ random contact failure
# q contacts chosen uniformly at random fail; the survivors' footprints are
# recomputed and the share of onset zones of radius s seen whole is taken
# again.  Seeds fixed.  This is a uniform failure model, not a measured one.
FAIL_Q = [0, 1, 2, 3, 4, 6, 8]
FAIL_S = [5.0, 10.0]
FAIL_DRAWS = 100
FAIL_DESIGNS = ['grid64_r8', 'crown64_r8', 'free66_r8', 'free46_r10',
                'free96_r10', 'free112_r9']


def dropout(name, centres, r, seed):
    rng = np.random.default_rng(seed)
    dist = dijkstra(R.G, directed=False, indices=centres, limit=r + 1e-9)
    Fm = (dist < r)[:, R.TRI[R.PF]].all(axis=2)      # contacts x patch faces
    n = len(centres)
    res = {}
    for q in FAIL_Q:
        vals = {str(s): [] for s in FAIL_S}
        for _ in range(1 if q == 0 else FAIL_DRAWS):
            keep = np.ones(n, bool)
            keep[rng.choice(n, size=q, replace=False)] = False
            cnt = Fm[keep].sum(axis=0)
            bad = np.unique(R.TRI[R.PF[cnt < 1]].ravel())
            d_bad = (np.full(len(PV), np.inf) if len(bad) == 0 else
                     dijkstra(R.G, directed=False, indices=bad, min_only=True,
                              limit=max(FAIL_S) + 1.0)[PV])
            for s in FAIL_S:
                vals[str(s)].append(fractions(d_bad, [s])[0])
        res[str(q)] = {s: dict(mean=round(float(np.mean(v)), 5),
                               p05=round(float(np.percentile(v, 5)), 5),
                               min=round(float(np.min(v)), 5))
                       for s, v in vals.items()}
    print('%-11s whole at s=5 mm, mean over draws, q=%s: %s'
          % (name, FAIL_Q, [res[str(q)]['5.0']['mean'] for q in FAIL_Q]),
          flush=True)
    return res


if __name__ == '__main__':
    out = dict(model='onset zone = geodesic disk B(x, s) inside X; centre '
                     'uniform over admissible vertices, weighted by area',
               patch_area_mm2=R.PATCH_AREA, s_mm=S_GRID.tolist(), designs={})
    for name, label, centres, r in DESIGNS:
        out['designs'][name] = run(name, label, centres, r)
    out['dropout'] = dict(model='q contacts fail uniformly at random; share '
                                'of onset zones of radius s seen whole by '
                                'the survivors', q=FAIL_Q, s_mm=FAIL_S,
                          draws=FAIL_DRAWS, designs={})
    spec = {d[0]: d for d in DESIGNS}
    for i, name in enumerate(FAIL_DESIGNS):
        _, _, centres, r = spec[name]
        out['dropout']['designs'][name] = dropout(name, centres, r, seed=1000 + i)
    json.dump(out, open(RES / 'soz_capture.json', 'w'), indent=1)
    print('wrote results/soz_capture.json')
