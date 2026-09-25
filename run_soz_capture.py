"""
run_soz_capture.py -- how often an array sees a seizure onset zone whole.

The design experiment of run_optimal.py measures coverage of a target patch
X as an area.  In the presurgical evaluation of focal epilepsy the question
behind that area is sharper: the onset zone is a region of unknown position
inside the hypothesised territory X, and the resection that aims at seizure
freedom can only be planned around what the implanted contacts saw.  This
script asks, for each array, where an onset zone of geodesic radius s can
sit inside X and still be

  seen whole       every triangle of it lies in some footprint (k = 1);
  seen twice       every triangle lies in two footprints, which is the same
                   as being seen whole after any one contact fails (k = 2);
  missed           no triangle of it lies in any footprint.

Everything is on the triangle rule of run_optimal.py: a triangle lies in the
footprint of a contact when all three of its vertices are within r of it.
Triangles are the atoms.  The onset zone about a triangle T0 of X is the set
of triangles whose centroids lie within distance s of the centroid of T0,
distances being shortest paths on the mesh with each centroid joined to its
three vertices; its centre is drawn with probability proportional to area
over the admissible triangles, those whose zone lies inside X.  At s = 0 the
zone is T0 itself, so the share seen whole k times is the share of the area
of X in R_k, and the shares seen whole and missed add to one.  Each share is
one multi-source Dijkstra run from the centroids of the unseen (or seen)
triangles, thresholded at every s.

The covering radii are on the same rule: e(p, T) is the distance from contact
p to the farthest vertex of T, and rho_k is the largest, over the triangles
of X, of the k-th smallest e(p, T).  X lies in R_k exactly when rho_k < r.

Designs: the documented 8x8 grid (results/cortex.json), and prefixes of the
placed and crown-only sequences of results/ecog_designs.json (run
run_ecog_designs.py first), including the fewest contacts that see X once and
twice at 8 and 10 mm.  No patient data, onset zones or outcomes are used; the
uniform prior over centres, and over failures, are assumptions of the model.

Writes results/soz_capture.json.
"""
import json
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import run_optimal as R

RES = R.RES
C = json.load(open(RES / 'cortex.json'))
ED = json.load(open(RES / 'ecog_designs.json'))

S_GRID = np.round(np.arange(0.0, 15.0 + 1e-9, 0.5), 2)

free = np.array(ED['free']['centres'])
crown = np.array(ED['crown']['centres'])
grid = np.array(C['grids']['parietal_8x8']['contact_vertices'])
dz = ED['designs']
n1_8, n1_10 = dz['k1_r8']['n'], dz['k1_r10']['n']
n2_8, n2_10 = dz['k2_r8']['n'], dz['k2_r10']['n']

# (name, label, contact vertices, footprint radius in mm, k it was built for)
DESIGNS = [
    ('grid64_r8', '8x8 grid, 64 contacts', grid, 8.0, None),
    ('grid64_r10', '8x8 grid, 64 contacts', grid, 10.0, None),
    ('crown64_r8', '64 crown contacts', crown[:64], 8.0, None),
    ('free64_r8', '64 placed contacts', free[:64], 8.0, None),
    ('free%d_r8' % n1_8, 'fewest placed, X in R_1', free[:n1_8], 8.0, 1),
    ('free%d_r8' % n2_8, 'fewest placed, X in R_2', free[:n2_8], 8.0, 2),
    ('free%d_r10' % n1_10, 'fewest placed, X in R_1', free[:n1_10], 10.0, 1),
    ('free%d_r10' % n2_10, 'fewest placed, X in R_2', free[:n2_10], 10.0, 2),
]

# ------------------------------------------------------------ the patch X
NT = len(R.TRI)
TX = R.TRI[R.PF]
AX = R.AREA[R.PF]
inX_t = np.zeros(NT, bool); inX_t[R.PF] = True
CENT = R.POS[R.TRI].mean(axis=1)

# the mesh graph with one extra node per triangle centroid, joined to the
# triangle's three vertices by straight segments
_rows = np.repeat(np.arange(NT), 3) + R.NV
_cols = R.TRI.ravel()
_w = np.linalg.norm(np.repeat(CENT, 3, axis=0) - R.POS[_cols], axis=1)
_G = R.G.tocoo()
NN = R.NV + NT
GC = coo_matrix((np.r_[_G.data, _w, _w],
                 (np.r_[_G.row, _rows, _cols], np.r_[_G.col, _cols, _rows])),
                shape=(NN, NN)).tocsr()
XC = R.NV + R.PF                        # centroid nodes of the triangles of X


def dist_to(tri_idx, limit=np.inf):
    """Distance from the centroid of every triangle of X to the nearest
    centroid of the given triangles (indices into R.TRI)."""
    if len(tri_idx) == 0:
        return np.full(len(R.PF), np.inf)
    d = dijkstra(GC, directed=False, indices=R.NV + np.asarray(tri_idx),
                 min_only=True, limit=limit)
    return d[XC]


# distance to the outside of X, for admissibility
D_OUT = dist_to(np.flatnonzero(~inX_t), limit=80.0)


def fractions(d_avoid, s_grid):
    """Area-weighted share of admissible centre triangles T0 whose zone of
    radius s avoids a set of triangles, given the centroid distance d_avoid
    from each triangle of X to that set.  The zone avoids the set exactly when
    d > s; T0 is admissible when its zone lies in X, D_OUT > s."""
    out = []
    for s in s_grid:
        adm = D_OUT > s
        ok = d_avoid[adm] > s
        out.append(round(float(AX[adm][ok].sum() / AX[adm].sum()), 5))
    return out


def tri_dist(centres):
    """e(p, T) = distance from contact p to the farthest vertex of T, for the
    triangles of X."""
    D = dijkstra(R.G, directed=False, indices=centres, limit=80.0)
    return D[:, TX].max(axis=2)


def run(name, label, centres, r, k_built):
    t0 = time.time()
    E = tri_dist(centres)
    Es = np.sort(E, axis=0)
    rec = dict(label=label, n_contacts=int(len(centres)), radius_mm=r,
               built_for_k=k_built,
               rho1_mm=round(float(Es[0].max()), 3),
               rho2_mm=round(float(Es[1].max()), 3))
    cnt = (E < r).sum(axis=0)
    rec['unseen_mm2'] = {str(k): round(float(AX[cnt < k].sum()), 2) for k in (1, 2)}
    for k in (1, 2):
        rec['whole_k%d' % k] = fractions(dist_to(R.PF[cnt < k]), S_GRID)
    d_seen = dist_to(R.PF[cnt >= 1])
    rec['missed'] = fractions(d_seen, S_GRID)
    # the largest onset zone that can be missed entirely: L(s) > 0 exactly
    # when some triangle has both distances above s
    rec['largest_hidden_s_mm'] = round(float(np.minimum(d_seen, D_OUT).max()), 2)
    rec['seconds'] = round(time.time() - t0, 1)
    print('%-12s n=%3d r=%4.1f rho1 %6.2f rho2 %6.2f unseen %7.1f  W1(5) %.3f  '
          'W2(5) %.3f  L(5) %.3f  h %.1f'
          % (name, len(centres), r, rec['rho1_mm'], rec['rho2_mm'],
             rec['unseen_mm2']['1'], rec['whole_k1'][10], rec['whole_k2'][10],
             rec['missed'][10], rec['largest_hidden_s_mm']), flush=True)
    return rec


# ------------------------------------------------ random contact failure
# q contacts chosen uniformly at random fail; the survivors' footprints are
# recomputed and the share of onset zones of radius s seen whole is taken
# again.  Seeds fixed.  This is a uniform failure model, not a measured one.
FAIL_Q = [0, 1, 2, 3, 4, 6, 8]
FAIL_S = [5.0, 10.0]
FAIL_DRAWS = 100
FAIL_DESIGNS = ['grid64_r8', 'crown64_r8', 'free%d_r8' % n1_8, 'free%d_r8' % n2_8,
                'free%d_r10' % n1_10, 'free%d_r10' % n2_10]


def dropout(name, centres, r, seed):
    rng = np.random.default_rng(seed)
    F = tri_dist(centres) < r                     # contacts x triangles of X
    n = len(centres)
    res = {}
    for q in FAIL_Q:
        vals = {str(s): [] for s in FAIL_S}
        for _ in range(1 if q == 0 else FAIL_DRAWS):
            keep = np.ones(n, bool)
            keep[rng.choice(n, size=q, replace=False)] = False
            cnt = F[keep].sum(axis=0)
            d_bad = dist_to(R.PF[cnt < 1], limit=max(FAIL_S) + 1.0)
            for s in FAIL_S:
                vals[str(s)].append(fractions(d_bad, [s])[0])
        res[str(q)] = {s: dict(mean=round(float(np.mean(v)), 5),
                               p05=round(float(np.percentile(v, 5)), 5),
                               min=round(float(np.min(v)), 5))
                       for s, v in vals.items()}
    print('%-12s whole at s=5 mm, mean over draws, q=%s: %s'
          % (name, FAIL_Q, [res[str(q)]['5.0']['mean'] for q in FAIL_Q]),
          flush=True)
    return res


if __name__ == '__main__':
    out = dict(model='triangle atoms; onset zone about a triangle T0 = triangles '
                     'whose centroids are within s of the centroid of T0; centre '
                     'uniform over admissible triangles, weighted by area',
               patch_area_mm2=R.PATCH_AREA, s_mm=S_GRID.tolist(),
               fewest=dict(k1_r8=n1_8, k2_r8=n2_8, k1_r10=n1_10, k2_r10=n2_10),
               designs={})
    for name, label, centres, r, kb in DESIGNS:
        out['designs'][name] = run(name, label, centres, r, kb)
    out['dropout'] = dict(model='q contacts fail uniformly at random; share '
                                'of onset zones of radius s seen whole by '
                                'the survivors', q=FAIL_Q, s_mm=FAIL_S,
                          draws=FAIL_DRAWS, designs={})
    spec = {d[0]: d for d in DESIGNS}
    for i, name in enumerate(FAIL_DESIGNS):
        _, _, centres, r, _ = spec[name]
        out['dropout']['designs'][name] = dropout(name, centres, r, seed=1000 + i)
    json.dump(out, open(RES / 'soz_capture.json', 'w'), indent=1)
    print('wrote results/soz_capture.json')
