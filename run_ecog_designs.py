"""
run_ecog_designs.py -- covering radii and designs of the ECoG paper, on the
triangle rule the rest of the repository uses.

run_optimal.py and its successors decide coverage triangle by triangle: a
triangle of the mesh lies in the footprint of a contact p when all three of its
vertices are within distance r of p.  The covering radius that matches that
rule is taken over triangles, with the distance from a contact to a triangle
being the distance to the triangle's farthest vertex,

    e(p, T) = max_{v in T} d(v, p),
    rho_k(P) = max_{T in X} (k-th smallest of e(p, T) over p in P),

and then X lies in the k-fold region exactly when rho_k(P) < r.  The vertex
covering radius stored in results/optimal.json (rho_mm, kfold_rho_mm) is
never larger, and exceeds this one by at most the longest edge of X, so it can
fall below r while a triangle is still unseen; that is why 64 placed contacts
have a vertex radius of 6.99 mm and still leave 1.0 mm^2 unseen at r = 8 mm.

This script computes, all on the triangle rule:

  * rho_1 and rho_2 of every prefix of the farthest-point sequence of
    results/optimal.json (sites anywhere on X);
  * the fewest contacts of that sequence with X inside R_1, scanned one
    contact at a time, at r = 8, 9, 10, 12, 14 mm, and the fewest with X
    inside R_2, with the certificate of each design (run_optimal.certificate);
  * the site floor of the gyral crowns, over every gyral vertex of the
    hemisphere that can reach X (not only those inside X), for k = 1 and 2,
    and the same on the vertex rule for comparison;
  * a crown-only sequence: at each step the admissible crown site with the
    smallest e(c, T*) to the worst covered triangle T* is added, with its
    covering radii, which level off at the site floor.

Writes results/ecog_designs.json.  A few minutes.
"""
import json
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

import run_optimal as R

RES = R.RES
O = json.load(open(RES / 'optimal.json'))
FREE = np.array(O['families']['free']['centres'])
RADII = [8.0, 9.0, 10.0, 12.0, 14.0]
REACH = 25.0          # no site farther than this from X can matter below
CHUNK = 100

TX = R.TRI[R.PF]                       # triangles of X, vertex indices
AX = R.AREA[R.PF]
PE = np.unique(R.TE[R.PF].ravel())
EDGE_MAX = float(R.LEN[PE].max())
EDGE_MED = float(np.median(R.LEN[PE]))


def tri_dist(sources, limit=np.inf):
    """e(p, T) for every source p and every triangle T of X, in chunks."""
    out = np.empty((len(sources), len(TX)))
    for i in range(0, len(sources), CHUNK):
        D = dijkstra(R.G, directed=False, indices=sources[i:i + CHUNK], limit=limit)
        out[i:i + CHUNK] = D[:, TX].max(axis=2)
    return out


def prefix_radii(E):
    """rho_1 and rho_2 (triangle rule) of every prefix of the rows of E."""
    m1 = np.full(E.shape[1], np.inf)
    m2 = np.full(E.shape[1], np.inf)
    r1, r2 = [], []
    for row in E:
        m2 = np.minimum(m2, np.maximum(m1, row))
        m1 = np.minimum(m1, row)
        r1.append(float(m1.max()))
        r2.append(float(m2.max()) if np.isfinite(m2.max()) else None)
    return r1, r2


def first_below(rho, r):
    for i, x in enumerate(rho):
        if x is not None and x < r:
            return i + 1
    return None


def unseen(E, n, k, r):
    cnt = (E[:n] < r).sum(axis=0)
    return float(AX[cnt < k].sum())


def summary(rec):
    return dict(shadow=rec['shadow'], mesh={k: [v['components'], v['b1']]
                                            for k, v in rec['mesh'].items()},
                disk_test_failures=rec['disk_test_failures'], faces=rec['faces'],
                certified_level=rec['certified_level'],
                dropout_margin=rec['dropout_margin'],
                uncovered_patch_area=rec['uncovered_patch_area'])


t0 = time.time()
out = dict(rule='triangle rule: T in the footprint of p iff every vertex of T '
                'is within r of p; e(p,T) = max over the vertices of T of d(v,p)',
           edge_max_mm=EDGE_MAX, edge_median_mm=EDGE_MED, radii_mm=RADII)

# ------------------------------------------------ the farthest-point sequence
E_free = tri_dist(FREE)
r1, r2 = prefix_radii(E_free)
out['free'] = dict(centres=FREE.tolist(), rho1_mm=[round(x, 4) for x in r1],
                   rho2_mm=[None if x is None else round(x, 4) for x in r2],
                   rho1_vertex_mm=O['families']['free']['rho_mm'])
print('free sequence: rho_1(64) = %.3f, rho_1(66) = %.3f, rho_2(160) = %s  [%.0f s]'
      % (r1[63], r1[65], r2[-1], time.time() - t0), flush=True)

designs = {}
for k in (1, 2):
    for r in RADII:
        n = first_below(r1 if k == 1 else r2, r)
        if n is None:
            designs['k%d_r%g' % (k, r)] = dict(k=k, radius_mm=r, n=None)
            continue
        rec = R.certificate(FREE[:n], r)
        d = dict(k=k, radius_mm=r, n=n,
                 rho_mm=round((r1 if k == 1 else r2)[n - 1], 4),
                 unseen_mm2=unseen(E_free, n, k, r),
                 unseen_mm2_one_fewer=unseen(E_free, n - 1, k, r),
                 off_crown=int((R.SULC[FREE[:n]] >= 0).sum()),
                 packing_bound=O['packing'].get(str(r), {}).get('size'),
                 certificate=summary(rec))
        designs['k%d_r%g' % (k, r)] = d
        print('k=%d r=%4.1f: n=%3d rho=%.3f unseen %.2f (n-1: %.2f)  D1=%s mesh1=%s  [%.0f s]'
              % (k, r, n, d['rho_mm'], d['unseen_mm2'], d['unseen_mm2_one_fewer'],
                 rec['shadow']['1'], d['certificate']['mesh']['1'], time.time() - t0),
              flush=True)
out['designs'] = designs

# ------------------------------------------------ the site floor of the crowns
reach = np.flatnonzero((R.SULC < 0) & (R.d_a <= R.PATCH_MM + REACH))
inside = np.flatnonzero((R.SULC < 0) & np.isin(np.arange(R.NV), R.PVf))
E_c = tri_dist(reach, limit=REACH)
part = np.partition(E_c, 1, axis=0)
f1, f2 = part[0], part[1]
w1 = int(np.argmax(f1))
# vertex rule, for comparison: every gyral vertex of the hemisphere
dv = dijkstra(R.G, directed=False, indices=np.flatnonzero(R.SULC < 0), min_only=True)
dv_in = dijkstra(R.G, directed=False, indices=inside, min_only=True)
out['crown_floor'] = dict(
    n_sites_within_reach=int(len(reach)), n_sites_inside_X=int(len(inside)),
    reach_mm=REACH,
    rho_C_mm=float(f1.max()), rho_C2_mm=float(f2.max()),
    worst_triangle_centroid_mni=R.POS[TX[w1]].mean(axis=0).round(2).tolist(),
    rho_C_vertex_all_crowns_mm=float(dv[R.PVf].max()),
    rho_C_vertex_crowns_inside_X_mm=float(dv_in[R.PVf].max()))
print('crown floor: triangle rule %.3f (k=2: %.3f); vertex rule, all crowns %.3f, '
      'crowns inside X only %.3f  [%.0f s]'
      % (f1.max(), f2.max(), dv[R.PVf].max(), dv_in[R.PVf].max(), time.time() - t0),
      flush=True)

# ------------------------------------------------ a crown-only sequence
NMAX = 160
chosen = [int(reach[np.argmin(R.d_a[reach])])]
m1 = E_c[np.searchsorted(reach, chosen[0])].copy()
m2 = np.full(len(TX), np.inf)
c1, c2 = [float(m1.max())], [None]
col = {int(v): i for i, v in enumerate(reach)}
floor_n = None
while len(chosen) < NMAX:
    # triangles already at their own floor f1 cannot be improved; take the
    # worst of the others, and add the admissible site with the smallest
    # e(c, T*) to it (never one already chosen: that would put T* at its floor)
    open_ = m1 > f1 + 1e-9
    if not open_.any():
        break
    if floor_n is None and m1.max() <= f1.max() + 1e-9:
        floor_n = len(chosen)
    worst = int(np.flatnonzero(open_)[np.argmax(m1[open_])])
    c = int(reach[np.argmin(E_c[:, worst])])
    chosen.append(c)
    row = E_c[col[c]]
    m2 = np.minimum(m2, np.maximum(m1, row))
    m1 = np.minimum(m1, row)
    c1.append(float(m1.max()))
    c2.append(float(m2.max()) if np.isfinite(m2.max()) else None)
if floor_n is None and m1.max() <= f1.max() + 1e-9:
    floor_n = len(chosen)
out['crown'] = dict(rule='add the admissible crown site with the smallest e(c,T) '
                         'to the worst covered triangle T of X not yet at its '
                         'own floor',
                    centres=chosen, rho1_mm=[round(x, 4) for x in c1],
                    rho2_mm=[None if x is None else round(x, 4) for x in c2],
                    reaches_floor_at=floor_n,
                    unseen_mm2_r8_at_64=(unseen(E_c[[col[c] for c in chosen]], 64, 1, 8.0)
                                         if len(chosen) >= 64 else None))
print('crown sequence: %d sites, rho_1 reaches the floor %.3f at n=%s; '
      'unseen at r=8 with 64: %s  [%.0f s]'
      % (len(chosen), f1.max(), floor_n, out['crown']['unseen_mm2_r8_at_64'],
         time.time() - t0), flush=True)

json.dump(out, open(RES / 'ecog_designs.json', 'w'), indent=1)
print('wrote results/ecog_designs.json')
