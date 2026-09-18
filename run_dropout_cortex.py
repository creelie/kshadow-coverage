"""
run_dropout_cortex.py -- what a failed contact actually costs, on real anatomy.

The paper opens with the clinical fact that motivates the whole certificate:
"a single failed or displaced contact can leave that zone unrecorded".  The
dropout guarantee (Proposition VII.1) answers that question when its two
hypotheses hold, and Section IX B exercises it on an idealised 8 x 8 grid.
On the cortical arrays of Section IX D the guarantee is vacuous: neither grid
reaches b_0(Delta_1(N)) = 1 at any radius tested, so the certified level and
the dropout margin are both 0 and the proposition promises nothing.

That is a statement about what the certificate can promise, not about what
actually happens.  This script measures what actually happens, directly on
the mesh, using the same footprint definition as run_cortex.py: the footprint
of a contact at radius r is the subcomplex of closed triangles all of whose
vertices are within graph distance r of the contact vertex.

For every contact i it reports
  - the private territory of i: the simplices of the surface that lie in the
    footprint of i and of no other contact.  If i fails, exactly this is lost
    from R_>=1, whatever the rest of the array does.
  - the k-private territory: simplices that drop below multiplicity k when i
    is removed, for k = 1, 2, 3.
  - the resulting topology of R_>=k, recomputed from scratch.

and for pairs it samples removals to measure the same quantities.  Every
number comes from the mesh, none from the nerve, so nothing here depends on
the good-cover hypothesis holding.

Writes results/dropout_cortex.json.
"""
import json, time, sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components

import distcache

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'

D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
POS = D['pos'].astype(float)
TRI = D['tri'].astype(np.int64)
NV = len(POS)
E = np.vstack([TRI[:, [0, 1]], TRI[:, [1, 2]], TRI[:, [2, 0]]])
E = np.unique(np.sort(E, axis=1), axis=0)
NE = len(E)

_key = E[:, 0] * NV + E[:, 1]
_order = np.argsort(_key); _keys = _key[_order]
def edge_ids(a, b):
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    return _order[np.searchsorted(_keys, lo * NV + hi)]
TE = np.stack([edge_ids(TRI[:, 0], TRI[:, 1]),
               edge_ids(TRI[:, 1], TRI[:, 2]),
               edge_ids(TRI[:, 2], TRI[:, 0])], axis=1)

# triangle areas, so "territory" is reported in mm^2 and not only in simplices
_a = POS[TRI[:, 1]] - POS[TRI[:, 0]]
_b = POS[TRI[:, 2]] - POS[TRI[:, 0]]
TAREA = 0.5 * np.linalg.norm(np.cross(_a, _b), axis=1)


def subcomplex_topology(vm, em, fm):
    """identical to run_cortex.py: components, chi, closed components, b1"""
    v = int(vm.sum())
    if v == 0:
        return dict(components=0, chi=0, closed=0, b1=0, vertices=0, edges=0, faces=0)
    e = int(em.sum()); f = int(fm.sum())
    loc = np.full(NV, -1); loc[np.flatnonzero(vm)] = np.arange(v)
    sub = E[em]
    A = coo_matrix((np.ones(len(sub)), (loc[sub[:, 0]], loc[sub[:, 1]])), shape=(v, v))
    c, lab = connected_components(A, directed=False)
    closed = 0
    if f:
        te = TE[fm].ravel()
        cnt = np.bincount(te, minlength=NE)
        bnd = np.flatnonzero(cnt == 1)
        has_face = np.zeros(c, bool); has_face[lab[loc[TRI[fm][:, 0]]]] = True
        has_bnd = np.zeros(c, bool)
        if len(bnd):
            has_bnd[lab[loc[E[bnd][:, 0]]]] = True
        closed = int((has_face & ~has_bnd).sum())
    chi = v - e + f
    return dict(components=int(c), chi=int(chi), closed=closed,
                b1=int(c - chi + closed), vertices=v, edges=e, faces=f)


def footprints(dist, r):
    """per-contact masks on vertices, edges, triangles, as in run_cortex.py"""
    n = dist.shape[0]
    Vin = dist < r
    Fm = Vin[:, TRI].all(axis=2)
    Vm = np.zeros((n, NV), bool); Em = np.zeros((n, NE), bool)
    for i in range(n):
        Vm[i, TRI[Fm[i]].ravel()] = True
        Em[i, TE[Fm[i]].ravel()] = True
    return Vm, Em, Fm


def region(mV, mE, mF, k):
    return mV >= k, mE >= k, mF >= k


def analyse(dist, r, kmax=3, pair_samples=400, seed=0):
    n = dist.shape[0]
    Vm, Em, Fm = footprints(dist, r)
    mV = Vm.sum(axis=0); mE = Em.sum(axis=0); mF = Fm.sum(axis=0)
    out = dict(n_electrodes=n, radius_mm=r)

    base = {}
    for k in range(1, kmax + 1):
        vm, em, fm = region(mV, mE, mF, k)
        t = subcomplex_topology(vm, em, fm)
        t['area_mm2'] = float(TAREA[fm].sum())
        base[str(k)] = t
    out['baseline'] = base

    # ---- single-contact removal
    singles = []
    for i in range(n):
        rec = dict(contact=i)
        for k in range(1, kmax + 1):
            lost_f = (mF >= k) & (mF - Fm[i] < k)      # faces dropping below k
            rec[f'k{k}_lost_faces'] = int(lost_f.sum())
            rec[f'k{k}_lost_area_mm2'] = round(float(TAREA[lost_f].sum()), 3)
        # topology of R_>=1 and R_>=2 after the removal
        for k in (1, 2):
            vm, em, fm = region(mV - Vm[i], mE - Em[i], mF - Fm[i], k)
            t = subcomplex_topology(vm, em, fm)
            rec[f'k{k}_components'] = t['components']
            rec[f'k{k}_b1'] = t['b1']
        singles.append(rec)
    out['singles'] = singles

    priv = np.array([s['k1_lost_area_mm2'] for s in singles])
    out['single_summary'] = dict(
        contacts_with_private_territory=int((priv > 0).sum()),
        private_area_max_mm2=float(priv.max()),
        private_area_median_mm2=float(np.median(priv)),
        private_area_total_mm2=float(priv.sum()),
        baseline_area_k1_mm2=base['1']['area_mm2'],
        worst_single_loss_fraction=float(priv.max() / base['1']['area_mm2']),
        components_k1_baseline=base['1']['components'],
        components_k1_max_after=int(max(s['k1_components'] for s in singles)),
        components_k1_min_after=int(min(s['k1_components'] for s in singles)),
    )

    # ---- pair removal, sampled
    rng = np.random.default_rng(seed)
    pairs = []
    seen = set()
    while len(pairs) < min(pair_samples, n * (n - 1) // 2):
        i, j = sorted(rng.choice(n, 2, replace=False).tolist())
        if (i, j) in seen:
            continue
        seen.add((i, j))
        lost_f = (mF >= 1) & (mF - Fm[i] - Fm[j] < 1)
        vm, em, fm = region(mV - Vm[i] - Vm[j], mE - Em[i] - Em[j], mF - Fm[i] - Fm[j], 1)
        t = subcomplex_topology(vm, em, fm)
        pairs.append(dict(pair=[i, j], k1_lost_area_mm2=round(float(TAREA[lost_f].sum()), 3),
                          k1_components=t['components'], k1_b1=t['b1']))
    la = np.array([p['k1_lost_area_mm2'] for p in pairs])
    out['pair_summary'] = dict(
        n_sampled=len(pairs),
        lost_area_max_mm2=float(la.max()), lost_area_median_mm2=float(np.median(la)),
        components_max_after=int(max(p['k1_components'] for p in pairs)),
        pairs_losing_area=int((la > 0).sum()))
    out['pairs'] = pairs[:50]
    return out


if __name__ == '__main__':
    radii = [float(x) for x in (sys.argv[1:] or ['8.0', '10.0', '12.0'])]
    res = {'radii_mm': radii, 'grids': {}}
    t0 = time.time()
    for gname in ('parietal_8x8', 'temporal_4x8'):
        dist = distcache.load(RES, gname)
        per = {}
        for r in radii:
            t1 = time.time()
            per[str(r)] = analyse(dist, r)
            s = per[str(r)]['single_summary']
            print(f"{gname} r={r}: baseline area {s['baseline_area_k1_mm2']:.0f} mm^2, "
                  f"{s['contacts_with_private_territory']}/{per[str(r)]['n_electrodes']} contacts "
                  f"have private territory, worst single loss "
                  f"{s['private_area_max_mm2']:.1f} mm^2 "
                  f"({100*s['worst_single_loss_fraction']:.2f}%), "
                  f"[{time.time()-t1:.0f}s]", flush=True)
        res['grids'][gname] = per
    res['total_time_s'] = round(time.time() - t0, 1)
    with open(RES / 'dropout_cortex.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote results/dropout_cortex.json')
