"""
run_design.py -- the design sweep on the template hemisphere.

run_cortex.py answers the question "what does this array certify?" for the
two arrays of the FieldTrip tutorial.  Both of them certify nothing: their
singly covered region is several pieces at every radius tested, so the
certified level is zero.  This script answers the design question instead.
Over a grid of contact pitches and footprint radii it asks which arrays reach

    b_0(Delta_1(N)) = 1 and b_1(Delta_1(N)) = 0,

a covered region that is one piece with no holes, and then how far up the
redundancy scale that survives.  The smallest array reaching each level is the
design the paper reports.

The surface, the grid placement, the footprint model and the topology routines
are the ones of run_cortex.py; the only new thing here is the sweep and a
faster disk test, which works on the vertices of the smallest footprint of a
face rather than on masks the length of the whole mesh.

Writes results/design.json.
"""
import json, sys, time, itertools
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import ConvexHull

from kshadow import nerve_from_sets, shadow_betti, dropout_margin

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'; RES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- mesh
D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
POS = D['pos'].astype(float)
TRI = D['tri'].astype(np.int64)
SULC = D['sulc'].astype(float) if 'sulc' in D else np.zeros(len(POS))
NV = len(POS)
E = np.vstack([TRI[:, [0, 1]], TRI[:, [1, 2]], TRI[:, [2, 0]]])
E = np.unique(np.sort(E, axis=1), axis=0)
NE = len(E)
LEN = np.linalg.norm(POS[E[:, 0]] - POS[E[:, 1]], axis=1)
G = csr_matrix((np.r_[LEN, LEN], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
               shape=(NV, NV))

_key = E[:, 0] * NV + E[:, 1]
_order = np.argsort(_key); _keys = _key[_order]


def edge_ids(a, b):
    k = np.minimum(a, b) * NV + np.maximum(a, b)
    return _order[np.searchsorted(_keys, k)]


TE = np.c_[edge_ids(TRI[:, 0], TRI[:, 1]),
           edge_ids(TRI[:, 1], TRI[:, 2]),
           edge_ids(TRI[:, 2], TRI[:, 0])]
AREA = 0.5 * np.linalg.norm(np.cross(POS[TRI[:, 1]] - POS[TRI[:, 0]],
                                     POS[TRI[:, 2]] - POS[TRI[:, 0]]), axis=1)
HULLV = np.unique(ConvexHull(POS).vertices)


# ---------------------------------------------------------------- grids
def fit_sphere(points):
    A = np.c_[2 * points, np.ones(len(points))]
    b = np.sum(points ** 2, axis=1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    return c, np.sqrt(sol[3] + c @ c)


ANCHOR = (-58.0, -38.0, 38.0)          # the parietal site of run_cortex.py
SPAN = 70.0                            # the 8 x 8 array at 10 mm pitch spans 70 mm


def place_grid(pitch, span=SPAN, anchor=ANCHOR):
    """Square lattice of the given pitch covering a patch of side `span`,
    placed exactly as in run_cortex.py.  Lattice points that land on the same
    gyral vertex are merged, so the contact count is what the array actually
    has on the surface."""
    n = int(round(span / pitch)) + 1
    a = int(np.argmin(np.sum((POS - np.asarray(anchor)) ** 2, axis=1)))
    hull_near = HULLV[np.sum((POS[HULLV] - POS[a]) ** 2, axis=1) < 60.0 ** 2]
    c, R = fit_sphere(POS[hull_near])
    u = POS[a] - c; u /= np.linalg.norm(u)
    e_ant = np.array([0.0, 1.0, 0.0]); e_ant -= u * (e_ant @ u); e_ant /= np.linalg.norm(e_ant)
    e_sup = np.cross(u, e_ant); e_sup /= np.linalg.norm(e_sup)
    if e_sup[2] < 0:
        e_sup = -e_sup
    gyral = np.flatnonzero(SULC < 0)
    seen, idx = set(), []
    for i in range(n):
        for j in range(n):
            v = pitch * (j - (n - 1) / 2) * e_ant + pitch * (i - (n - 1) / 2) * e_sup
            rho = np.linalg.norm(v)
            p = c + R * (np.cos(rho / R) * u + (np.sin(rho / R) * v / rho if rho > 0 else 0))
            w = int(gyral[np.argmin(np.sum((POS[gyral] - p) ** 2, axis=1))])
            if w not in seen:
                seen.add(w)
                idx.append(w)
    return np.array(idx), n


# ------------------------------------------------- topology of a subcomplex
def topology(vsel, esel, fsel):
    """Components, Euler characteristic, closed components and b_1 over GF(2)
    of the subcomplex given by global vertex, edge and triangle index arrays."""
    v = len(vsel)
    if v == 0:
        return dict(components=0, chi=0, closed=0, b1=0, vertices=0, edges=0, faces=0)
    loc = {int(x): i for i, x in enumerate(vsel)}
    e, f = len(esel), len(fsel)
    if e:
        se = E[esel]
        r = np.fromiter((loc[int(x)] for x in se[:, 0]), int, e)
        cc = np.fromiter((loc[int(x)] for x in se[:, 1]), int, e)
        A = coo_matrix((np.ones(e), (r, cc)), shape=(v, v))
        ncomp, lab = connected_components(A, directed=False)
    else:
        ncomp, lab = v, np.arange(v)
    closed = 0
    if f:
        te = TE[fsel].ravel()
        uniq, cnt = np.unique(te, return_counts=True)
        bnd = uniq[cnt == 1]
        first = np.fromiter((loc[int(x)] for x in TRI[fsel][:, 0]), int, f)
        has_face = np.zeros(ncomp, bool); has_face[lab[first]] = True
        has_bnd = np.zeros(ncomp, bool)
        if len(bnd):
            bv = np.fromiter((loc[int(x)] for x in E[bnd][:, 0]), int, len(bnd))
            has_bnd[lab[bv]] = True
        closed = int((has_face & ~has_bnd).sum())
    chi = v - e + f
    return dict(components=int(ncomp), chi=int(chi), closed=closed,
                b1=int(ncomp - chi + closed), vertices=v, edges=e, faces=f)


def is_disk(t):
    return t['components'] == 1 and t['chi'] == 1 and t['closed'] == 0


def certified_level(N, kmax):
    """Largest k such that (b_0, b_1)(Delta_j(N)) = (1, 0) for every j <= k."""
    out, k = {}, 0
    for j in range(1, kmax + 1):
        b0, b1, _ = shadow_betti(N, j)
        out[j] = (b0, b1)
        if b0 == 1 and b1 == 0 and k == j - 1:
            k = j
    return k, out


# ---------------------------------------------------------------- the sweep
PITCHES = [float(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1
                              else ['10', '8', '6', '5', '4'])]
RADII = [float(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2
                            else ['8', '10', '12', '14', '15', '16', '18', '20'])]
KMAX = 4
MAXFACE = KMAX + 2

out = {'surface': dict(vertices=NV, edges=NE, triangles=len(TRI)),
       'anchor_mni': list(ANCHOR), 'span_mm': SPAN,
       'pitches_mm': PITCHES, 'radii_mm': RADII, 'kmax': KMAX, 'designs': []}

for pitch in PITCHES:
    idx, n_lat = place_grid(pitch)
    n_el = len(idx)
    print('== pitch %.1f mm: %d lattice points, %d distinct contacts'
          % (pitch, n_lat * n_lat, n_el), flush=True)
    t0 = time.time()
    dist = dijkstra(G, directed=False, indices=idx, limit=max(RADII) + 1e-9)
    print('   dijkstra %.0f s' % (time.time() - t0), flush=True)
    geo = []
    for i in range(n_el):
        d = np.sort(dist[i, idx])
        geo.append(d[1] if len(d) > 1 and np.isfinite(d[1]) else np.nan)
    geo = np.array(geo)
    for r in RADII:
        t1 = time.time()
        Vin = dist < r
        Fm = Vin[:, TRI].all(axis=2)
        if not Fm.any():
            continue
        Fidx = [np.flatnonzero(Fm[i]) for i in range(n_el)]
        Vidx, Eidx = [], []
        for i in range(n_el):
            Vidx.append(np.unique(TRI[Fidx[i]].ravel()))
            Eidx.append(np.unique(TE[Fidx[i]].ravel()))
        live = [i for i in range(n_el) if len(Fidx[i])]
        # mesh reference: the union, and the region covered at least twice
        cnt = np.zeros(len(TRI), np.int32)
        for i in live:
            cnt[Fidx[i]] += 1
        mesh = {}
        for k in (1, 2, 3):
            fsel = np.flatnonzero(cnt >= k)
            if len(fsel) == 0:
                mesh[k] = dict(components=0, b1=0, area=0.0)
                continue
            vsel = np.unique(TRI[fsel].ravel())
            esel = np.unique(TE[fsel].ravel())
            t = topology(vsel, esel, fsel)
            mesh[k] = dict(components=t['components'], b1=t['b1'],
                           area=float(AREA[fsel].sum()))
        # the nerve, from the vertex sets of the footprints
        vbits = []
        for i in range(n_el):
            m = np.zeros(NV, bool)
            m[Vidx[i]] = True
            vbits.append(int.from_bytes(np.packbits(m).tobytes(), 'big'))
        N = nerve_from_sets(vbits, max_size=MAXFACE)
        kstar, betti = certified_level(N, KMAX)
        q = dropout_margin(N, n_el, kmax=KMAX)
        # disk test, on the vertices of the smallest footprint of each face
        bad = 0
        for f in N:
            if len(f) == 1:
                i = next(iter(f))
                t = topology(Vidx[i], Eidx[i], Fidx[i])
            else:
                fl = sorted(f, key=lambda i: len(Vidx[i]))
                vs, es, fs = Vidx[fl[0]], Eidx[fl[0]], Fidx[fl[0]]
                for i in fl[1:]:
                    vs = np.intersect1d(vs, Vidx[i], assume_unique=True)
                    if len(vs) == 0:
                        break
                    es = np.intersect1d(es, Eidx[i], assume_unique=True)
                    fs = np.intersect1d(fs, Fidx[i], assume_unique=True)
                t = topology(vs, es, fs)
            if not is_disk(t):
                bad += 1
        rec = dict(pitch_mm=pitch, radius_mm=r, n_contacts=n_el,
                   lattice=n_lat, geodesic_nn_mm=float(np.nanmedian(geo)),
                   faces=len(N), max_face=int(max(len(f) for f in N)),
                   disk_test_failures=bad,
                   shadow={str(k): list(betti[k]) for k in betti},
                   mesh={str(k): mesh[k] for k in mesh},
                   certified_level=kstar, dropout_margin=q,
                   seconds=round(time.time() - t1, 1))
        out['designs'].append(rec)
        print('   r=%4.1f  n=%3d  faces=%6d  fail=%4d  Delta1=%s  mesh1=(%d,%d)  k*=%d  q=%d  [%ds]'
              % (r, n_el, len(N), bad, betti[1], mesh[1]['components'], mesh[1]['b1'],
                 kstar, q, time.time() - t1), flush=True)

json.dump(out, open(RES / 'design.json', 'w'), indent=1)
print('wrote results/design.json')
