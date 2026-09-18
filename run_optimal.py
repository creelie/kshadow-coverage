"""
run_optimal.py -- optimal coverage of a cortical patch, and its price.

Section IX.D of the paper measures two published arrays and finds that
neither certifies anything: the singly covered region is several pieces at
every radius tested.  The lattice sweep of run_design.py shows that no choice
of pitch or radius repairs this for a grid laid on the envelope of the
hemisphere, because the contacts sit on gyral crowns and the sulcal fundi are
geodesically far from every crown.

This script asks the design question directly.  On a fixed target patch of the
pial surface it builds contact sets by farthest-point sampling in the geodesic
metric of the surface, which is the greedy algorithm for the k-centre
problem, and reports

  rho_n   the covering radius of the first n contacts, exactly, by Dijkstra;
  the certificate at radius r for each n: the disk test on every face of the
  nerve, the Betti numbers of Delta_k(N), the mesh reference computed without
  the nerve, the certified level and the dropout margin.

Two candidate sets are used.  The free one allows a contact at any vertex of
the patch, which is what a conformable or sulcal array can do.  The crown one
allows a contact only where the FreeSurfer sulcal depth is negative, which is
what a subdural sheet can do.  The two answers are different, and the
difference is the point.

The greedy sequence also carries its own lower bound.  The first n points are
pairwise at least rho_{n-1} apart, so any n-1 points of the patch leave some
point of it at distance at least rho_{n-1}/2: no placement of n-1 contacts
beats the greedy one by more than a factor two.

Writes results/optimal.json.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from kshadow import nerve_from_sets, shadow_betti, dropout_margin

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'; RES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- mesh
D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
POS = D['pos'].astype(float)
TRI = D['tri'].astype(np.int64)
SULC = D['sulc'].astype(float)
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


# ------------------------------------------------- topology of a subcomplex
def topology(vsel, esel, fsel):
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


# ---------------------------------------------------------------- the patch
ANCHOR = (-58.0, -38.0, 38.0)
_argv = sys.argv[1:] if sys.argv[0].endswith('run_optimal.py') else []
PATCH_MM = float(_argv[0]) if len(_argv) > 0 else 35.0
NMAX = int(_argv[1]) if len(_argv) > 1 else 160

a = int(np.argmin(np.sum((POS - np.asarray(ANCHOR)) ** 2, axis=1)))
d_a = dijkstra(G, directed=False, indices=a)
PV = np.flatnonzero(d_a <= PATCH_MM)
inpatch = np.zeros(NV, bool); inpatch[PV] = True
PF = np.flatnonzero(inpatch[TRI].all(axis=1))
PE = np.unique(TE[PF].ravel())
PVf = np.unique(TRI[PF].ravel())
patch_top = topology(PVf, PE, PF)
PATCH_AREA = float(AREA[PF].sum())
print('patch: %d vertices, %d triangles, %.0f mm^2, %s'
      % (len(PVf), len(PF), PATCH_AREA,
         'a disk' if is_disk(patch_top) else 'NOT a disk: %s' % patch_top), flush=True)


# ------------------------------------------------- farthest-point sampling
def greedy(candidates, target, nmax, start=None):
    """Farthest-point sampling in the geodesic metric of the whole surface.
    `candidates` are the vertices a contact may occupy, `target` the vertices
    that have to be covered.  Returns the centres in order and rho_n, the
    covering radius of the first n of them, for every n."""
    cand = np.asarray(candidates)
    tgt = np.asarray(target)
    centres = [int(start if start is not None else cand[np.argmin(d_a[cand])])]
    dmin_t = np.full(len(tgt), np.inf)
    dmin_c = np.full(len(cand), np.inf)
    rho = []
    for step in range(nmax):
        d = dijkstra(G, directed=False, indices=centres[-1])
        dmin_t = np.minimum(dmin_t, d[tgt])
        dmin_c = np.minimum(dmin_c, d[cand])
        rho.append(float(dmin_t.max()))
        if step + 1 >= nmax:
            break
        nxt = int(cand[np.argmax(dmin_c)])
        if nxt in centres:
            break
        centres.append(nxt)
    return np.array(centres), rho


# ------------------------------------------------- certificate at (P, r)
KMAX = 4
MAXFACE = KMAX + 2


def certificate(centres, r, need_faces=True):
    dist = dijkstra(G, directed=False, indices=centres, limit=r + 1e-9)
    n = len(centres)
    Vin = dist < r
    Fm = Vin[:, TRI].all(axis=2)
    Fidx = [np.flatnonzero(Fm[i]) for i in range(n)]
    Vidx = [np.unique(TRI[f].ravel()) for f in Fidx]
    Eidx = [np.unique(TE[f].ravel()) for f in Fidx]
    cnt = np.zeros(len(TRI), np.int32)
    for f in Fidx:
        cnt[f] += 1
    mesh = {}
    for k in (1, 2, 3):
        fsel = np.flatnonzero(cnt >= k)
        if len(fsel) == 0:
            mesh[str(k)] = dict(components=0, b1=0, area=0.0)
            continue
        t = topology(np.unique(TRI[fsel].ravel()), np.unique(TE[fsel].ravel()), fsel)
        mesh[str(k)] = dict(components=t['components'], b1=t['b1'],
                            area=float(AREA[fsel].sum()))
    covered = np.zeros(len(TRI), bool); covered[np.flatnonzero(cnt >= 1)] = True
    uncovered = PF[~covered[PF]]
    vbits = []
    for i in range(n):
        m = np.zeros(NV, bool); m[Vidx[i]] = True
        vbits.append(int.from_bytes(np.packbits(m).tobytes(), 'big'))
    N = nerve_from_sets(vbits, max_size=MAXFACE)
    betti, kstar = {}, 0
    for j in range(1, KMAX + 1):
        b0, b1, _ = shadow_betti(N, j)
        betti[str(j)] = [b0, b1]
        if b0 == 1 and b1 == 0 and kstar == j - 1:
            kstar = j
    q = dropout_margin(N, n, kmax=KMAX)
    bad = []
    if need_faces:
        for f in N:
            if len(f) == 1:
                i = next(iter(f))
                t = topology(Vidx[i], Eidx[i], Fidx[i])
            else:
                fl = sorted(f, key=lambda i: len(Vidx[i]))
                vs, es, fs = Vidx[fl[0]], Eidx[fl[0]], Fidx[fl[0]]
                for i in fl[1:]:
                    vs = np.intersect1d(vs, Vidx[i], assume_unique=True)
                    es = np.intersect1d(es, Eidx[i], assume_unique=True)
                    fs = np.intersect1d(fs, Fidx[i], assume_unique=True)
                t = topology(vs, es, fs)
            if not is_disk(t):
                bad.append(sorted(int(x) for x in f))
    return dict(n_contacts=n, radius_mm=r, faces=len(N),
                max_face=int(max(len(f) for f in N)),
                disk_test_failures=len(bad),
                disk_test_examples=bad[:5],
                shadow=betti, mesh=mesh, certified_level=kstar, dropout_margin=q,
                uncovered_patch_triangles=int(len(uncovered)),
                uncovered_patch_area=float(AREA[uncovered].sum()),
                patch_area=PATCH_AREA)


# ---------------------------------------------------------------- run
out = dict(anchor_mni=list(ANCHOR), patch_radius_mm=PATCH_MM,
           patch=dict(vertices=int(len(PVf)), triangles=int(len(PF)),
                      area_mm2=PATCH_AREA, is_disk=bool(is_disk(patch_top)),
                      **{k: patch_top[k] for k in ('components', 'chi', 'closed', 'b1')}),
           kmax=KMAX, families={})

gyral = np.flatnonzero((SULC < 0) & inpatch)

if __name__ == '__main__':
  for name, cand in (('free', PVf), ('crown', gyral)):
    t0 = time.time()
    centres, rho = greedy(cand, PVf, NMAX)
    print('%s: %d centres, rho_1=%.1f rho_%d=%.2f mm  [%.0f s]'
          % (name, len(centres), rho[0], len(rho), rho[-1], time.time() - t0), flush=True)
    out['families'][name] = dict(n_candidates=int(len(cand)),
                                 centres=centres.tolist(),
                                 centre_mni=POS[centres].round(2).tolist(),
                                 rho_mm=[round(x, 4) for x in rho],
                                 certificates=[])

  json.dump(out, open(RES / 'optimal.json', 'w'), indent=1)
  print('wrote results/optimal.json (greedy stage)')
