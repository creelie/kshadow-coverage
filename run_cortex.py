"""
run_cortex.py -- the certificate on a real cortical surface.

Surface: the left pial surface of the FieldTrip template anatomy (colin27,
FreeSurfer reconstruction, MNI coordinates in mm; data/colin27_lh_pial.npz,
extracted by fetch_template_surface.py).  The script first checks that the
mesh is a closed combinatorial surface (every edge in exactly two triangles,
every vertex link a single cycle) and records its Euler characteristic.

Arrays: an 8 x 8 grid and a 4 x 8 grid, both at 10 mm pitch, on the lateral
parietal and the lateral temporal convexity, the layout documented for the
FieldTrip SubjectUCI29 tutorial.  A grid is placed by fitting a sphere to
the convex hull of the hemisphere near an anchor point, laying the lattice
out on that sphere by the exponential map (neighbours exactly 10 mm apart
along the envelope, as on a subdural sheet) and assigning every lattice point
the nearest gyral pial vertex (FreeSurfer sulcal depth below zero).

Footprints: the discrete geodesic disk of radius r of a contact is the
subcomplex of the mesh formed by the closed triangles all of whose vertices
are within graph distance r of the contact vertex (graph distance = shortest
path in the edge graph with Euclidean edge lengths, Dijkstra).  A second,
deliberately naive model uses Euclidean distance in place of graph distance.

For each grid, model and radius the script computes
  - the nerve N of the footprints (a set of contacts is a face iff their
    footprints share a vertex),
  - the Betti numbers of Delta_k(N) for every k <= KMAX, the redundancy
    barcode of Sub_k(N) and the dropout margin (library kshadow.py),
  - the disk test on every face sigma of N: whether the intersection of the
    footprints of sigma is connected, has Euler characteristic 1 and is not a
    closed component of the surface (Lemma "disk test" of the paper), which
    is exactly the hypothesis of the k-shadow nerve theorem on the mesh,
  - the mesh reference: for every k, b0 and b1 of the subcomplex R_k of all
    simplices contained in at least k footprints, computed directly,
  - the Gaussian curvature of the surface under the array (angle deficit) and
    the bound pi/(2 sqrt K) of the geodesic theorem.

Writes results/cortex.json and results/cortex_dist_<grid>.{npy,npz} (figures
are drawn by make_cortex_figures.py).  Runtime about three minutes.
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import ConvexHull

from kshadow import (nerve_from_sets, shadow_betti, subdivision_persistence,
                     betti_from_bars, dropout_margin)
import distcache

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'; RES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- mesh
D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
POS = D['pos'].astype(float)
TRI = D['tri'].astype(np.int64)
NV = len(POS)
E = np.vstack([TRI[:, [0, 1]], TRI[:, [1, 2]], TRI[:, [2, 0]]])
E = np.unique(np.sort(E, axis=1), axis=0)
NE = len(E)
LEN = np.linalg.norm(POS[E[:, 0]] - POS[E[:, 1]], axis=1)
G = csr_matrix((np.r_[LEN, LEN], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(NV, NV))

# edge index of every triangle edge
_key = E[:, 0] * NV + E[:, 1]
_order = np.argsort(_key); _keys = _key[_order]
def edge_ids(a, b):
    k = np.minimum(a, b) * NV + np.maximum(a, b)
    return _order[np.searchsorted(_keys, k)]
TE = np.stack([edge_ids(TRI[:, 0], TRI[:, 1]), edge_ids(TRI[:, 1], TRI[:, 2]), edge_ids(TRI[:, 2], TRI[:, 0])], axis=1)


def surface_check():
    """Every edge in exactly two triangles, every vertex link a single cycle,
    Euler characteristic of the whole mesh."""
    cnt = np.bincount(TE.ravel(), minlength=NE)
    two = bool((cnt == 2).all())
    # link connectivity: nodes are the directed edges (v -> w); a triangle
    # (v, a, b) joins (v -> a) with (v -> b); the links are single cycles iff
    # the number of connected components equals the number of vertices
    def nid(v, w):
        return v * NV + w
    src, dst = [], []
    for k in range(3):
        v = TRI[:, k]; a = TRI[:, (k + 1) % 3]; b = TRI[:, (k + 2) % 3]
        src.append(nid(v, a)); dst.append(nid(v, b))
    src = np.concatenate(src); dst = np.concatenate(dst)
    ids, inv = np.unique(np.r_[src, dst], return_inverse=True)
    A = coo_matrix((np.ones(len(src)), (inv[:len(src)], inv[len(src):])), shape=(len(ids), len(ids)))
    ncomp = connected_components(A, directed=False, return_labels=False)
    chi = NV - NE + len(TRI)
    return dict(edges_in_two_triangles=two, vertex_links_single_cycles=bool(ncomp == NV),
                euler_characteristic=int(chi))


SURF = surface_check()
print('surface check', SURF, flush=True)
assert SURF['edges_in_two_triangles'] and SURF['vertex_links_single_cycles']


# vertex normals (area weighted) and Gaussian curvature by angle deficit
def gaussian_curvature():
    deficit = np.full(NV, 2 * np.pi)
    area = np.zeros(NV)
    a, b, c = POS[TRI[:, 0]], POS[TRI[:, 1]], POS[TRI[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    for k in range(3):
        p = POS[TRI[:, k]]; q = POS[TRI[:, (k + 1) % 3]]; s = POS[TRI[:, (k + 2) % 3]]
        u = q - p; v = s - p
        cosang = np.sum(u * v, axis=1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
        np.add.at(deficit, TRI[:, k], -np.arccos(np.clip(cosang, -1, 1)))
        np.add.at(area, TRI[:, k], area2 / 6.0)
    return deficit / np.maximum(area, 1e-12)


KGAUSS = gaussian_curvature()
SULC = D['sulc'].astype(float)
HULLV = np.unique(ConvexHull(POS).simplices)


# ---------------------------------------------------------------- grids
def fit_sphere(points):
    """Least-squares sphere through a point cloud: centre c and radius R."""
    A = np.c_[2 * points, np.ones(len(points))]
    b = np.sum(points ** 2, axis=1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    return c, np.sqrt(sol[3] + c @ c)


def place_grid(anchor, rows, cols, pitch=10.0):
    """Lattice of the given pitch on the lateral convexity: sphere fitted to
    the convex-hull vertices within 60 mm of the anchor, lattice laid out on
    the sphere by the exponential map at the anchor direction, every lattice
    point assigned the nearest gyral pial vertex.  Returns the vertex indices,
    the anchor vertex, the sphere and the lattice points on it."""
    a = int(np.argmin(np.sum((POS - np.asarray(anchor)) ** 2, axis=1)))
    hull_near = HULLV[np.sum((POS[HULLV] - POS[a]) ** 2, axis=1) < 60.0 ** 2]
    c, R = fit_sphere(POS[hull_near])
    u = POS[a] - c; u /= np.linalg.norm(u)
    e_ant = np.array([0.0, 1.0, 0.0]); e_ant -= u * (e_ant @ u); e_ant /= np.linalg.norm(e_ant)
    e_sup = np.cross(u, e_ant); e_sup /= np.linalg.norm(e_sup)
    if e_sup[2] < 0:
        e_sup = -e_sup
    gyral = np.flatnonzero(SULC < 0)
    idx, sphere_pts = [], []
    for i in range(rows):
        for j in range(cols):
            v = pitch * (j - (cols - 1) / 2) * e_ant + pitch * (i - (rows - 1) / 2) * e_sup
            rho = np.linalg.norm(v)
            p = c + R * (np.cos(rho / R) * u + (np.sin(rho / R) * v / rho if rho > 0 else 0))
            sphere_pts.append(p)
            idx.append(int(gyral[np.argmin(np.sum((POS[gyral] - p) ** 2, axis=1))]))
    return np.array(idx), a, dict(centre=c.round(3).tolist(), radius=float(R)), np.array(sphere_pts)


GRIDS = {
    'parietal_8x8': dict(anchor=(-58.0, -38.0, 38.0), rows=8, cols=8),
    'temporal_4x8': dict(anchor=(-66.0, -22.0, -8.0), rows=4, cols=8),
}
RADII = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
EUC_RADII = [r for r in RADII if r <= 12.0]        # the Euclidean model at 15 mm has faces of size ten and above; its chain complex does not fit in memory
KMAX = 6


# ---------------------------------------------------------------- topology of subcomplexes
def subcomplex_topology(vm, em, fm):
    """Components, Euler characteristic, number of closed components and b1
    (over GF(2)) of the subcomplex of the mesh with vertex mask vm, edge mask
    em and triangle mask fm.  A component is closed if it contains triangles
    and no edge of it lies in fewer than two of its triangles; b2 equals the
    number of closed components and b1 = c - chi + b2."""
    v = int(vm.sum())
    if v == 0:
        return dict(components=0, chi=0, closed=0, b1=0, vertices=0, edges=0, faces=0, sizes=[])
    e = int(em.sum()); f = int(fm.sum())
    loc = np.full(NV, -1); loc[np.flatnonzero(vm)] = np.arange(v)
    sub = E[em]
    A = coo_matrix((np.ones(len(sub)), (loc[sub[:, 0]], loc[sub[:, 1]])), shape=(v, v))
    c, lab = connected_components(A, directed=False)
    closed = 0
    if f:
        te = TE[fm].ravel()
        cnt = np.bincount(te, minlength=NE)
        bnd = np.flatnonzero((cnt == 1))                 # boundary edges of the kept triangles
        has_face = np.zeros(c, bool); has_face[lab[loc[TRI[fm][:, 0]]]] = True
        has_bnd = np.zeros(c, bool)
        if len(bnd):
            has_bnd[lab[loc[E[bnd][:, 0]]]] = True
        closed = int((has_face & ~has_bnd).sum())
    chi = v - e + f
    sizes = sorted(np.bincount(lab).tolist(), reverse=True)
    return dict(components=int(c), chi=int(chi), closed=closed, b1=int(c - chi + closed),
                vertices=v, edges=e, faces=f, sizes=sizes[:6])


def is_disk(t):
    return t['components'] == 1 and t['chi'] == 1 and t['closed'] == 0


# ---------------------------------------------------------------- main
results = {'surface': dict(source=str(D['source']), commit=str(D['source_commit']),
                           vertices=NV, edges=NE, triangles=len(TRI), coordsys=str(D['coordsys']),
                           unit=str(D['unit']), **SURF),
           'radii_mm': RADII, 'euclidean_radii_mm': EUC_RADII, 'kmax': KMAX, 'grids': {}}
t_all = time.time()
for gname, spec in GRIDS.items():
    print('==', gname, flush=True)
    idx, anchor_v, sphere, sphere_pts = place_grid(spec['anchor'], spec['rows'], spec['cols'])
    n_el = len(idx)
    assert len(set(idx.tolist())) == n_el, 'two contacts fell on the same vertex'
    dist = dijkstra(G, directed=False, indices=idx, limit=max(RADII) + 1e-9)
    rows, cols = spec['rows'], spec['cols']
    pitch_geo, pitch_euc = [], []
    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            for nb in ([k + 1] if j + 1 < cols else []) + ([k + cols] if i + 1 < rows else []):
                pitch_geo.append(dist[k, idx[nb]])
                pitch_euc.append(np.linalg.norm(POS[idx[k]] - POS[idx[nb]]))
    pitch_geo = np.array(pitch_geo); pitch_euc = np.array(pitch_euc)
    n_pairs = len(pitch_geo)
    pitch_geo = pitch_geo[np.isfinite(pitch_geo)]
    under = np.flatnonzero(np.isfinite(dist).any(axis=0))       # vertices within the largest radius of the array
    Kloc = KGAUSS[under]
    curv = dict(vertices_under_array=int(len(under)),
                K_median=float(np.median(np.abs(Kloc))),
                K_p95=float(np.percentile(np.abs(Kloc), 95)),
                K_max=float(np.abs(Kloc).max()),
                Kpos_p95=float(np.percentile(np.maximum(Kloc, 0), 95)),
                bound_p95_mm=float(np.pi / (2 * np.sqrt(max(np.percentile(np.maximum(Kloc, 0), 95), 1e-12)))),
                bound_max_mm=float(np.pi / (2 * np.sqrt(max(Kloc.max(), 1e-12)))))
    dist_euc = np.linalg.norm(POS[None, :, :] - POS[idx][:, None, :], axis=2)
    grid_out = dict(n_electrodes=n_el, rows=rows, cols=cols, pitch_mm=10.0, anchor_mni=spec['anchor'],
                    anchor_vertex=int(anchor_v), fitted_sphere=sphere,
                    contact_vertices=idx.tolist(), contact_mni=POS[idx].round(3).tolist(),
                    euclidean_pitch_mm=dict(min=float(pitch_euc.min()), median=float(np.median(pitch_euc)),
                                            max=float(pitch_euc.max()), n=int(len(pitch_euc))),
                    geodesic_pitch_mm=dict(min=float(pitch_geo.min()), median=float(np.median(pitch_geo)),
                                           max=float(pitch_geo.max()), within_limit=int(len(pitch_geo)),
                                           beyond_limit=int(n_pairs - len(pitch_geo)), limit_mm=max(RADII)),
                    curvature=curv, models={})
    for model, DM in (('geodesic', dist), ('euclidean', dist_euc)):
        per_radius = []
        for r in (RADII if model == 'geodesic' else EUC_RADII):
            t0 = time.time()
            Vin = DM < r                                           # n_el x vertices within distance r
            Fm = Vin[:, TRI].all(axis=2)                           # n_el x triangles: triangles fully inside
            Vm = np.zeros((n_el, NV), bool); Em = np.zeros((n_el, NE), bool)
            for i in range(n_el):
                Vm[i, TRI[Fm[i]].ravel()] = True
                Em[i, TE[Fm[i]].ravel()] = True
            bits = [int.from_bytes(np.packbits(Vm[i]).tobytes(), 'big') for i in range(n_el)]
            N = nerve_from_sets(bits)
            sizes = {}
            for f in N:
                sizes[len(f)] = sizes.get(len(f), 0) + 1
            top = max(sizes)
            # disk test on every face
            bad = []
            for f in N:
                vm = np.ones(NV, bool); em = np.ones(NE, bool); fm = np.ones(len(TRI), bool)
                for i in f:
                    vm &= Vm[i]; em &= Em[i]; fm &= Fm[i]
                t = subcomplex_topology(vm, em, fm)
                if not is_disk(t):
                    bad.append(dict(face=sorted(f), **t))
            # k-shadow Betti numbers and the barcode
            betti = {}
            for k in range(1, min(KMAX, top) + 1):
                b0, b1, (nv, ne, nt) = shadow_betti(N, k)
                betti[k] = dict(b0=b0, b1=b1, V=nv, E=ne, T=nt)
            bars = subdivision_persistence(N)
            from_bars = {k: [betti_from_bars(bars, k)[0], betti_from_bars(bars, k)[1]] for k in range(1, min(KMAX, top) + 1)}
            # mesh reference: the subcomplex R_k of all simplices contained in at least k footprints
            vmult = Vm.sum(axis=0); emult = Em.sum(axis=0); fmult = Fm.sum(axis=0)
            truth = {}
            for k in range(1, min(KMAX, top) + 1):
                t = subcomplex_topology(vmult >= k, emult >= k, fmult >= k)
                truth[k] = [t['components'], t['b1'], t['closed']]
            q = dropout_margin(N, n_el)
            certified = 0
            for k in range(1, min(KMAX, top) + 1):
                if betti[k]['b0'] == 1 and betti[k]['b1'] == 0:
                    certified = k
                else:
                    break
            rec = dict(radius_mm=r, faces_by_size=[sizes.get(s, 0) for s in range(1, top + 1)],
                       disk_test_failures=len(bad), failures=bad,
                       failures_without_triangle=sum(1 for b in bad if b['faces'] == 0),
                       failures_second_component_at_most_3_vertices=sum(
                           1 for b in bad if b['components'] >= 2 and b['sizes'][1] <= 3),
                       shadow=betti, barcode_betti=from_bars,
                       barcode={str(d): [[int(lo), int(hi)] for (lo, hi) in bars[d]] for d in bars},
                       mesh_reference=truth, dropout_margin=q, certified_level=certified,
                       covered_vertices=int((vmult >= 1).sum()), covered_triangles=int((fmult >= 1).sum()),
                       agree={k: (betti[k]['b0'] == truth[k][0] and betti[k]['b1'] == truth[k][1]) for k in betti},
                       time_s=round(time.time() - t0, 2))
            per_radius.append(rec)
            print('  %s r=%4.1f faces %s fail %d margin %d cert %d shadow %s mesh %s (%.1f s)' % (
                model, r, rec['faces_by_size'], len(bad), q, certified,
                {k: (v['b0'], v['b1']) for k, v in betti.items()},
                {k: (v[0], v[1]) for k, v in truth.items()}, rec['time_s']), flush=True)
        grid_out['models'][model] = per_radius
    results['grids'][gname] = grid_out
    distcache.save(RES, gname, dist)

results['total_time_s'] = round(time.time() - t_all, 1)
(RES / 'cortex.json').write_text(json.dumps(results, indent=1))
print('wrote results/cortex.json in %.0f s' % (time.time() - t_all))
