"""Electrode array on the unit sphere with geodesic-disk (spherical cap)
footprints.  The nerve is computed exactly from spherical geometry (minimum
enclosing cap of the centres, Helly number 3 inside an open hemisphere via
the gnomonic projection); the ground truth is a raster of the multiplicity
function in the gnomonic chart, which is a homeomorphism of the open
hemisphere onto the plane.  Writes results/sphere.json, figures/sphere_*.png."""
import json, os, sys, time, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kshadow import (HellyNerve, shadow_betti, subdivision_persistence,
                     betti_from_bars, dropout_margin, raster_betti)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
rng = np.random.default_rng(7)


def to_sphere(u, v):
    p = np.stack([u, v, np.ones_like(u)], axis=-1)
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


def gdist(p, q):
    return np.arccos(np.clip(np.sum(p * q, axis=-1), -1.0, 1.0))


def min_cap_radius(pts):
    """Radius of the smallest spherical cap containing the given unit
    vectors (2 or 3 points inside an open hemisphere)."""
    if len(pts) == 2:
        return 0.5 * gdist(pts[0], pts[1])
    a, b, c = pts
    best = np.inf
    for p, q, s in ((a, b, c), (a, c, b), (b, c, a)):
        mid = p + q
        mid = mid / np.linalg.norm(mid)
        rad = 0.5 * gdist(p, q)
        if gdist(mid, s) <= rad + 1e-12:
            best = min(best, rad)
    if best < np.inf:
        return best
    nrm = np.cross(b - a, c - a)
    nn = np.linalg.norm(nrm)
    if nn < 1e-14:
        return max(gdist(a, b), gdist(b, c), gdist(a, c)) / 2
    ctr = nrm / nn
    rad = gdist(ctr, a)
    return min(rad, np.pi - rad)


def nerve_of_caps(centers, r):
    n = len(centers)
    faces = set(frozenset([i]) for i in range(n))
    adj = {i: set() for i in range(n)}
    for i, j in itertools.combinations(range(n), 2):
        if gdist(centers[i], centers[j]) < 2 * r:
            faces.add(frozenset([i, j])); adj[i].add(j); adj[j].add(i)
    tri = set()
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k > j and min_cap_radius(centers[[i, j, k]]) < r:
                    tri.add(frozenset([i, j, k]))
    faces |= tri
    return HellyNerve(faces, tri, adj, None)


# electrodes: jittered 8x8 grid in the gnomonic chart
g = np.linspace(-0.55, 0.55, 8)
uv = np.array([(a, b) for a in g for b in g]) + rng.uniform(-0.02, 0.02, (64, 2))
C = to_sphere(uv[:, 0], uv[:, 1])
n = len(C)
max_colat = float(np.degrees(np.arccos(C[:, 2]).max()))

# raster in the chart at two resolutions
L = 1.0
RES = (1000, 3000)
grids = {}
for res in RES:
    xs = np.linspace(-L, L, res)
    GX, GY = np.meshgrid(xs, xs)
    grids[res] = to_sphere(GX, GY)

out = {'n': n, 'chart_half_width': L, 'max_colatitude_deg': max_colat, 'runs': {}}
for r in (0.10, 0.13, 0.16):
    t0 = time.time()
    N = nerve_of_caps(C, r)
    sizes = {}
    for s in N:
        sizes[len(s)] = sizes.get(len(s), 0) + 1
    mults = {}
    for res in RES:
        mm = np.zeros((res, res), dtype=np.int16)
        for i in range(n):
            mm += (gdist(grids[res], C[i]) < r)
        mults[res] = mm
    rec = {'r': r, 'cap_edge_max_colatitude_deg': max_colat + np.degrees(r),
           'face_counts': {str(k): v for k, v in sorted(sizes.items())},
           'shadow': {}, 'raster': {str(res): {} for res in RES}}
    kmax = max(sizes)
    for k in range(1, kmax + 2):
        b0, b1, sz = shadow_betti(N, k)
        rec['shadow'][str(k)] = {'b0': b0, 'b1': b1, 'V': sz[0], 'E': sz[1], 'T': sz[2]}
        for res in RES:
            rb0, rb1 = raster_betti(mults[res] >= k)
            rec['raster'][str(res)][str(k)] = {'b0': rb0, 'b1': rb1}
    mult = mults[RES[0]]
    bars = subdivision_persistence(N)
    rec['bars'] = {str(d): bars[d] for d in bars}
    rec['bars_betti'] = {str(k): betti_from_bars(bars, k) for k in range(1, kmax + 2)}
    rec['dropout_margin'] = dropout_margin(N, n)
    rec['agree'] = {str(res): [k for k in rec['shadow']
                               if (rec['shadow'][k]['b0'], rec['shadow'][k]['b1']) ==
                               (rec['raster'][str(res)][k]['b0'], rec['raster'][str(res)][k]['b1'])]
                    for res in RES}
    rec['time'] = time.time() - t0
    out['runs'][str(r)] = rec
    print(r, json.dumps({k: v for k, v in rec.items() if k != 'bars'}), flush=True)

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    im = ax[0].imshow(mult, origin='lower', extent=(-L, L, -L, L), cmap='viridis', interpolation='nearest')
    ax[0].plot(uv[:, 0], uv[:, 1], 'k.', ms=3)
    ax[0].set_title('multiplicity, cap radius $r=%.2f$' % r)
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    for j, k in enumerate((1, 2, 3)):
        ax[j + 1].imshow(mult >= k, origin='lower', extent=(-L, L, -L, L), cmap='Greys', interpolation='nearest')
        ax[j + 1].plot(uv[:, 0], uv[:, 1], 'r.', ms=3)
        s = rec['shadow'][str(k)]
        ax[j + 1].set_title(r'$R_{\geq %d}$: $b_0=%d$, $b_1=%d$' % (k, s['b0'], s['b1']))
    for a in ax:
        a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig('figures/sphere_r%03d.png' % int(round(100 * r)), dpi=200)
    plt.close()

# 3D view
fig = plt.figure(figsize=(5.6, 5.0))
ax = fig.add_subplot(111, projection='3d')
th = np.linspace(0, np.pi / 2.2, 40); ph = np.linspace(0, 2 * np.pi, 80)
TH, PH = np.meshgrid(th, ph)
ax.plot_surface(np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH),
                color='lightsteelblue', alpha=0.6, linewidth=0)
ax.scatter(C[:, 0], C[:, 1], C[:, 2] + 0.01, c='crimson', s=12)
ax.set_box_aspect((1, 1, 0.9)); ax.view_init(elev=40, azim=-60)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
plt.tight_layout(); plt.savefig('figures/sphere_3d.png', dpi=200); plt.close()

json.dump(out, open('results/sphere.json', 'w'), indent=1, default=str)
print('done')
