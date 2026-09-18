"""
run_anisotropic.py -- isotropic against anisotropic footprints on the same
5 x 5 centres at 8 mm pitch: circular footprints of radius 6 mm, and
elliptical footprints with semi-axes a in [5, 8] mm, b in [3, 5.5] mm and
orientations in [-45, 45] degrees drawn once from a fixed seed.

Ellipses are convex, so the nerve is determined by its triangles (Helly).
A set of ellipses has a common interior point if and only if
    t* = min_x max_i f_i(x) < 1,   f_i(x) = ((x-c_i) . u_i / a_i)^2 + ((x-c_i) . v_i / b_i)^2,
a convex minimax problem solved here in epigraph form by sequential
quadratic programming from several starting points; the margin |t* - 1| is
recorded for every pair and triple, and the smallest margin over all tests
is reported so that a near-tangency would be visible.  The Betti numbers of
Delta_k(N) for k = 1, 2, 3, the dropout margin, and a raster of the
multiplicity function at two resolutions (ground truth) are computed by the
library.  Writes results/anisotropic.json and figures/anisotropic.png.
"""
import json, itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from kshadow import shadow_betti, dropout_margin, raster_betti

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'; RES.mkdir(exist_ok=True)
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
xs = np.linspace(4, 36, 5)
centres = np.array([(x, y) for y in xs for x in xs])
n = len(centres)
iso = dict(a=np.full(n, 6.0), b=np.full(n, 6.0), ang=np.zeros(n))
aniso = dict(a=rng.uniform(5.0, 8.0, n), b=rng.uniform(3.0, 5.5, n), ang=rng.uniform(-np.pi / 4, np.pi / 4, n))


def quad(model, i, x):
    d = x - centres[i]
    c, s = np.cos(model['ang'][i]), np.sin(model['ang'][i])
    u = c * d[0] + s * d[1]; v = -s * d[0] + c * d[1]
    return (u / model['a'][i]) ** 2 + (v / model['b'][i]) ** 2


def common_point(model, idx):
    """t* = min_x max_{i in idx} f_i(x), and the minimiser."""
    idx = list(idx)
    best = None
    starts = [centres[idx].mean(axis=0)] + [centres[i] for i in idx]
    for x0 in starts:
        t0 = max(quad(model, i, x0) for i in idx)
        z0 = np.r_[x0, t0]
        cons = [{'type': 'ineq', 'fun': (lambda z, i=i: z[2] - quad(model, i, z[:2]))} for i in idx]
        res = minimize(lambda z: z[2], z0, constraints=cons, method='SLSQP', options=dict(ftol=1e-12, maxiter=500))
        t = max(quad(model, i, res.x[:2]) for i in idx)   # certified value at the returned point
        if best is None or t < best[0]:
            best = (t, res.x[:2])
    return best


def nerve(model):
    faces = {frozenset([i]) for i in range(n)}
    margins = []
    adj = {i: set() for i in range(n)}
    for i, j in itertools.combinations(range(n), 2):
        if np.linalg.norm(centres[i] - centres[j]) > model['a'][i] + model['a'][j] + 1e-9:
            continue          # bounding circles are disjoint
        t, _ = common_point(model, (i, j))
        margins.append(abs(t - 1))
        if t < 1:
            faces.add(frozenset([i, j])); adj[i].add(j); adj[j].add(i)
    for i, j, k in itertools.combinations(range(n), 3):
        if j in adj[i] and k in adj[i] and k in adj[j]:
            t, _ = common_point(model, (i, j, k))
            margins.append(abs(t - 1))
            if t < 1:
                faces.add(frozenset([i, j, k]))
    # Helly: a larger set is a face iff all its triples are
    tri = {f for f in faces if len(f) == 3}
    level = list(tri); size = 3
    while level:
        nxt = set()
        for f in level:
            for v in set.intersection(*(adj[x] for x in f)):
                if v > max(f):
                    g = f | {v}
                    if all(frozenset(c) in tri for c in itertools.combinations(sorted(g), 3)):
                        nxt.add(g)
        faces |= nxt; level = list(nxt); size += 1
    return faces, min(margins) if margins else None


def multiplicity(model, res):
    g = np.linspace(0, 40, res)
    X, Y = np.meshgrid(g, g)
    m = np.zeros_like(X, dtype=int)
    for i in range(n):
        d0, d1 = X - centres[i, 0], Y - centres[i, 1]
        c, s = np.cos(model['ang'][i]), np.sin(model['ang'][i])
        u = c * d0 + s * d1; v = -s * d0 + c * d1
        m += ((u / model['a'][i]) ** 2 + (v / model['b'][i]) ** 2 < 1)
    return m, g


out = {'centres': centres.tolist(), 'pitch_mm': 8.0, 'seed': 42}
for name, model in (('isotropic', iso), ('anisotropic', aniso)):
    N, margin = nerve(model)
    sizes = {}
    for f in N:
        sizes[len(f)] = sizes.get(len(f), 0) + 1
    rec = dict(a=model['a'].round(4).tolist(), b=model['b'].round(4).tolist(), angle_deg=np.degrees(model['ang']).round(3).tolist(),
               faces_by_size=[sizes.get(s, 0) for s in range(1, max(sizes) + 1)], least_margin=margin, shadow={}, raster={})
    for k in (1, 2, 3):
        b0, b1, (nv, ne, nt) = shadow_betti(N, k)
        rec['shadow'][k] = dict(b0=b0, b1=b1, V=nv, E=ne, T=nt)
    for res in (800, 1600):
        m, _ = multiplicity(model, res)
        rec['raster'][res] = {k: list(raster_betti(m >= k)) for k in (1, 2, 3)}
    rec['dropout_margin'] = dropout_margin(N, n)
    out[name] = rec
    print(name, rec['faces_by_size'], 'least margin %.3g' % margin, rec['shadow'], rec['raster'], 'dropout', rec['dropout_margin'])
(RES / 'anisotropic.json').write_text(json.dumps(out, indent=1))

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
vmax = 0
fields = {}
for name, model in (('isotropic', iso), ('anisotropic', aniso)):
    m, g = multiplicity(model, 600)
    fields[name] = (m, g); vmax = max(vmax, m.max())
for ax, (name, model), lab in zip(axes, (('isotropic', iso), ('anisotropic', aniso)), ('(a)', '(b)')):
    m, g = fields[name]
    im = ax.pcolormesh(g, g, m, cmap='viridis', vmin=0, vmax=vmax, shading='nearest', rasterized=True)
    for i in range(n):
        ax.add_patch(Ellipse(centres[i], 2 * model['a'][i], 2 * model['b'][i], angle=np.degrees(model['ang'][i]),
                             fc='none', ec='white', lw=0.5, alpha=0.8))
    N, _ = nerve(model)
    for f in N:
        if len(f) == 2:
            i, j = tuple(f)
            ax.plot(*zip(centres[i], centres[j]), color='#ff7f0e', lw=0.6, alpha=0.9)
    ax.plot(centres[:, 0], centres[:, 1], 'o', ms=2.5, color='white', mec='k', mew=0.3)
    r = out[name]
    ax.set_title('%s %s: $(b_0,b_1)=(%d,%d)$' % (
        lab, name, r['shadow'][1]['b0'], r['shadow'][1]['b1']), fontsize=8)
    ax.set_aspect('equal'); ax.set_xlim(0, 40); ax.set_ylim(0, 40)
    ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$y$ (mm)')
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, ticks=range(0, vmax + 1))
cb.set_label('multiplicity $m(x)$')
fig.savefig(FIG / 'anisotropic.png', dpi=400, bbox_inches='tight')
print('figures/anisotropic.png')
