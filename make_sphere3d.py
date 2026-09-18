"""
make_sphere3d.py -- three-dimensional views of the spherical-cap arrays of
run_sphere.py: the sixty-four centres on the unit sphere (the same jittered
8 x 8 gnomonic grid, seed 7) with the multiplicity of the caps of radius
r = 0.13 and r = 0.16 drawn on a fine triangulation of the sphere, lit and
depth-sorted.  Writes figures/sphere_3d.png.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors as mcolors

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
g = np.linspace(-0.55, 0.55, 8)
uv = np.array([(a, b) for a in g for b in g]) + rng.uniform(-0.02, 0.02, (64, 2))
C = np.stack([uv[:, 0], uv[:, 1], np.ones(64)], axis=1)
C /= np.linalg.norm(C, axis=1, keepdims=True)

# a fine triangulation of the sphere (icosahedron subdivided)
def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(level):
        cache = {}; v = list(v); nf = []
        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = v[a] + v[b]; p /= np.linalg.norm(p); v.append(p); cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(v); f = np.array(nf)
    return v, f

V, F = icosphere(6)
LIGHT = np.array([0.4, -0.5, 0.75]); LIGHT /= np.linalg.norm(LIGHT)
CMAP = plt.get_cmap('viridis')
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

fig = plt.figure(figsize=(7.0, 3.6))
vmax = 6
for k, r in enumerate((0.13, 0.16)):
    ax = fig.add_subplot(1, 2, k + 1, projection='3d')
    ax.computed_zorder = False
    ang = np.arccos(np.clip(V @ C.T, -1, 1))            # geodesic distances to the centres
    mult = (ang < r).sum(axis=1)
    fm = mult[F].min(axis=1)
    base = np.tile(np.array([0.82, 0.82, 0.84]), (len(F), 1))
    base[fm > 0] = CMAP(np.clip(fm[fm > 0], 0, vmax) / vmax)[:, :3]
    nrm = V[F].mean(axis=1); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    fac = 0.35 + 0.65 * np.clip(nrm @ LIGHT, 0, 1)
    cols = np.clip(base * fac[:, None], 0, 1)
    # draw only the front hemisphere with respect to the camera
    cam = np.array([0.55, -0.6, 0.58]); cam /= np.linalg.norm(cam)
    keep = nrm @ cam > -0.05
    pc = Poly3DCollection(V[F[keep]], facecolors=cols[keep], edgecolors=cols[keep], linewidths=0.05, antialiased=False)
    ax.add_collection3d(pc)
    P = C * 1.02
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=7, c='#d62728', edgecolors='k', linewidths=0.2, depthshade=False, zorder=10)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=np.degrees(np.arcsin(cam[2])), azim=np.degrees(np.arctan2(cam[1], cam[0])))
    ax.set_title('(%s) caps of radius $r=%.2f$' % ('ab'[k], r), pad=-6)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0, vmax)); sm.set_array([])
cb = fig.colorbar(sm, ax=fig.axes, fraction=0.025, pad=0.01, shrink=0.6, ticks=range(0, vmax + 1))
cb.set_label('multiplicity $m(x)$')
plt.subplots_adjust(left=0.0, right=0.9, top=1.0, bottom=0.0, wspace=0.0)
fig.savefig(FIG / 'sphere_3d.png', dpi=400)
print('figures/sphere_3d.png', len(F), 'triangles')
