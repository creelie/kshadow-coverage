"""
make_sphere3d_pv.py -- the spherical-cap arrays, rendered with VTK.

Same construction as make_sphere3d.py: the sixty-four centres of run_sphere.py
on the unit sphere, the jittered 8 x 8 gnomonic grid with seed 7, and the
multiplicity of the caps of geodesic radius r = 0.13 and r = 0.16 on a
subdivided icosahedron.  The difference is the renderer.  Depth sorting a
triangulated sphere in matplotlib puts faces in the wrong order wherever the
silhouette folds, and flat shading loses the curvature entirely; VTK keeps a
depth buffer and interpolates normals, so the sphere reads as a sphere and
the caps sit on it rather than beside it.

Writes figures/sphere_3d.png.
"""
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import pyvista as pv
pv.OFF_SCREEN = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, cm

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                     'figure.dpi': 200, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

# ---- the same centres as run_sphere.py
rng = np.random.default_rng(7)
g = np.linspace(-0.55, 0.55, 8)
uv = np.array([(a, b) for a in g for b in g]) + rng.uniform(-0.02, 0.02, (64, 2))
C = np.stack([uv[:, 0], uv[:, 1], np.ones(64)], axis=1)
C /= np.linalg.norm(C, axis=1, keepdims=True)


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(level):
        cache = {}; vl = list(v); nf = []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = vl[a] + vl[b]
                p = p / np.linalg.norm(p)
                vl.append(p)
                cache[key] = len(vl) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(vl); f = np.array(nf)
    return v, f


V, F = icosphere(6)
FC = V[F].mean(axis=1)
FC /= np.linalg.norm(FC, axis=1, keepdims=True)
faces = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()

SHOT = (1300, 1300)
VMAX = 6


def render(r):
    """multiplicity of the caps of geodesic radius r, as a face scalar"""
    m = (np.arccos(np.clip(FC @ C.T, -1, 1)) < r).sum(axis=1)
    mesh = pv.PolyData(V, faces)
    mesh.cell_data['m'] = m.astype(float)
    p = pv.Plotter(off_screen=True, window_size=SHOT)
    p.set_background('white')
    p.enable_anti_aliasing('ssaa')
    # the uncovered sphere in grey, the covered caps on the viridis scale
    grey = mesh.extract_cells(np.flatnonzero(m == 0))
    p.add_mesh(grey, color='#c9c9cc', smooth_shading=True, specular=0.30,
               specular_power=24, ambient=0.28, diffuse=0.78,
               show_scalar_bar=False)
    capped = mesh.extract_cells(np.flatnonzero(m > 0))
    p.add_mesh(capped, scalars='m', cmap='viridis', clim=(0, VMAX),
               smooth_shading=True, specular=0.34, specular_power=26,
               ambient=0.28, diffuse=0.78, show_scalar_bar=False)
    pts = pv.PolyData(C * 1.006)
    p.add_mesh(pts.glyph(geom=pv.Sphere(radius=0.012, theta_resolution=18,
                                        phi_resolution=18),
                         scale=False, orient=False),
               color='#b02020', smooth_shading=True, specular=0.5,
               specular_power=30)
    p.camera_position = 'xy'
    p.camera.elevation = 28
    p.camera.azimuth = 12
    p.camera.zoom(1.45)
    return p.screenshot(None, return_img=True)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
for ax, r, lab in ((axes[0], 0.13, '(a) caps of geodesic radius $r=0.13$'),
                   (axes[1], 0.16, '(b) caps of geodesic radius $r=0.16$')):
    ax.imshow(render(r))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(lab, fontsize=9, pad=3)
cax = fig.add_axes([0.915, 0.16, 0.014, 0.66])
cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, VMAX), cmap='viridis'),
                  cax=cax, ticks=range(0, VMAX + 1))
cb.set_label('multiplicity $m(x)$', fontsize=8)
cb.ax.tick_params(labelsize=7)
fig.subplots_adjust(left=0.005, right=0.90, wspace=0.02)
fig.savefig(FIG / 'sphere_3d.png')
print('wrote sphere_3d.png')
