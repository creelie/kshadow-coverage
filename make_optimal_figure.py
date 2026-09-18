"""
make_optimal_figure.py -- the figure of the optimal-coverage section.

Panels (a) and (b) are VTK renders on the camera of the other cortical
figures; (c) to (f) are drawn from results/optimal.json.  Writes
figures/fig_optimal.png and copies it into ../../tex/figures/.
"""
import json, os, shutil
from pathlib import Path

import numpy as np

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import pyvista as pv
pv.OFF_SCREEN = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
RES = ROOT / 'results'
# When this tree sits beside the manuscript, finished figures are copied
# into it as well; in a standalone checkout that directory is absent and
# the copy is skipped.
TEXFIG = ROOT.parent.parent / 'tex' / 'figures'

O = json.load(open(RES / 'optimal.json'))
D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
POS = D['pos'].astype(float); TRI = D['tri'].astype(np.int64)
SULC = D['sulc'].astype(float)

plt.rcParams.update({'font.family': 'serif', 'font.size': 8,
                     'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'figure.dpi': 200, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})

SHOT = (1500, 1150)
AZIM, ELEV, ZOOM = 180.0, 8.0, 1.62
CONTACT_R = 1.9
LAT = np.flatnonzero(POS[TRI].mean(axis=1)[:, 0] < -5.0)
TL = TRI[LAT]
PATCH = '#1f5f8b'
COVER = '#2e86c1'
CONTACT = '#c0392b'


def _faces(tri):
    return np.hstack([np.full((len(tri), 1), 3, dtype=np.int64), tri]).ravel()


def _plotter():
    p = pv.Plotter(off_screen=True, window_size=SHOT)
    p.set_background('white')
    p.enable_anti_aliasing('ssaa')
    return p


def _camera(p):
    p.camera_position = 'yz'
    p.camera.azimuth = AZIM
    p.camera.elevation = ELEV
    p.camera.zoom(ZOOM)


def _surface(p, tri, scalars=None, cmap=None, clim=None, color=None):
    m = pv.PolyData(POS, _faces(tri))
    if scalars is not None:
        m.cell_data['s'] = scalars
    p.add_mesh(m, scalars='s' if scalars is not None else None, cmap=cmap,
               clim=clim, color=color, smooth_shading=True, specular=0.32,
               specular_power=22, ambient=0.30, diffuse=0.76,
               show_scalar_bar=False)


def _contacts(p, idx, color, radius=CONTACT_R):
    pts = pv.PolyData(POS[idx])
    g = pts.glyph(geom=pv.Sphere(radius=radius, theta_resolution=20,
                                 phi_resolution=20), scale=False, orient=False)
    p.add_mesh(g, color=color, smooth_shading=True, specular=0.5,
               specular_power=30, ambient=0.25, diffuse=0.8)


def render(face_masks, contacts=()):
    """grey hemisphere, then each (mask, colour) painted on top"""
    p = _plotter()
    used = np.zeros(len(LAT), bool)
    for m, _ in face_masks:
        used |= m
    rest = TL[~used]
    if len(rest):
        _surface(p, rest, scalars=SULC[rest].mean(axis=1), cmap='Greys',
                 clim=(-1.5, 1.8))
    for m, col in face_masks:
        if m.any():
            _surface(p, TL[m], color=col)
    for idx, col in contacts:
        _contacts(p, idx, col)
    _camera(p)
    return p.screenshot(None, return_img=True)


# ----------------------------------------------------------- the geometry
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

NV = len(POS)
E = np.vstack([TRI[:, [0, 1]], TRI[:, [1, 2]], TRI[:, [2, 0]]])
E = np.unique(np.sort(E, axis=1), axis=0)
LEN = np.linalg.norm(POS[E[:, 0]] - POS[E[:, 1]], axis=1)
G = csr_matrix((np.r_[LEN, LEN], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
               shape=(NV, NV))

anchor = np.asarray(O['anchor_mni'])
a = int(np.argmin(np.sum((POS - anchor) ** 2, axis=1)))
d_a = dijkstra(G, directed=False, indices=a)
inpatch = d_a <= O['patch_radius_mm']
patch_faces_all = inpatch[TRI].all(axis=1)
patch_mask = patch_faces_all[LAT]

RBEST = 8.0
C = json.load(open(RES / 'cortex.json'))
grid_idx = np.array(C['grids']['parietal_8x8']['contact_vertices'])
NBEST = len(grid_idx)
centres = np.array(O['families']['free']['centres'][:NBEST])


def seen(idx, r):
    d = dijkstra(G, directed=False, indices=idx, limit=r + 1e-9)
    return (d < r)[:, TRI].all(axis=2).any(axis=0)[LAT]


cov_grid = seen(grid_idx, RBEST)
cov_free = seen(centres, RBEST)
print('patch faces %d; grid misses %d; greedy misses %d'
      % (patch_mask.sum(), (patch_mask & ~cov_grid).sum(),
         (patch_mask & ~cov_free).sum()))

MISS = '#d9534f'
img_a = render([(patch_mask & cov_grid, COVER), (patch_mask & ~cov_grid, MISS)],
               contacts=((grid_idx, CONTACT),))
img_b = render([(patch_mask & cov_free, COVER), (patch_mask & ~cov_free, MISS)],
               contacts=((centres, CONTACT),))

# ----------------------------------------------------------- the figure
fig = plt.figure(figsize=(7.1, 5.05))
outer = fig.add_gridspec(2, 1, height_ratios=[2.15, 2.30], hspace=0.30)
top = outer[0].subgridspec(1, 2, wspace=0.02)
bot = outer[1].subgridspec(1, 3, wspace=0.40)


def titled(ax, s, text, fontsize=8):
    ax.set_title('(%s) %s' % (s, text), fontsize=fontsize, loc='left', pad=3)


for j, (img, lab, ttl) in enumerate((
        (img_a, 'a', r'the published $8\times8$ grid, $r=8$ mm'),
        (img_b, 'b', 'the same $64$ contacts, placed by radius'))):
    ax = fig.add_subplot(top[0, j])
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    titled(ax, lab, ttl)

# (c) covering radius against contact count
ax = fig.add_subplot(bot[0, 0])
for fam, col, ls, lab in (('free', 'tab:blue', '-', 'any site'),
                          ('crown', 'tab:orange', '--', 'gyral crowns only')):
    rho = O['families'][fam]['rho_mm']
    ax.plot(range(1, len(rho) + 1), rho, ls, color=col, lw=1.4, label=lab)
floor = O['crown_floor']['covering_radius_mm']
ax.axhline(floor, color='tab:red', lw=1.0, ls=':')
ax.text(len(rho) * 0.97, floor * 1.06, '%.2f mm' % floor, ha='right',
        fontsize=6.5, color='tab:red')
ax.set_xlabel('contacts $n$', fontsize=7.5)
ax.set_ylabel(r'covering radius $\rho_n$ (mm)', fontsize=7.5)
ax.set_xlim(1, len(rho)); ax.set_ylim(0, 36)
ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.2, frameon=False, loc='upper right', handlelength=1.6)
titled(ax, 'c', 'the crown floor')

# (d) contacts needed against footprint radius, with the packing bound
ax = fig.add_subplot(bot[0, 1])
rs, ns = [], []
for key, c in sorted(O['curves'].items()):
    if not key.startswith('k1_r') or c.get('minimal') is None:
        continue
    rs.append(c['radius_mm']); ns.append(c['minimal']['n_contacts'])
o = np.argsort(rs); rs = np.array(rs)[o]; ns = np.array(ns)[o]
ax.plot(rs, ns, 'o-', color='tab:blue', ms=4, lw=1.4, label='certified design')
pk = O.get('packing', {})
prs = sorted(float(k) for k in pk)
ax.plot(prs, [pk[('%g' % r) if ('%g' % r) in pk else str(r)]['size'] for r in prs],
        's--', color='0.35', ms=3.5, lw=1.2, label='packing bound')
ax.axvline(floor, color='tab:red', lw=1.0, ls=':')
ax.text(floor - 0.25, 20, 'crown floor', rotation=90, ha='right', va='bottom',
        fontsize=6.2, color='tab:red')
ax.set_xlabel('footprint radius $r$ (mm)', fontsize=7.5)
ax.set_ylabel('contacts', fontsize=7.5)
ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.2, frameon=False, loc='upper right', handlelength=1.6)
titled(ax, 'd', 'the price of a certificate')

# (e) the matched comparison: same contacts, same radius, same target
ax = fig.add_subplot(bot[0, 2])
mc = O['matched_comparison']
rs2 = sorted({m['radius_mm'] for m in mc})
w = 0.35
for off, arr, col, lab in ((-w / 2, 'grid_8x8', 'tab:orange', r'$8\times8$ grid'),
                           (+w / 2, 'greedy', 'tab:blue', 'placed by radius')):
    y = [100 * next(m['uncovered_fraction'] for m in mc
                    if m['array'] == arr and m['radius_mm'] == rr) for rr in rs2]
    ax.bar(np.arange(len(rs2)) + off, y, w, color=col, label=lab)
    for x, v in zip(np.arange(len(rs2)) + off, y):
        ax.text(x, v + 1.2, ('%.1f' % v) if v >= 0.05 else '0', ha='center',
                fontsize=6.0)
ax.set_xticks(range(len(rs2)))
ax.set_xticklabels(['$r=%g$' % rr for rr in rs2])
ax.set_ylabel('patch unseen (%)', fontsize=7.5)
ax.set_ylim(0, 66)
ax.grid(alpha=0.25, lw=0.5, axis='y'); ax.set_axisbelow(True)
ax.legend(fontsize=6.2, frameon=False, loc='upper right', handlelength=1.4)
titled(ax, 'e', '64 contacts, two placements')

fig.savefig(FIG / 'fig_optimal.png')
plt.close(fig)
if TEXFIG.is_dir():
    shutil.copy(FIG / 'fig_optimal.png', TEXFIG / 'fig_optimal.png')
print('wrote fig_optimal.png')
