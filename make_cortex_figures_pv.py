"""
make_cortex_figures_pv.py -- the cortical figures, rendered with VTK.

The earlier versions of these three figures were drawn with matplotlib's
Poly3DCollection, which sorts triangles by depth instead of keeping a depth
buffer.  On a folded surface that produces visible sorting artefacts where a
gyral crown passes in front of a sulcus, and it cannot do smooth shading, so
the hemisphere reads as a grey mass with a folded silhouette.  Rendering the
same meshes through VTK gives a real depth buffer, per-vertex normals, a
specular term and supersampled antialiasing.

Every panel of every figure uses one camera, one light and one colour scale,
so a contact sits at the same screen position in each; panel labels and
colour bars are drawn outside the render, never over the surface.

Produces, as drop-in replacements for the files of make_cortex_figures.py:
  figures/cortex_overview.png   the two grids, and the multiplicity at 8 mm
  figures/cortex_regions.png    R_{>=k}, k = 1..4, on the parietal grid
  figures/cortex_models.png     Euclidean and geodesic footprint of contact 35
"""
import json, os
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
RES = ROOT / 'results'

R = json.load(open(RES / 'cortex.json'))
D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
POS = D['pos'].astype(float); TRI = D['tri'].astype(np.int64)
SULC = D['sulc'].astype(float)
NV = len(POS)

grids = R['grids']
idx_p = np.array(grids['parietal_8x8']['contact_vertices'])
idx_t = np.array(grids['temporal_4x8']['contact_vertices'])
import distcache
dist_p = distcache.load(RES, 'parietal_8x8')
dist_t = distcache.load(RES, 'temporal_4x8')

plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                     'figure.dpi': 200, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

# one camera for every panel in the paper
SHOT = (1500, 1150)
AZIM, ELEV, ZOOM = 180.0, 8.0, 1.62
CONTACT_R = 1.9          # mm, drawn radius of a contact marker


def _faces(tri):
    return np.hstack([np.full((len(tri), 1), 3, dtype=np.int64), tri]).ravel()


def _plotter():
    p = pv.Plotter(off_screen=True, window_size=SHOT)
    p.set_background('white')
    p.enable_anti_aliasing('ssaa')
    return p


def _camera(p, focus=None, span=None):
    """
    The whole-hemisphere camera used by every overview panel.  Passing a
    focus point and a span keeps the same viewing direction but recentres
    on that point and narrows the field to `span` mm, which is what the
    single-contact panels need: at hemisphere zoom an 8 mm footprint is a
    few pixels across and the tearing of the Euclidean ball is invisible.
    """
    p.camera_position = 'yz'
    p.camera.azimuth = AZIM
    p.camera.elevation = ELEV
    if focus is None:
        p.camera.zoom(ZOOM)
        return
    pos = np.asarray(p.camera.position, float)
    fp = np.asarray(p.camera.focal_point, float)
    d = pos - fp
    d = d / np.linalg.norm(d)
    p.camera.focal_point = tuple(focus)
    p.camera.position = tuple(np.asarray(focus, float) + d * 200.0)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = span


def _add_surface(p, tri, scalars=None, cmap=None, clim=None, color=None,
                 opacity=1.0):
    m = pv.PolyData(POS, _faces(tri))
    if scalars is not None:
        m.cell_data['s'] = scalars
    return p.add_mesh(m, scalars='s' if scalars is not None else None,
                      cmap=cmap, clim=clim, color=color, opacity=opacity,
                      smooth_shading=True, specular=0.32, specular_power=22,
                      ambient=0.30, diffuse=0.76, show_scalar_bar=False)


def _add_contacts(p, idx, color):
    pts = pv.PolyData(POS[idx])
    glyph = pts.glyph(geom=pv.Sphere(radius=CONTACT_R, theta_resolution=20,
                                     phi_resolution=20), scale=False, orient=False)
    p.add_mesh(glyph, color=color, smooth_shading=True, specular=0.5,
               specular_power=30, ambient=0.25, diffuse=0.8)


def footprint_faces(dist, r):
    """triangles all of whose vertices lie within r of the contact"""
    return (dist[:, TRI] < r).all(axis=2)


# lateral aspect only; the medial wall never faces this camera
LAT = np.flatnonzero(POS[TRI].mean(axis=1)[:, 0] < -5.0)
TL = TRI[LAT]
SULC_F = SULC[TL].mean(axis=1)


def render_base(contacts=(), fname=None):
    p = _plotter()
    _add_surface(p, TL, scalars=SULC_F, cmap='Greys', clim=(-1.5, 1.8))
    for idx, col in contacts:
        _add_contacts(p, idx, col)
    _camera(p)
    return p.screenshot(fname, return_img=True)


def render_scalar(face_mask, values, cmap, clim, contacts=(), fname=None):
    """grey surface with a coloured overlay on the masked faces"""
    p = _plotter()
    rest = TL[~face_mask]
    if len(rest):
        _add_surface(p, rest, scalars=SULC[rest].mean(axis=1),
                     cmap='Greys', clim=(-1.5, 1.8))
    sel = TL[face_mask]
    if len(sel):
        _add_surface(p, sel, scalars=values[face_mask], cmap=cmap, clim=clim)
    for idx, col in contacts:
        _add_contacts(p, idx, col)
    _camera(p)
    return p.screenshot(fname, return_img=True)


def render_patch(face_mask, color, contacts=(), fname=None, focus=None, span=None):
    p = _plotter()
    rest = TL[~face_mask]
    if len(rest):
        _add_surface(p, rest, scalars=SULC[rest].mean(axis=1),
                     cmap='Greys', clim=(-1.5, 1.8))
    sel = TL[face_mask]
    if len(sel):
        _add_surface(p, sel, color=color)
    for idx, col in contacts:
        _add_contacts(p, idx, col)
    _camera(p, focus, span)
    return p.screenshot(fname, return_img=True)


def panel(ax, img, label):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(label, fontsize=9, pad=3)


PARIETAL = '#c0392b'
TEMPORAL = '#e08020'

# ------------------------------------------------------------ figure: overview
mult_p = footprint_faces(dist_p, 8.0)[:, LAT].sum(axis=0)
mult_t = footprint_faces(dist_t, 8.0)[:, LAT].sum(axis=0)
mult = mult_p + mult_t
cov = mult > 0

a = render_base(contacts=((idx_p, PARIETAL), (idx_t, TEMPORAL)))
b = render_scalar(cov, mult.astype(float), 'viridis', (0, 6),
                  contacts=((idx_p, PARIETAL), (idx_t, TEMPORAL)))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
panel(axes[0], a, '(a) the two grids on the pial surface')
panel(axes[1], b, '(b) multiplicity of the geodesic footprints, $r=8$ mm')
cax = fig.add_axes([0.915, 0.18, 0.013, 0.62])
cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, 6), cmap='viridis'),
                  cax=cax)
cb.set_label('contacts seeing a face, $m$', fontsize=7.5)
cb.ax.tick_params(labelsize=7)
fig.subplots_adjust(left=0.005, right=0.90, wspace=0.02)
fig.savefig(FIG / 'cortex_overview.png')
plt.close(fig)
print('wrote cortex_overview.png')

# ------------------------------------------------------------ figure: regions
fp = footprint_faces(dist_p, 8.0)[:, LAT]
mp = fp.sum(axis=0)

# Betti numbers of Delta_k(N) and of the mesh reference at r = 8 mm, as
# computed by run_cortex.py, so the titles carry the certificate itself
rec8 = [e for e in R['grids']['parietal_8x8']['models']['geodesic']
        if abs(e['radius_mm'] - 8.0) < 1e-9][0]
shadow = rec8['shadow']; meshref = rec8['mesh_reference']


def betti(d, k):
    v = d.get(str(k)) or d.get(k)
    if v is None:
        return None
    if isinstance(v, dict):
        return (v.get('b0'), v.get('b1'))
    return tuple(v[:2])


fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.3))
for j, k in enumerate((1, 2, 3, 4)):
    img = render_patch(mp >= k, '#1f5f8b', contacts=((idx_p, PARIETAL),))
    bs, bm = betti(shadow, k), betti(meshref, k)
    lab = r'$R_{\geq%d}$' % k
    if bs is not None:
        lab += '\n' + r'$\Delta_%d(N)$: $(%s,%s)$' % (k, bs[0], bs[1])
    if bm is not None:
        lab += r', mesh $(%s,%s)$' % (bm[0], bm[1])
    panel(axes[j], img, lab)
fig.subplots_adjust(left=0.004, right=0.996, wspace=0.02)
fig.savefig(FIG / 'cortex_regions.png')
plt.close(fig)
print('wrote cortex_regions.png')

# ------------------------------------------------------------ figure: models
C = 35
geo = (dist_p[C][TRI] < 8.0).all(axis=1)[LAT]
euc = (np.linalg.norm(POS - POS[idx_p[C]], axis=1)[TRI] < 8.0).all(axis=1)[LAT]
FOCUS = POS[idx_p[C]]
SPAN = 15.0
img_g = render_patch(geo, '#2e86c1', contacts=((idx_p[[C]], PARIETAL),), focus=FOCUS, span=SPAN)
img_e = render_patch(euc, '#d35400', contacts=((idx_p[[C]], PARIETAL),), focus=FOCUS, span=SPAN)

fig = plt.figure(figsize=(7.1, 2.55))
axA = fig.add_axes([0.005, 0.02, 0.29, 0.86])
panel(axA, img_e, '(a) Euclidean ball, contact 35')
axB = fig.add_axes([0.305, 0.02, 0.29, 0.86])
panel(axB, img_g, '(b) geodesic disk, same contact')
axC = fig.add_axes([0.705, 0.20, 0.28, 0.66])
SER = (('parietal_8x8', 'geodesic', 'parietal, geodesic', 'tab:blue', '-', 'o'),
       ('parietal_8x8', 'euclidean', 'parietal, Euclidean', 'tab:red', '-', 'o'),
       ('temporal_4x8', 'geodesic', 'temporal, geodesic', 'tab:blue', '--', 's'),
       ('temporal_4x8', 'euclidean', 'temporal, Euclidean', 'tab:red', '--', 's'))
for gname, model, lab, col, ls, mk in SER:
    recs = R['grids'][gname]['models'][model]
    rs = [e['radius_mm'] for e in recs]
    fl = [max(e['disk_test_failures'], 0.5) for e in recs]
    axC.plot(rs, fl, color=col, ls=ls, marker=mk, ms=3.2, lw=1.3, label=lab)
axC.set_yscale('log')
axC.set_xlabel('footprint radius $r$ (mm)', fontsize=8)
axC.set_ylabel('faces failing the disk test', fontsize=8)
axC.tick_params(labelsize=7)
axC.grid(alpha=0.25, lw=0.5)
axC.legend(fontsize=5.6, frameon=False, loc='upper left')
axC.set_title('(c) disk-test failures', fontsize=9, pad=3)
fig.savefig(FIG / 'cortex_models.png')
plt.close(fig)
print('wrote cortex_models.png')
