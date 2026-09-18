"""
make_cortex_figures.py -- three-dimensional figures for the cortical-surface
certificate of run_cortex.py: the left hemisphere of the template with the
two grids and the multiplicity of the geodesic footprints, the regions R_k
of the parietal grid, the geodesic and the Euclidean footprint of one contact
together with the disk-test failure counts of the two models, the radius
sweep against the mesh reference, and the redundancy barcodes.  Reads
results/cortex.json, the cortex_dist_* caches and data/colin27_lh_pial.npz;
writes figures/cortex_sweep.png and cortex_barcodes.png, which are the two
plots of this group, and, under *_mpl.png names, the earlier matplotlib
renders of the three three-dimensional panels.  Those three are produced for
the paper by make_cortex_figures_pv.py, which renders them through VTK; the
two scripts therefore never write the same file.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors as mcolors

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
R = json.load(open(ROOT / 'results' / 'cortex.json'))
D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
POS = D['pos'].astype(float); TRI = D['tri'].astype(np.int64); NV = len(POS)

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 9,
                     'figure.dpi': 200, 'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04})

LIGHT = np.array([-0.55, 0.35, 0.75]); LIGHT /= np.linalg.norm(LIGHT)
GREY = np.array([0.80, 0.80, 0.82])
CMAP = plt.get_cmap('viridis')


def face_normals(tri):
    a, b, c = POS[tri[:, 0]], POS[tri[:, 1]], POS[tri[:, 2]]
    n = np.cross(b - a, c - a)
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


def shade(base_rgb, tri, ambient=0.35, light=LIGHT):
    lam = np.clip(face_normals(tri) @ light, 0, 1)
    fac = ambient + (1 - ambient) * lam
    rgb = np.asarray(base_rgb)
    if rgb.ndim == 1:
        rgb = np.tile(rgb, (len(tri), 1))
    return np.clip(rgb * fac[:, None], 0, 1)


def draw_mesh(ax, tri, colors):
    pc = Poly3DCollection(POS[tri], facecolors=colors, edgecolors=colors, linewidths=0.05, antialiased=False, zorder=1)
    ax.computed_zorder = False
    ax.add_collection3d(pc)
    return pc


def set_view(ax, centre, half, elev=8, azim=180):
    ax.computed_zorder = False
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


grids = R['grids']
pariet = grids['parietal_8x8']; tempo = grids['temporal_4x8']
idx_p = np.array(pariet['contact_vertices']); idx_t = np.array(tempo['contact_vertices'])
import distcache
dist_p = distcache.load(ROOT / 'results', 'parietal_8x8')
dist_t = distcache.load(ROOT / 'results', 'temporal_4x8')


def tri_mult(dist, r):
    """Number of footprints (triangles fully within distance r) containing each triangle."""
    return (dist[:, TRI] < r).all(axis=2).sum(axis=0)


# lateral triangles only (the medial wall is hidden in a lateral view anyway)
TL = TRI[np.flatnonzero(POS[TRI].mean(axis=1)[:, 0] < -5.0)]

# ---------------------------------------------------------------- figure 1: overview
fig = plt.figure(figsize=(7.0, 3.2))
for k, (title, r) in enumerate((('(a) the two grids on the pial surface', None),
                                ('(b) multiplicity of the geodesic footprints, $r=8$ mm', 8.0))):
    ax = fig.add_subplot(1, 2, k + 1, projection='3d')
    if r is None:
        draw_mesh(ax, TL, shade(GREY, TL))
    else:
        keep = np.flatnonzero(POS[TRI].mean(axis=1)[:, 0] < -5.0)
        fm = (tri_mult(dist_p, r) + tri_mult(dist_t, r))[keep]
        base = np.tile(GREY, (len(TL), 1))
        vmax = 6
        cols = CMAP(np.clip(fm, 0, vmax) / vmax)[:, :3]
        base[fm > 0] = cols[fm > 0]
        draw_mesh(ax, TL, shade(base, TL))
    for idx, c in ((idx_p, '#d62728'), (idx_t, '#ff7f0e')):
        P = POS[idx] + np.array([-1.5, 0, 0])
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=6, c=c, depthshade=False, edgecolors='k', linewidths=0.2, zorder=10)
    set_view(ax, centre=(-30, -20, 15), half=72, elev=8, azim=180)
    ax.set_title(title, pad=-10)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0, 6)); sm.set_array([])
cb = fig.colorbar(sm, ax=fig.axes, fraction=0.025, pad=0.01, shrink=0.55, ticks=range(0, 7))
cb.set_label('multiplicity $m$')
plt.subplots_adjust(left=0.0, right=0.9, top=1.0, bottom=0.0, wspace=0.0)
fig.savefig(FIG / 'cortex_overview_mpl.png')
plt.close(fig)
print('cortex_overview.png')

# ---------------------------------------------------------------- figure 2: regions of the parietal grid
r = 8.0
tm = tri_mult(dist_p, r)
centre = POS[idx_p].mean(axis=0)
near = np.flatnonzero(np.linalg.norm(POS[TL].mean(axis=1) - centre, axis=1) < 80)
TN = TL[near]; tmN = tm[near]
rec = [x for x in pariet['models']['geodesic'] if x['radius_mm'] == r][0]
fig = plt.figure(figsize=(7.0, 6.2))
for k in range(1, 5):
    ax = fig.add_subplot(2, 2, k, projection='3d')
    base = np.tile(GREY, (len(TN), 1))
    base[tmN >= k] = np.array(mcolors.to_rgb(['#1f77b4', '#2ca02c', '#9467bd', '#d62728'][k - 1]))
    draw_mesh(ax, TN, shade(base, TN))
    P = POS[idx_p] + np.array([-1.5, 0, 0])
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=8, c='k', depthshade=False, zorder=10)
    set_view(ax, centre=centre, half=44, elev=10, azim=180)
    b = rec['shadow'][str(k)]; t = rec['mesh_reference'][str(k)]
    ax.set_title(r'(%s) $R_{\geq %d}$:  $\Delta_%d(N)$ gives $(b_0,b_1)=(%d,%d)$, mesh $(%d,%d)$' % (
        'abcd'[k - 1], k, k, b['b0'], b['b1'], t[0], t[1]), pad=6)
plt.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.0, wspace=0.0, hspace=0.22)
fig.savefig(FIG / 'cortex_regions_mpl.png')
plt.close(fig)
print('cortex_regions.png')

# ---------------------------------------------------------------- figure 3: the two footprint models of one contact
dist_e = np.linalg.norm(POS[None, :, :] - POS[idx_p][:, None, :], axis=2)
r = 8.0
rec_e = [x for x in pariet['models']['euclidean'] if x['radius_mm'] == r][0]
singles = [f for f in rec_e['failures'] if len(f['face']) == 1 and f['components'] >= 2]
singles.sort(key=lambda f: f['sizes'][1], reverse=True)
cnum = singles[0]['face'][0]
c = POS[idx_p[cnum]]
sel = np.flatnonzero(np.linalg.norm(POS[TRI].mean(axis=1) - c, axis=1) < 16)
TS = TRI[sel]
nrm = face_normals(TRI[np.linalg.norm(POS[TRI].mean(axis=1) - c, axis=1) < 5]).mean(axis=0); nrm /= np.linalg.norm(nrm)
az0 = np.degrees(np.arctan2(nrm[1], nrm[0])); el0 = np.degrees(np.arcsin(np.clip(nrm[2], -1, 1)))
fig = plt.figure(figsize=(7.0, 2.7))
for k, (dist, title, col) in enumerate(((dist_e, '(a) Euclidean ball, contact %d, $r=8$ mm' % cnum, '#fdae6b'),
                                        (dist_p, '(b) geodesic disk, contact %d, $r=8$ mm' % cnum, '#9ecae1'))):
    ax = fig.add_subplot(1, 3, k + 1, projection='3d')
    inside = (dist[cnum][TS] < r).all(axis=1)
    base = np.tile(GREY, (len(TS), 1)); base[inside] = mcolors.to_rgb(col)
    draw_mesh(ax, TS, shade(base, TS, ambient=0.45, light=nrm * 0.6 + LIGHT * 0.4))
    ax.scatter([c[0] - 1], [c[1]], [c[2]], s=16, c='k', depthshade=False, zorder=30)
    set_view(ax, centre=c, half=12, elev=el0 + 25, azim=az0 + 20)
    ax.set_title(title, fontsize=8, pad=-4)
ax = fig.add_subplot(1, 3, 3)
for g, gl, ls in ((pariet, 'parietal', '-'), (tempo, 'temporal', '--')):
    for model, col in (('geodesic', '#1f77b4'), ('euclidean', '#d62728')):
        recs = g['models'][model]
        rr = [x['radius_mm'] for x in recs]
        nf = [x['disk_test_failures'] for x in recs]
        ax.plot(rr, nf, ls, color=col, lw=1.3, marker='o', ms=3, label='%s, %s' % (gl, model))
ax.set_yscale('symlog', linthresh=1)
ax.set_xlabel('footprint radius $r$ (mm)'); ax.set_ylabel('faces of $N$ failing the disk test')
ax.set_xlim(3.5, 15.5); ax.set_ylim(-0.3, 3000)
leg = ax.legend(fontsize=6.5, frameon=True, loc='upper left')
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('none')
leg.get_frame().set_alpha(0.92)
leg.set_zorder(20)
ax.set_title('(c) disk-test failures', fontsize=8)
plt.subplots_adjust(left=0.0, right=0.99, top=0.9, bottom=0.17, wspace=0.05)
fig.savefig(FIG / 'cortex_models_mpl.png')
plt.close(fig)
print('cortex_models.png contact', cnum, 'components', singles[0]['sizes'])

# ---------------------------------------------------------------- figure 4: the radius sweep, geodesic model
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
for col, (gname, g) in enumerate((('parietal 8 x 8', pariet), ('temporal 4 x 8', tempo))):
    recs = g['models']['geodesic']
    rr = [x['radius_mm'] for x in recs]
    for row, (bname, bi) in enumerate((('b0', 0), ('b1', 1))):
        ax = axes[row, col]
        for k, c in zip((1, 2, 3), ('#1f77b4', '#2ca02c', '#d62728')):
            sh = [x['shadow'].get(str(k), {'b0': 0, 'b1': 0})[bname] for x in recs]
            tr = [x['mesh_reference'].get(str(k), [0, 0])[bi] for x in recs]
            ax.plot(rr, sh, '-', color=c, lw=1.2, label=r'$\Delta_%d(N)$' % k)
            ax.plot(rr, tr, 'o', color=c, ms=4, mfc='none', mew=1.0, label=r'mesh, $k=%d$' % k)
        ax.set_ylabel(r'$b_%d$' % bi)
        if row == 0:
            ax.set_title('(%s) %s' % ('ab'[col], gname))
            ax2 = ax.twinx()
            ax2.bar(rr, [x['disk_test_failures'] for x in recs], width=0.45, color='0.8', zorder=0)
            ax2.set_ylabel('faces failing the disk test', color='0.45', fontsize=7.5)
            ax2.tick_params(axis='y', colors='0.45', labelsize=7)
            ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        if row == 1:
            ax.set_xlabel('footprint radius $r$ (mm)')
        if row == 0 and col == 0:
            handles, labels = ax.get_legend_handles_labels()
# one legend for the whole figure, above the panels, so that nothing sits on
# top of a curve: the two columns plot the same six series
fig.legend(handles, labels, fontsize=7, ncol=6, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, 1.005),
           handlelength=1.6, columnspacing=1.3, handletextpad=0.5)
plt.tight_layout(rect=(0, 0, 1, 0.945))
fig.savefig(FIG / 'cortex_sweep.png')
plt.close(fig)
print('cortex_sweep.png')

# ---------------------------------------------------------------- figure 5: redundancy barcodes on the cortex
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, (gname, g) in zip(axes, (('(a) parietal 8 x 8, $r=8$ mm', pariet), ('(b) temporal 4 x 8, $r=8$ mm', tempo))):
    rec8 = [x for x in g['models']['geodesic'] if x['radius_mm'] == 8.0][0]
    bars = rec8['barcode']
    y = 0
    for d, c in (('0', '#1f77b4'), ('1', '#d62728')):
        cnt = Counter(tuple(b) for b in bars[d])
        for (lo, hi), m in sorted(cnt.items(), key=lambda t: (-t[0][1], -t[0][0])):
            ax.plot([lo - 0.4, hi + 0.4], [y, y], lw=2.0, color=c)
            ax.text(hi + 0.5, y, r'$\times%d$' % m, va='center', fontsize=6.5)
            y += 1
    ax.set_yticks([]); ax.set_xlabel('redundancy level $k$'); ax.set_title(gname)
    ax.set_xlim(0.4, 6.5)
    ax.plot([], [], color='#1f77b4', lw=2, label='$H_0$'); ax.plot([], [], color='#d62728', lw=2, label='$H_1$')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
plt.tight_layout()
fig.savefig(FIG / 'cortex_barcodes.png')
plt.close(fig)
print('cortex_barcodes.png')
