"""
make_field_figures.py -- figures of the planar sensor fields, drawn from the
same generators (fields_v2.py) and the same exact nerve (kshadow.py) as the
tables: Field A at k = 2 with its eight holes marked, Field D at k = 2 with
its dead zone, a near-tangent cluster of Field C at two raster resolutions,
the 41-clique of Field E, the sizes of the nerves of Fields A to D, and a
schematic of six footprints with the 1-skeletons of N and Delta_2(N).
Writes figures/field_A_k2.png, field_D_k2.png, field_C_tangency.png,
field_E_clique.png, nerve_sizes.png, six_footprints.png.
"""
import json, itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import ndimage
import networkx as nx

from fields_v2 import FIELDS
from kshadow import nerve_of_disks, raster_multiplicity, shadow_2skeleton, shadow_betti

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
RESULTS = json.load(open(ROOT / 'results' / 'fields.json'))


MULT_CMAP = 'viridis'


def _mult_panel(ax, m, win, vmax):
    """the multiplicity field itself, on the scale used everywhere else"""
    return ax.imshow(np.clip(m, 0, vmax), origin='lower',
                     extent=(win[0], win[1], win[2], win[3]), cmap=MULT_CMAP,
                     vmin=0, vmax=vmax, interpolation='bilinear')


def _holes_of(m, k, xs, ys):
    """enclosed background components of the raster of R_{>=k}"""
    region = m >= k
    lab, nb = ndimage.label(~region, structure=np.ones((3, 3), int))
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    out = []
    for l in range(1, nb + 1):
        if l in border:
            continue
        yy, xx = np.nonzero(lab == l)
        out.append((xs[xx].mean(), ys[yy].mean()))
    return out


def region_figure(name, k, fname, title):
    """
    The k-fold region drawn on the multiplicity field rather than as a flat
    two-colour mask: the field carries the whole of m(x), a single white
    contour carries the boundary of R_{>=k}, and the holes counted by b_1 are
    ringed at one size.  The earlier version used a saturated red and green,
    which is both a poor pair for colour-blind readers and throws away every
    level of m above k.
    """
    pts, R, desc, win = FIELDS[name]()
    m, xs, ys = raster_multiplicity(pts, R, win, 1400)
    vmax = int(max(4, m.max()))
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    im = _mult_panel(ax, m, win, vmax)
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, (m >= k).astype(float), levels=[0.5], colors='white',
               linewidths=1.3)
    ax.plot(pts[:, 0], pts[:, 1], '.', color='white', ms=2.4, alpha=0.85)
    holes = _holes_of(m, k, xs, ys)
    span = min(win[1] - win[0], win[3] - win[2])
    rad = 0.030 * span
    for cx, cy in holes:
        ax.add_patch(Circle((cx, cy), rad, fc='none', ec='#ff4d4d', lw=1.15))
    sh = RESULTS[name]['shadow'][str(k)]
    ax.set_title('%s:  $(b_0,b_1)(\\Delta_%d(N))=(%d,%d)$,  raster holes %d'
                 % (title, k, sh['b0'], sh['b1'], len(holes)), fontsize=8.5)
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                      ticks=range(0, vmax + 1))
    cb.set_label('multiplicity $m(x)$', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG / fname, dpi=400, bbox_inches='tight'); plt.close(fig)
    print(fname, 'raster holes', len(holes))


region_figure('A_ring_grid_seed3', 2, 'field_A_k2.png', 'Field A, $k=2$')
region_figure('D_hex_dead_zone', 2, 'field_D_k2.png', 'Field D, $k=2$')

# ---------------------------------------------------------------- Field C: near-tangent pairs and the raster
pts, R, desc, win = FIELDS['C_poisson_random']()
d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
iu = np.triu_indices(len(pts), 1)
near = [(i, j) for i, j in zip(*iu) if abs(d[i, j] - 2 * R) < 0.05]
print('Field C near-tangent pairs (|d - 2R| < 0.05):', len(near))
# the densest cluster of near-tangent pairs
cp = np.array([(pts[i] + pts[j]) / 2 for i, j in near])
best = max(range(len(cp)), key=lambda a: np.sum(np.linalg.norm(cp - cp[a], axis=1) < 2.5))
c0 = cp[best]
fig, ax = plt.subplots(figsize=(4.3, 4.0))
m, xs, ys = raster_multiplicity(pts, R, win, 1400)
vmax = int(max(4, m.max()))
im = _mult_panel(ax, m, win, vmax)
X, Y = np.meshgrid(xs, ys)
ax.contour(X, Y, (m >= 2).astype(float), levels=[0.5], colors='white', linewidths=1.2)
ax.plot(pts[:, 0], pts[:, 1], '.', color='white', ms=2.0, alpha=0.85)
for i, j in near:
    ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
            color='#ff4d4d', lw=1.0, solid_capstyle='round')
ax.set_title('Field C, $k=2$: the %d pairs within $0.05$ of tangency' % len(near),
             fontsize=8.5)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=range(0, vmax + 1))
cb.set_label('multiplicity $m(x)$', fontsize=8)
cb.ax.tick_params(labelsize=7)
fig.savefig(FIG / 'field_C_tangency.png', dpi=400, bbox_inches='tight'); plt.close(fig)
print('field_C_tangency.png')

# ---------------------------------------------------------------- Field E: the clique
pts, R, desc, win = FIELDS['E_two_cluster']()
N = nerve_of_disks(pts, R, max_size=2)
G = nx.Graph(); G.add_nodes_from(range(len(pts))); G.add_edges_from(tuple(s) for s in N if len(s) == 2)
clique = max(nx.find_cliques(G), key=len)
fig, ax = plt.subplots(figsize=(3.4, 3.4))
m, xs, ys = raster_multiplicity(pts, R, win, 1000)
ax.imshow(np.clip(m, 0, 12), origin='lower', extent=win, cmap='Greys', vmin=0, vmax=12, interpolation='nearest')
for i in range(len(pts)):
    ax.add_patch(Circle(pts[i], R, fc='none', ec='#d62728' if i in clique else '0.4', lw=0.35 if i in clique else 0.3, alpha=0.9))
ax.plot(pts[:, 0], pts[:, 1], 'k.', ms=1.5)
ax.plot(pts[clique, 0], pts[clique, 1], 'o', ms=2.5, color='#d62728')
ax.set_title('Field E: a clique of %d mutually overlapping sensors' % len(clique), fontsize=7.5)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
fig.savefig(FIG / 'field_E_clique.png', dpi=400, bbox_inches='tight'); plt.close(fig)
print('field_E_clique.png clique', len(clique))

# ---------------------------------------------------------------- nerve sizes
fig, ax = plt.subplots(figsize=(3.4, 2.2))
names = [('A_ring_grid_seed3', 'A'), ('B_ring_grid_seed7', 'B'), ('C_poisson_random', 'C'), ('D_hex_dead_zone', 'D')]
w = 0.25
for j, (key, lab) in enumerate((('1', 'vertices'), ('2', 'edges'), ('3', 'triangles'))):
    vals = [RESULTS[nm]['face_counts'].get(key, 0) for nm, _ in names]
    ax.bar(np.arange(4) + (j - 1) * w, vals, w, label=lab)
    for x, v in zip(np.arange(4) + (j - 1) * w, vals):
        ax.text(x, v * 1.08, str(v), ha='center', fontsize=5.5)
ax.set_yscale('log'); ax.set_xticks(range(4)); ax.set_xticklabels(['Field ' + l for _, l in names])
ax.set_ylabel('count'); ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.22))
ax.set_ylim(50, 6000)
fig.savefig(FIG / 'nerve_sizes.png', dpi=400, bbox_inches='tight'); plt.close(fig)
print('nerve_sizes.png')

# ---------------------------------------------------------------- six footprints: N and Delta_2(N)
P6 = np.array([(0, 0), (1.6, 0.3), (0.9, 1.5), (2.7, 1.4), (3.9, 0.4), (4.1, 1.9)])
R6 = 1.0
N6 = nerve_of_disks(P6, R6)
fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.15))
EXT6 = (-1.3, 5.4, -1.3, 3.2)
NAVY, ORANGE = '#1f4e79', '#c2570a'

ax = axes[0]
m, xs, ys = raster_multiplicity(P6, R6, EXT6, 700)
im6 = ax.imshow(np.clip(m, 0, 4), origin='lower', extent=EXT6,
                cmap=MULT_CMAP, vmin=0, vmax=4, interpolation='bilinear')
for i, p in enumerate(P6):
    ax.add_patch(Circle(p, R6, fc='none', ec='white', lw=0.9, alpha=0.9))
    ax.text(p[0], p[1], str(i + 1), ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')
cax6 = ax.inset_axes([0.0, -0.115, 1.0, 0.052])
cb6 = fig.colorbar(im6, cax=cax6, orientation='horizontal', ticks=range(0, 5))
cb6.set_label('multiplicity $m(x)$', fontsize=8)
cb6.ax.tick_params(labelsize=7)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('(a) footprints and $m(x)$', fontsize=9)

# ---- (b) the nerve, drawn on the footprints it comes from
ax = axes[1]
G = nx.Graph(); G.add_nodes_from(range(6))
G.add_edges_from(tuple(s) for s in N6 if len(s) == 2)
tris = [tuple(sorted(s)) for s in N6 if len(s) == 3]
for p in P6:
    ax.add_patch(Circle(p, R6, fc='#eef3f8', ec='#b9c8d8', lw=0.6, alpha=0.85,
                        zorder=0))
for t in tris:
    ax.fill(P6[list(t), 0], P6[list(t), 1], color='#7fa8d4', alpha=0.55,
            ec=NAVY, lw=0.6, zorder=1)
for a, b in G.edges():
    ax.plot(P6[[a, b], 0], P6[[a, b], 1], color='white', lw=3.4, zorder=2,
            solid_capstyle='round')
    ax.plot(P6[[a, b], 0], P6[[a, b], 1], color=NAVY, lw=1.7, zorder=3,
            solid_capstyle='round')
for i, p in enumerate(P6):
    ax.add_patch(Circle(p, 0.26, fc='white', ec=NAVY, lw=1.3, zorder=4))
    ax.text(p[0], p[1], str(i + 1), ha='center', va='center', fontsize=8.5,
            color=NAVY, fontweight='bold', zorder=5)
ax.text(0.015, 0.015, '$6$ vertices, $%d$ edges, $%d$ triangles'
        % (G.number_of_edges(), len(tris)), transform=ax.transAxes,
        fontsize=7, color=NAVY, ha='left', va='bottom')
ax.set_aspect('equal'); ax.set_axis_off()
ax.set_xlim(EXT6[0], EXT6[1]); ax.set_ylim(EXT6[2], EXT6[3])
ax.set_title('(b) the nerve $N$', fontsize=9)
print('N6 edges', G.number_of_edges(), 'triangles', len(tris))

# ---- (c) the 2-shadow, drawn on the nerve its vertices come from
ax = axes[2]
v, e, t = shadow_2skeleton(N6, 2)
pos = {s: P6[list(s)].mean(axis=0) for s in v}
H = nx.Graph(); H.add_nodes_from(v); H.add_edges_from(e)
for a, b in G.edges():                       # the nerve, as a ghost
    ax.plot(P6[[a, b], 0], P6[[a, b], 1], color='#c8cfd8', lw=1.0, zorder=0)
ax.plot(P6[:, 0], P6[:, 1], 'o', ms=3.0, mfc='white', mec='#c8cfd8', mew=1.0,
        zorder=0)
for a, b, c in t:
    ax.fill([pos[a][0], pos[b][0], pos[c][0]],
            [pos[a][1], pos[b][1], pos[c][1]], color='#f4b183', alpha=0.6,
            ec=ORANGE, lw=0.6, zorder=1)
for a, b in H.edges():
    xy = np.array([pos[a], pos[b]])
    ax.plot(xy[:, 0], xy[:, 1], color='white', lw=3.2, zorder=2,
            solid_capstyle='round')
    ax.plot(xy[:, 0], xy[:, 1], color=ORANGE, lw=1.6, zorder=3,
            solid_capstyle='round')
for s in v:
    q = pos[s]
    ax.add_patch(Circle(q, 0.27, fc='white', ec=ORANGE, lw=1.3, zorder=4))
    ax.text(q[0], q[1], ''.join(str(i + 1) for i in sorted(s)), ha='center',
            va='center', fontsize=6.6, color=ORANGE, fontweight='bold',
            zorder=5)
b0, b1, _ = shadow_betti(N6, 2)
ax.text(0.015, 0.015, '$b_0=%d$, $b_1=%d$, matching $R_{\\geq2}$' % (b0, b1),
        transform=ax.transAxes, fontsize=7, color=ORANGE, ha='left',
        va='bottom')
PC = np.array(list(pos.values()))
cx = 0.5 * (PC[:, 0].max() + PC[:, 0].min()); cy = 0.5 * (PC[:, 1].max() + PC[:, 1].min())
ratio = (EXT6[1] - EXT6[0]) / (EXT6[3] - EXT6[2])
hx = 0.5 * (PC[:, 0].max() - PC[:, 0].min()) * 1.22
hy = 0.5 * (PC[:, 1].max() - PC[:, 1].min()) * 1.45
hx = max(hx, hy * ratio); hy = hx / ratio
ax.set_aspect('equal'); ax.set_axis_off()
ax.set_xlim(cx - hx, cx + hx); ax.set_ylim(cy - hy, cy + hy)
ax.set_title('(c) the $2$-shadow $\\Delta_2(N)$', fontsize=9)
plt.tight_layout(); fig.savefig(FIG / 'six_footprints.png', dpi=400); plt.close(fig)
print('six_footprints.png', len(v), len(e), len(t), (b0, b1))
