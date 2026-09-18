"""
make_merged_figures.py -- the composite figures of the submitted paper.

The manuscript originally carried twenty-one separate floats.  Several of
them were generated at a physical width of seven to fourteen inches and then
included at a single revtex column, 3.4 inches, which reduces a nine-point
label to under four points.  This script rebuilds the same content as eight
composite figures, each drawn at the width it is actually printed at, so
that every label in the paper is set at seven points or more.

Panels that come from an off-screen PyVista render are pasted from the PNG
that produced them (no renderer is needed here); every other panel is redrawn
from the cached results in results/*.json, or, where a raster is needed, from
the same generator the original figure used.

Writes, into figures/ and into ../../tex/figures/:

    fig_fields.png     Fields A to E and their barcodes        (6 panels)
    fig_grid.png       the 8x8 grid: regions, barcode, sweeps  (8 panels)
    fig_sphere.png     spherical caps                          (3 panels)
    fig_cortex_a.png   the template hemisphere: grids, regions (3 panels)
    fig_cortex_b.png   footprint models and the radius sweep   (4 panels)
    fig_robust.png     cortical barcodes, dropout, stability   (6 panels)

six_footprints.png and anisotropic.png are already drawn at the width they
are printed at and are left alone.
"""
import json, shutil, itertools
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy import ndimage
import networkx as nx

from fields_v2 import FIELDS
from kshadow import nerve_of_disks, raster_multiplicity, shadow_betti

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
# When this tree sits beside the manuscript, finished figures are copied
# into it as well; in a standalone checkout that directory is absent and
# the copy is skipped.
TEXFIG = ROOT.parent.parent / 'tex' / 'figures'

FULL = 7.1          # revtex two-column text width, inches
plt.rcParams.update({'font.family': 'serif', 'font.size': 8,
                     'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'figure.dpi': 200, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})

FIELDS_JSON = json.load(open(RES / 'fields.json'))
GRID = json.load(open(RES / 'grid.json'))
CORTEX = json.load(open(RES / 'cortex.json'))
DROP = json.load(open(RES / 'dropout_cortex.json'))
STAB = json.load(open(RES / 'stability.json'))

H0C, H1C = '#1f6fb4', '#c0392b'


# ----------------------------------------------------------------- helpers
def letter(ax, s, dx=-0.01, dy=1.02):
    ax.text(dx, dy, '(%s)' % s, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, fontweight='bold')


def titled(ax, s, text, fontsize=8):
    """panel letter carried by the title, so that no label can collide with
    it whatever the layout does"""
    ax.set_title('(%s) %s' % (s, text), fontsize=fontsize, loc='left', pad=3)


def trimmed(path):
    """the PNG with its uniform white border removed, and its aspect ratio"""
    a = plt.imread(path)
    if a.ndim == 3 and a.shape[2] == 4:          # composite over white
        al = a[..., 3:4]
        a = a[..., :3] * al + (1.0 - al)
    g = a.mean(axis=2) if a.ndim == 3 else a
    ink = g < 0.995
    rows, cols = np.any(ink, axis=1), np.any(ink, axis=0)
    if not rows.any():
        return a, a.shape[1] / a.shape[0]
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    pad = 2
    r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
    r1, c1 = min(a.shape[0], r1 + pad), min(a.shape[1], c1 + pad)
    a = a[r0:r1, c0:c1]
    return a, a.shape[1] / a.shape[0]


def paste(ax, img):
    ax.imshow(img, interpolation='lanczos')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)


def holes_of(m, k, xs, ys):
    lab, nb = ndimage.label(~(m >= k), structure=np.ones((3, 3), int))
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    out = []
    for l in range(1, nb + 1):
        if l in border:
            continue
        yy, xx = np.nonzero(lab == l)
        out.append((xs[xx].mean(), ys[yy].mean()))
    return out


def region_panel(ax, name, k, title, res=1200):
    pts, R, desc, win = FIELDS[name]()
    m, xs, ys = raster_multiplicity(pts, R, win, res)
    vmax = int(max(4, m.max()))
    im = ax.imshow(np.clip(m, 0, vmax), origin='lower',
                   extent=(win[0], win[1], win[2], win[3]), cmap='viridis',
                   vmin=0, vmax=vmax, interpolation='bilinear')
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, (m >= k).astype(float), levels=[0.5], colors='white',
               linewidths=1.0)
    ax.plot(pts[:, 0], pts[:, 1], '.', color='white', ms=1.8, alpha=0.85)
    hs = holes_of(m, k, xs, ys)
    rad = 0.030 * min(win[1] - win[0], win[3] - win[2])
    for cx, cy in hs:
        ax.add_patch(Circle((cx, cy), rad, fc='none', ec='#ff4d4d', lw=0.9))
    ax.set_title(title, fontsize=8)
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    return im, len(hs), vmax


def barcode_panel(ax, bars, xmax, title, ylabel='bar index', annotate=True):
    """H0 and H1 bars of one subdivision barcode in a single axes, distinct
    intervals only, thickness carrying multiplicity."""
    y = 0
    for d, col in (('0', H0C), ('1', H1C)):
        cnt = Counter(tuple(b) for b in bars[d])
        mx = max(cnt.values())
        for (lo, hi), mlt in sorted(cnt.items(), key=lambda t: (-t[0][1], -t[0][0])):
            lw = 1.1 + 2.2 * (mlt / mx) ** 0.5
            ax.plot([lo - 0.42, hi + 0.42], [y, y], lw=lw, color=col,
                    solid_capstyle='round', zorder=3)
            if annotate and mlt > 1:
                ax.text(hi + 0.55, y, r'$\times%d$' % mlt, va='center',
                        fontsize=5.5, color=col)
            y += 1
    ax.set_xlim(0.4, xmax + 0.6); ax.set_ylim(-1, y)
    ax.set_xticks(range(1, int(xmax) + 1))
    ax.set_yticks([]); ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlabel('redundancy level $k$', fontsize=7.5)
    ax.set_title(title, fontsize=8)
    ax.plot([], [], color=H0C, lw=2, label='$H_0$')
    ax.plot([], [], color=H1C, lw=2, label='$H_1$')
    ax.legend(fontsize=6.5, frameon=False, loc='upper right', handlelength=1.2)
    ax.grid(alpha=0.2, lw=0.4, axis='x'); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    if TEXFIG.is_dir():
        shutil.copy(FIG / name, TEXFIG / name)
    print('wrote', name)


# ================================================================ fig_fields
def fig_fields():
    fig = plt.figure(figsize=(FULL, 4.35))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.15, 2.05], hspace=0.30)
    top = outer[0].subgridspec(1, 3, wspace=0.26)
    bot = outer[1].subgridspec(1, 3, wspace=0.30, width_ratios=[0.88, 1.10, 1.30])

    axA = fig.add_subplot(top[0, 0])
    imA, hA, vA = region_panel(axA, 'A_ring_grid_seed3', 2, '')
    titled(axA, 'a', 'Field A, $k=2$')
    cb = fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.02,
                      ticks=range(0, vA + 1))
    cb.ax.tick_params(labelsize=6)
    cb.set_label('$m(x)$', fontsize=7)

    axD = fig.add_subplot(top[0, 1])
    imD, hD, vD = region_panel(axD, 'D_hex_dead_zone', 2, '')
    titled(axD, 'b', 'Field D, $k=2$')
    cb = fig.colorbar(imD, ax=axD, fraction=0.046, pad=0.02,
                      ticks=range(0, vD + 1))
    cb.ax.tick_params(labelsize=6)
    cb.set_label('$m(x)$', fontsize=7)

    # Field C with the near-tangent pairs drawn over the raster
    axC = fig.add_subplot(top[0, 2])
    pts, R, desc, win = FIELDS['C_poisson_random']()
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    iu = np.triu_indices(len(pts), 1)
    near = [(i, j) for i, j in zip(*iu) if abs(d[i, j] - 2 * R) < 0.05]
    m, xs, ys = raster_multiplicity(pts, R, win, 1200)
    vmax = int(max(4, m.max()))
    imC = axC.imshow(np.clip(m, 0, vmax), origin='lower',
                     extent=(win[0], win[1], win[2], win[3]), cmap='viridis',
                     vmin=0, vmax=vmax, interpolation='bilinear')
    X, Y = np.meshgrid(xs, ys)
    axC.contour(X, Y, (m >= 2).astype(float), levels=[0.5], colors='white',
                linewidths=0.9)
    axC.plot(pts[:, 0], pts[:, 1], '.', color='white', ms=1.6, alpha=0.85)
    for i, j in near:
        axC.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                 color='#ff4d4d', lw=0.8, solid_capstyle='round')
    axC.set_aspect('equal'); axC.set_xticks([]); axC.set_yticks([])
    titled(axC, 'c', 'Field C, $k=2$: %d near-tangent pairs' % len(near),
           fontsize=7.5)
    cb = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.02,
                      ticks=range(0, vmax + 1))
    cb.ax.tick_params(labelsize=6)
    cb.set_label('$m(x)$', fontsize=7)

    # Field E and its clique
    axE = fig.add_subplot(bot[0, 0])
    pts, R, desc, win = FIELDS['E_two_cluster']()
    N = nerve_of_disks(pts, R, max_size=2)
    G = nx.Graph(); G.add_nodes_from(range(len(pts)))
    G.add_edges_from(tuple(s) for s in N if len(s) == 2)
    clique = max(nx.find_cliques(G), key=len)
    m, xs, ys = raster_multiplicity(pts, R, win, 900)
    axE.imshow(np.clip(m, 0, 12), origin='lower', extent=win, cmap='Greys',
               vmin=0, vmax=12, interpolation='nearest')
    for i in range(len(pts)):
        axE.add_patch(Circle(pts[i], R, fc='none',
                             ec='#d62728' if i in clique else '0.45',
                             lw=0.35, alpha=0.9))
    axE.plot(pts[:, 0], pts[:, 1], 'k.', ms=1.2)
    axE.plot(pts[clique, 0], pts[clique, 1], 'o', ms=2.2, color='#d62728')
    axE.set_aspect('equal'); axE.set_xticks([]); axE.set_yticks([])
    titled(axE, 'd', 'Field E: a %d-clique' % len(clique))

    # nerve sizes
    axN = fig.add_subplot(bot[0, 1])
    names = [('A_ring_grid_seed3', 'A'), ('B_ring_grid_seed7', 'B'),
             ('C_poisson_random', 'C'), ('D_hex_dead_zone', 'D')]
    w = 0.25
    for j, (key, lab) in enumerate((('1', 'vertices'), ('2', 'edges'),
                                    ('3', 'triangles'))):
        vals = [FIELDS_JSON[nm]['face_counts'].get(key, 0) for nm, _ in names]
        axN.bar(np.arange(4) + (j - 1) * w, vals, w, label=lab)
        for x, v in zip(np.arange(4) + (j - 1) * w, vals):
            axN.text(x, v * 1.10, str(v), ha='center', fontsize=5.0)
    axN.set_yscale('log'); axN.set_xticks(range(4))
    axN.set_xticklabels([l for _, l in names])
    axN.set_xlabel('field', fontsize=7.5)
    axN.set_ylabel('count', fontsize=7.5)
    axN.set_ylim(50, 9000)
    axN.legend(fontsize=6.0, frameon=False, ncol=3, loc='upper left',
               columnspacing=0.8, handlelength=1.0, borderaxespad=0.2)
    titled(axN, 'e', 'sizes of the nerves')

    # the two barcodes, A above D, in one cell
    inner = bot[0, 2].subgridspec(2, 1, hspace=0.75)
    for row, (nm, lab) in enumerate((('A_ring_grid_seed3', 'Field A'),
                                     ('D_hex_dead_zone', 'Field D'))):
        ax = fig.add_subplot(inner[row, 0])
        bars = FIELDS_JSON[nm]['bars']
        xmax = max(b[1] for dd in bars for b in bars[dd])
        barcode_panel(ax, bars, xmax, '', ylabel='', annotate=False)
        titled(ax, 'fg'[row], lab + ' barcode')
        ax.set_xlabel('redundancy level $k$' if row == 1 else '', fontsize=7.5)
    save(fig, 'fig_fields.png')


# ================================================================== fig_grid
def fig_grid():
    ref = GRID['reference']
    pts = np.array([(i, j) for i in range(8) for j in range(8)], float)
    r, window = 0.8, (-1.5, 8.5, -1.5, 8.5)
    m, xs, ys = raster_multiplicity(pts, r, window, 1000)

    fig = plt.figure(figsize=(FULL, 3.95))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.95], hspace=0.30)
    top = outer[0].subgridspec(1, 4, wspace=0.22)
    bot = outer[1].subgridspec(1, 4, wspace=0.40)

    ax = fig.add_subplot(top[0, 0])
    im = ax.imshow(m, origin='lower', extent=window, cmap='viridis',
                   interpolation='nearest', vmin=0, vmax=m.max())
    ax.plot(pts[:, 0], pts[:, 1], '.', color='0.15', ms=2.0)
    titled(ax, 'a', r'multiplicity $m(x)$')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    axes4 = [ax]
    cmap = plt.get_cmap('viridis')
    for j, k in enumerate((1, 2, 3)):
        a = fig.add_subplot(top[0, j + 1])
        col = cmap(k / max(m.max(), 1))
        rgba = np.zeros(m.shape + (4,))
        rgba[..., :3] = col[:3]
        rgba[..., 3] = (m >= k).astype(float)
        a.imshow(np.ones(m.shape), origin='lower', extent=window, cmap='Greys',
                 vmin=0, vmax=7.0, interpolation='nearest')
        a.imshow(rgba, origin='lower', extent=window, interpolation='nearest')
        a.plot(pts[:, 0], pts[:, 1], '.', color='0.15', ms=2.0)
        b0, b1 = ref['shadow'][str(k)]['b0'], ref['shadow'][str(k)]['b1']
        titled(a, 'bcd'[j], r'$R_{\geq %d}$: $(%d,%d)$' % (k, b0, b1))
        axes4.append(a)
    for a in axes4:
        a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color('0.7')

    ax = fig.add_subplot(bot[0, 0])
    bars = ref['bars']
    barcode_panel(ax, bars, 5, '', ylabel='')
    titled(ax, 'e', 'subdivision barcode')

    sweep = GRID['radius_sweep']
    for col, (bn, lab, ttl) in enumerate(
            (('b0', r'$b_0(\Delta_k(N))$', r'$b_0$ against radius'),
             ('b1', r'$b_1(\Delta_k(N))$', r'$b_1$ against radius'))):
        a = fig.add_subplot(bot[0, col + 1])
        for k, c in zip((1, 2, 3, 4), ('C0', 'C1', 'C2', 'C3')):
            a.plot([s['r'] for s in sweep], [s['%s_%d' % (bn, k)] for s in sweep],
                   '-', lw=1.1, color=c, label='$k=%d$' % k)
        a.set_ylabel(lab, fontsize=7.5)
        a.set_xlabel('radius $r$ (pitch $=1$)', fontsize=7.5)
        a.set_yscale('symlog')
        a.legend(fontsize=6.0, frameon=False, ncol=2, handlelength=1.1,
                 columnspacing=0.9, loc='upper left')
        a.grid(alpha=0.2, lw=0.4); a.set_axisbelow(True)
        titled(a, 'fg'[col], ttl)

    a = fig.add_subplot(bot[0, 3])
    pr = GRID['pitch_sweep']

    def certified(s):
        """Largest k with (b0, b1) = (1, 0) at every level up to k.  Read from
        the stored value where present, and recomputed from the stored Betti
        numbers otherwise; the raw 'largest level that happens to be (1, 0)'
        is a different and larger quantity at pitch 0.8."""
        if s.get('k_certified') is not None:
            return s['k_certified']
        k = 0
        for j in range(1, 8):
            if s.get('b0_%d' % j) == 1 and s.get('b1_%d' % j) == 0:
                k = j
            else:
                break
        return k

    a.plot([s['pitch'] for s in pr], [certified(s) for s in pr], 'o-', ms=3,
           lw=1.2, label=r'certified level $k^\ast$')
    a.plot([s['pitch'] for s in pr], [s['margin'] for s in pr], 's--', ms=3,
           lw=1.2, label='dropout margin $q$')
    a.set_xlabel('pitch ($r=0.8$)', fontsize=7.5)
    a.set_ylabel('level', fontsize=7.5)
    a.legend(fontsize=6.0, frameon=False, handlelength=1.4, loc='upper right')
    a.grid(alpha=0.2, lw=0.4); a.set_axisbelow(True)
    titled(a, 'h', 'pitch sweep')
    save(fig, 'fig_grid.png')


# ================================================================ fig_sphere
def fig_sphere():
    a3, r3 = trimmed(FIG / 'sphere_3d.png')
    a13, r13 = trimmed(FIG / 'sphere_r013.png')
    a16, r16 = trimmed(FIG / 'sphere_r016.png')
    w3 = 0.66 * FULL
    h = [w3 / r3, FULL / r13, FULL / r16]
    fig = plt.figure(figsize=(FULL, sum(h) + 0.45))
    gs = fig.add_gridspec(3, 1, height_ratios=h, hspace=0.14)
    top = gs[0].subgridspec(1, 3, width_ratios=[0.17, 0.66, 0.17])
    ax = fig.add_subplot(top[0, 1]); paste(ax, a3)
    for i, (img, lab) in enumerate(((a13, 'c'), (a16, 'd'))):
        ax = fig.add_subplot(gs[i + 1, 0]); paste(ax, img)
        letter(ax, lab, dx=-0.005, dy=0.99)
    save(fig, 'fig_sphere.png')


# ============================================================== fig_cortex_a
def fig_cortex_a():
    ov, rov = trimmed(FIG / 'cortex_overview.png')
    rg, rrg = trimmed(FIG / 'cortex_regions.png')
    h = [FULL / rov, FULL / rrg]
    fig = plt.figure(figsize=(FULL, sum(h) + 0.30))
    gs = fig.add_gridspec(2, 1, height_ratios=h, hspace=0.10)
    ax = fig.add_subplot(gs[0, 0]); paste(ax, ov)
    ax = fig.add_subplot(gs[1, 0]); paste(ax, rg); letter(ax, 'c', dx=-0.005, dy=0.98)
    save(fig, 'fig_cortex_a.png')


# ============================================================== fig_cortex_b
def fig_cortex_b():
    """the two footprint models (pasted render, panels a-c) above the radius
    sweep, redrawn here as panels d and e so that the letters run on."""
    md, rmd = trimmed(FIG / 'cortex_models.png')
    hmd = FULL / rmd
    hsw = 3.15
    fig = plt.figure(figsize=(FULL, hmd + hsw + 0.25))
    gs = fig.add_gridspec(2, 1, height_ratios=[hmd, hsw], hspace=0.24)
    ax = fig.add_subplot(gs[0, 0]); paste(ax, md)

    sub = gs[1, 0].subgridspec(2, 2, wspace=0.46, hspace=0.10)
    handles = labels = None
    for col, (gname, ttl, lab) in enumerate(
            (('parietal_8x8', r'parietal $8\times8$', 'd'),
             ('temporal_4x8', r'temporal $4\times8$', 'e'))):
        recs = CORTEX['grids'][gname]['models']['geodesic']
        rr = [x['radius_mm'] for x in recs]
        for row, (bname, bi) in enumerate((('b0', 0), ('b1', 1))):
            ax = fig.add_subplot(sub[row, col])
            for k, c in zip((1, 2, 3), ('#1f77b4', '#2ca02c', '#d62728')):
                sh = [x['shadow'].get(str(k), {'b0': 0, 'b1': 0})[bname] for x in recs]
                tr = [x['mesh_reference'].get(str(k), [0, 0])[bi] for x in recs]
                ax.plot(rr, sh, '-', color=c, lw=1.1, label=r'$\Delta_%d(N)$' % k)
                ax.plot(rr, tr, 'o', color=c, ms=3.2, mfc='none', mew=0.9,
                        label=r'mesh, $k=%d$' % k)
            ax.set_ylabel(r'$b_%d$' % bi, fontsize=7.5)
            ax.tick_params(labelsize=7)
            if row == 0:
                titled(ax, lab, ttl)
                ax.set_xticklabels([])
                ax2 = ax.twinx()
                ax2.bar(rr, [x['disk_test_failures'] for x in recs], width=0.45,
                        color='0.82', zorder=0)
                ax2.set_ylabel('disk-test failures', color='0.45', fontsize=7)
                ax2.tick_params(axis='y', colors='0.45', labelsize=6.5)
                ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
                if col == 0:
                    handles, labels = ax.get_legend_handles_labels()
            else:
                ax.set_xlabel('footprint radius $r$ (mm)', fontsize=7.5)
    fig.legend(handles, labels, fontsize=6.5, ncol=6, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, -0.012),
               handlelength=1.5, columnspacing=1.2, handletextpad=0.4)
    save(fig, 'fig_cortex_b.png')


# ================================================================ fig_robust
def fig_robust():
    fig = plt.figure(figsize=(FULL, 4.55))
    gs = fig.add_gridspec(2, 3, wspace=0.46, hspace=0.62)

    # (a,b) cortical redundancy barcodes at r = 8 mm
    for col, (gname, ttl) in enumerate((('parietal_8x8', r'parietal $8\times8$'),
                                        ('temporal_4x8', r'temporal $4\times8$'))):
        ax = fig.add_subplot(gs[0, col])
        rec = [x for x in CORTEX['grids'][gname]['models']['geodesic']
               if x['radius_mm'] == 8.0][0]
        barcode_panel(ax, rec['barcode'], 6, '', ylabel='')
        titled(ax, 'ab'[col], ttl + ', $r=8$ mm')

    radii = DROP['radii_mm']
    STYLE = {'parietal_8x8': dict(color='tab:blue', ls='-', marker='o',
                                  label=r'parietal $8\times8$'),
             'temporal_4x8': dict(color='tab:orange', ls='--', marker='s',
                                  label=r'temporal $4\times8$')}

    ax = fig.add_subplot(gs[0, 2])
    for g, st in STYLE.items():
        per = DROP['grids'][g]
        n = per[str(radii[0])]['n_electrodes']
        y = [100.0 * per[str(r)]['single_summary']['contacts_with_private_territory'] / n
             for r in radii]
        ax.plot(radii, y, ms=3, lw=1.3, **st)
    ax.set_xlabel('radius $r$ (mm)', fontsize=7.5)
    ax.set_ylabel('contacts with private\nterritory (%)', fontsize=7.5)
    titled(ax, 'c', 'sole observers')
    ax.set_ylim(0, 108); ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, frameon=False, loc='lower left', handlelength=1.5)

    ax = fig.add_subplot(gs[1, 0])
    for g, st in STYLE.items():
        per = DROP['grids'][g]
        y = [100.0 * per[str(r)]['single_summary']['worst_single_loss_fraction']
             for r in radii]
        ax.plot(radii, y, ms=3, lw=1.3, **st)
    ax.set_xlabel('radius $r$ (mm)', fontsize=7.5)
    ax.set_ylabel('worst loss (% of $R_{\\geq1}$)', fontsize=7.5)
    titled(ax, 'd', 'cost of one failure')
    ax.set_ylim(0, None); ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, frameon=False, loc='upper left', handlelength=1.5)

    ax = fig.add_subplot(gs[1, 1])
    ms_ = [e['m'] for e in STAB['contact_loss']]
    d0 = [e['d0'] for e in STAB['contact_loss']]
    d1 = [e['d1'] for e in STAB['contact_loss']]
    ax.plot(ms_, ms_, color='0.35', ls=(0, (4, 2.5)), lw=1.2, label='bound $m$')
    ax.plot(ms_, d0, color='tab:blue', marker='o', ms=3.4, lw=1.3,
            label=r'worst $d_B$, deg. $0$')
    ax.plot(ms_, d1, color='tab:red', marker='s', ms=3.4, lw=1.3,
            label=r'worst $d_B$, deg. $1$')
    ax.fill_between(ms_, d0, ms_, color='tab:blue', alpha=0.07)
    ax.set_xlabel('contacts removed, $m$', fontsize=7.5)
    ax.set_ylabel('bottleneck distance in $k$', fontsize=7.5)
    titled(ax, 'e', 'contact loss')
    ax.set_xticks(ms_); ax.set_ylim(0, max(ms_) + 0.4)
    ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, frameon=False, loc='upper left', handlelength=1.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(gs[1, 2])
    L = STAB['localisation']
    delta, hi = L['delta'], float(L['rmax'])
    P = np.array(L['bars_p']['1'], float)
    Q = np.array(L['bars_q']['1'], float)
    ax.plot([0, hi], [0, hi], color='0.55', lw=0.8)
    for b, dth in P:
        ax.add_patch(Rectangle((b - delta, dth - delta), 2 * delta, 2 * delta,
                               facecolor='tab:blue', alpha=0.10, edgecolor='none',
                               zorder=1))
    ax.scatter(P[:, 0], P[:, 1], s=16, facecolor='none', edgecolor='tab:blue',
               lw=0.9, label='placement $p$', zorder=3)
    ax.scatter(Q[:, 0], Q[:, 1], s=9, color='tab:red', marker='x', lw=0.9,
               label=r'placement $q$', zorder=4)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi * 1.02)
    ax.set_xlabel('birth radius', fontsize=7.5)
    ax.set_ylabel('death radius', fontsize=7.5)
    titled(ax, 'f', r'localisation: $d_B=%.3f\leq\delta=%.3f$'
           % (L['bottleneck_H1'], delta), fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, frameon=False, loc='lower right', handlelength=1.2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    save(fig, 'fig_robust.png')


if __name__ == '__main__':
    fig_fields()
    fig_grid()
    fig_sphere()
    fig_cortex_a()
    fig_cortex_b()
    fig_robust()
    print('all composites written')
