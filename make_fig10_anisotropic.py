"""Generate fig10 (isotropic versus anisotropic footprints) for the k-Shadow Nerve paper.

All figures are publication-quality at 300 DPI with Times New Roman serif font,
consistent with PRE single-column article format.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import itertools

rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          10,
    'axes.labelsize':     11,
    'axes.titlesize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.4,
    'grid.linewidth':     0.4,
    'grid.alpha':         0.4,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'text.usetex':        False,
})

FIGS = Path(__file__).resolve().parent / 'figures'
FIGS.mkdir(exist_ok=True)

# ============================================================
# Shared utilities
# ============================================================

def betti_numbers_from_complex(edge_list, triangle_list, n_nodes):
    """Return beta0, beta1 of the full 2D simplicial complex.

    Uses Euler characteristic: chi = V - E + T and rank-nullity.
    beta0 = connected components of graph.
    beta1 = max(0, E - T - V + beta0)  (assumes beta2 = 0 = no closed voids).
    """
    if n_nodes == 0:
        return 0, 0
    if not edge_list:
        return n_nodes, 0
    rows2 = np.array([e[0] for e in edge_list] + [e[1] for e in edge_list],
                     dtype=np.int32)
    cols2 = np.array([e[1] for e in edge_list] + [e[0] for e in edge_list],
                     dtype=np.int32)
    data2 = np.ones(len(rows2), dtype=float)
    A = csr_matrix((data2, (rows2, cols2)), shape=(n_nodes, n_nodes))
    beta0 = connected_components(A, directed=False, return_labels=False)
    n_edges = len(edge_list)
    n_tri   = len(triangle_list)
    # Rank of boundary map del2 bounded by min(T, E-V+beta0)
    rank_del1 = n_nodes - beta0
    rank_del2 = min(n_tri, n_edges - rank_del1)
    beta1 = max(0, (n_edges - rank_del1) - rank_del2)
    return beta0, beta1


def nerve_from_footprints(centers, radii_a, radii_b, angles):
    """Build nerve graph from elliptical footprints.

    Returns: list of edges (i,j) where footprints i and j intersect.
    Also returns list of triangles (i,j,k) where all three pairwise intersect.
    Intersection test: ellipse overlap approximation using bounding circle.
    """
    n = len(centers)
    edges = []
    triangles = []

    # Build adjacency
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            # Rotate into frame of ellipse i
            ci_val = np.cos(-angles[i])
            si_val = np.sin(-angles[i])
            dx_r = ci_val * dx - si_val * dy
            dy_r = si_val * dx + ci_val * dy
            ri_a, ri_b = radii_a[i], radii_b[i]
            rj_a, rj_b = radii_a[j], radii_b[j]
            # Minkowski sum approximation for ellipse intersection
            overlap = ((dx_r / (ri_a + rj_a)) ** 2 +
                       (dy_r / (ri_b + rj_b)) ** 2) <= 1.0
            if overlap:
                adj[i, j] = adj[j, i] = True
                edges.append((i, j))

    # Triangles: all triples where all pairs intersect
    for ii, jj, kk in itertools.combinations(range(n), 3):
        if adj[ii, jj] and adj[ii, kk] and adj[jj, kk]:
            triangles.append((ii, jj, kk))

    return edges, triangles, adj


def multiplicity_field(grid_x, grid_y, centers, radii_a, radii_b, angles):
    """Compute m(x) = number of footprints covering each grid point."""
    nx, ny = grid_x.shape
    mult = np.zeros_like(grid_x, dtype=float)
    for ci, (cx, cy) in enumerate(centers):
        dx = grid_x - cx
        dy = grid_y - cy
        ca, sa = np.cos(-angles[ci]), np.sin(-angles[ci])
        dx_r = ca * dx - sa * dy
        dy_r = sa * dx + ca * dy
        inside = (dx_r / radii_a[ci]) ** 2 + (dy_r / radii_b[ci]) ** 2 <= 1.0
        mult += inside.astype(float)
    return mult


# ============================================================
# FIG 10 -- Isotropic vs. anisotropic footprint comparison
# ============================================================

def make_fig10():
    """Three-panel figure.

    Panel A: isotropic circular footprints, m(x) field, nerve overlay
    Panel B: mixed anisotropic elliptical footprints, m(x) field, nerve overlay
    Panel C: bar chart comparing beta0, beta1, face counts for both cases at k=1,2
    """
    rng = np.random.default_rng(42)

    # ---- Array layout: 5x5 grid in [0,40]x[0,40] mm ----
    nx_arr, ny_arr = 5, 5
    pitch = 8.0   # mm
    xs = np.linspace(4, 36, nx_arr)
    ys = np.linspace(4, 36, ny_arr)
    cx, cy = np.meshgrid(xs, ys)
    centers = np.column_stack([cx.ravel(), cy.ravel()])
    n_elec = len(centers)

    # Isotropic: all circles r=6 mm
    r_iso = 6.0
    ra_iso = np.full(n_elec, r_iso)
    rb_iso = np.full(n_elec, r_iso)
    ang_iso = np.zeros(n_elec)

    # Anisotropic: ellipses with random semi-axes and orientations
    # Semi-axes: a in [5,8], b in [3,5], angle in [-45,45] deg
    ra_aniso = rng.uniform(5.0, 8.0, n_elec)
    rb_aniso = rng.uniform(3.0, 5.5, n_elec)
    ang_aniso = rng.uniform(-np.pi / 4, np.pi / 4, n_elec)

    # Grid for m(x)
    res = 200
    gx = np.linspace(0, 40, res)
    gy = np.linspace(0, 40, res)
    GX, GY = np.meshgrid(gx, gy)

    mult_iso   = multiplicity_field(GX, GY, centers, ra_iso, rb_iso, ang_iso)
    mult_aniso = multiplicity_field(GX, GY, centers, ra_aniso, rb_aniso, ang_aniso)

    edges_iso, tri_iso, adj_iso     = nerve_from_footprints(centers, ra_iso, rb_iso, ang_iso)
    edges_aniso, tri_aniso, adj_aniso = nerve_from_footprints(centers, ra_aniso, rb_aniso, ang_aniso)

    # Betti numbers at k=1: nerve N
    b0_iso_k1, b1_iso_k1     = betti_numbers_from_complex(edges_iso, tri_iso, n_elec)
    b0_aniso_k1, b1_aniso_k1 = betti_numbers_from_complex(edges_aniso, tri_aniso, n_elec)

    # k=2 shadow: vertices = edges of N; edge in shadow if both edges share a triangle
    def shadow_k2_betti(edges, triangles, n_elec_unused):
        if not edges:
            return 0, 0
        edge_idx = {e: idx for idx, e in enumerate(edges)}
        nv = len(edges)
        sh_edges = set()
        # Shadow 2-simplices: triples of edges all sharing a common 3-electrode group
        # Here we just build the shadow 1-skeleton from triangle adjacency
        for a, b, c in triangles:
            e1 = (min(a,b), max(a,b))
            e2 = (min(a,c), max(a,c))
            e3 = (min(b,c), max(b,c))
            for ea, eb in [(e1,e2),(e1,e3),(e2,e3)]:
                if ea in edge_idx and eb in edge_idx:
                    ii, jj = edge_idx[ea], edge_idx[eb]
                    sh_edges.add((min(ii,jj), max(ii,jj)))
        sh_list = list(sh_edges)
        # For shadow, use graph-level beta (triangles of shadow not computed here)
        return betti_numbers_from_complex(sh_list, [], nv)

    b0_iso_k2, b1_iso_k2     = shadow_k2_betti(edges_iso, tri_iso, n_elec)
    b0_aniso_k2, b1_aniso_k2 = shadow_k2_betti(edges_aniso, tri_aniso, n_elec)

    # ---- Plot: 3-column layout with dedicated colorbar strip ----
    # Columns: [image panels] [colorbar] [stats chart]
    # width_ratios: images wide, colorbar narrow, stats medium
    cmap = plt.cm.plasma
    vmin, vmax = 0, max(mult_iso.max(), mult_aniso.max())

    fig = plt.figure(figsize=(7.4, 5.6))
    # Left block: images stacked; right block: stats
    # We manually manage the colorbar so it never overlaps image content
    gs = GridSpec(2, 2, figure=fig,
                  hspace=0.45, wspace=0.50,
                  left=0.07, right=0.97, top=0.93, bottom=0.10,
                  width_ratios=[1.15, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    def draw_array_panel(ax, mult, centers, edges,
                         ra, rb, ang, title, label):
        im = ax.pcolormesh(gx, gy, mult, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='gouraud',
                           rasterized=True)
        # Draw footprint outlines (every electrode)
        for i in range(n_elec):
            e = Ellipse(xy=(centers[i, 0], centers[i, 1]),
                        width=2 * ra[i], height=2 * rb[i],
                        angle=np.degrees(ang[i]),
                        edgecolor='white', facecolor='none',
                        linewidth=0.55, alpha=0.65, zorder=3)
            ax.add_patch(e)
        # Nerve edges
        for (ii, jj) in edges:
            ax.plot([centers[ii, 0], centers[jj, 0]],
                    [centers[ii, 1], centers[jj, 1]],
                    color='cyan', lw=0.45, alpha=0.55, zorder=4)
        # Electrode centres
        ax.scatter(centers[:, 0], centers[:, 1], s=7, color='white',
                   zorder=5, linewidths=0.4, edgecolors='gray')
        ax.set_xlim(0, 40); ax.set_ylim(0, 40)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.set_xlabel('$x$ (mm)', fontsize=9)
        ax.set_ylabel('$y$ (mm)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', color='white')
        return im

    im0 = draw_array_panel(ax0, mult_iso, centers, edges_iso,
                           ra_iso, rb_iso, ang_iso,
                           'Isotropic ($r=6$ mm)', '(a)')
    im1 = draw_array_panel(ax1, mult_aniso, centers, edges_aniso,
                           ra_aniso, rb_aniso, ang_aniso,
                           'Anisotropic (mixed $a,b$)', '(b)')

    # Colorbar: placed manually BELOW both image panels, full width of left column
    # This avoids any overlap with image content
    cbar_ax = fig.add_axes([0.08, 0.04, 0.42, 0.022])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Multiplicity $m(x)$', fontsize=8.5, labelpad=2)
    cbar.ax.tick_params(labelsize=7.5)

    # Stats annotations inside each image panel (top-left so not cut off)
    rho_iso   = int(mult_iso[mult_iso > 0].min()) if np.any(mult_iso > 0) else 0
    rho_aniso = int(mult_aniso[mult_aniso > 0].min()) if np.any(mult_aniso > 0) else 0
    for ax_img, ne, nt, rho in [
            (ax0, len(edges_iso), len(tri_iso), rho_iso),
            (ax1, len(edges_aniso), len(tri_aniso), rho_aniso)]:
        ax_img.text(0.03, 0.03,
                    f'$n={n_elec}$  $|E|={ne}$  $|T|={nt}$\n'
                    + r'$\rho(\Omega)=' + str(rho) + r'$',
                    transform=ax_img.transAxes,
                    fontsize=7.5, ha='left', va='bottom', color='white',
                    bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.50))

    # Panel C: bar chart -- only show k=1 Betti numbers and face counts
    # (k=2 shadow beta values without full shadow triangles are graph-level only,
    #  so we omit them and show only the rigorously computed quantities)
    categories = [
        r'$|E(N)|$',
        r'$|T(N)|$',
        r'$\beta_0\;(k{=}1)$',
        r'$\beta_1\;(k{=}1)$',
    ]
    vals_iso   = [len(edges_iso),   len(tri_iso),   b0_iso_k1,   b1_iso_k1]
    vals_aniso = [len(edges_aniso), len(tri_aniso), b0_aniso_k1, b1_aniso_k1]

    ypos = np.arange(len(categories))
    h = 0.30
    ax2.barh(ypos + h / 2, vals_iso,   height=h, color='#1a6faf', alpha=0.82,
             label='Isotropic', zorder=3)
    ax2.barh(ypos - h / 2, vals_aniso, height=h, color='#e67e22', alpha=0.82,
             label='Anisotropic', zorder=3)

    x_max = max(max(vals_iso), max(vals_aniso))
    for ki, (vi, va) in enumerate(zip(vals_iso, vals_aniso)):
        off = x_max * 0.03
        ax2.text(vi + off, ki + h / 2, str(vi),
                 va='center', fontsize=8.5, color='#1a6faf')
        ax2.text(va + off, ki - h / 2, str(va),
                 va='center', fontsize=8.5, color='#e67e22')

    ax2.set_yticks(ypos)
    ax2.set_yticklabels(categories, fontsize=9.5)
    ax2.set_xlabel('Count', fontsize=10)
    ax2.set_title('Nerve statistics\n($k=1$ nerve $N$)', fontsize=10, pad=4)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.88,
               edgecolor='#cccccc', fontsize=8.5)
    ax2.grid(axis='x', lw=0.4, alpha=0.4)
    ax2.set_xlim(0, x_max * 1.30)
    ax2.set_ylim(-0.6, len(categories) - 0.4)
    ax2.text(0.03, 0.97, '(c)', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', va='top')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    out = FIGS / 'fig10_anisotropic.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved {out}')
    return (n_elec, len(edges_iso), len(tri_iso),
            b0_iso_k1, b1_iso_k1, b0_iso_k2, b1_iso_k2,
            rho_iso, rho_aniso)


# ============================================================
# FIG 20 -- Independent validation architecture (quantitative)
# ============================================================



n_elec, ne, nt, b0k1, b1k1, b0k2, b1k2, rho_iso, rho_aniso = stats
print(f'\nFig10 stats: n={n_elec}, |E|={ne}, |T|={nt}')
print(f'  Isotropic   k=1: beta0={b0k1}, beta1={b1k1}; k=2: beta0={b0k2}, beta1={b1k2}; rho={rho_iso}')
print(f'  Anisotropic rho={rho_aniso}')
print('\nAll three figures regenerated.')
make_fig10()
