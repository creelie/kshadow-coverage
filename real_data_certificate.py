"""Execution-ready UCI29 real-data k-shadow certificate.

Expected directory: a locally downloaded FieldTrip SubjectUCI29 folder containing
  SubjectUCI29_elec_acpc_fr.mat
  freesurfer/surf/lh.pial
  freesurfer/surf/rh.pial (optional)

The script intentionally does not fabricate results when the source files are absent.
It uses the brain-shift-corrected electrode structure from the FieldTrip tutorial and
restricts the analysis to LPG/LTG surface electrodes.

Figures generated (always, without real data):
  figures/fig23_complexity_scaling.png  -- O(n) nerve vs O(2^n) naive complexity
  figures/fig24_persistence_barcode.png -- redundancy persistence barcode schematic
"""
from pathlib import Path
import heapq, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Global plot style: publication-quality, PRE-consistent
# ---------------------------------------------------------------------------
rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          11,
    'axes.labelsize':     12,
    'axes.titlesize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.6,
    'grid.linewidth':     0.5,
    'grid.alpha':         0.4,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'text.usetex':        False,
})

ROOT  = Path(__file__).resolve().parent
FIGS  = ROOT / 'figures'
FIGS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 23 -- Complexity scaling: O(n) nerve vs O(2^n) naive enumeration
# ---------------------------------------------------------------------------
def make_fig23():
    n = np.arange(1, 101)
    # Nerve under bounded local overlap: O(n * kappa) where kappa ~ constant
    # We model the realistic nerve as cn with c = average local clique budget
    kappa = 8          # average contacts per electrode in 4x4 ECoG pitch
    nerve_ops = kappa * n
    naive_ops = 2.0 ** (n / 10.0)   # scaled so curves cross near n=55 for visibility

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.semilogy(n, nerve_ops, color='#1a6faf', lw=2.0, label=r'Nerve enum. $O(n\kappa)$')
    ax.semilogy(n, naive_ops, color='#c0392b', lw=2.0, ls='--',
                label=r'Naive $2^{n/10}$ (scaled)')

    # Shade the gap
    ax.fill_between(n, nerve_ops, naive_ops,
                    where=(naive_ops > nerve_ops),
                    alpha=0.12, color='#c0392b', label='Complexity gap')

    # Annotate crossover
    cross_idx = np.argmin(np.abs(nerve_ops - naive_ops))
    ax.axvline(n[cross_idx], color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.text(n[cross_idx] + 1.5, 8e2, f'$n={n[cross_idx]}$',
            fontsize=9, color='gray', va='center')

    ax.set_xlabel('Number of electrodes $n$')
    ax.set_ylabel('Operations (log scale)')
    ax.set_title('Complexity: nerve enumeration vs.\ naive', pad=6)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_xlim(1, 100)
    ax.set_ylim(0.8, 2e4)
    ax.grid(True, which='both')

    # Minor ticks
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs='all', numticks=10))
    ax.tick_params(which='minor', length=2)

    fig.tight_layout()
    out = FIGS / 'fig23_complexity_scaling.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 24 -- Redundancy persistence barcode (schematic)
# ---------------------------------------------------------------------------
def make_fig24():
    # Bars: (label, birth, death, color, linestyle)
    bars = [
        # H0 -- connected components of R_{>=k}
        (r'$H_0$ component $\gamma_1$',   0.0,  6.0,  '#1a6faf', '-'),
        (r'$H_0$ component $\gamma_2$',   0.0,  4.5,  '#1a6faf', '-'),
        (r'$H_0$ component $\gamma_3$',   0.0,  2.8,  '#1a6faf', '-'),
        (r'$H_0$ component $\gamma_4$',   0.0, 12.0,  '#1a6faf', '-'),  # infinite bar
        # H1 -- loops / redundant loops
        (r'$H_1$ loop $\ell_1$',          1.2,  5.5,  '#c0392b', '-'),
        (r'$H_1$ loop $\ell_2$',          2.1,  3.8,  '#c0392b', '-'),
        (r'$H_1$ loop $\ell_3$',          3.5, 12.0,  '#c0392b', '-'),  # long-lived
        # H2 -- voids
        (r'$H_2$ void $v_1$',             4.0,  6.7,  '#27ae60', '--'),
    ]
    INF_VAL = 12.0    # represents infinity (draw as arrow)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bar_h   = 0.35
    y_pos   = list(range(len(bars)))

    for i, (label, birth, death, col, ls) in enumerate(bars):
        is_inf = (death >= INF_VAL)
        draw_d = INF_VAL - 0.3 if is_inf else death
        ax.barh(i, draw_d - birth, left=birth, height=bar_h,
                color=col, alpha=0.75, linewidth=0)
        # Draw bar outline
        ax.plot([birth, draw_d], [i - bar_h/2]*2, color=col, lw=0.6, ls=ls)
        ax.plot([birth, draw_d], [i + bar_h/2]*2, color=col, lw=0.6, ls=ls)
        ax.plot([birth, birth], [i - bar_h/2, i + bar_h/2], color=col, lw=0.8)
        if is_inf:
            ax.annotate('', xy=(draw_d + 0.4, i),
                        xytext=(draw_d, i),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.0))
        else:
            ax.plot([draw_d, draw_d], [i - bar_h/2, i + bar_h/2], color=col, lw=0.8)

    # Dropout threshold line
    q_thresh = 3.0
    ax.axvline(q_thresh, color='#8e44ad', lw=1.2, ls='--', zorder=5)
    ax.text(q_thresh + 0.1, len(bars) - 0.5,
            r'$q$-dropout threshold', fontsize=8.5,
            color='#8e44ad', va='top', rotation=0)

    # Axis labels and ticks
    ax.set_yticks(y_pos)
    ax.set_yticklabels([b[0] for b in bars], fontsize=9)
    ax.set_xlabel(r'Redundancy filtration parameter $k$', fontsize=11)
    ax.set_title('Redundancy persistence barcode', pad=6)
    ax.set_xlim(-0.2, 13.0)
    ax.set_ylim(-0.7, len(bars) - 0.3)

    # Legend patches
    h0_p  = mpatches.Patch(color='#1a6faf', alpha=0.75, label='$H_0$ (components)')
    h1_p  = mpatches.Patch(color='#c0392b', alpha=0.75, label='$H_1$ (loops)')
    h2_p  = mpatches.Patch(color='#27ae60', alpha=0.75, label='$H_2$ (voids)')
    inf_p = mpatches.Patch(color='none', label=r'$\rightarrow$ infinite bar')
    ax.legend(handles=[h0_p, h1_p, h2_p], loc='lower right',
              frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=8.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', lw=0.5, alpha=0.35)

    fig.tight_layout()
    out = FIGS / 'fig24_persistence_barcode.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Always generate the two publication figures
# ---------------------------------------------------------------------------
import matplotlib.ticker   # needed for LogLocator in fig23
make_fig23()
make_fig24()

# ---------------------------------------------------------------------------
# Real-data pipeline (requires SubjectUCI29 dataset)
# ---------------------------------------------------------------------------
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = Path('SubjectUCI29')
MAT  = DATA / 'SubjectUCI29_elec_acpc_fr.mat'
PIAL = DATA / 'freesurfer' / 'surf' / 'lh.pial'
OUT  = ROOT / 'real_data_results'

if not MAT.exists() or not PIAL.exists():
    raise SystemExit(
        'REAL-DATA RUN NOT EXECUTED: missing SubjectUCI29_elec_acpc_fr.mat and/or '
        'freesurfer/surf/lh.pial.\n'
        'Download the public FieldTrip SubjectUCI29 dataset, place it at '
        './SubjectUCI29, then rerun.\n'
        'Figures fig23 and fig24 have already been written to figures/.'
    )

OUT.mkdir(exist_ok=True)


def read_fs_surface(path):
    with open(path, 'rb') as f:
        b = int.from_bytes(f.read(3), 'big')
        if b not in (16777214, 16777215):
            raise ValueError('Not a FreeSurfer triangle surface')
        f.readline(); f.readline()
        nv = int.from_bytes(f.read(4), 'big', signed=False)
        nf = int.from_bytes(f.read(4), 'big', signed=False)
        v  = np.frombuffer(f.read(nv * 12), dtype='>f4').reshape(nv, 3).astype(float)
        tri = np.frombuffer(f.read(nf * 12), dtype='>i4').reshape(nf, 3).astype(np.int64)
    return v, tri


def field(obj, name):
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj[name]
    raise KeyError(name)


def unwrap(x):
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.flat[0]
    return x


def to_labels(x):
    x = unwrap(x)
    if isinstance(x, np.ndarray):
        return [str(unwrap(z)) for z in x.ravel()]
    return [str(x)]


raw = loadmat(MAT, squeeze_me=False, struct_as_record=False)
if 'elec_acpc_fr' not in raw:
    raise SystemExit('Could not find elec_acpc_fr in MATLAB file')
e      = unwrap(raw['elec_acpc_fr'])
labels = to_labels(field(e, 'label'))
pos    = np.asarray(unwrap(field(e, 'chanpos')), float)

keep   = np.array([lab.startswith('LPG') or lab.startswith('LTG') for lab in labels])
labels = [lab for lab, k in zip(labels, keep) if k]
pos    = pos[keep]
if len(labels) == 0:
    raise SystemExit('No LPG/LTG surface electrodes found')

V, F = read_fs_surface(PIAL)
edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
edges = np.unique(np.sort(edges, axis=1), axis=0)
d     = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
G     = csr_matrix(
    (np.r_[d, d],
     (np.r_[edges[:, 0], edges[:, 1]],
      np.r_[edges[:, 1], edges[:, 0]])),
    shape=(len(V), len(V))
)

idx = np.empty(len(pos), int)
for i, p in enumerate(pos):
    idx[i] = np.argmin(np.sum((V - p) ** 2, axis=1))


def dijkstra_cut(source, cutoff):
    dist = {source: 0.0}
    pq   = [(0.0, source)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u):
            continue
        if du > cutoff:
            continue
        for q in range(G.indptr[u], G.indptr[u + 1]):
            v  = int(G.indices[q])
            nd = du + float(G.data[q])
            if nd <= cutoff and nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


radii   = [6.0, 8.0, 10.0, 12.0, 15.0]
summary = []
for r in radii:
    balls   = [set(dijkstra_cut(s, r).keys()) for s in idx]
    mult    = np.zeros(len(V), dtype=np.uint16)
    for b in balls:
        mult[list(b)] += 1
    covered = np.flatnonzero(mult >= 1)
    double  = np.flatnonzero(mult >= 2)
    rho     = int(mult.min())

    pairs = []
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            if balls[i].intersection(balls[j]):
                pairs.append((i, j))

    pair_index = {p: q for q, p in enumerate(pairs)}
    sh_edges   = []
    for a, (i, j) in enumerate(pairs):
        for kk in range(j + 1, len(balls)):
            if ((min(i, kk), max(i, kk)) in pair_index and
                    (min(j, kk), max(j, kk)) in pair_index):
                if balls[i].intersection(balls[j]).intersection(balls[kk]):
                    sh_edges.append((a, pair_index[(min(i, kk), max(i, kk))]))
                    sh_edges.append((a, pair_index[(min(j, kk), max(j, kk))]))

    if pairs and sh_edges:
        A = csr_matrix(
            (np.ones(len(sh_edges)),
             (np.array(sh_edges)[:, 0], np.array(sh_edges)[:, 1])),
            shape=(len(pairs), len(pairs))
        )
        A    = A.maximum(A.T)
        ncomp = connected_components(A, directed=False, return_labels=False)
    elif pairs:
        ncomp = len(pairs)
    else:
        ncomp = 0

    summary.append(dict(
        radius_mm=r,
        n_surface_electrodes=len(labels),
        covered_vertices=int(len(covered)),
        double_vertices=int(len(double)),
        min_vertex_multiplicity=rho,
        shadow_vertices=len(pairs),
        shadow_components=int(ncomp),
    ))

(OUT / 'certificate_summary.json').write_text(
    json.dumps({'dataset': 'SubjectUCI29',
                'surface_electrodes': labels,
                'results': summary}, indent=2)
)
print(json.dumps(summary, indent=2))
