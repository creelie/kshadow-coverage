"""Synthetic flat electrode grids: exact k-shadow certificate versus raster
ground truth, redundancy barcode, radius and pitch sweeps, and a direct
check of the dropout guarantee.  Writes results/grid.json and figures."""
import json, os, sys, time, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kshadow import *

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
rng = np.random.default_rng(2026)
res = {}


def components_after(p, rad):
    """Number of components of the union of open disks = b0 of the nerve
    1-skeleton (Nerve Lemma at k = 1)."""
    Nq = nerve_of_disks(p, rad, max_size=2)
    parent = list(range(len(p)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for s in Nq:
        if len(s) == 2:
            a, b = tuple(s)
            parent[find(a)] = find(b)
    return len(set(find(i) for i in range(len(p))))

def grid(nx, ny, pitch):
    return np.array([(i * pitch, j * pitch) for i in range(nx) for j in range(ny)], float)

# --------------------------------------------------------------------------
# 1. reference 8x8 array, pitch 1, radius 0.8: multiplicity map, shadow vs
#    raster at every k, redundancy barcode, dropout check
# --------------------------------------------------------------------------
pts = grid(8, 8, 1.0)
r = 0.8
window = (-1.5, 8.5, -1.5, 8.5)
N = nerve_of_disks(pts, r)
sizes = {}
for s in N:
    sizes[len(s)] = sizes.get(len(s), 0) + 1
ref = {'n': 64, 'pitch': 1.0, 'r': 0.8, 'face_counts': {str(k): v for k, v in sorted(sizes.items())}}
ref['shadow'] = {}
for k in range(1, 6):
    b0, b1, sz = shadow_betti(N, k)
    ref['shadow'][str(k)] = {'b0': b0, 'b1': b1, 'V': sz[0], 'E': sz[1], 'T': sz[2]}
ref['raster'] = {}
for rr in (600, 1200):
    m, xs, ys = raster_multiplicity(pts, r, window, rr)
    ref['raster'][str(rr)] = {str(k): raster_betti(m >= k) for k in range(1, 6)}
bars = subdivision_persistence(N)
ref['bars'] = {str(d): bars[d] for d in bars}
ref['bars_betti'] = {str(k): betti_from_bars(bars, k) for k in range(1, 6)}
ref['dropout_margin'] = dropout_margin(N, 64)
print('reference', json.dumps({k: v for k, v in ref.items() if k != 'bars'}), flush=True)

# multiplicity figure
m, xs, ys = raster_multiplicity(pts, r, window, 1200)
fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.9))
im = ax[0].imshow(m, origin='lower', extent=window, cmap='viridis',
                  interpolation='nearest', vmin=0, vmax=m.max())
ax[0].plot(pts[:, 0], pts[:, 1], '.', color='0.15', ms=3.5)
ax[0].set_title(r'multiplicity $m(x)$', fontsize=11)
cb = plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)
cb.ax.tick_params(labelsize=8)
# the three regions in the colours of the multiplicity scale they come from,
# rather than as black silhouettes: the k-th region is drawn in the viridis
# colour of level k, so the panels read as a nested family
cmap = plt.get_cmap('viridis')
for j, k in enumerate((1, 2, 3)):
    col = cmap(k / max(m.max(), 1))
    rgba = np.zeros(m.shape + (4,))
    rgba[..., :3] = col[:3]
    rgba[..., 3] = (m >= k).astype(float)
    ax[j + 1].imshow(np.ones(m.shape), origin='lower', extent=window,
                     cmap='Greys', vmin=0, vmax=7.0, interpolation='nearest')
    ax[j + 1].imshow(rgba, origin='lower', extent=window, interpolation='nearest')
    ax[j + 1].plot(pts[:, 0], pts[:, 1], '.', color='0.15', ms=3.5)
    b0, b1 = ref['shadow'][str(k)]['b0'], ref['shadow'][str(k)]['b1']
    ax[j + 1].set_title(r'$R_{\geq %d}$:  $b_0=%d$,  $b_1=%d$' % (k, b0, b1),
                        fontsize=11)
for a in ax:
    a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color('0.7')
plt.tight_layout()
plt.savefig('figures/grid_multiplicity.png', dpi=220)
plt.close()

# barcode figure
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for j, d in enumerate((0, 1)):
    lst = bars[d]
    y = 0
    for lo, hi in sorted(lst, key=lambda x: (-x[1], -x[0])):
        ax[j].plot([lo - 0.45, hi + 0.45], [y, y], lw=2.0, color='C0' if d == 0 else 'C3')
        y += 1
    ax[j].set_xlabel('redundancy level $k$')
    ax[j].set_ylabel('bar index')
    ax[j].set_title('$H_%d$ bars (%d)' % (d, len(lst)))
    ax[j].set_xlim(0.4, 5.6)
    ax[j].set_xticks(range(1, 6))
plt.tight_layout()
plt.savefig('figures/grid_barcode.png', dpi=200)
plt.close()

# dropout guarantee check: q = dropout margin; remove random q electrodes
q = ref['dropout_margin']
trials = 300
fails = 0
fails_qp1 = 0
for t in range(trials):
    keep = np.ones(64, bool)
    keep[rng.choice(64, q, replace=False)] = False
    b0 = components_after(pts[keep], r)
    fails += (b0 != 1)
    keep = np.ones(64, bool)
    keep[rng.choice(64, q + 1, replace=False)] = False
    b0 = components_after(pts[keep], r)
    fails_qp1 += (b0 != 1)
ref['dropout_check'] = {'q': q, 'trials': trials, 'disconnected_at_q': int(fails),
                        'disconnected_at_q_plus_1': int(fails_qp1)}
# exhaustive check for q = margin when feasible
if q <= 2:
    bad = 0
    tot = 0
    for rem in itertools.combinations(range(64), q):
        keep = np.ones(64, bool); keep[list(rem)] = False
        b0 = components_after(pts[keep], r)
        tot += 1; bad += (b0 != 1)
    ref['dropout_check']['exhaustive_q'] = {'subsets': tot, 'disconnected': bad}
print('dropout', ref['dropout_check'], flush=True)
res['reference'] = ref

# --------------------------------------------------------------------------
# 2. radius sweep on the 8x8 array: nerve is piecewise constant in r
# --------------------------------------------------------------------------
radii = np.round(np.arange(0.50, 1.201, 0.025), 3)
sweep = []
prev_key = None
for rr in radii:
    Nr = nerve_of_disks(pts, rr)
    key = (len(Nr), tuple(sorted((len(s) for s in Nr))))
    row = {'r': float(rr), 'faces': len(Nr), 'changed': key != prev_key}
    for k in (1, 2, 3, 4):
        b0, b1, _ = shadow_betti(Nr, k)
        row['b0_%d' % k] = b0; row['b1_%d' % k] = b1
    row['margin'] = dropout_margin(Nr, 64)
    sweep.append(row)
    prev_key = key
res['radius_sweep'] = sweep
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for k, c in zip((1, 2, 3, 4), ('C0', 'C1', 'C2', 'C3')):
    ax[0].plot([s['r'] for s in sweep], [s['b0_%d' % k] for s in sweep], '-o', ms=3, color=c, label='$k=%d$' % k)
    ax[1].plot([s['r'] for s in sweep], [s['b1_%d' % k] for s in sweep], '-o', ms=3, color=c, label='$k=%d$' % k)
ax[0].set_ylabel(r'$b_0(\Delta_k(N))$'); ax[1].set_ylabel(r'$b_1(\Delta_k(N))$')
for a in ax:
    a.set_xlabel('footprint radius $r$ (pitch $=1$)'); a.legend(fontsize=8); a.set_yscale('symlog')
plt.tight_layout(); plt.savefig('figures/grid_radius_sweep.png', dpi=200); plt.close()

# --------------------------------------------------------------------------
# 3. pitch sweep with fixed radius: smallest k with b0=1,b1=0 and margin
# --------------------------------------------------------------------------
pitch_rows = []
for p in np.round(np.arange(0.6, 1.45, 0.05), 3):
    pp = grid(8, 8, p)
    Np = nerve_of_disks(pp, 0.8)
    row = {'pitch': float(p)}
    # The certified level is the largest k such that (b0, b1) = (1, 0) at
    # EVERY level up to k, not the largest level at which it happens to hold:
    # the property is not monotone in k.  At pitch 0.8, for instance,
    # Delta_1 and Delta_5 are both (1, 0) while Delta_2, Delta_3 and Delta_4
    # are not, so the certified level is 1 and not 5.
    kcert = 0
    khit = 0
    for k in range(1, 8):
        b0, b1, _ = shadow_betti(Np, k)
        row['b0_%d' % k] = b0; row['b1_%d' % k] = b1
        if b0 == 1 and b1 == 0:
            khit = k
            if kcert == k - 1:
                kcert = k
        if b0 == 0:
            break
    row['k_certified'] = kcert
    row['k_highest_hit'] = khit
    row['margin'] = dropout_margin(Np, 64)
    pitch_rows.append(row)
res['pitch_sweep'] = pitch_rows
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot([s['pitch'] for s in pitch_rows], [s['k_certified'] for s in pitch_rows], 'o-', label=r'certified level $k^\ast$')
ax.plot([s['pitch'] for s in pitch_rows], [s['margin'] for s in pitch_rows], 's--', label='dropout margin $q$')
ax.set_xlabel('electrode pitch (radius $r=0.8$)'); ax.set_ylabel('level'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/grid_pitch_sweep.png', dpi=200); plt.close()

json.dump(res, open('results/grid.json', 'w'), indent=1, default=str)
print('done')
