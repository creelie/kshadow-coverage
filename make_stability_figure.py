"""
make_stability_figure.py -- the figure for the two stability statements.

Panel (a): the worst bottleneck distance found between the redundancy barcode
of the full array and that of the array with m contacts removed, against the
bound m of the contact-loss proposition.

Panel (b): the degree-one persistence diagrams of the radius filtration of
the 2-shadow complex for a placement and for the same placement with every
contact moved by delta in a random direction, with a box of half-width delta
about each unperturbed point: the localisation proposition says every
perturbed point can be matched inside one of those boxes, or to the diagonal.

Writes figures/fig_stability.pdf and .png.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 9,
                     'figure.dpi': 200, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04})

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
S = json.load(open(ROOT / 'results' / 'stability.json'))

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))

# ------------------------------------------------------------------ (a)
ax = axes[0]
ms = [e['m'] for e in S['contact_loss']]
d0 = [e['d0'] for e in S['contact_loss']]
d1 = [e['d1'] for e in S['contact_loss']]
ax.plot(ms, ms, color='0.35', ls=(0, (4, 2.5)), lw=1.3, label='bound $m$')
ax.plot(ms, d0, color='tab:blue', marker='o', ms=4.5, lw=1.6,
        label=r'worst $d_B$, degree $0$')
ax.plot(ms, d1, color='tab:red', marker='s', ms=4.5, lw=1.6,
        label=r'worst $d_B$, degree $1$')
ax.fill_between(ms, d0, ms, color='tab:blue', alpha=0.07)
ax.set_xlabel('contacts removed, $m$')
ax.set_ylabel('bottleneck distance in $k$')
ax.set_title('(a) losing $m$ contacts moves the barcode by at most $m$',
             fontsize=8.5)
ax.set_xticks(ms)
ax.set_ylim(0, max(ms) + 0.4)
ax.grid(alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=7, frameon=False, loc='upper left')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

# ------------------------------------------------------------------ (b)
ax = axes[1]
L = S['localisation']
delta = L['delta']
P = np.array(L['bars_p']['1'], float)
Q = np.array(L['bars_q']['1'], float)
lo = 0.0
hi = float(L['rmax'])
ax.plot([lo, hi], [lo, hi], color='0.55', lw=0.9)
for b, d in P:
    ax.add_patch(Rectangle((b - delta, d - delta), 2 * delta, 2 * delta,
                           facecolor='tab:blue', alpha=0.10, edgecolor='none',
                           zorder=1))
ax.scatter(P[:, 0], P[:, 1], s=26, facecolor='none', edgecolor='tab:blue',
           lw=1.1, label='placement $p$', zorder=3)
ax.scatter(Q[:, 0], Q[:, 1], s=13, color='tab:red', marker='x', lw=1.1,
           label=r'placement $q$, $|p_i-q_i|=\delta$', zorder=4)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi * 1.02)
ax.set_xlabel('birth radius')
ax.set_ylabel('death radius')
ax.set_title(r'(b) degree-one diagram of $\Delta_2$, $d_B=%.3f\leq\delta=%.3f$'
             % (L['bottleneck_H1'], delta), fontsize=8.5)
ax.grid(alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=7, frameon=False, loc='lower right')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

fig.tight_layout()
fig.savefig(FIG / 'fig_stability.pdf')
fig.savefig(FIG / 'fig_stability.png')
print('wrote figures/fig_stability.pdf')
print('H0 bottleneck %.4f, H1 %.4f, delta %.4f'
      % (L['bottleneck_H0'], L['bottleneck_H1'], delta))
