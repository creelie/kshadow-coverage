"""
make_dropout_figure.py -- the figure for the cortical dropout experiment.

One question: does a 10 mm pitch array on folded cortex ever acquire
redundancy as the footprint radius grows?  Two panels, both answering it,
same two series, same colours, solid/dashed as a second encoding so the
identity of a curve does not rest on colour alone.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 9,
                     'figure.dpi': 200, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04})

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)

d = json.load(open(RES / 'dropout_cortex.json'))
radii = d['radii_mm']

STYLE = {'parietal_8x8': dict(color='tab:blue', ls='-', marker='o',
                              label=r'parietal $8\times8$ (64 contacts)'),
         'temporal_4x8': dict(color='tab:orange', ls='--', marker='s',
                              label=r'temporal $4\times8$ (32 contacts)')}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# (a) how many contacts are the sole observer of some cortical territory
ax = axes[0]
for g, st in STYLE.items():
    per = d['grids'][g]
    n = per[str(radii[0])]['n_electrodes']
    y = [100.0 * per[str(r)]['single_summary']['contacts_with_private_territory'] / n
         for r in radii]
    ax.plot(radii, y, ms=4, lw=1.6, **st)
ax.set_xlabel('footprint radius $r$ (mm)')
ax.set_ylabel('contacts with private territory (%)')
ax.set_title('(a) contacts holding territory no other contact sees', fontsize=8.5)
ax.set_ylim(0, 105)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=6.5, frameon=False, loc='lower left')

# (b) what the worst single failure costs, as a share of the covered region
ax = axes[1]
for g, st in STYLE.items():
    per = d['grids'][g]
    y = [100.0 * per[str(r)]['single_summary']['worst_single_loss_fraction'] for r in radii]
    ax.plot(radii, y, ms=4, lw=1.6, **st)
ax.set_xlabel('footprint radius $r$ (mm)')
ax.set_ylabel('worst single-contact loss (% of $R_{\geq1}$)')
ax.set_title('(b) cost of the worst single failure', fontsize=8.5)
ax.set_ylim(0, None)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=6.5, frameon=False, loc='upper left')

fig.tight_layout()
fig.savefig(FIG / 'fig_dropout_cortex.pdf')
fig.savefig(FIG / 'fig_dropout_cortex.png')
print('wrote figures/fig_dropout_cortex.pdf')

# a small table for the paper
print('\nr(mm)  parietal: priv/64  worst mm^2  (%)   temporal: priv/32  worst mm^2  (%)')
for r in radii:
    p = d['grids']['parietal_8x8'][str(r)]['single_summary']
    t = d['grids']['temporal_4x8'][str(r)]['single_summary']
    print(f"{r:5.0f}  {p['contacts_with_private_territory']:14d} "
          f"{p['private_area_max_mm2']:10.1f} {100*p['worst_single_loss_fraction']:6.2f}"
          f"{t['contacts_with_private_territory']:16d} "
          f"{t['private_area_max_mm2']:10.1f} {100*t['worst_single_loss_fraction']:6.2f}")
