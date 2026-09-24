"""make_signal_figure.py -- figures/fig_signal.png from results/signal_rebuild.json,
results/signal_explore.json and results/failure_correlation.json."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
S = json.load(open(ROOT / 'results' / 'signal_rebuild.json'))
X = json.load(open(ROOT / 'results' / 'signal_explore.json'))
C = json.load(open(ROOT / 'results' / 'failure_correlation.json'))

plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.linewidth': 0.7,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
rng = np.random.default_rng(1)

# (a) within-channel natural experiment
a = ax[0]
w = X['X2_within_channel_t1_centred']
rows = [('$\\lambda$, pre-specified', S['P_lambda_t1']['rho'], 'C3'),
        ('$\\lambda$, recording removed', w['lambda']['rho'], 'C3'),
        ('$-$mean$(d_1,d_2,d_3)$', w['minus_mean_d123']['rho'], 'C0'),
        ('$\\lambda$ given $d_1,d_2,d_3$', w['lambda_given_d123']['rho'], 'C3'),
        ('$-\\pi$ (not additive)', w['minus_pi']['rho'], 'C2')]
for i, (label, rho, col) in enumerate(rows):
    v = np.array([x for x in rho.values() if np.isfinite(x)])
    y = len(rows) - 1 - i
    a.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=6, color=col, alpha=0.55,
              edgecolors='none')
    a.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color='k', lw=1.4)
a.axvline(0, color='0.5', lw=0.6, ls=':')
a.axhline(len(rows) - 1.5, color='0.8', lw=0.6)
a.set_yticks(range(len(rows)))
a.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
a.set_xlabel('Spearman $\\rho$ with rebuild quality,\nper channel across recordings')
a.set_title('(a) within each channel', loc='left', fontsize=9)

# (b) out-of-sample prediction
b = ax[1]
models = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
labels = ['$d_1$', '$d_{1..6}$', '+$\\lambda$', '+$\\lambda,\\pi$', '$d_{1..6}$+site',
          '+$\\lambda,\\pi$']
cv = [X['X3_prediction_z'][m]['cv_r2_t1'] for m in models]
te = [X['X3_prediction_z'][m].get('test_r2_t2', np.nan) for m in models]
xs = np.arange(len(models))
b.plot(xs, cv, 'o', color='C0', ms=5, label='first, by-subject CV')
b.plot(xs, te, 's', color='C1', ms=4.5, mfc='white', mew=1.2, label='second, held out')
b.axhline(0, color='0.5', lw=0.6, ls=':')
b.axvline(3.5, color='0.8', lw=0.6)
b.set_xticks(xs)
b.set_xticklabels(labels, rotation=40, ha='right', fontsize=7.5)
b.set_ylabel('$R^2$ of rebuild quality\n(centred within recording)')
b.legend(frameon=False, fontsize=6.8, loc='center left', bbox_to_anchor=(0.0, 0.62))
b.set_title('(b) out of sample', loc='left', fontsize=9)

# (c) pair correlation of failure
c = ax[2]
edges = np.array(C['bin_edges_deg'])
mid = 0.5 * (edges[:-1] + edges[1:])
for ses, col, mk, ls, name in (('ses-t1', 'C3', 'o', '-', 'first'),
                               ('ses-t2', 'C0', 's', '--', 'second')):
    g = np.array([np.nan if v is None else v for v in C[ses]['g']])
    ok = np.isfinite(g)
    c.plot(mid[ok], g[ok], ls=ls, marker=mk, color=col, ms=4, lw=1.3, label=name)
lo = np.array([np.nan if v is None else v for v in C['ses-t1']['null_lo']])
hi = np.array([np.nan if v is None else v for v in C['ses-t1']['null_hi']])
ok = np.isfinite(lo)
c.fill_between(mid[ok], lo[ok], hi[ok], color='0.85', lw=0, label='null 95%')
c.axhline(1, color='0.5', lw=0.6, ls=':')
xi = C['ses-t1']['xi_deg']
c.axvline(xi, color='C3', lw=0.7, ls=':')
c.annotate('$\\xi\\approx$%.0f$^\\circ$' % xi, (xi, 1.75), xytext=(4, 0),
           textcoords='offset points', fontsize=7.5, color='0.2')
c.set_xlabel('separation (deg)')
c.set_ylabel('co-failure / expected')
c.legend(frameon=False, fontsize=6.8, loc='lower left')
c.set_ylim(0.45, 1.95)
c.set_title('(c) co-failure', loc='left', fontsize=9)

fig.tight_layout(w_pad=1.2)
out = ROOT / 'figures' / 'fig_signal.png'
fig.savefig(out, dpi=300)
print('wrote', out)
