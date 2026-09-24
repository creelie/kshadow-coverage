"""make_signal_figure.py -- figures/fig_signal.png from results/signal_rebuild.json,
results/signal_explore.json and results/failure_margins.json."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
S = json.load(open(ROOT / 'results' / 'signal_rebuild.json'))
X = json.load(open(ROOT / 'results' / 'signal_explore.json'))
G = json.load(open(ROOT / 'results' / 'failure_margins.json'))

plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.linewidth': 0.7,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
rng = np.random.default_rng(1)

# (a) within-channel natural experiment
a = ax[0]
w = X['X2_within_channel_t1']
rows = [('pre-specified: $\\lambda$', S['P_lambda_t1']['rho'], 'C3'),
        ('$\\lambda$ given $d_1$', w['lambda_given_d1']['rho'], 'C3'),
        ('$\\lambda$ given $d_1,d_2,d_3$', w['lambda_given_d123']['rho'], 'C3'),
        ('$-$mean$(d_1,d_2,d_3)$', w['minus_mean_d123']['rho'], 'C0'),
        ('$-\\pi$ (not additive)', w['minus_pi']['rho'], 'C2')]
for i, (label, rho, col) in enumerate(rows):
    v = np.array([x for x in rho.values() if np.isfinite(x)])
    y = len(rows) - 1 - i
    a.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=6, color=col, alpha=0.55,
              edgecolors='none')
    a.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color='k', lw=1.4)
a.axvline(0, color='0.5', lw=0.6, ls=':')
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
b.plot(xs, cv, 'o', color='C0', ms=5, label='first sessions, by-subject CV')
b.plot(xs, te, 's', color='C1', ms=4.5, mfc='white', mew=1.2, label='second sessions, held out')
b.axhline(0, color='0.5', lw=0.6, ls=':')
b.axvline(3.5, color='0.8', lw=0.6)
b.set_xticks(xs)
b.set_xticklabels(labels, rotation=40, ha='right', fontsize=7.5)
b.set_ylabel('$R^2$ of rebuild quality\n(centred within recording)')
b.legend(frameon=False, fontsize=6.8, loc='upper left')
b.set_title('(b) what predicts a rebuild', loc='left', fontsize=9)

# (c) clustering beyond a rate gradient
c = ax[2]
stats_ = [('A', 'adjacent\nfailed pairs'), ('U', 'target\nuncovered'), ('rho95', 'reach at\n95%')]
g = G['ses-t1']
for i, (key, label) in enumerate(stats_):
    obs = g['observed'][key]
    for j, (null, col, mk, name) in enumerate((('uniform', 'C0', 'o', 'uniform null'),
                                               ('margin_preserving', 'C3', 's',
                                                'rates kept null'))):
        e = g[null][key]
        z = (obs - e['null_mean']) / e['null_sd']
        c.plot([i + (j - 0.5) * 0.28], [z], mk, color=col, ms=5,
               label=name if i == 0 else None)
c.axhline(0, color='0.5', lw=0.6, ls=':')
c.axhline(1.96, color='0.7', lw=0.6, ls='--')
c.set_xticks(range(len(stats_)))
c.set_xticklabels([s[1] for s in stats_], fontsize=7.5)
c.set_xlim(-0.5, len(stats_) - 0.5)
c.set_ylabel('observed minus null mean (null s.d.)')
c.legend(frameon=False, fontsize=6.8, loc='upper right')
c.set_title('(c) clustering, first sessions', loc='left', fontsize=9)

fig.tight_layout(w_pad=1.2)
out = ROOT / 'figures' / 'fig_signal.png'
fig.savefig(out, dpi=300)
print('wrote', out)
