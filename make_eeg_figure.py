"""make_eeg_figure.py -- figures/fig_eeg.png from results/eeg_failures.json and
results/eeg_design.json (the random-failure reach curve of panel b is
recomputed here with its own seed)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_eeg_failures as R
import run_eeg_design as E

ROOT = Path(__file__).resolve().parent
F = json.load(open(ROOT / 'results' / 'eeg_failures.json'))
D = json.load(open(ROOT / 'results' / 'eeg_design.json'))
names, U, recs = R._setup()
t1 = [r for r in recs if r['ses'] == 'ses-t1']

plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.linewidth': 0.7,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(7.0, 6.1))

# (a) failure rate on the cap
a = ax[0, 0]
rate = np.array([F['heterogeneity']['rates'][n] for n in names])
col = np.degrees(np.arccos(U[:, 2]))
az = np.arctan2(U[:, 1], U[:, 0])
px, py = col * np.cos(az), col * np.sin(az)
t = np.linspace(0, 2 * np.pi, 400)
a.plot(92 * np.cos(t), 92 * np.sin(t), color='0.35', lw=0.8, ls='--')
sc = a.scatter(px, py, c=rate, s=18 + 260 * rate, cmap='viridis_r', vmin=0, vmax=rate.max(),
               edgecolors='k', linewidths=0.4, zorder=3)
for i in np.argsort(-rate)[:6]:
    off = (-20, 7) if px[i] < 0 else (7, 7)
    a.annotate(names[i], (px[i], py[i]), xytext=off, textcoords='offset points', fontsize=7)
a.set_aspect('equal')
a.set_xlim(-125, 125)
a.set_ylim(-125, 125)
a.set_xticks([])
a.set_yticks([])
a.spines['left'].set_visible(False)
a.spines['bottom'].set_visible(False)
a.text(0, 112, 'nose', ha='center', fontsize=7, color='0.35')
cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
cb.set_label('fraction of first sessions marked bad')
a.set_title('(a) where contacts fail', loc='left', fontsize=9)

# (b) reach needed to keep the whole target seen
b = ax[0, 1]
obs = np.sort(np.array(D['E1']['ses-t1']['observed']))
rng = np.random.default_rng(20260930)
null = []
for r in t1:
    for _ in range(40):
        surv = np.ones(64, bool)
        if r['f']:
            surv[rng.choice(64, r['f'], replace=False)] = False
        null.append(E.reach1(U[surv]))
null = np.sort(np.array(null))
grid = np.linspace(14, 60, 400)
b.plot(grid, [(obs <= g).mean() for g in grid], color='C3', lw=1.6, label='real failures')
b.plot(grid, [(null <= g).mean() for g in grid], color='C0', lw=1.6, ls='--',
       label='random failures, same counts')
b.axhline(0.95, color='0.5', lw=0.6, ls=':')
q_obs = D['E1']['ses-t1']['reach95_observed']
q_null = D['E1']['ses-t1']['reach95_null']
b.plot([q_obs], [0.95], 'o', color='C3', ms=4)
b.plot([q_null], [0.95], 'o', color='C0', ms=4)
b.annotate('%.1f deg' % q_obs, (q_obs, 0.95), xytext=(2, -12), textcoords='offset points', fontsize=7, color='C3')
b.annotate('%.1f deg' % q_null, (q_null, 0.95), xytext=(-34, 5), textcoords='offset points', fontsize=7, color='C0')
b.set_xlabel('footprint radius (deg)')
b.set_ylabel('recordings with the whole target seen')
b.set_ylim(0, 1.02)
b.legend(frameon=False, loc='lower right', fontsize=7.5)
b.set_title('(b) reach the real failures demand', loc='left', fontsize=9)

# (c) clustering
c = ax[1, 0]
pr = F['runs']['23.5']['cohorts']['t1']['per_recording']
A = np.array([p['A'] for p in pr])
EA = np.array([p['EA'] for p in pr])
m = max(A.max(), EA.max()) * 1.05 + 1
c.plot([0, m], [0, m], color='0.5', lw=0.7)
c.scatter(EA, A + rng.uniform(-0.15, 0.15, len(A)), s=9, color='C3', alpha=0.7, edgecolors='none')
c.set_xlabel('adjacent failed pairs expected at random')
c.set_ylabel('adjacent failed pairs observed')
c.set_xlim(0, m)
c.set_ylim(0, m)
c.set_title('(c) failures come in neighbours', loc='left', fontsize=9)

# (d) optimised layout against real failures
d = ax[1, 1]
u0 = np.array(D['E2']['U_per_recording']['D0'])
u1 = np.array(D['E2']['U_per_recording']['D1'])
sel = np.array([r['ses'] == 'ses-t1' for r in recs])
u0, u1 = 100 * u0[sel], 100 * u1[sel]
both0 = (u0 == 0) & (u1 == 0)
lim = max(u0.max(), u1.max()) * 1.3
d.plot([0, lim], [0, lim], color='0.5', lw=0.7)
worse = u1 > u0
d.scatter(u0[~worse & ~both0], u1[~worse & ~both0], s=11, color='C0', alpha=0.8,
          edgecolors='none', label='optimised layout better or equal')
d.scatter(u0[worse], u1[worse], s=11, color='C3', alpha=0.8, edgecolors='none',
          label='optimised layout worse')
d.set_xscale('symlog', linthresh=0.01)
d.set_yscale('symlog', linthresh=0.01)
d.set_xlim(-0.002, lim)
d.set_ylim(-0.002, lim)
d.set_xlabel('target unseen, 10-20 cap (%)')
d.set_ylabel('target unseen, optimised layout (%)')
d.legend(frameon=False, fontsize=7, loc='upper left')
d.text(0.98, 0.03, '%d recordings unaffected in both' % both0.sum(), transform=d.transAxes,
       ha='right', fontsize=7, color='0.35')
d.set_title('(d) tighter coverage, not more robust', loc='left', fontsize=9)

fig.tight_layout(w_pad=1.5, h_pad=1.6)
out = ROOT / 'figures' / 'fig_eeg.png'
fig.savefig(out, dpi=300)
print('wrote', out)
