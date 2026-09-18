"""Figures derived from results/*.json: field barcodes and the pitch sweep
with the certified level k* (largest k such that b0 = 1, b1 = 0 for all
levels up to k)."""
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

f = json.load(open('results/fields.json'))
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

H0C, H1C = '#1f6fb4', '#c0392b'
fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8))
for row, (name, label) in enumerate((('A_ring_grid_seed3', 'Field A'),
                                     ('D_hex_dead_zone', 'Field D'))):
    bars = f[name]['bars']
    for col, d in enumerate(('0', '1')):
        ax = axes[row, col]
        col_d = H0C if d == '0' else H1C
        cnt = Counter(tuple(b) for b in bars[d])
        items = sorted(cnt.items(), key=lambda x: (-x[0][1], -x[0][0]))
        y = 0
        for (lo, hi), mlt in items:
            # a bar whose thickness carries its multiplicity, so that a class
            # occurring forty times is not drawn like one occurring once
            lw = 2.0 + 3.2 * (mlt / max(c for _, c in items)) ** 0.5
            ax.plot([lo - 0.45, hi + 0.45], [y, y], lw=lw, color=col_d,
                    solid_capstyle='round', zorder=3)
            ax.text(hi + 0.62, y, r'$\times%d$' % mlt, va='center',
                    fontsize=8.5, color='0.25')
            y += 1
        for x in range(1, 8):
            ax.axvline(x, color='0.88', lw=0.7, zorder=0)
        ax.set_yticks([])
        ax.set_ylim(-0.8, max(y, 1) - 0.2)
        ax.set_xticks(range(1, 8))
        ax.set_xlim(0.3, 8.4)
        ax.tick_params(labelsize=8.5)
        if row == 1:
            ax.set_xlabel('redundancy level $k$', fontsize=9.5)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('0.6')
        ax.set_title('%s:  $H_%s$,  %d classes in %d distinct intervals'
                     % (label, d, len(bars[d]), len(cnt)), fontsize=10)
plt.tight_layout()
plt.savefig('figures/field_barcodes.png', dpi=300)
plt.close()

g = json.load(open('results/grid.json'))
rows = g['pitch_sweep']
ks, qs, ps = [], [], []
for s in rows:
    # largest k with (b0, b1) = (1, 0) at every level up to k; the loop stops
    # at the first level that fails, which is what makes this the certified
    # level rather than the highest level that happens to be (1, 0)
    kcert = 0
    for k in range(1, 8):
        if s.get('b0_%d' % k) == 1 and s.get('b1_%d' % k) == 0:
            kcert = k
        else:
            break
    ks.append(kcert); qs.append(s['margin']); ps.append(s['pitch'])
    s['k_certified'] = kcert
json.dump(g, open('results/grid.json', 'w'), indent=1)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.step(ps, ks, where='mid', label=r'certified level $k^\ast$')
ax.step(ps, qs, where='mid', ls='--', label='dropout margin $q$')
ax.set_xlabel('electrode pitch (footprint radius $r=0.8$)')
ax.set_ylabel('level')
ax.set_yticks(range(0, 7))
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/grid_pitch_sweep.png', dpi=200)
plt.close()
print('ok', list(zip(ps, ks, qs)))
