"""
make_ecog_paper_data.py -- the numbers and plot tables of the ECoG paper.

Reads results/optimal.json, results/cortex.json, results/dropout_cortex.json
and results/soz_capture.json, and writes

  paper/ecog/numbers_ecog.tex     every number the manuscript quotes, as macros
  paper/ecog/tikz/data/*.dat      the tables the pgfplots figures read

so that nothing in paper/ecog/ecog_coverage.tex or its figures is typed by
hand.  Run run_soz_capture.py first.  Then

    cd paper/ecog && ./build.sh
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
OUT = ROOT / 'paper' / 'ecog'
DAT = OUT / 'tikz' / 'data'
DAT.mkdir(parents=True, exist_ok=True)

O = json.load(open(RES / 'optimal.json'))
C = json.load(open(RES / 'cortex.json'))
DC = json.load(open(RES / 'dropout_cortex.json'))
S = json.load(open(RES / 'soz_capture.json'))

macros = []


def mac(name, value):
    macros.append('\\newcommand{\\%s}{%s}' % (name, value))


def f(x, nd=1):
    return ('%.' + str(nd) + 'f') % x


def pct(x, nd=1):
    return f(100.0 * x, nd)


def table(name, header, rows):
    with open(DAT / name, 'w') as fh:
        fh.write(' '.join(header) + '\n')
        for r in rows:
            fh.write(' '.join(str(v) for v in r) + '\n')


# ------------------------------------------------------------ the target
P = O['patch']
mac('patchArea', '%d' % round(P['area_mm2']))
mac('patchVerts', P['vertices'])
mac('patchTris', P['triangles'])
mac('patchRadius', '%d' % O['patch_radius_mm'])
a = O['anchor_mni']
mac('anchorMNI', '(%d, %d, %d)' % tuple(int(x) for x in a))
mac('meshVerts', '175\\,409')
mac('meshTris', '350\\,814')

# ------------------------------------------------------------ crown floor
cf = O['crown_floor']
mac('crownFloor', f(cf['covering_radius_mm'], 2))
mac('crownFloorExact', f(cf['covering_radius_mm'], 3))
mac('crownVerts', cf['n_crown_vertices'])
w = cf['worst_vertex_mni']
mac('crownWorstMNI', '(%s, %s, %s)' % (f(w[0]), f(w[1]), f(w[2])))

# ------------------------------------------------------------ covering radius
fr = O['families']['free']['rho_mm']
cr = O['families']['crown']['rho_mm']
table('rho_free.dat', ['n', 'rho'], [(i + 1, x) for i, x in enumerate(fr)])
table('rho_crown.dat', ['n', 'rho'], [(i + 1, x) for i, x in enumerate(cr)])
k2 = O['families']['free']['kfold_rho_mm']
table('rho2_free.dat', ['n', 'rho'],
      [(int(n), round(v['2'], 4)) for n, v in sorted(k2.items(), key=lambda t: int(t[0]))])
k2c = O['families']['crown']['kfold_rho_mm']
table('rho2_crown.dat', ['n', 'rho'],
      [(int(n), round(v['2'], 4)) for n, v in sorted(k2c.items(), key=lambda t: int(t[0]))])
# first n from which the crown sequence sits on its floor
flat = next(i + 1 for i, x in enumerate(cr) if abs(x - cf['covering_radius_mm']) < 1e-3)
mac('crownFlatN', flat)
mac('rhoFreeOne', f(fr[0], 0))
mac('rhoFreeLast', f(fr[-1], 2))
mac('nMaxGreedy', len(fr))
mac('rhoFreeSixtyFour', f(fr[63], 2))
mac('rhoCrownSixtyFour', f(cr[63], 2))
mac('rhoTwoFreeSixtyFour', f(k2['64']['2'], 2))
mac('rhoTwoFreeOneTwentyEight', f(k2['128']['2'], 2))
mac('rhoTwoFreeOneSixty', f(k2['160']['2'], 2))

# ------------------------------------------------------------ minimal designs
names = {14.0: 'Fourteen', 12.0: 'Twelve', 10.0: 'Ten', 8.0: 'Eight'}
rows = []
for key in ('k1_r14', 'k1_r12', 'k1_r10', 'k1_r8'):
    m = O['curves'][key]['minimal']
    r = m['radius_mm']
    n = m['n_contacts']
    rho = fr[n - 1]
    bound = O['packing'][str(r)]['size']
    rows.append((r, n, round(rho, 2), bound))
    w_ = names[r]
    mac('minN' + w_, n)
    mac('minRho' + w_, f(rho, 2))
    mac('pack' + w_, bound)
    mac('diskFail' + w_, m['disk_test_failures'])
table('min_design.dat', ['r', 'n', 'rho', 'bound'], rows)
table('packing.dat', ['r', 'bound'],
      sorted((float(r), v['size']) for r, v in O['packing'].items()))
mac('packNine', O['packing']['9.0']['size'])

# ------------------------------------------------------------ matched designs
mrow = {}
for m in O['matched_comparison']:
    mrow[(m['array'], m['radius_mm'])] = m
def _lab(p):
    return '0' if p == 0 else ('%.2f' % p if p < 0.1 else '%.1f' % p)


table('matched.dat', ['r', 'grid', 'greedy', 'gridlab', 'greedylab'],
      [(int(r), round(100 * mrow[('grid_8x8', r)]['uncovered_fraction'], 2),
        round(100 * mrow[('greedy', r)]['uncovered_fraction'], 3),
        _lab(100 * mrow[('grid_8x8', r)]['uncovered_fraction']),
        _lab(100 * mrow[('greedy', r)]['uncovered_fraction']))
       for r in (8.0, 10.0, 12.0)])
for r, w_ in ((8.0, 'Eight'), (10.0, 'Ten'), (12.0, 'Twelve')):
    g, q = mrow[('grid_8x8', r)], mrow[('greedy', r)]
    mac('gridUnseen' + w_, f(g['uncovered_mm2']))
    mac('gridUnseenPct' + w_, pct(g['uncovered_fraction']))
    mac('gridBetti' + w_, '(%d, %d)' % tuple(g['shadow1']))
    mac('gridK' + w_, g['certified_level'])
    mac('greedyUnseen' + w_, f(q['uncovered_mm2']))
    mac('greedyUnseenPct' + w_, pct(q['uncovered_fraction'], 2))
    mac('greedyBetti' + w_, '(%d, %d)' % tuple(q['shadow1']))
    mac('greedyK' + w_, q['certified_level'])
    mac('greedyQ' + w_, q['dropout_margin'])
mac('gridSeenPctEight', pct(1 - mrow[('grid_8x8', 8.0)]['uncovered_fraction'], 0))

# ------------------------------------------------------------ double coverage
k2c_ = {c['n_contacts']: c for c in O['curves']['k2']}
d96, d112 = k2c_[96], k2c_[112]
mac('dblAN', 96)
mac('dblAR', '10')
mac('dblAOnce', f(d96['uncovered_area_mm2']['2']))
mac('dblAK', d96['certified_level'])
mac('dblBN', 112)
mac('dblBR', '9')
mac('dblBOnce', f(d112['uncovered_area_mm2']['2']))
mac('dblBShadow', '(%d, %d)' % tuple(d112['shadow']['2']))
mac('dblBMesh', '(%d, %d)' % (d112['mesh']['2']['components'], d112['mesh']['2']['b1']))
mac('dblBDisk', d112['disk_test_failures'])
mac('dblBFaces', d112['faces'])

# ------------------------------------------------------------ gyral / sulcal share
# how many of the free sites of a design are off the gyral crowns
import numpy as np  # noqa: E402
D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
sulc = D['sulc']
free_c = np.array(O['families']['free']['centres'])
off_pct = []
for n, w_ in ((66, 'SixtySix'), (112, 'OneTwelve'), (46, 'FortySix')):
    off = int((sulc[free_c[:n]] >= 0).sum())
    mac('offCrown' + w_, off)
    off_pct.append(100.0 * off / n)
mac('offCrownPctLo', '%d' % round(min(off_pct)))
mac('offCrownPctHi', '%d' % round(max(off_pct)))

# ------------------------------------------------------------ published grid
G = C['grids']['parietal_8x8']
mac('gridPitch', '%d' % G['pitch_mm'])
mac('gridEuPitchMedian', f(G['euclidean_pitch_mm']['median'], 2))
gp = G['geodesic_pitch_mm']
mac('gridGeoPitchMedian', f(gp['median'], 2))
mac('gridGeoPitchOver', gp['beyond_limit'])
mac('gridGeoPitchPairs', gp['within_limit'] + gp['beyond_limit'])

# ------------------------------------------------------------ private territory
pr = DC['grids']['parietal_8x8']
rows = []
for r in DC['radii_mm']:
    s_ = pr[str(float(r))]['single_summary']
    rows.append((r, s_['contacts_with_private_territory'], s_['private_area_max_mm2'],
                 round(100 * s_['worst_single_loss_fraction'], 3)))
table('private.dat', ['r', 'npriv', 'worst', 'pct'], rows)
s8 = pr['8.0']['single_summary']
mac('privCountEight', s8['contacts_with_private_territory'])
mac('privWorstEight', f(s8['private_area_max_mm2']))
mac('privMedianEight', f(s8['private_area_median_mm2']))

# ------------------------------------------------------------ onset-zone capture
s_mm = S['s_mm']
Dz = S['designs']
for name, rec in Dz.items():
    table('capture_%s.dat' % name, ['s', 'whole', 'twice', 'missed'],
          [(s, rec['whole_k1'][i], rec['whole_k2'][i], rec['missed'][i])
           for i, s in enumerate(s_mm)])
i5, i10 = s_mm.index(5.0), s_mm.index(10.0)
table('design_rho.dat', ['name', 'n', 'r', 'rho1', 'rho2'],
      [(k, v['n_contacts'], v['radius_mm'], v['rho1_mm'], v['rho2_mm'])
       for k, v in Dz.items()])
tag = dict(grid64_r8='GridEight', crown64_r8='CrownEight', free64_r8='FreeSixtyFour',
           free66_r8='FreeSixtySix', grid64_r10='GridTen', free46_r10='FreeFortySix',
           free96_r10='FreeNinetySix', free112_r9='FreeOneTwelve')
for name, rec in Dz.items():
    t = tag[name]
    mac('capWholeFive' + t, pct(rec['whole_k1'][i5]))
    mac('capWholeTen' + t, pct(rec['whole_k1'][i10]))
    mac('capTwiceFive' + t, pct(rec['whole_k2'][i5]))
    mac('capTwiceTen' + t, pct(rec['whole_k2'][i10]))
    mac('capMissFive' + t, pct(rec['missed'][i5]))
    mac('capMissTen' + t, pct(rec['missed'][i10]))
    mac('capHidden' + t, f(rec['largest_hidden_s_mm']))
    mac('capRhoOne' + t, f(rec['rho1_mm'], 2))
    mac('capRhoTwo' + t, f(rec['rho2_mm'], 2))
    mac('capUnseen' + t, f(rec['unseen_mm2']['1']))
    mac('capOnce' + t, f(rec['unseen_mm2']['2']))
# the smallest onset radius the grid can miss whole at r = 8 mm with
# probability at least one in ten
g8 = Dz['grid64_r8']
mac('capGridMissZero', pct(g8['missed'][0]))
mac('gridHideBound', f(g8['rho1_mm'] - g8['radius_mm'], 2))

# ------------------------------------------------------------ random dropout
DR = S['dropout']
qs = DR['q']
mac('dropDraws', DR['draws'])
mac('dropQmax', max(qs))
cols = ['q']
rows = [[q] for q in qs]
for name in DR['designs']:
    for s in ('5.0', '10.0'):
        cols += ['%s_s%s_mean' % (name, s[:-2]), '%s_s%s_p05' % (name, s[:-2])]
        for j, q in enumerate(qs):
            v = DR['designs'][name][str(q)][s]
            rows[j] += [v['mean'], v['p05']]
table('dropout.dat', cols, rows)
mac('dropMinFreeOneTwelve',
    f(100 * min(DR['designs']['free112_r9'][str(q)]['5.0']['mean'] for q in qs), 1))
for name in DR['designs']:
    t = tag[name]
    for q, w_ in ((1, 'One'), (4, 'Four'), (8, 'Eight')):
        v = DR['designs'][name][str(q)]['5.0']
        mac('dropMean%s%s' % (w_, t), pct(v['mean']))
        mac('dropLow%s%s' % (w_, t), pct(v['p05']))

(OUT / 'numbers_ecog.tex').write_text(
    '%% generated by make_ecog_paper_data.py from results/*.json; do not edit\n'
    + '\n'.join(macros) + '\n')
print('wrote %s (%d macros) and %d tables in %s'
      % (OUT / 'numbers_ecog.tex', len(macros), len(list(DAT.glob('*.dat'))), DAT))

# ------------------------------------------------------------ cortex renders
# the two surface renders of figures/fig_optimal.png (panels a and b, drawn
# off screen by make_optimal_figure.py with PyVista); the plots below them
# are redrawn in TikZ, so only the renders are kept.
from PIL import Image  # noqa: E402
im = Image.open(ROOT / 'figures' / 'fig_optimal.png')
w_, h_ = im.size
top = im.crop((0, 0, w_, int(0.385 * h_)))
(OUT / 'figures').mkdir(exist_ok=True)
top.save(OUT / 'figures' / 'fig5_cortex.png')
print('wrote %s (%d x %d)' % (OUT / 'figures' / 'fig5_cortex.png', *top.size))
