"""
make_ecog_paper_data.py -- the numbers and plot tables of the ECoG paper.

Reads results/ecog_designs.json, results/soz_capture.json, results/optimal.json,
results/cortex.json and results/dropout_cortex.json, and writes

  paper/ecog/numbers_ecog.tex     every number the manuscript quotes, as macros
  paper/ecog/tikz/data/*.dat      the tables the pgfplots figures read
  paper/ecog/figures/fig5_cortex.png   the two surface renders of
                                  figures/fig_optimal.png

so that nothing in paper/ecog/ecog_coverage.tex or its figures is typed by
hand.  Macros and tables are named by role (the grid, the crown design, the
fewest contacts seeing X once or twice at 8 or 10 mm), not by contact count, so
the TikZ sources do not change when a count does.  Run run_ecog_designs.py and
run_soz_capture.py first.  Then

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
ED = json.load(open(RES / 'ecog_designs.json'))
S = json.load(open(RES / 'soz_capture.json'))

# the tables this script writes (the fold_* files belong to fold_geometry.py)
for p in DAT.glob('*.dat'):
    if not p.name.startswith('fold_'):
        p.unlink()

macros = []


def mac(name, value):
    macros.append('\\newcommand{\\%s}{%s}' % (name, value))


def f(x, nd=1):
    return ('%.' + str(nd) + 'f') % x


def pct(x, nd=1):
    """A share as a percentage, never rounded onto 0 or 100 unless exact."""
    s = f(100.0 * x, nd)
    if x < 1 and float(s) >= 100:
        return '{>}' + f(100 - 10 ** -nd, nd)
    if x > 0 and float(s) <= 0:
        return '{<}' + f(10 ** -nd, nd)
    return s


def table(name, header, rows):
    with open(DAT / name, 'w') as fh:
        fh.write(' '.join(header) + '\n')
        for r in rows:
            fh.write(' '.join(str(v) for v in r) + '\n')


RW = {8.0: 'Eight', 9.0: 'Nine', 10.0: 'Ten', 12.0: 'Twelve', 14.0: 'Fourteen'}

# ------------------------------------------------------------ the target
P = O['patch']
mac('patchArea', '%d' % round(P['area_mm2']))
mac('patchVerts', P['vertices'])
mac('patchTris', P['triangles'])
mac('patchRadius', '%d' % O['patch_radius_mm'])
mac('anchorMNI', '(%d, %d, %d)' % tuple(int(x) for x in O['anchor_mni']))
mac('meshVerts', '175\\,409')
mac('meshTris', '350\\,814')
mac('edgeMax', f(ED['edge_max_mm'], 2))
mac('edgeMedian', f(ED['edge_median_mm'], 2))

# ------------------------------------------------------------ crown floor
cf = ED['crown_floor']
mac('crownFloor', f(cf['rho_C_mm'], 2))
mac('crownFloorTwo', f(cf['rho_C2_mm'], 2))
mac('crownFloorVertex', f(cf['rho_C_vertex_all_crowns_mm'], 2))
mac('crownFloorInside', f(cf['rho_C_vertex_crowns_inside_X_mm'], 2))
mac('crownSites', cf['n_sites_within_reach'])
mac('crownSitesInside', cf['n_sites_inside_X'])
mac('crownReach', '%d' % cf['reach_mm'])
w = cf['worst_triangle_centroid_mni']
mac('crownWorstMNI', '(%s, %s, %s)' % (f(w[0]), f(w[1]), f(w[2])))
cr = ED['crown']
mac('crownFlatN', cr['reaches_floor_at'])
mac('crownUnseenSixtyFour', f(cr['unseen_mm2_r8_at_64']))

# ------------------------------------------------------------ covering radii
fr1, fr2 = ED['free']['rho1_mm'], ED['free']['rho2_mm']
table('rho_free.dat', ['n', 'rho'], [(i + 1, x) for i, x in enumerate(fr1)])
table('rho2_free.dat', ['n', 'rho'],
      [(i + 1, x) for i, x in enumerate(fr2) if x is not None])
table('rho_crown.dat', ['n', 'rho'],
      [(i + 1, x) for i, x in enumerate(cr['rho1_mm'])
       if x is not None and math.isfinite(x) and x < 40])
mac('nMaxGreedy', len(fr1))
mac('rhoFreeSixtyFour', f(fr1[63], 2))
mac('rhoFreeSixtyFourVertex', f(ED['free']['rho1_vertex_mm'][63], 2))

# ------------------------------------------------------------ fewest contacts
dz = ED['designs']
rows = []
for r in ED['radii_mm']:
    d1, d2 = dz['k1_r%g' % r], dz['k2_r%g' % r]
    w_ = RW[r]
    for k, d in ((1, d1), (2, d2)):
        tag = ('One' if k == 1 else 'Two') + w_
        c = d['certificate']
        mac('minN' + tag, d['n'])
        mac('minRho' + tag, f(d['rho_mm'], 2))
        mac('minOneFewer' + tag, f(d['unseen_mm2_one_fewer'], 2))
        mac('minOff' + tag, d['off_crown'])
        mac('minDisk' + tag, c['disk_test_failures'])
        mac('minFaces' + tag, c['faces'])
        kk = str(k)
        mac('minShadow' + tag, '(%d, %d)' % tuple(c['shadow'][kk]))
        mac('minMesh' + tag, '(%d, %d)' % tuple(c['mesh'][kk]))
    mac('pack' + w_, d1['packing_bound'])
    rows.append((r, d1['n'], d1['rho_mm'], d2['n'], d2['rho_mm'], d1['packing_bound']))
table('min_design.dat', ['r', 'n1', 'rho1', 'n2', 'rho2', 'bound'], rows)
# the share of sites off the crowns, over the designs below the crown floor
below = [d for d in dz.values() if d['n'] and d['radius_mm'] < cf['rho_C_mm']]
offp = [100.0 * d['off_crown'] / d['n'] for d in below]
mac('offCrownPctLo', '%d' % round(min(offp)))
mac('offCrownPctHi', '%d' % round(max(offp)))

# ------------------------------------------------------------ matched designs
mrow = {(m['array'], m['radius_mm']): m for m in O['matched_comparison']}


def _lab(p):
    return '0' if p == 0 else ('%.2f' % p if p < 0.1 else '%.1f' % p)


table('matched.dat', ['r', 'grid', 'greedy', 'gridlab', 'greedylab'],
      [(int(r), round(100 * mrow[('grid_8x8', r)]['uncovered_fraction'], 2),
        round(100 * mrow[('greedy', r)]['uncovered_fraction'], 3),
        _lab(100 * mrow[('grid_8x8', r)]['uncovered_fraction']),
        _lab(100 * mrow[('greedy', r)]['uncovered_fraction']))
       for r in (8.0, 10.0, 12.0)])
for r in (8.0, 10.0, 12.0):
    g, q = mrow[('grid_8x8', r)], mrow[('greedy', r)]
    w_ = RW[r]
    mac('gridUnseen' + w_, f(g['uncovered_mm2']))
    mac('gridUnseenPct' + w_, pct(g['uncovered_fraction']))
    mac('gridBetti' + w_, '(%d, %d)' % tuple(g['shadow1']))
    mac('greedyUnseen' + w_, f(q['uncovered_mm2']))
    mac('greedyUnseenPct' + w_, pct(q['uncovered_fraction'], 2))

# ------------------------------------------------------------ documented grid
G = C['grids']['parietal_8x8']
mac('gridPitch', '%d' % G['pitch_mm'])
mac('gridEuPitchMedian', f(G['euclidean_pitch_mm']['median'], 2))
mac('gridGeoPitchMedian', f(G['geodesic_pitch_mm']['median'], 2))
# how many grid contacts lie outside X: a sheet's contacts are not confined to
# the crowns inside the territory it is meant to cover
import numpy as np  # noqa: E402
import run_optimal as R  # noqa: E402  (loads the mesh and the patch X)
gv = np.array(G['contact_vertices'])
mac('gridOutsideX', int((~np.isin(gv, R.PVf)).sum()))

# ------------------------------------------------------------ onset-zone capture
s_mm = S['s_mm']
Dz = S['designs']
fw = S['fewest']
ROLE = {'grid64_r8': 'GridEight', 'grid64_r10': 'GridTen',
        'crown64_r8': 'CrownEight', 'free64_r8': 'FreeSixtyFour',
        'free%d_r8' % fw['k1_r8']: 'OneEight', 'free%d_r8' % fw['k2_r8']: 'TwoEight',
        'free%d_r10' % fw['k1_r10']: 'OneTen', 'free%d_r10' % fw['k2_r10']: 'TwoTen'}
assert set(ROLE) == set(Dz), (sorted(ROLE), sorted(Dz))
i5, i10 = s_mm.index(5.0), s_mm.index(10.0)
for name, rec in Dz.items():
    t = ROLE[name]
    table('capture_%s.dat' % t, ['s', 'whole', 'twice', 'missed'],
          [(s, rec['whole_k1'][i], rec['whole_k2'][i], rec['missed'][i])
           for i, s in enumerate(s_mm)])
    mac('capN' + t, rec['n_contacts'])
    mac('capWholeFive' + t, pct(rec['whole_k1'][i5]))
    mac('capWholeTen' + t, pct(rec['whole_k1'][i10]))
    mac('capTwiceFive' + t, pct(rec['whole_k2'][i5]))
    mac('capMissFive' + t, pct(rec['missed'][i5]))
    mac('capHidden' + t, f(rec['largest_hidden_s_mm']))
    mac('capRhoOne' + t, f(rec['rho1_mm'], 2))
    mac('capRhoTwo' + t, f(rec['rho2_mm'], 2))
    mac('capUnseen' + t, f(rec['unseen_mm2']['1']))
    mac('capOnce' + t, f(rec['unseen_mm2']['2']))
g8 = Dz['grid64_r8']
mac('gridHideBound', f(g8['hide_bound_mm'], 2))
mac('gridRhoRatio', '%d' % round(g8['rho1_mm'] / Dz['free64_r8']['rho1_mm']))

# ------------------------------------------------------------ random dropout
DR = S['dropout']
qs = DR['q']
mac('dropDraws', DR['draws'])
mac('dropQmax', max(qs))
cols = ['q']
rows = [[q] for q in qs]
for name in DR['designs']:
    t = ROLE[name]
    for s in ('5.0', '10.0'):
        cols += ['%s_s%s_mean' % (t, s[:-2]), '%s_s%s_p05' % (t, s[:-2])]
        for j, q in enumerate(qs):
            v = DR['designs'][name][str(q)][s]
            rows[j] += [v['mean'], v['p05']]
    for q, w_ in ((1, 'One'), (4, 'Four'), (8, 'Eight')):
        v = DR['designs'][name][str(q)]['5.0']
        mac('dropMean%s%s' % (w_, t), pct(v['mean']))
        mac('dropLow%s%s' % (w_, t), pct(v['p05']))
table('dropout.dat', cols, rows)
# the lowest mean share over all q, for each double-coverage design, s = 5 mm
for name in DR['designs']:
    t = ROLE[name]
    if t.startswith('Two'):
        mac('dropMin' + t, pct(min(DR['designs'][name][str(q)]['5.0']['mean']
                                   for q in qs)))

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
