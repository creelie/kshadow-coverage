"""
make_paper_numbers.py -- paper/numbers.tex, one LaTeX macro per number the
manuscript quotes, read from results/*.json so that nothing in the text is
typed by hand.

    python3 make_paper_numbers.py
"""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
OUT = ROOT / 'paper' / 'numbers.tex'
WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six']


def load(name):
    return json.load(open(RES / name))


def pval(p):
    """p-values as printed: two significant figures, scientific below 1e-3,
    and Monte Carlo floors as inequalities."""
    if p is None:
        return 'n/a'
    if p < 1e-3:
        m, e = ('%.1e' % p).split('e')
        return r'%s\times 10^{%d}' % (m, int(e))
    return '%.3f' % p if p < 0.1 else '%.2f' % p


def main():
    m = {}
    F = load('eeg_failures.json')
    D = load('eeg_design.json')
    O = load('optimal.json')
    run = F['runs']['23.5']['cohorts']
    m['nRec'] = F['source']['recordings']
    m['nTone'] = F['source']['t1']
    m['nTtwo'] = F['source']['t2']
    m['thetaOne'] = '%.1f' % F['geometry']['covering_radii_deg'][0]
    m['thetaTwo'] = '%.1f' % F['geometry']['covering_radii_deg'][1]
    m['thetaThree'] = '%.1f' % F['geometry']['covering_radii_deg'][2]
    m['nWithFail'] = run['t1']['with_failures']
    fcount = [int(r['n_bad']) for r in csv.DictReader(
        (l for l in open(ROOT / 'data' / 'ds003775_channel_status.tsv') if not l.startswith('#')),
        delimiter='\t') if r['session'] == 'ses-t1']
    m['failMedian'] = '%d' % statistics.median(fcount)
    m['failMax'] = max(fcount)
    m['failNone'] = sum(f == 0 for f in fcount)
    m['PoneP'] = pval(run['t1']['U']['wilcoxon']['p_greater'])
    m['PoneMCratio'] = '%.1f' % run['t1']['U']['montecarlo']['ratio']
    m['PoneMCp'] = pval(run['t1']['U']['montecarlo']['p_greater'])
    m['PoneMCn'] = F['plan']['m_global']
    m['PonePos'] = run['t1']['U']['wilcoxon']['positive']
    m['PoneNeg'] = run['t1']['U']['wilcoxon']['negative']
    m['PoneMeanObs'] = '%.2f' % (100 * run['t1']['U']['mean_observed'])
    m['PoneMeanNull'] = '%.2f' % (100 * run['t1']['U']['mean_null'])
    m['PoneTwoP'] = pval(run['t2']['U']['wilcoxon']['p_greater'])
    m['SoneP'] = pval(run['t1']['A']['wilcoxon']['p_greater'])
    m['SoneObs'] = int(run['t1']['A']['montecarlo']['observed_sum'])
    m['SoneNull'] = '%.0f' % run['t1']['A']['montecarlo']['null_mean']
    m['SoneTwoP'] = pval(run['t2']['A']['wilcoxon']['p_greater'])
    qc = run['t1']['qc']
    m['qcPassLost'] = qc['pass90_and_lost']
    m['qcFailIntact'] = qc['fail90_and_intact']
    m['qcPassIntact'] = qc['pass90_and_intact']
    m['qcFailLost'] = qc['fail90_and_lost']
    h = F['heterogeneity']
    m['hetChi'] = '%.0f' % h['chi2']
    m['hetNull'] = '%.0f' % h['null_mean']
    m['hetRho'] = '%.2f' % h['spearman_rate_vs_colatitude']['rho']
    m['maxRateName'] = max(h['rates'], key=h['rates'].get)
    m['maxRate'] = '%.0f' % (100 * max(h['rates'].values()))
    m['recJac'] = '%.3f' % F['recurrence']['mean_jaccard']
    m['recNull'] = '%.3f' % F['recurrence']['null_mean']
    m['recP'] = pval(F['recurrence']['p'])
    c = F['certificate']
    m['certExact'] = c['agree_with_exact']
    m['certRaster'] = c['agree_with_raster']
    m['stressSets'] = c['random_stress']['sets']
    m['stressAgree'] = c['random_stress']['agree']
    ho = F['holes']
    m['holesObs'] = '%.2f' % ho['holes_observed_mean']
    m['holesNull'] = '%.2f' % ho['holes_null_mean']
    m['holesP'] = pval(ho['wilcoxon_less'])
    e1 = D['E1']['ses-t1']
    m['reachNinetyFiveObs'] = '%.1f' % e1['reach95_observed']
    m['reachNinetyFiveNull'] = '%.1f' % e1['reach95_null']
    m['reachNinetyFiveNullLo'] = '%.1f' % e1['reach95_null_lo']
    m['reachNinetyFiveNullHi'] = '%.1f' % e1['reach95_null_hi']
    m['reachNinetyObs'] = '%.1f' % e1['reach90_observed']
    m['reachNinetyNull'] = '%.1f' % e1['reach90_null']
    m['reachP'] = pval(e1['wilcoxon_greater'])
    e2 = D['E2']
    m['DoneTwo'] = '%.1f' % e2['designs']['D1']['covering_radii'][1]
    m['DzeroTwo'] = '%.1f' % e2['designs']['D0']['covering_radii'][1]
    m['DoneP'] = pval(e2['D1_vs_D0']['U']['p_greater'])
    m['DoneBetter'] = e2['D1_vs_D0']['U']['improved']
    m['DoneWorse'] = e2['D1_vs_D0']['U']['worse']
    cv = e2['D2_crossvalidated']
    m['DtwoPa'] = pval(cv['0']['test_U']['p_greater'])
    m['DtwoPb'] = pval(cv['1']['test_U']['p_greater'])

    mc = {(x['array'], x['radius_mm']): x for x in O['matched_comparison']}
    g8, o8 = mc[('grid_8x8', 8.0)], mc[('greedy', 8.0)]
    m['patchArea'] = '%.0f' % O['patch']['area_mm2']
    m['gridUnseen'] = '%.0f' % g8['uncovered_mm2']
    m['gridUnseenPct'] = '%.1f' % (100 * g8['uncovered_fraction'])
    m['gridBetti'] = '(%d, %d)' % tuple(g8['shadow1'])
    m['greedyUnseen'] = '%.1f' % o8['uncovered_mm2']
    m['greedyBetti'] = '(%d, %d)' % tuple(o8['shadow1'])
    m['crownFloor'] = '%.2f' % O['crown_floor']['covering_radius_mm']
    m['crownVerts'] = O['crown_floor']['n_crown_vertices']
    for key, tag in (('k1_r8', 'Eight'), ('k1_r10', 'Ten'), ('k1_r12', 'Twelve'),
                     ('k1_r14', 'Fourteen')):
        first = next(x for x in O['curves'][key]['scan'] if x['ok'])
        m['minN%s' % tag] = first['n']
    for key, tag in (('8.0', 'Eight'), ('10.0', 'Ten'), ('14.0', 'Fourteen')):
        m['pack%s' % tag] = O['packing'][key]['size']
    two = {(x['n_contacts'], x['radius_mm']): x for x in O['curves']['k2']}
    a, b = two[(96, 10.0)], two[(112, 9.0)]
    m['twoAUncov'] = '%.1f' % a['uncovered_area_mm2']['2']
    m['twoADisk'] = a['disk_test_failures']
    m['twoAFaces'] = a['faces']
    m['twoBUncov'] = '%.1f' % b['uncovered_area_mm2']['2']
    m['twoBDisk'] = b['disk_test_failures']
    m['twoBFaces'] = b['faces']
    m['twoBShadow'] = '(%d, %d)' % tuple(b['shadow']['2'])
    m['twoBMesh'] = '(%d, %d)' % (b['mesh']['2']['components'], b['mesh']['2']['b1'])

    S = load('signal_rebuild.json')
    m['sigRec'] = S['recordings']
    m['sigRows'] = S['rows']
    m['sigMedianR'] = '%.3f' % S['overall']['median_r']
    m['sigRecTwo'] = S['recordings'] - m['nTone']
    for key, tag in (('P_lambda_t1', 'P'), ('S1_pi_t1', 'Sone'), ('S3_lambda_t2', 'Sthree')):
        t = S[key]
        m['sig%sCh' % tag] = t['channels']
        m['sig%sRho' % tag] = '%.3f' % t['median_rho']
        m['sig%sPos' % tag] = t['positive']
        m['sig%sNeg' % tag] = t['negative']
        m['sig%sP' % tag] = pval(t.get('p'))
    t = S['S2_lambda_given_d_t1']
    m['sigStwoRec'] = t['recordings']
    m['sigStwoRho'] = '%.3f' % t['median_partial_rho']
    m['sigStwoPos'] = t['positive']
    m['sigStwoP'] = pval(t['p'])

    X = load('signal_explore.json')
    xz = X['X3_prediction_z']
    gain = [xz[b][k] - xz[a][k] for a, b in (('M2', 'M4'), ('M5', 'M6'))
            for k in ('cv_r2_t1', 'test_r2_t2')]
    m['topoGainBound'] = '%.3f' % (math.ceil(1000 * max(abs(g) for g in gain)) / 1000)
    m['siteGain'] = '%.2f' % (xz['M5']['cv_r2_t1'] - xz['M2']['cv_r2_t1'])
    m['lamDev'] = '%.4f' % X['X1_lambda_is_pairwise']['max_abs_deviation']
    for block, prefix in (('X2_within_channel_t1', 'xw'), ('X2_within_channel_t2', 'xv'),
                          ('X2_within_channel_t1_centred', 'xc'),
                          ('X2_within_channel_t2_centred', 'xd')):
        w = X[block]
        for key, tag in (('lambda', 'Lam'), ('minus_d1', 'Done'), ('minus_mean_d123', 'Dthree'),
                         ('n_within_2theta', 'Ntwo'), ('minus_pi', 'Pi'),
                         ('lambda_given_d1', 'LamGivenDone'),
                         ('lambda_given_d123', 'LamGivenDthree'),
                         ('lambda_given_n_good', 'LamGivenNgood')):
            m['%s%sRho' % (prefix, tag)] = '%.3f' % w[key]['median_rho']
            m['%s%sP' % (prefix, tag)] = pval(w[key].get('p_greater'))
            m['%s%sCh' % (prefix, tag)] = w[key]['channels']
            m['%s%sPos' % (prefix, tag)] = w[key]['positive']
    for out, tag in (('X3_prediction_z', 'z'), ('X3_prediction_nrmse', 'e')):
        for model, v in X[out].items():
            name = 'M' + WORDS[int(model[1:])]
            m['cv%s%s' % (tag, name)] = '%.3f' % v['cv_r2_t1']
            if 'test_r2_t2' in v:
                m['te%s%s' % (tag, name)] = '%.3f' % v['test_r2_t2']

    if not (RES / 'failure_margins.json').exists():
        print('warning: results/failure_margins.json missing; its macros are left out')
        G = None
    else:
        G = load('failure_margins.json')
    for ses, tag in (('ses-t1', 'One'), ('ses-t2', 'Two')) if G else ():
        g = G[ses]
        for key, kt in (('A', 'A'), ('U', 'U'), ('rho95', 'Reach')):
            fmt = '%d' if key == 'A' else '%.1f' if key == 'rho95' else '%.2f'
            m['mg%s%sObs' % (kt, tag)] = fmt % (
                g['observed'][key] if key != 'U' else 100 * g['observed'][key] / g['recordings'])
            for null, nt in (('margin_preserving', 'Marg'), ('uniform', 'Unif')):
                e = g[null][key]
                scale = 100 / g['recordings'] if key == 'U' else 1
                fmt = '%.0f' if key == 'A' else '%.1f' if key == 'rho95' else '%.2f'
                m['mg%s%s%sMean' % (kt, tag, nt)] = fmt % (e['null_mean'] * scale)
                m['mg%s%s%sLo' % (kt, tag, nt)] = fmt % (e['null_q025'] * scale)
                m['mg%s%s%sHi' % (kt, tag, nt)] = fmt % (e['null_q975'] * scale)
                m['mg%s%s%sP' % (kt, tag, nt)] = pval(e['p_greater'])
    if G:
        m['mgNull'] = G['m_null']
        m['mgPfloor'] = '%.3f' % (1 / (1 + G['m_null']))
    C = load('failure_correlation.json')
    edges = C['bin_edges_deg']
    first = next(i for i, v in enumerate(C['ses-t1']['g']) if v is not None)
    m['gBinLo'] = '%d' % edges[first]
    m['gBinHi'] = '%d' % edges[first + 1]
    for ses, tag in (('ses-t1', 'One'), ('ses-t2', 'Two')):
        c = C[ses]
        m['gFirst%s' % tag] = '%.2f' % c['g'][first]
        m['gSecond%s' % tag] = '%.2f' % c['g'][first + 1]
        m['gThird%s' % tag] = '%.2f' % c['g'][first + 2]
        m['xi%s' % tag] = '%.0f' % c['xi_deg']
        m['xi%sLo' % tag] = '%.0f' % c['xi_boot_lo']
        m['xi%sHi' % tag] = '%.0f' % c['xi_boot_hi']
    m['gNullHiFirst'] = '%.2f' % C['ses-t1']['null_hi'][first]
    import numpy as np
    import run_eeg_failures as R
    _, U = R.load_positions()
    Dg = np.degrees(np.arccos(np.clip(U @ U.T, -1, 1)))
    np.fill_diagonal(Dg, np.inf)
    m['nnSpacing'] = '%.0f' % np.median(Dg.min(1))

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as fh:
        fh.write('%% generated by make_paper_numbers.py from results/*.json; do not edit\n')
        for k, v in m.items():
            assert k.isalpha(), k
            fh.write('\\newcommand{\\%s}{%s}\n' % (k, v))
    print('%d macros written to %s' % (len(m), OUT))


if __name__ == '__main__':
    main()
