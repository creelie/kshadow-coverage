"""
run_signal_explore.py -- what the signal-level result can and cannot say
about topology.  Exploratory: written after run_signal_rebuild.py had been
run, and after its pre-specified tests were recorded as they came out.  It
reads only results/signal_rebuild_rows.csv and the two data tables, so it
needs no EEG signals.

X1  lambda is pairwise.  The redundancy depth of run_signal_rebuild.py is
    the mean over the cap of c of the number of surviving caps covering the
    point.  Exchanging the mean and the sum gives

        lambda(c, S) = sum_{j in S} K(angle(c, j)),

    K(g) the fraction of a cap of radius theta covered by a second cap of
    the same radius at angular distance g (a lens area, zero for g >= 2
    theta).  So lambda is a kernel sum over pairwise distances and cannot
    carry information that the distances from c to the survivors do not.
    Checked here on every row against K evaluated by quadrature.  The
    private territory pi, by contrast, is the part of the cap outside a
    union of lenses and depends on how the lenses overlap one another.

X2  The within-channel natural experiment of test P, repeated with purely
    metric predictors in place of lambda: the nearest distance d1 (sign
    reversed, so that positive means "closer is better"), the mean of the
    three nearest distances, the number of survivors within 2 theta, and
    lambda given d1 and given (d1, d2, d3) by partial Spearman correlation.
    One-sided Wilcoxon over channels as in P.

X3  Out-of-sample comparison.  Rebuild quality z, centred within each
    recording (which removes every recording-level effect: subject, noise,
    number of failures), is predicted by nested least-squares models:
        M1  d1
        M2  d1..d6, the six nearest surviving distances, and their squares
        M3  M2 + lambda
        M4  M2 + lambda + pi
        M5  M2 + channel identity (64 indicators: position, rim, anatomy)
        M6  M5 + lambda + pi
    scored by leave-one-subject-out cross-validated R^2 on the first
    sessions, with the second sessions as an untouched test set for models
    fitted on all first sessions.  The question: once the distance profile
    is in the model, does anything topological add predictive power?

Usage:  python3 run_signal_explore.py      -> results/signal_explore.json
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import integrate, stats

import run_signal_rebuild as S

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
THETA = np.radians(S.THETA)
NN = 6


def lens_fraction(g, th=THETA):
    """Fraction of a cap of radius th covered by another of radius th at
    angular separation g."""
    if g >= 2 * th:
        return 0.0
    if g < 1e-12:
        return 1.0
    area = 2 * np.pi * (1 - np.cos(th))

    def f(rho):
        s = np.sin(rho) * np.sin(g)
        if s <= 0:
            return 2 * np.pi * np.sin(rho) * (np.cos(rho) > np.cos(th))
        cosphi = (np.cos(th) - np.cos(rho) * np.cos(g)) / s
        return np.sin(rho) * 2 * np.arccos(np.clip(cosphi, -1, 1))

    return integrate.quad(f, 0, th, limit=200)[0] / area


def load():
    names, U, status, _ = S.load_tables()
    idx = {n: i for i, n in enumerate(names)}
    rows = list(csv.DictReader(open(RES / 'signal_rebuild_rows.csv', newline='')))
    grid = np.linspace(0, 2 * THETA, 4001)
    Kg = np.array([lens_fraction(g) for g in grid])
    out = []
    for r in rows:
        c = idx[r['channel']]
        bad = set(status[r['recording']]['bad'])
        surv = [i for i, n in enumerate(names) if n not in bad and i != c]
        g = np.sort(np.arccos(np.clip(U[surv] @ U[c], -1, 1)))
        out.append(dict(
            rec=r['recording'], sub=r['recording'].split('_')[0], ses=r['session'],
            ch=r['channel'], ci=c, z=float(r['z']), nrmse=float(r['nrmse']),
            lam=float(r['lambda']), pi=float(r['pi']), mu=int(r['mu']),
            d=np.degrees(g[:NN]), n2=int((g < 2 * THETA).sum()),
            lam_kernel=float(np.interp(g, grid, Kg).sum())))
    return names, out


# ------------------------------------------------------------------ X1
def x1(rows):
    dev = np.array([r['lam_kernel'] - r['lam'] for r in rows])
    return dict(rows=len(rows), max_abs_deviation=float(np.abs(dev).max()),
                mean_deviation=float(dev.mean()),
                note='lambda = sum_j K(angle(c,j)); deviation is Monte Carlo error '
                     'of the 400,000-point cap sampling in run_signal_rebuild.py')


# ------------------------------------------------------------------ X2
def partial_spearman(y, x, ctrl):
    ry = stats.rankdata(y)
    rx = stats.rankdata(x)
    C = np.column_stack([stats.rankdata(c) for c in np.atleast_2d(ctrl)] +
                        [np.ones(len(y))])

    def resid(a):
        return a - C @ np.linalg.lstsq(C, a, rcond=None)[0]

    a, b = resid(ry), resid(rx)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def summarise(rho):
    rho = {k: v for k, v in rho.items() if np.isfinite(v)}
    v = np.array(list(rho.values()))
    out = dict(channels=len(v), median_rho=float(np.median(v)) if len(v) else None,
               positive=int((v > 0).sum()), negative=int((v < 0).sum()), rho=rho)
    if len(v) >= 5:
        out['p_greater'] = float(stats.wilcoxon(v, alternative='greater').pvalue)
    return out


def x2(rows, session='ses-t1'):
    by = {}
    for r in rows:
        if r['ses'] == session:
            by.setdefault(r['ch'], []).append(r)
    preds = {
        'lambda': lambda rr: np.array([r['lam'] for r in rr]),
        'minus_d1': lambda rr: -np.array([r['d'][0] for r in rr]),
        'minus_mean_d123': lambda rr: -np.array([r['d'][:3].mean() for r in rr]),
        'n_within_2theta': lambda rr: np.array([r['n2'] for r in rr], float),
        'minus_pi': lambda rr: -np.array([r['pi'] for r in rr]),
    }
    out = {}
    for name, f in preds.items():
        rho = {}
        for ch, rr in by.items():
            v = f(rr)
            if len(rr) >= 10 and len(np.unique(v)) >= 3:
                rho[ch] = float(stats.spearmanr(v, [r['z'] for r in rr])[0])
        out[name] = summarise(rho)
    for label, k in (('lambda_given_d1', 1), ('lambda_given_d123', 3)):
        rho = {}
        for ch, rr in by.items():
            lam = np.array([r['lam'] for r in rr])
            if len(rr) >= 10 and len(np.unique(lam)) >= 3:
                D = np.array([r['d'][:k] for r in rr]).T
                rho[ch] = partial_spearman([r['z'] for r in rr], lam, D)
        out[label] = summarise(rho)
    return out


# ------------------------------------------------------------------ X3
def design(rows, model):
    d = np.array([r['d'] for r in rows])
    cols = [np.ones(len(rows))]
    if model == 'M1':
        cols += [d[:, 0], d[:, 0] ** 2]
    else:
        cols += list(d.T) + list((d ** 2).T)
    if model in ('M3', 'M4', 'M6'):
        lam = np.array([r['lam'] for r in rows])
        cols += [lam, lam ** 2]
    if model in ('M4', 'M6'):
        pi = np.array([r['pi'] for r in rows])
        cols += [pi, (pi > 0).astype(float)]
    if model in ('M5', 'M6'):
        ci = np.array([r['ci'] for r in rows])
        cols += [(ci == j).astype(float) for j in range(1, 64)]
    return np.column_stack(cols)


def centred(rows, key='z'):
    y = np.array([r[key] for r in rows])
    rec = np.array([r['rec'] for r in rows])
    out = y.copy()
    for u in np.unique(rec):
        m = rec == u
        out[m] -= y[m].mean()
    return out


def centred_design(rows, model):
    X = design(rows, model)
    rec = np.array([r['rec'] for r in rows])
    Xc = X.copy()
    for u in np.unique(rec):
        m = rec == u
        Xc[m] -= X[m].mean(0)
    return Xc[:, 1:]            # the intercept is absorbed by centring


def fit(X, y, ridge=1e-6):
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / (y ** 2).sum())


def x3(rows, key='z'):
    t1 = [r for r in rows if r['ses'] == 'ses-t1']
    t2 = [r for r in rows if r['ses'] == 'ses-t2']
    y1, y2 = centred(t1, key), centred(t2, key)
    subs = np.array([r['sub'] for r in t1])
    out = {}
    for m in ('M1', 'M2', 'M3', 'M4', 'M5', 'M6'):
        X1 = centred_design(t1, m)
        yh = np.empty_like(y1)
        for s in np.unique(subs):
            te = subs == s
            yh[te] = X1[te] @ fit(X1[~te], y1[~te])
        entry = dict(cv_r2_t1=r2(y1, yh), parameters=int(X1.shape[1]))
        if t2:
            X2 = centred_design(t2, m)
            entry['test_r2_t2'] = r2(y2, X2 @ fit(X1, y1))
        out[m] = entry
    return out


def main():
    names, rows = load()
    res = dict(rows=len(rows), recordings=len({r['rec'] for r in rows}),
               status='exploratory, written after the pre-specified tests were run')
    res['X1_lambda_is_pairwise'] = x1(rows)
    res['X2_within_channel_t1'] = x2(rows, 'ses-t1')
    res['X2_within_channel_t2'] = x2(rows, 'ses-t2')
    res['X3_prediction_z'] = x3(rows, 'z')
    res['X3_prediction_nrmse'] = x3(rows, 'nrmse')
    with open(RES / 'signal_explore.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
