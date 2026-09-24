"""
run_failure_margins.py -- is the clustering of real contact failures more
than a gradient in failure rate?  Exploratory: written after
run_eeg_failures.py and run_eeg_design.py had been run.

The pre-specified tests of run_eeg_failures.py compare each recording's
failure set with a uniformly random set of the same size.  That null
ignores that channels fail at very different rates (test S3: chi-square
p < 1e-4, rate against colatitude Spearman 0.82), and the channels that fail
most are the rim channels, which are each other's neighbours.  A rate
gradient alone therefore produces adjacent failures, larger uncovered
areas at the rim, and a larger required reach, with no interaction between
failures at all.

X4  Margin-preserving null.  The failures of a cohort form a binary matrix,
    recordings x channels.  Curveball swaps (Strona et al. 2014) sample
    uniformly among matrices with the same row sums (each recording keeps
    its number of failures) and the same column sums (each channel keeps
    its failure count).  Under this null a channel's failures are placed
    with no regard to which other channels failed in the same recording.
    Statistics, at the primary radius 23.5 deg:
        A      failed pairs whose caps meet, summed over the cohort
        U      uncovered target fraction, summed over the cohort
        rho95  the 95th percentile over recordings of the 1-fold reach
               rho_1(S) needed to see all of the target from the survivors
    Upper-tail Monte Carlo p-values against 1000 null cohorts; the uniform
    null of run_eeg_failures.py is reported beside it for comparison.

Usage:  python3 run_failure_margins.py        -> results/failure_margins.json
"""
import itertools
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import run_eeg_failures as R
import run_eeg_design as D

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
SEED = 20260928
THETA = R.THETA_PRIMARY
M_NULL = 1000
TRADES = 1000              # curveball trades between recorded null cohorts
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def curveball(M, rng, trades):
    rows = [set(np.flatnonzero(r)) for r in M]
    n = len(rows)
    for _ in range(trades):
        a, b = rng.choice(n, 2, replace=False)
        ab = rows[a] - rows[b]
        ba = rows[b] - rows[a]
        if not ab or not ba:
            continue
        pool = list(ab | ba)
        rng.shuffle(pool)
        keep = rows[a] & rows[b]
        rows[a] = keep | set(pool[:len(ab)])
        rows[b] = keep | set(pool[len(ab):])
    out = np.zeros_like(M)
    for i, r in enumerate(rows):
        out[i, list(r)] = True
    return out


_G = {}


def _init(U, cells, adj):
    _G.update(U=U, cells=cells, adj=adj)


def _stats(M):
    return stats_of(M, _G['U'], _G['cells'], _G['adj'])


def stats_of(M, U, cells, adj):
    masks = np.array([R.mask_of(np.flatnonzero(r)) for r in M], dtype=np.uint64)
    A = int(R.adjacency_count(masks, adj).sum())
    Us = float(cells.uncovered(masks).sum())
    reach = np.array([D.reach1(U[~r]) for r in M])
    return dict(A=A, U=Us, rho95=float(np.percentile(reach, 95)),
                rho90=float(np.percentile(reach, 90)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    names, U = R.load_positions()
    n = len(names)
    recs = R.load_status(names)
    P = R.fib_sphere(R.N_FIB)
    P = P[P[:, 2] >= np.cos(np.radians(R.T_COLAT))]
    cells = R.Cells(R.signatures(P, U, THETA))
    adj = [(i, j) for i, j in itertools.combinations(range(n), 2)
           if R.gdist(U[i], U[j]) < 2 * np.radians(THETA)]
    out = dict(theta=THETA, seed=SEED, m_null=M_NULL, trades=TRADES,
               status='exploratory', points_in_target=int(len(P)))
    for ses in ('ses-t1', 'ses-t2'):
        coh = [r for r in recs if r['ses'] == ses]
        M = np.zeros((len(coh), n), bool)
        for i, r in enumerate(coh):
            M[i, r['bad']] = True
        obs = stats_of(M, U, cells, adj)
        mats_m, mats_u = [], []
        X = curveball(M, rng, 20 * TRADES)          # burn-in
        for k in range(M_NULL):
            X = curveball(X, rng, TRADES)
            assert (X.sum(0) == M.sum(0)).all() and (X.sum(1) == M.sum(1)).all()
            mats_m.append(X)
            Y = np.zeros_like(M)
            for i, f in enumerate(M.sum(1)):
                Y[i, rng.choice(n, f, replace=False)] = True
            mats_u.append(Y)
        with Pool(WORKERS, initializer=_init, initargs=(U, cells, adj)) as pool:
            null_m = pool.map(_stats, mats_m, chunksize=10)
            print('%s margin-preserving null done (%.0f s)' % (ses, time.time() - t0), flush=True)
            null_u = pool.map(_stats, mats_u, chunksize=10)
            print('%s uniform null done (%.0f s)' % (ses, time.time() - t0), flush=True)
        res = dict(recordings=len(coh), observed=obs)
        for label, null in (('margin_preserving', null_m), ('uniform', null_u)):
            e = {}
            for key in ('A', 'U', 'rho95', 'rho90'):
                v = np.array([x[key] for x in null])
                e[key] = dict(null_mean=float(v.mean()), null_sd=float(v.std()),
                              null_q025=float(np.percentile(v, 2.5)),
                              null_q975=float(np.percentile(v, 97.5)),
                              p_greater=float((1 + (v >= obs[key]).sum()) / (1 + len(v))))
            res[label] = e
        out[ses] = res
    out['seconds'] = time.time() - t0
    with open(RES / 'failure_margins.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({s: {k: out[s][k] for k in ('observed', 'margin_preserving', 'uniform')}
                      for s in ('ses-t1', 'ses-t2')}, indent=1))


if __name__ == '__main__':
    main()
