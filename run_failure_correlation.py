"""
run_failure_correlation.py -- how far apart do channels still fail together?
Exploratory: written after run_failure_margins.py showed that real failures
cluster beyond what the channels' own failure rates explain.

X5  Pair correlation of failure.  For a pair of channels at angular
    separation gamma, count the recordings in which both failed.  Summed over
    the pairs in a separation bin and divided by the same count averaged over
    null cohorts that keep every recording's failure count and every
    channel's failure count (curveball swaps, as in run_failure_margins.py),
    this is g(gamma): 1 where co-failure is what the rates predict, above 1
    where neighbours fail together.  95% band from the null cohorts.  The
    correlation length xi is the separation at which g - 1 falls to 1/e of
    its value in the first bin, by linear interpolation, with a 95% interval
    from 2000 bootstrap resamples of the recordings.

Usage:  python3 run_failure_correlation.py     -> results/failure_correlation.json
"""
import itertools
import json
from pathlib import Path

import numpy as np

import run_eeg_failures as R
from run_failure_margins import curveball

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
SEED = 20260929
M_NULL = 1000
TRADES = 1000
N_BOOT = 2000
EDGES = np.arange(0, 100.1, 10.0)       # degrees; nearest BioSemi pairs are ~15-25


def pair_table(U):
    n = len(U)
    I, J = np.array(list(itertools.combinations(range(n), 2))).T
    gam = np.degrees(np.arccos(np.clip(np.sum(U[I] * U[J], 1), -1, 1)))
    return I, J, gam, np.digitize(gam, EDGES) - 1


def binned_cofail(M, I, J, b, nb):
    co = (M[:, I] & M[:, J]).sum(0).astype(float)
    return np.bincount(b, weights=co, minlength=nb)[:nb]


def xi_of(g):
    """Separation where g - 1 falls to 1/e of its value in the first bin that
    holds any pair, by linear interpolation between bin centres; None if that
    bin shows no excess or the excess never falls that far."""
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    ok = np.isfinite(g)
    ex, centres = g[ok] - 1, centres[ok]
    if not len(ex) or ex[0] <= 0:
        return None
    target = ex[0] / np.e
    for k in range(1, len(ex)):
        if ex[k] <= target:
            t = (ex[k - 1] - target) / (ex[k - 1] - ex[k])
            return float(centres[k - 1] + t * (centres[k] - centres[k - 1]))
    return None


def nan_to_none(a):
    return [None if not np.isfinite(x) else float(x) for x in a]


def main():
    rng = np.random.default_rng(SEED)
    names, U = R.load_positions()
    recs = R.load_status(names)
    I, J, gam, b = pair_table(U)
    nb = len(EDGES) - 1
    out = dict(seed=SEED, m_null=M_NULL, n_boot=N_BOOT, bin_edges_deg=EDGES.tolist(),
               pairs_per_bin=np.bincount(b, minlength=nb)[:nb].tolist(), status='exploratory')
    for ses in ('ses-t1', 'ses-t2'):
        coh = [r for r in recs if r['ses'] == ses]
        M = np.zeros((len(coh), len(U)), bool)
        for i, r in enumerate(coh):
            M[i, r['bad']] = True
        obs = binned_cofail(M, I, J, b, nb)
        null = []
        X = curveball(M, rng, 20 * TRADES)
        for _ in range(M_NULL):
            X = curveball(X, rng, TRADES)
            null.append(binned_cofail(X, I, J, b, nb))
        null = np.array(null)
        exp = null.mean(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            g = obs / exp
            band = null / exp
        boot = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(M), len(M))
            with np.errstate(invalid='ignore', divide='ignore'):
                boot.append(xi_of(binned_cofail(M[idx], I, J, b, nb) / exp))
        boot = np.array([x for x in boot if x is not None])
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
        out[ses] = dict(
            recordings=len(coh), observed=obs.tolist(), expected=exp.tolist(),
            g=nan_to_none(g), null_lo=nan_to_none(np.nanpercentile(band, 2.5, 0)),
            null_hi=nan_to_none(np.nanpercentile(band, 97.5, 0)),
            p_greater=((1 + (null >= obs).sum(0)) / (1 + M_NULL)).tolist(),
            xi_deg=xi_of(g), xi_boot_lo=float(lo), xi_boot_hi=float(hi),
            xi_boot_defined=int(len(boot)))
        print(ses, 'g =', np.round(g, 2), 'xi =', out[ses]['xi_deg'],
              '[%.1f, %.1f]' % (out[ses]['xi_boot_lo'], out[ses]['xi_boot_hi']), flush=True)
    with open(RES / 'failure_correlation.json', 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
