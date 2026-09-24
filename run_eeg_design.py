"""
run_eeg_design.py -- what the real failures ask of a design, and whether a
better layout answers it.  Exploratory: written after run_eeg_failures.py,
whose pre-specified primary test is reported as it came out.

E1  Required reach (model-free apart from the cap footprint).  For each
    recording, rho_k(S) is the smallest cap radius at which the surviving
    channels S see every point of the target T at least k times.  Real
    failure sets are compared with uniformly random sets of the same size:
    one-sided Wilcoxon signed-rank on rho_k(S_obs) - E_null[rho_k | f], and
    the reach needed to keep the whole target seen in 90% and 95% of
    recordings.  Continuous, so it avoids the zero inflation of U.

E2  Layouts, all with 64 contacts:
      D0  the BioSemi 10-20 cap;
      D1  positions chosen to minimise the 2-fold covering radius of T,
          anywhere within the cap's own extent (colatitude <= 115 deg);
          uses no failure data;
      D2  the same optimisation restricted to colatitude <= c, with c chosen
          from {70, 75, 80, 85, 92} on one half of the subjects and scored
          on the other half (two folds, split by subject number parity).
    Real failures are carried to a new layout by one stated assumption:
    a contact fails in a recording exactly when the BioSemi channel nearest
    to it failed there (failure is a property of the scalp site).  Outcome:
    U, the uncovered fraction of T at the primary radius 23.5 deg, and
    rho_1.  Test: one-sided Wilcoxon signed-rank, D0 minus design, over the
    held-out t1 recordings.

Stages:  e1 | opt D1 | opt 70 | opt 75 | opt 80 | opt 85 | opt 92 | eval.
Writes results/eeg_design.json.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

import run_eeg_failures as R

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
SEED = 20260927
THETA = 23.5
N_COARSE = 400_000       # full-sphere Fibonacci points for E1 and optimisation
M_NULL = 400
CAP_EXTENT = 115.0
C_GRID = (70.0, 75.0, 80.0, 85.0, 92.0)
ITERS = 6000
N_OPT = 30_000           # full-sphere points for the optimiser (~15,500 in T)


def target_points(n):
    P = R.fib_sphere(n)
    return P[P[:, 2] >= np.cos(np.radians(R.T_COLAT))]


BOUNDARY = None


def reach1(X):
    """Exact 1-fold covering radius (deg) of T by the points X: the largest
    distance to the nearest point is attained at a Voronoi vertex inside T or
    on the boundary circle of T, which is sampled every 0.02 deg."""
    global BOUNDARY
    from scipy.spatial import SphericalVoronoi
    if BOUNDARY is None:
        t = np.radians(R.T_COLAT)
        a = np.radians(np.arange(0, 360, 0.02))
        BOUNDARY = np.stack([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a),
                             np.full_like(a, np.cos(t))], 1)
    sv = SphericalVoronoi(X, radius=1.0, threshold=1e-9)
    V = sv.vertices
    V = V[V[:, 2] >= np.cos(np.radians(R.T_COLAT))]
    Q = np.vstack([V, BOUNDARY])
    return float(np.degrees(np.arccos(np.clip(Q @ X.T, -1, 1)).min(1).max()))


def kth_reach(D, surv_mask, k):
    """k-fold covering radius (deg) of the points by the surviving columns."""
    Ds = D[:, surv_mask]
    if Ds.shape[1] < k:
        return 180.0
    return float(np.degrees(np.partition(Ds, k - 1, axis=1)[:, k - 1].max()))


def slerp_toward(a, b, frac):
    ang = np.arccos(np.clip(a @ b, -1, 1))
    if ang < 1e-9:
        return a
    t = frac * ang
    w = b - (a @ b) * a
    w /= np.linalg.norm(w)
    return np.cos(t) * a + np.sin(t) * w


def clamp_colat(p, cmax):
    c = np.degrees(np.arccos(np.clip(p[2], -1, 1)))
    if c <= cmax:
        return p
    az = np.arctan2(p[1], p[0])
    th = np.radians(cmax)
    return np.array([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)])


def optimise(U0, P, cmax, iters, rng):
    """Minimax local search for the 2-fold covering radius of P: move one of
    the two contacts nearest the worst-served point towards it, keep the move
    if the radius drops; when the step size has collapsed, jitter the best
    layout found so far by up to 1 deg per contact and continue (basin
    hopping).  Returns the best layout seen."""
    X = np.array([clamp_colat(u, cmax) for u in U0])
    D = np.arccos(np.clip(P @ X.T, -1, 1))

    def second(DD):
        return np.partition(DD, 1, axis=1)[:, 1]

    s = second(D)
    cur = s.max()
    bestX, best = X.copy(), cur
    eta = 0.5
    for it in range(iters):
        w = int(np.argmax(s))
        order = np.argsort(D[w])[:2]
        moved = False
        for j in order[::-1]:
            old = X[j].copy()
            oldcol = D[:, j].copy()
            X[j] = clamp_colat(slerp_toward(X[j], P[w], eta * rng.uniform(0.3, 1.0)), cmax)
            D[:, j] = np.arccos(np.clip(P @ X[j], -1, 1))
            s2 = second(D)
            if s2.max() < cur - 1e-12:
                s, cur, moved = s2, s2.max(), True
                break
            X[j] = old
            D[:, j] = oldcol
        if cur < best:
            bestX, best = X.copy(), cur
        if not moved:
            eta *= 0.95
            if eta < 2e-3:
                X = bestX.copy()
                for j in range(len(X)):
                    v = rng.normal(size=3)
                    v -= (v @ X[j]) * X[j]
                    v /= np.linalg.norm(v)
                    a = np.radians(rng.uniform(0, 1.0))
                    X[j] = clamp_colat(np.cos(a) * X[j] + np.sin(a) * v, cmax)
                D = np.arccos(np.clip(P @ X.T, -1, 1))
                s = second(D)
                cur = s.max()
                eta = 0.5
    return bestX, float(np.degrees(best))


def transfer(X, U0):
    """Index of the BioSemi channel nearest to each new contact."""
    return np.argmax(X @ U0.T, axis=1)


def failed_new(owner, bad):
    b = np.zeros(64, bool)
    b[bad] = True
    return b[owner]


def stage_e1():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    names, U0, recs = R._setup()
    n = len(names)
    P = target_points(N_COARSE)
    D0 = np.arccos(np.clip(P @ U0.T, -1, 1)).astype(np.float32)
    out = dict(points_in_target=int(len(P)),
               grid_spacing_deg=float(np.degrees(np.sqrt(4 * np.pi / N_COARSE))),
               seed=SEED, m_null=M_NULL)

    # ---------------------------------------------------------------- E1
    fs = sorted(set(r['f'] for r in recs))
    null = {}
    for f in fs:
        vals = []
        for _ in range(M_NULL):
            surv = np.ones(n, bool)
            if f:
                surv[rng.choice(n, f, replace=False)] = False
            vals.append(reach1(U0[surv]))
        null[f] = np.array(vals)
    e1 = {}
    for cname in ('ses-t1', 'ses-t2'):
        coh = [r for r in recs if r['ses'] == cname]
        obs = []
        for r in coh:
            surv = np.ones(n, bool)
            surv[r['bad']] = False
            obs.append(reach1(U0[surv]))
        obs = np.array(obs)
        ex = np.array([null[r['f']].mean() for r in coh])
        sel = np.array([r['f'] >= 1 for r in coh])
        d = (obs - ex)[sel]
        nz = d[d != 0]
        qs = []
        for _ in range(2000):
            draw = np.array([null[r['f']][rng.integers(M_NULL)] for r in coh])
            qs.append(np.quantile(draw, [0.90, 0.95]))
        qs = np.array(qs)
        e1[cname] = dict(
            intact=reach1(U0),
            observed_mean=float(obs.mean()), null_mean=float(ex.mean()),
            wilcoxon_greater=float(stats.wilcoxon(nz, alternative='greater').pvalue),
            wilcoxon_two_sided=float(stats.wilcoxon(nz).pvalue),
            positive=int((nz > 0).sum()), negative=int((nz < 0).sum()),
            reach90_observed=float(np.quantile(obs, 0.90)),
            reach95_observed=float(np.quantile(obs, 0.95)),
            reach90_null=float(qs[:, 0].mean()), reach95_null=float(qs[:, 1].mean()),
            reach95_null_lo=float(np.quantile(qs[:, 1], 0.025)),
            reach95_null_hi=float(np.quantile(qs[:, 1], 0.975)),
            observed=[float(x) for x in obs])
        print('E1 %s: rho1 obs %.2f null %.2f p=%.3g; 95%% reach obs %.2f null %.2f [%.2f, %.2f]' % (
            cname, obs.mean(), ex.mean(), e1[cname]['wilcoxon_greater'],
            e1[cname]['reach95_observed'], e1[cname]['reach95_null'],
            e1[cname]['reach95_null_lo'], e1[cname]['reach95_null_hi']), flush=True)
    out['E1'] = e1
    with open(RES / 'eeg_design_e1.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('E1 done in %.0f s' % (time.time() - t0))


def stage_opt(which):
    t0 = time.time()
    rng = np.random.default_rng(SEED + (0 if which == 'D1' else int(which)))
    names, U0, recs = R._setup()
    Pc = target_points(N_OPT)
    cmax = CAP_EXTENT if which == 'D1' else float(which)
    X, r2 = optimise(U0, Pc, cmax, ITERS, rng)
    key = 'D1' if which == 'D1' else 'D2_%d' % int(cmax)
    np.save(RES / ('eeg_design_%s.npy' % key), X)
    print('%s: cmax %.0f, coarse 2-fold radius %.2f deg (%.0f s)' % (key, cmax, r2, time.time() - t0))


def stage_eval():
    t0 = time.time()
    names, U0, recs = R._setup()
    out = json.load(open(RES / 'eeg_design_e1.json'))
    P = target_points(N_COARSE)
    designs = {'D0': dict(X=U0.copy(), cmax=CAP_EXTENT)}
    designs['D1'] = dict(X=np.load(RES / 'eeg_design_D1.npy'), cmax=CAP_EXTENT)
    for c in C_GRID:
        designs['D2_%d' % int(c)] = dict(X=np.load(RES / ('eeg_design_D2_%d.npy' % int(c))), cmax=c)
    Pf = target_points(R.N_FIB)
    evals = {}
    for key, d in designs.items():
        X = d['X']
        Dd = np.arccos(np.clip(P @ X.T, -1, 1)).astype(np.float32)
        cells = R.Cells(R.signatures(Pf, X, THETA))
        owner = transfer(X, U0)
        Uv, rho = [], []
        for r in recs:
            fm = failed_new(owner, r['bad'])
            Uv.append(float(cells.uncovered([R.mask_of(np.flatnonzero(fm))])[0]))
            rho.append(reach1(X[~fm]))
        cr = [reach1(X)] + [kth_reach(Dd, np.ones(64, bool), k) for k in (2, 3)]
        evals[key] = dict(cmax=d['cmax'], covering_radii=cr, U=Uv, rho1=rho,
                          owners_distinct=int(len(set(owner.tolist()))),
                          positions=[[float(v) for v in x] for x in X])
        print('%s cmax %.0f: covering radii %s; mean U %.4f' % (
            key, d['cmax'], np.round(cr, 2), np.mean(Uv)), flush=True)

    recs_idx = {r['rec']: i for i, r in enumerate(recs)}
    t1 = [r for r in recs if r['ses'] == 'ses-t1']

    def paired(a_key, b_key, subset, what='U'):
        a = np.array([evals[a_key][what][recs_idx[r['rec']]] for r in subset])
        b = np.array([evals[b_key][what][recs_idx[r['rec']]] for r in subset])
        d = a - b
        nz = d[d != 0]
        p = float(stats.wilcoxon(nz, alternative='greater').pvalue) if len(nz) else 1.0
        return dict(n=len(subset), a_mean=float(a.mean()), b_mean=float(b.mean()),
                    improved=int((d > 0).sum()), worse=int((d < 0).sum()),
                    a_lost=int((a > 0).sum()) if what == 'U' else None,
                    b_lost=int((b > 0).sum()) if what == 'U' else None,
                    p_greater=p)

    out['E2'] = dict(designs={k: {kk: vv for kk, vv in v.items() if kk not in ('U', 'rho1')}
                              for k, v in evals.items()},
                     D1_vs_D0=dict(U=paired('D0', 'D1', t1), rho1=paired('D0', 'D1', t1, 'rho1')))
    folds = {}
    for fold in (0, 1):
        train = [r for r in t1 if int(r['sub'][-3:]) % 2 == fold]
        test = [r for r in t1 if int(r['sub'][-3:]) % 2 != fold]
        score = {c: float(np.mean([evals['D2_%d' % int(c)]['U'][recs_idx[r['rec']]]
                                   for r in train])) for c in C_GRID}
        cbest = min(score, key=lambda c: (score[c], -c))
        key = 'D2_%d' % int(cbest)
        folds[str(fold)] = dict(train_scores={str(int(c)): v for c, v in score.items()},
                                chosen_cmax=cbest,
                                test_U=paired('D0', key, test),
                                test_rho1=paired('D0', key, test, 'rho1'))
        print('fold %d: c=%.0f; test U D0 %.4f -> %.4f, p=%.3g' % (
            fold, cbest, folds[str(fold)]['test_U']['a_mean'],
            folds[str(fold)]['test_U']['b_mean'], folds[str(fold)]['test_U']['p_greater']),
            flush=True)
    out['E2']['D2_crossvalidated'] = folds
    out['E2']['U_per_recording'] = {k: v['U'] for k, v in evals.items()}
    out['seconds_eval'] = time.time() - t0
    with open(RES / 'eeg_design.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote results/eeg_design.json in %.0f s' % out['seconds_eval'])


if __name__ == '__main__':
    import sys
    st = sys.argv[1]
    if st == 'e1':
        stage_e1()
    elif st == 'opt':
        stage_opt(sys.argv[2])
    elif st == 'eval':
        stage_eval()
