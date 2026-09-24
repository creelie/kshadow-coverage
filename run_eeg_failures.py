"""
run_eeg_failures.py -- the k-shadow certificate on real contact failures.

Data
----
data/ds003775_channel_status.tsv, built by fetch_ds003775_status.py: for each
of the 153 recordings of OpenNeuro ds003775 v1.2.1 (111 subjects, session t1;
42 of them again 2-3 months later, session t2) the set of the 64 BioSemi
channels that the dataset's own automated pipeline marked "bad".  These are
real contact failures on a real cap, decided by the data curators, not by us.
data/biosemi64_unit.tsv: the 64 positions on the unit sphere.

Model
-----
Each channel is given a spherical-cap footprint of angular radius theta.
The target T is the scalp region bounded by the Fpz-T7-Oz-T8 circumference
of the 10-20 system, colatitude <= 92 degrees.  For the intact cap the
covering radii of T are theta_1 = 16.0, theta_2 = 23.0 and theta_3 = 25.3
degrees (computed below): above theta_k every point of T is seen by at
least k channels.  A recording with failure set F keeps the caps of the
surviving channels S = all minus F.

Plan, fixed before any outcome was computed
-------------------------------------------
Primary radius theta* = 23.5 deg (theta_2 rounded up to the next half
degree), at which the intact cap sees all of T at least twice, so that no
single failure can uncover any of T.  Sensitivity radii 16.5 and 26.0 deg.
Primary cohort: session t1 (111 independent subjects).  Replication: t2.

Null model: F replaced by a uniformly random subset of the same size.

P1  (primary) Uncovered target fraction U(S), the area of T seen by no
    surviving channel.  Test: one-sided Wilcoxon signed-rank on
    U_obs - E_null[U | |F|] over recordings with |F| >= 1, and a Monte Carlo
    test of sum U_obs against 10,000 independent null replicates of the
    whole cohort.  alpha = 0.05.
S1  Spatial clustering: A(F) = number of failed pairs whose caps meet at
    theta*.  Same two tests.
S2  Channel-count quality control against the coverage outcome: the
    descriptor's rule "more than 90% of channels kept" against U(S) = 0.
S3  Heterogeneity of failure across channels: Monte Carlo chi-square test
    that keeps each recording's failure count; Spearman correlation of
    per-channel failure rate with colatitude.
S4  Recurrence within subject: mean Jaccard index of the t1 and t2 failure
    sets over the 42 subjects, against 10,000 random re-pairings.
S5  The certificate on the observed surviving caps: b0, b1 of Delta_1 and
    Delta_2 of the nerve, computed from intersection data alone, checked
    against the exact topology of the k-fold region computed from the
    arrangement of the boundary circles (arrangement.py), and against a
    raster of the multiplicity function in a stereographic chart.

Nerve: caps of radius theta < 30 deg, so any subfamily whose triples meet
lies in the open hemisphere about any of its centres (every cap is within
3*theta < 90 deg of it), the gnomonic chart maps the caps to planar convex
sets, and Helly's theorem makes the nerve the Helly completion of its
triangles (as in run_sphere.py).

Stages, each well inside a few minutes:
    python3 run_eeg_failures.py main         # P1, S1-S4
    python3 run_eeg_failures.py cert         # S5 on the 153 recordings
    python3 run_eeg_failures.py b1           # holes, real against random
    python3 run_eeg_failures.py stress 0 500 # certificate vs exact, random sets
    python3 run_eeg_failures.py stress 500 1000
    python3 run_eeg_failures.py merge        # -> results/eeg_failures.json
or `python3 run_eeg_failures.py` for everything in one go.
"""
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from kshadow import HellyNerve, shadow_betti, raster_betti
from arrangement import caps_to_disks, region_betti

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)

SEED = 20260924
THETA_PRIMARY = 23.5
THETA_SENS = (16.5, 26.0)
T_COLAT = 92.0
N_FIB = 4_000_000          # full-sphere Fibonacci points; ~2.07e6 fall in T
M_POOL = 20_000            # null draws per failure count
M_GLOBAL = 10_000          # cohort-level null replicates
M_PERM = 10_000            # permutations for S3, S4
M_B1 = 200                 # null draws per recording for the b1 comparison
CHART_RES = 4000
CHART_HALF = 2.7          # stereographic radius; colatitude 2*atan(2.7) = 139.4 deg


# ------------------------------------------------------------------ data
def load_positions():
    names, U = [], []
    with open(ROOT / 'data' / 'biosemi64_unit.tsv') as fh:
        rd = csv.reader((l for l in fh if not l.startswith('#')), delimiter='\t')
        next(rd)
        for r in rd:
            names.append(r[0])
            U.append([float(x) for x in r[1:4]])
    U = np.asarray(U)
    return names, U / np.linalg.norm(U, axis=1, keepdims=True)


def load_status(names):
    idx = {n: i for i, n in enumerate(names)}
    recs = []
    with open(ROOT / 'data' / 'ds003775_channel_status.tsv') as fh:
        rd = csv.DictReader((l for l in fh if not l.startswith('#')), delimiter='\t')
        for r in rd:
            bad = [b for b in r['bad_channels'].split(',') if b]
            assert len(bad) == int(r['n_bad'])
            recs.append(dict(rec=r['recording'], sub=r['subject'], ses=r['session'],
                             bad=sorted(idx[b] for b in bad)))
    return recs


def fib_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def mask_of(ids):
    m = np.uint64(0)
    for i in ids:
        m |= np.uint64(1) << np.uint64(i)
    return m


FULL = np.uint64(0xFFFFFFFFFFFFFFFF)


# ------------------------------------------------------------------ geometry
def signatures(P, U, theta_deg):
    """uint64 bitmask per point: which caps of radius theta contain it."""
    c = np.cos(np.radians(theta_deg))
    sig = np.zeros(len(P), dtype=np.uint64)
    for j in range(len(U)):
        sig[(P @ U[j]) > c] |= np.uint64(1) << np.uint64(j)
    return sig


def covering_radii(P, U, kmax=3):
    best = np.full((len(P), kmax), 9.0)
    for j in range(len(U)):
        d = np.arccos(np.clip(P @ U[j], -1, 1))
        st = np.concatenate([best, d[:, None]], 1)
        st.sort(1)
        best = st[:, :kmax]
    return np.degrees(best.max(0))


class Cells:
    """The arrangement of caps restricted to T, as distinct signatures with
    their area fractions; U(S) is then a masked sum."""

    def __init__(self, sig):
        self.sig, cnt = np.unique(sig, return_counts=True)
        self.frac = cnt / cnt.sum()

    def uncovered(self, fail_masks):
        fail_masks = np.atleast_1d(np.asarray(fail_masks, dtype=np.uint64))
        out = np.empty(len(fail_masks))
        for a in range(0, len(fail_masks), 2000):
            fm = fail_masks[a:a + 2000]
            surv = FULL ^ fm
            hit = (self.sig[None, :] & surv[:, None]) == 0
            out[a:a + 2000] = hit @ self.frac
        return out


def random_masks(rng, n, f, m):
    if f == 0:
        return np.zeros(m, dtype=np.uint64)
    keys = rng.random((m, n))
    ids = np.argpartition(keys, f - 1, axis=1)[:, :f]
    bits = (np.uint64(1) << ids.astype(np.uint64))
    return np.bitwise_or.reduce(bits, axis=1)


def adjacency_count(masks, adj_pairs):
    masks = np.atleast_1d(np.asarray(masks, dtype=np.uint64))
    out = np.zeros(len(masks), dtype=np.int64)
    one = np.uint64(1)
    for i, j in adj_pairs:
        bi = (masks >> np.uint64(i)) & one
        bj = (masks >> np.uint64(j)) & one
        out += (bi & bj).astype(np.int64)
    return out


# ------------------------------------------------------------------ nerve
def gdist(p, q):
    return np.arccos(np.clip(np.sum(p * q, axis=-1), -1.0, 1.0))


def min_cap_radius(pts):
    if len(pts) == 2:
        return 0.5 * gdist(pts[0], pts[1])
    a, b, c = pts
    best = np.inf
    for p, q, s in ((a, b, c), (a, c, b), (b, c, a)):
        mid = p + q
        mid = mid / np.linalg.norm(mid)
        rad = 0.5 * gdist(p, q)
        if gdist(mid, s) <= rad + 1e-12:
            best = min(best, rad)
    if best < np.inf:
        return best
    nrm = np.cross(b - a, c - a)
    nn = np.linalg.norm(nrm)
    if nn < 1e-14:
        return max(gdist(a, b), gdist(b, c), gdist(a, c)) / 2
    ctr = nrm / nn
    rad = gdist(ctr, a)
    return min(rad, np.pi - rad)


def full_nerve(U, theta):
    n = len(U)
    faces = set(frozenset([i]) for i in range(n))
    adj = {i: set() for i in range(n)}
    for i, j in itertools.combinations(range(n), 2):
        if gdist(U[i], U[j]) < 2 * theta:
            faces.add(frozenset([i, j]))
            adj[i].add(j)
            adj[j].add(i)
    tri = set()
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k > j and min_cap_radius(U[[i, j, k]]) < theta:
                    tri.add(frozenset([i, j, k]))
    faces |= tri
    N = HellyNerve(faces, tri, adj, None)
    assert N.complete
    return set(N.faces), adj


def sub_nerve(faces, surv):
    s = set(surv)
    return {f for f in faces if f <= s}


def chart_signatures(U, theta_deg, res):
    """Stereographic chart from the south pole; returns grid signatures and
    a mask of pixels whose preimage is on the sphere."""
    L = CHART_HALF
    xs = np.linspace(-L, L, res)
    X, Y = np.meshgrid(xs, xs)
    r2 = X * X + Y * Y
    P = np.stack([2 * X, 2 * Y, 1 - r2], -1) / (1 + r2)[..., None]
    P = P.reshape(-1, 3)
    return signatures(P, U, theta_deg).reshape(res, res)


# ------------------------------------------------------------------ main
def stage_main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    names, U = load_positions()
    n = len(names)
    recs = load_status(names)
    colat = np.degrees(np.arccos(U[:, 2]))

    P = fib_sphere(N_FIB)
    P = P[P[:, 2] >= np.cos(np.radians(T_COLAT))]
    th = covering_radii(P, U, 3)
    print('points in T %d; covering radii %s' % (len(P), np.round(th, 3)), flush=True)

    cohorts = {'t1': [r for r in recs if r['ses'] == 'ses-t1'],
               't2': [r for r in recs if r['ses'] == 'ses-t2']}
    for r in recs:
        r['mask'] = mask_of(r['bad'])
        r['f'] = len(r['bad'])

    out = dict(
        source=dict(dataset='OpenNeuro ds003775 v1.2.1',
                    doi='10.18112/openneuro.ds003775.v1.2.1',
                    descriptor_doi='10.1016/j.dib.2022.108647',
                    recordings=len(recs), t1=len(cohorts['t1']), t2=len(cohorts['t2'])),
        geometry=dict(target_colatitude_deg=T_COLAT, points_in_target=int(len(P)),
                      covering_radii_deg=[float(x) for x in th]),
        plan=dict(theta_primary=THETA_PRIMARY, theta_sensitivity=list(THETA_SENS),
                  seed=SEED, m_pool=M_POOL, m_global=M_GLOBAL, m_perm=M_PERM,
                  primary='one-sided Wilcoxon signed-rank on U_obs - E_null[U|f], '
                          't1, f>=1; Monte Carlo test of sum U; alpha 0.05'),
        runs={})

    for theta in (THETA_PRIMARY,) + THETA_SENS:
        print('== theta %.1f' % theta, flush=True)
        cells = Cells(signatures(P, U, theta))
        thr = np.radians(theta)
        adj_pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)
                     if gdist(U[i], U[j]) < 2 * thr]
        fs = sorted(set(r['f'] for r in recs))
        pool_U, pool_A = {}, {}
        for f in fs:
            mk = random_masks(rng, n, f, M_POOL)
            pool_U[f] = cells.uncovered(mk)
            pool_A[f] = adjacency_count(mk, adj_pairs)
        run = dict(cells=int(len(cells.sig)), nerve_edges=len(adj_pairs),
                   intact_uncovered=float(cells.uncovered([np.uint64(0)])[0]),
                   cohorts={})
        for cname, coh in cohorts.items():
            Uo = cells.uncovered([r['mask'] for r in coh])
            Ao = adjacency_count([r['mask'] for r in coh], adj_pairs)
            EU = np.array([pool_U[r['f']].mean() for r in coh])
            EA = np.array([pool_A[r['f']].mean() for r in coh])
            f_arr = np.array([r['f'] for r in coh])
            sel = f_arr >= 1

            def wil(d):
                d = d[sel]
                nz = d[d != 0]
                if len(nz) == 0:
                    return dict(n=int(sel.sum()), nonzero=0)
                g = stats.wilcoxon(nz, alternative='greater')
                t2 = stats.wilcoxon(nz, alternative='two-sided')
                return dict(n=int(sel.sum()), nonzero=int(len(nz)),
                            statistic=float(g.statistic), p_greater=float(g.pvalue),
                            p_two_sided=float(t2.pvalue),
                            median_diff=float(np.median(d)),
                            positive=int((nz > 0).sum()), negative=int((nz < 0).sum()))

            def mc(obs, pool):
                sums = np.zeros(M_GLOBAL)
                for r in coh:
                    sums += pool[r['f']][rng.integers(0, M_POOL, M_GLOBAL)]
                o = float(obs.sum())
                return dict(observed_sum=o, null_mean=float(sums.mean()),
                            null_sd=float(sums.std()),
                            ratio=float(o / sums.mean()) if sums.mean() > 0 else None,
                            p_greater=float((1 + (sums >= o - 1e-12).sum()) / (1 + M_GLOBAL)))

            per_rec_p = [float((1 + (pool_U[r['f']] >= Uo[i] - 1e-12).sum()) / (1 + M_POOL))
                         for i, r in enumerate(coh)]
            passes90 = np.array([(64 - r['f']) / 64 > 0.90 for r in coh])
            intact = Uo <= 0
            run['cohorts'][cname] = dict(
                recordings=len(coh), with_failures=int(sel.sum()),
                U=dict(wilcoxon=wil(Uo - EU), montecarlo=mc(Uo, pool_U),
                       observed_positive=int((Uo > 0).sum()),
                       expected_positive=float(sum((pool_U[r['f']] > 0).mean() for r in coh)),
                       mean_observed=float(Uo.mean()), mean_null=float(EU.mean())),
                A=dict(wilcoxon=wil((Ao - EA).astype(float)), montecarlo=mc(Ao, pool_A),
                       mean_observed=float(Ao.mean()), mean_null=float(EA.mean())),
                qc=dict(pass90_and_intact=int((passes90 & intact).sum()),
                        pass90_and_lost=int((passes90 & ~intact).sum()),
                        fail90_and_intact=int((~passes90 & intact).sum()),
                        fail90_and_lost=int((~passes90 & ~intact).sum())),
                per_recording=[dict(rec=r['rec'], f=r['f'], U=float(Uo[i]),
                                    EU=float(EU[i]), A=int(Ao[i]), EA=float(EA[i]),
                                    p_U=per_rec_p[i])
                               for i, r in enumerate(coh)])
            print('  %s: U obs %.4f null %.4f  p(W)=%s  p(MC)=%.4g' % (
                cname, Uo.mean(), EU.mean(),
                run['cohorts'][cname]['U']['wilcoxon'].get('p_greater'),
                run['cohorts'][cname]['U']['montecarlo']['p_greater']), flush=True)
        out['runs']['%.1f' % theta] = run

    # ---------------------------------------------------------- S3 heterogeneity
    t1 = cohorts['t1']
    cnt = np.zeros(n)
    for r in t1:
        cnt[r['bad']] += 1
    fs = np.array([r['f'] for r in t1])
    exp = fs.sum() / n

    def chi(c):
        return float(((c - exp) ** 2 / exp).sum())

    obs_chi = chi(cnt)
    null = np.empty(M_PERM)
    for b in range(M_PERM):
        c = np.zeros(n)
        for f in fs:
            if f:
                c[rng.choice(n, f, replace=False)] += 1
        null[b] = chi(c)
    rho = stats.spearmanr(cnt / len(t1), colat)
    order = np.argsort(-cnt)
    out['heterogeneity'] = dict(
        chi2=obs_chi, null_mean=float(null.mean()),
        p=float((1 + (null >= obs_chi).sum()) / (1 + M_PERM)),
        spearman_rate_vs_colatitude=dict(rho=float(rho.statistic), p=float(rho.pvalue)),
        rates={names[i]: float(cnt[i] / len(t1)) for i in range(n)},
        top10=[(names[i], int(cnt[i])) for i in order[:10]])
    print('S3 chi2 %.1f (null %.1f) p=%.4g; spearman rho %.3f p=%.3g' % (
        obs_chi, null.mean(), out['heterogeneity']['p'], rho.statistic, rho.pvalue), flush=True)

    # ---------------------------------------------------------- S4 recurrence
    by_sub = {}
    for r in recs:
        by_sub.setdefault(r['sub'], {})[r['ses']] = set(r['bad'])
    pairs = [(v['ses-t1'], v['ses-t2']) for v in by_sub.values() if 'ses-t2' in v]

    def jac(a, b):
        u = a | b
        return len(a & b) / len(u) if u else np.nan

    obs_j = float(np.nanmean([jac(a, b) for a, b in pairs]))
    A1 = [a for a, _ in pairs]
    A2 = [b for _, b in pairs]
    nullj = np.empty(M_PERM)
    for b in range(M_PERM):
        pi = rng.permutation(len(pairs))
        nullj[b] = np.nanmean([jac(A1[i], A2[pi[i]]) for i in range(len(pairs))])
    out['recurrence'] = dict(subjects=len(pairs), mean_jaccard=obs_j,
                             null_mean=float(nullj.mean()),
                             p=float((1 + (nullj >= obs_j).sum()) / (1 + M_PERM)))
    print('S4 jaccard %.3f null %.3f p=%.4g' % (obs_j, nullj.mean(), out['recurrence']['p']),
          flush=True)

    out['seconds'] = time.time() - t0
    with open(RES / 'eeg_failures_main.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote results/eeg_failures_main.json in %.0f s' % out['seconds'])


def _setup():
    names, U = load_positions()
    recs = load_status(names)
    for r in recs:
        r['mask'] = mask_of(r['bad'])
        r['f'] = len(r['bad'])
    return names, U, recs


def stage_cert():
    """S5: certificate against the exact arrangement and the raster."""
    t0 = time.time()
    names, U, recs = _setup()
    n = len(names)
    colat = np.degrees(np.arccos(U[:, 2]))
    theta = np.radians(THETA_PRIMARY)
    faces, _ = full_nerve(U, theta)
    reach = float(colat.max()) + THETA_PRIMARY
    assert reach < np.degrees(2 * np.arctan(CHART_HALF)), reach
    csig = chart_signatures(U, THETA_PRIMARY, CHART_RES)
    cert, agree, agree_x = [], 0, 0
    for r in recs:
        surv = [i for i in range(n) if i not in set(r['bad'])]
        Ns = sub_nerve(faces, surv)
        mult = np.bitwise_count(csig & (FULL ^ r['mask']))
        O, Rr = caps_to_disks(U[surv], theta)
        ex = region_betti(O, Rr, 2)
        row = dict(rec=r['rec'], f=r['f'])
        ok = okx = True
        for k in (1, 2):
            b0, b1, _ = shadow_betti(Ns, k)
            rb0, rb1 = raster_betti(mult >= k)
            row['k%d' % k] = [b0, b1]
            row['exact_k%d' % k] = list(ex[k])
            row['raster_k%d' % k] = [rb0, rb1]
            ok &= (b0, b1) == (rb0, rb1)
            okx &= (b0, b1) == tuple(ex[k])
        row['agree'] = bool(ok)
        row['agree_exact'] = bool(okx)
        agree += ok
        agree_x += okx
        cert.append(row)
    fsz = {}
    for fc in faces:
        fsz[len(fc)] = fsz.get(len(fc), 0) + 1
    O, Rr = caps_to_disks(U, theta)
    out = dict(theta=THETA_PRIMARY, face_counts={str(k): v for k, v in sorted(fsz.items())},
               intact_betti={str(k): list(shadow_betti(faces, k)[:2]) for k in (1, 2, 3)},
               intact_exact={str(k): list(v) for k, v in region_betti(O, Rr, 3).items()},
               chart_res=CHART_RES, chart_half_width=CHART_HALF, recordings=len(cert),
               agree_with_exact=int(agree_x), agree_with_raster=int(agree), rows=cert,
               seconds=time.time() - t0)
    with open(RES / 'eeg_failures_cert.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('S5 certificate = exact on %d/%d, = raster on %d/%d (%.0f s)'
          % (agree_x, len(cert), agree, len(cert), out['seconds']))


def stage_b1():
    """Holes of Delta_1 after real against random failures, t1 cohort."""
    t0 = time.time()
    names, U, recs = _setup()
    n = len(names)
    rng = np.random.default_rng(SEED + 2)
    faces, _ = full_nerve(U, np.radians(THETA_PRIMARY))
    t1 = [r for r in recs if r['ses'] == 'ses-t1']
    obs, null = [], []
    for r in t1:
        obs.append(shadow_betti(sub_nerve(faces, [i for i in range(n) if i not in set(r['bad'])]), 1)[1])
        vals = []
        for _ in range(M_B1):
            fail = set(rng.choice(n, r['f'], replace=False).tolist()) if r['f'] else set()
            vals.append(shadow_betti(sub_nerve(faces, [i for i in range(n) if i not in fail]), 1)[1])
        null.append(float(np.mean(vals)))
    obs = np.array(obs, float)
    null = np.array(null)
    sel = np.array([r['f'] >= 1 for r in t1])
    d = (obs - null)[sel]
    nz = d[d != 0]
    out = dict(theta=THETA_PRIMARY, draws=M_B1, seed=SEED + 2,
               holes_observed_mean=float(obs.mean()), holes_null_mean=float(null.mean()),
               recordings_with_hole=int((obs > 0).sum()),
               wilcoxon_less=float(stats.wilcoxon(nz, alternative='less').pvalue),
               wilcoxon_greater=float(stats.wilcoxon(nz, alternative='greater').pvalue),
               seconds=time.time() - t0)
    with open(RES / 'eeg_failures_b1.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('holes: observed %.3f, null %.3f, p(less) %.3g (%.0f s)'
          % (obs.mean(), null.mean(), out['wilcoxon_less'], out['seconds']))


def stage_stress(start, stop):
    """Certificate against the exact arrangement on random failure sets."""
    t0 = time.time()
    names, U, recs = _setup()
    n = len(names)
    theta = np.radians(THETA_PRIMARY)
    faces, _ = full_nerve(U, theta)
    rs = np.random.default_rng(SEED + 1)
    draws = [(int(rs.integers(0, 33)), rs.random(n)) for _ in range(stop)]
    agree = 0
    for f, key in draws[start:stop]:
        fail = set(np.argsort(key)[:f].tolist())
        surv = [i for i in range(n) if i not in fail]
        Ns = sub_nerve(faces, surv)
        O, Rr = caps_to_disks(U[surv], theta)
        ex = region_betti(O, Rr, 3)
        agree += all(tuple(shadow_betti(Ns, k)[:2]) == tuple(ex[k]) for k in (1, 2, 3))
    path = RES / ('eeg_failures_stress_%d_%d.json' % (start, stop))
    with open(path, 'w') as fh:
        json.dump(dict(start=start, stop=stop, sets=stop - start, agree=int(agree),
                       levels=[1, 2, 3], seed=SEED + 1), fh)
    print('stress %d-%d: %d/%d agree (%.0f s)' % (start, stop, agree, stop - start, time.time() - t0))


def stage_merge():
    out = json.load(open(RES / 'eeg_failures_main.json'))
    out['certificate'] = json.load(open(RES / 'eeg_failures_cert.json'))
    out['holes'] = json.load(open(RES / 'eeg_failures_b1.json'))
    sets = agree = 0
    for p in sorted(RES.glob('eeg_failures_stress_*.json')):
        d = json.load(open(p))
        sets += d['sets']
        agree += d['agree']
    out['certificate']['random_stress'] = dict(sets=sets, agree=agree, levels=[1, 2, 3])
    with open(RES / 'eeg_failures.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote results/eeg_failures.json')


if __name__ == '__main__':
    import sys
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('main', 'all'):
        stage_main()
    if st in ('cert', 'all'):
        stage_cert()
    if st in ('b1', 'all'):
        stage_b1()
    if st == 'stress':
        stage_stress(int(sys.argv[2]), int(sys.argv[3]))
    elif st == 'all':
        stage_stress(0, 1000)
    if st in ('merge', 'all'):
        stage_merge()
