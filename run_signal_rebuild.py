"""
run_signal_rebuild.py -- does local redundancy predict how well a dropped
channel can be rebuilt from its neighbours?

Runs on your own machine; the EEG signals of OpenNeuro ds003775 could not be
reached from the environment in which the rest of the analysis was done.
On Windows use run_signal_rebuild.ps1, which sets everything up.

Data
----
The curated, cleaned epochs of ds003775 v1.2.1 (1-45 Hz band-pass, average
reference, 60 epochs of 4 s at 1024 Hz per recording), one EEGLAB .set file
per recording, about 115 MB each: 12.7 GB for the 111 first sessions, 17.4
GB for all 153 recordings.  Files are fetched from OpenNeuro's public
storage and checked against the size and MD5 recorded for version 1.2.1 in
data/ds003775_epochs_manifest.tsv; a file that does not match is not used.
Channels the curators marked bad were interpolated by their pipeline, so
they are used neither as targets nor as inputs.

What is measured, per recording and per good channel c
------------------------------------------------------
Rebuild.  c is removed and its signal is predicted from all other good
channels by spherical-spline interpolation (Perrin et al. 1989; stiffness 4,
50 Legendre terms, regularisation 1e-5, the MNE-Python defaults), on the
same BioSemi positions used everywhere else in the paper.  Quality is the
Pearson correlation r between true and rebuilt signal over all 240 s, used
through Fisher's z = artanh r, and the normalised error ||x - xhat||/||x||.

Redundancy at the primary radius theta* = 23.5 deg, from the surviving set
S = good channels minus c:
  lambda  redundancy depth: the mean, over the cap of c, of the number of
          caps of S covering the point (the multiplicity function of the
          k-shadow filtration, averaged over the footprint that is lost);
  pi      private territory: the fraction of the cap of c covered by no cap
          of S;
  mu      the number of channels of S within theta* of c;
  d       the angular distance from c to the nearest channel of S (the
          usual predictor of interpolation error).

Plan, fixed before any signal was loaded
----------------------------------------
P   (primary) A natural experiment inside each channel.  A channel's own
    position never changes; what changes between recordings is which of
    its neighbours failed.  For every channel that is a good target in at
    least 10 first-session recordings with at least 3 distinct values of
    lambda, the Spearman correlation between lambda and z across those
    recordings; one-sided Wilcoxon signed-rank test that these correlations
    are positive, over channels.  alpha = 0.05.
S1  The same with pi, one-sided in the negative direction.
S2  Beyond distance: in each first-session recording, the partial Spearman
    correlation of z with lambda given d across channels; one-sided
    Wilcoxon over recordings.
S3  P repeated on the second sessions.

Usage
-----
    python run_signal_rebuild.py                    # first sessions, download as needed
    python run_signal_rebuild.py --sessions t1,t2   # all 153 recordings
    python run_signal_rebuild.py --max-recordings 20
    python run_signal_rebuild.py --data-dir D:/ds003775   # an existing copy
    python run_signal_rebuild.py --delete-after     # keep disk use to one file
    python run_signal_rebuild.py --stats-only       # recompute the tests

Per-channel rows go to results/signal_rebuild_rows.csv as each recording is
finished, so an interrupted run resumes where it stopped; the tests go to
results/signal_rebuild.json.  Send that JSON file back.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legval
from scipy import stats

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)
ROWS = RES / 'signal_rebuild_rows.csv'
BUCKET = 'https://s3.amazonaws.com/openneuro.org/ds003775/'
THETA = 23.5
N_CAP_SAMPLE = 400_000
FIELDS = ['recording', 'session', 'channel', 'n_good', 'r', 'z', 'nrmse',
          'lambda', 'pi', 'mu', 'd']


# ------------------------------------------------------------------ tables
def read_tsv(path):
    with open(path, newline='') as fh:
        lines = [l for l in fh if not l.startswith('#')]
    return list(csv.DictReader(lines, delimiter='\t'))


def load_tables():
    pos = read_tsv(ROOT / 'data' / 'biosemi64_unit.tsv')
    names = [p['name'] for p in pos]
    U = np.array([[float(p['x']), float(p['y']), float(p['z'])] for p in pos])
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    status = {}
    for r in read_tsv(ROOT / 'data' / 'ds003775_channel_status.tsv'):
        status[r['recording']] = dict(session=r['session'],
                                      bad=[b for b in r['bad_channels'].split(',') if b])
    manifest = {}
    for m in read_tsv(ROOT / 'data' / 'ds003775_epochs_manifest.tsv'):
        rec = Path(m['path']).name.split('_task-')[0]
        manifest[rec] = dict(path=m['path'], size=int(m['size']), md5=m['md5'])
    return names, U, status, manifest


# ------------------------------------------------------------------ files
def md5sum(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry, data_dir, verify=True, tries=3):
    dest = Path(data_dir) / entry['path']
    if dest.exists():
        if not verify:
            return dest
        if dest.stat().st_size == entry['size'] and md5sum(dest) == entry['md5']:
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.part')
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(BUCKET + entry['path'], timeout=120) as resp, \
                    open(tmp, 'wb') as out:
                while True:
                    block = resp.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
            if tmp.stat().st_size != entry['size']:
                raise IOError('size %d, expected %d' % (tmp.stat().st_size, entry['size']))
            if verify and md5sum(tmp) != entry['md5']:
                raise IOError('MD5 does not match version 1.2.1')
            tmp.replace(dest)
            return dest
        except Exception as exc:
            print('    download attempt %d failed: %s' % (attempt + 1, exc), flush=True)
            time.sleep(5)
    return None


def load_signals(path, names):
    import mne
    mne.set_log_level('ERROR')
    ep = mne.read_epochs_eeglab(str(path))
    have = {c.lower(): c for c in ep.ch_names}
    missing = [n for n in names if n.lower() not in have]
    if missing:
        raise ValueError('channels missing from file: %s' % missing)
    ep = ep.pick([have[n.lower()] for n in names])
    X = ep.get_data()                                  # epochs x 64 x time
    return np.concatenate(list(X), axis=1)             # 64 x (epochs*time)


# ------------------------------------------------------------------ spline
def calc_g(cosang, stiffness=4, n_terms=50):
    f = [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness * 4 * np.pi)
         for n in range(1, n_terms + 1)]
    return legval(cosang, [0] + f)


def spline_weights(pos_from, pos_to, alpha=1e-5):
    """Spherical-spline interpolation matrix, as MNE-Python computes it."""
    G_from = calc_g(np.clip(pos_from @ pos_from.T, -1, 1))
    G_to = calc_g(np.clip(pos_to @ pos_from.T, -1, 1))
    G_from.flat[::len(G_from) + 1] += alpha
    n = len(pos_from)
    C = np.block([[G_from, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    Ci = np.linalg.pinv(C)
    return np.hstack([G_to, np.ones((len(pos_to), 1))]) @ Ci[:, :-1]


# ------------------------------------------------------------------ geometry
class Caps:
    def __init__(self, U, theta_deg, n=N_CAP_SAMPLE):
        i = np.arange(n) + 0.5
        z = 1 - 2 * i / n
        phi = i * np.pi * (3 - np.sqrt(5))
        r = np.sqrt(1 - z * z)
        P = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
        c = np.cos(np.radians(theta_deg))
        inside = (P @ U.T) > c                           # points x 64
        keep = inside.any(1)
        self.inside = inside[keep]
        self.U = U
        self.theta = np.radians(theta_deg)
        self.idx = [np.flatnonzero(self.inside[:, j]) for j in range(len(U))]

    def measures(self, c, surv):
        pts = self.idx[c]
        cnt = self.inside[np.ix_(pts, surv)].sum(1)
        ang = np.arccos(np.clip(self.U[surv] @ self.U[c], -1, 1))
        return dict(**{'lambda': float(cnt.mean())}, pi=float((cnt == 0).mean()),
                    mu=int((ang < self.theta).sum()), d=float(np.degrees(ang.min())))


# ------------------------------------------------------------------ analysis
def rebuild_recording(X, names, U, bad, caps):
    good = [i for i, n in enumerate(names) if n not in set(bad)]
    rows = []
    for c in good:
        surv = [i for i in good if i != c]
        w = spline_weights(U[surv], U[[c]])[0]
        x = X[c]
        xh = w @ X[surv]
        r = float(np.corrcoef(x, xh)[0, 1])
        rows.append(dict(channel=names[c], n_good=len(good), r=r,
                         z=float(np.arctanh(min(r, 0.999999))),
                         nrmse=float(np.linalg.norm(x - xh) / np.linalg.norm(x)),
                         **caps.measures(c, surv)))
    return rows


def partial_spearman(y, x, ctrl):
    ry, rx, rc = (stats.rankdata(v) for v in (y, x, ctrl))
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(stats.pearsonr(resid(ry, rc), resid(rx, rc))[0])


def within_channel(rows, session, var, alternative):
    by = {}
    for r in rows:
        if r['session'] == session:
            by.setdefault(r['channel'], []).append(r)
    rhos = {}
    for ch, rr in by.items():
        v = np.array([float(r[var]) for r in rr])
        z = np.array([float(r['z']) for r in rr])
        if len(rr) >= 10 and len(np.unique(v)) >= 3:
            rhos[ch] = float(stats.spearmanr(v, z)[0])
    vals = np.array([x for x in rhos.values() if np.isfinite(x)])
    out = dict(channels=len(vals), median_rho=float(np.median(vals)) if len(vals) else None,
               positive=int((vals > 0).sum()), negative=int((vals < 0).sum()), rho=rhos)
    if len(vals) >= 5:
        out['p'] = float(stats.wilcoxon(vals, alternative=alternative).pvalue)
    return out


def run_stats(rows):
    out = dict(rows=len(rows),
               recordings=len({r['recording'] for r in rows}),
               plan='P: within-channel Spearman(lambda, z), one-sided Wilcoxon > 0 over '
                    'channels, first sessions, alpha 0.05')
    out['P_lambda_t1'] = within_channel(rows, 'ses-t1', 'lambda', 'greater')
    out['S1_pi_t1'] = within_channel(rows, 'ses-t1', 'pi', 'less')
    out['S3_lambda_t2'] = within_channel(rows, 'ses-t2', 'lambda', 'greater')
    parts = []
    by_rec = {}
    for r in rows:
        if r['session'] == 'ses-t1':
            by_rec.setdefault(r['recording'], []).append(r)
    for rec, rr in by_rec.items():
        y = np.array([float(r['z']) for r in rr])
        x = np.array([float(r['lambda']) for r in rr])
        d = np.array([float(r['d']) for r in rr])
        if len(rr) >= 20 and len(np.unique(x)) >= 3:
            parts.append(partial_spearman(y, x, d))
    parts = np.array(parts)
    out['S2_lambda_given_d_t1'] = dict(
        recordings=len(parts), median_partial_rho=float(np.median(parts)) if len(parts) else None,
        positive=int((parts > 0).sum()), negative=int((parts < 0).sum()),
        p=float(stats.wilcoxon(parts, alternative='greater').pvalue) if len(parts) >= 5 else None)
    allz = np.array([float(r['z']) for r in rows])
    out['overall'] = dict(median_r=float(np.median(np.tanh(allz))),
                          spearman_lambda_z=float(stats.spearmanr(
                              [float(r['lambda']) for r in rows], allz)[0]),
                          spearman_d_z=float(stats.spearmanr(
                              [float(r['d']) for r in rows], allz)[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', default='t1')
    ap.add_argument('--max-recordings', type=int, default=0)
    ap.add_argument('--data-dir', default=str(ROOT / '_ds003775_epochs'))
    ap.add_argument('--delete-after', action='store_true')
    ap.add_argument('--no-verify', action='store_true')
    ap.add_argument('--stats-only', action='store_true')
    a = ap.parse_args()

    names, U, status, manifest = load_tables()
    done = set()
    rows = []
    if ROWS.exists():
        with open(ROWS, newline='') as fh:
            rows = list(csv.DictReader(fh))
        done = {r['recording'] for r in rows}
    if not a.stats_only:
        sess = {'ses-' + s.strip() for s in a.sessions.split(',')}
        todo = [rec for rec in sorted(status) if status[rec]['session'] in sess and rec not in done]
        if a.max_recordings:
            todo = todo[:max(0, a.max_recordings - len(done))]
        gb = sum(manifest[r]['size'] for r in todo) / 1e9
        print('%d recordings to process, %.1f GB to fetch at most' % (len(todo), gb), flush=True)
        caps = Caps(U, THETA)
        new = not ROWS.exists()
        with open(ROWS, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for k, rec in enumerate(todo):
                t0 = time.time()
                print('[%d/%d] %s' % (k + 1, len(todo), rec), flush=True)
                path = fetch(manifest[rec], a.data_dir, verify=not a.no_verify)
                if path is None:
                    print('    skipped: could not obtain a verified file', flush=True)
                    continue
                try:
                    X = load_signals(path, names)
                except Exception as exc:
                    print('    skipped: %s' % exc, flush=True)
                    continue
                out = rebuild_recording(X, names, U, status[rec]['bad'], caps)
                for r in out:
                    r.update(recording=rec, session=status[rec]['session'])
                    w.writerow({f: r[f] for f in FIELDS})
                fh.flush()
                rows.extend({f: r[f] for f in FIELDS} for r in out)
                if a.delete_after:
                    path.unlink()
                print('    %d channels, median r %.3f (%.0f s)' % (
                    len(out), np.median([r['r'] for r in out]), time.time() - t0), flush=True)
    res = run_stats(rows)
    res['environment'] = dict(python=sys.version.split()[0], numpy=np.__version__)
    try:
        import mne
        res['environment']['mne'] = mne.__version__
    except ImportError:
        pass
    with open(RES / 'signal_rebuild.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    P = res['P_lambda_t1']
    print('primary: %d channels, median rho %s, p = %s' % (
        P['channels'], P.get('median_rho'), P.get('p')))
    print('wrote results/signal_rebuild.json')


if __name__ == '__main__':
    main()
