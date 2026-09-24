"""
fetch_ds003775_status.py -- rebuild data/ds003775_channel_status.tsv and
data/biosemi64_unit.tsv from their public sources.

Source of the channel status
----------------------------
OpenNeuro ds003775, "SRM Resting-state EEG", version 1.2.1,
doi:10.18112/openneuro.ds003775.v1.2.1, licence CC0, described in
Hatlestad-Hall, Rygvold and Andersson, Data in Brief 45:108647 (2022),
doi:10.1016/j.dib.2022.108647.  OpenNeuro mirrors every dataset as a git
repository; tag 1.2.1 of github.com/OpenNeuroDatasets/ds003775 is commit
cb354b249827d17fa0575b2b42521ba25982c16b.  The per-recording files
derivatives/cleaned_epochs/sub-*/ses-*/eeg/*_desc-epochs_channels.tsv are
plain text in that repository (the EEG signals themselves are git-annex
objects and are not needed here).  Each lists the 64 channels of the BioSemi
ActiveTwo cap with status "good" or "bad", as decided by the dataset's own
automated cleaning pipeline (code/bidsify-srm-restingstate/s2_preprocess.m).

Usage:
    python3 fetch_ds003775_status.py                 # clones the mirror
    python3 fetch_ds003775_status.py /path/to/clone  # uses a local clone

As a check that the table is the published data, the script recomputes the
two channel-retention figures quoted in Section 2.3 of the data descriptor:
98 of 153 epoched files (64.1%) keep more than 90% of their channels and 36
(23.5%) keep more than 75% and at most 90%.

Source of the electrode positions
---------------------------------
MNE-Python's standard 'biosemi64' montage: the 64 BioSemi positions of the
10-20 system on a sphere, Cz at the pole and the Fpz-T7-Oz-T8 circumference
at colatitude 92 degrees.  Stored as unit vectors.
"""
import csv
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)

MIRROR = 'https://github.com/OpenNeuroDatasets/ds003775.git'
TAG = '1.2.1'
COMMIT = 'cb354b249827d17fa0575b2b42521ba25982c16b'


def clone(dest):
    subprocess.run(['git', 'clone', '-q', '--depth', '1', '--branch', TAG,
                    MIRROR, str(dest)], check=True)


def main():
    if len(sys.argv) > 1:
        repo = Path(sys.argv[1])
    else:
        repo = ROOT / '_ds003775'
        if not repo.exists():
            clone(repo)
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        print('warning: clone is at %s, expected %s (tag %s)' % (head, COMMIT, TAG))

    files = sorted(glob.glob(str(repo / 'derivatives' / 'cleaned_epochs' / 'sub-*' /
                                 'ses-*' / 'eeg' / '*_desc-epochs_channels.tsv')))
    rows, names = [], None
    for f in files:
        recs = list(csv.DictReader(open(f, newline=''), delimiter='\t'))
        nm = [r['name'].strip() for r in recs]
        st = [r['status'].strip() for r in recs]
        if names is None:
            names = nm
        assert nm == names, f
        assert set(st) <= {'good', 'bad'}, f
        p = Path(f)
        sub = p.parts[-4]
        ses = p.parts[-3]
        bad = [n for n, s in zip(nm, st) if s == 'bad']
        rows.append((sub + '_' + ses, sub, ses, len(bad), ','.join(bad)))

    n = len(rows)
    good = [64 - r[3] for r in rows]
    gt90 = sum(g / 64 > 0.90 for g in good)
    mid = sum(0.75 < g / 64 <= 0.90 for g in good)
    print('recordings %d; >90%% kept: %d (%.1f%%); 75-90%%: %d (%.1f%%)'
          % (n, gt90, 100 * gt90 / n, mid, 100 * mid / n))
    assert (n, gt90, mid) == (153, 98, 36), 'does not match the data descriptor'

    with open(DATA / 'ds003775_channel_status.tsv', 'w', newline='') as fh:
        fh.write('# OpenNeuro ds003775 v%s, doi:10.18112/openneuro.ds003775.v1.2.1, '
                 'CC0; git mirror commit %s\n' % (TAG, head))
        fh.write('# channel order: %s\n' % ','.join(names))
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(['recording', 'subject', 'session', 'n_bad', 'bad_channels'])
        w.writerows(rows)

    import mne
    m = mne.channels.make_standard_montage('biosemi64')
    pos = m.get_positions()['ch_pos']
    assert sorted(pos) == sorted(names)
    with open(DATA / 'biosemi64_unit.tsv', 'w', newline='') as fh:
        fh.write('# MNE-Python %s standard montage biosemi64, unit sphere\n' % mne.__version__)
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(['name', 'x', 'y', 'z'])
        for nme in names:
            u = np.asarray(pos[nme], float)
            u = u / np.linalg.norm(u)
            w.writerow([nme] + ['%.12f' % c for c in u])
    print('wrote data/ds003775_channel_status.tsv and data/biosemi64_unit.tsv')


if __name__ == '__main__':
    main()
