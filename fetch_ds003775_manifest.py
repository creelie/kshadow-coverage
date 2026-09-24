"""
fetch_ds003775_manifest.py -- rebuild data/ds003775_epochs_manifest.tsv, the
list of cleaned-epoch files of OpenNeuro ds003775 v1.2.1 that
run_signal_rebuild.py downloads.

For each derivatives/cleaned_epochs/.../*_desc-epochs_eeg.set file at tag
1.2.1 of the public git mirror (github.com/OpenNeuroDatasets/ds003775,
commit cb354b249827d17fa0575b2b42521ba25982c16b) the table records

    path        the file's path in the dataset
    size, md5   from its git-annex key (MD5E-s<size>--<md5>.set)
    version_id  the S3 object version that holds exactly this content, from
                the key's .log.rmet entry on the mirror's git-annex branch

The version id is needed because later versions of the dataset dropped the
derivatives folder, so the plain URL
https://s3.amazonaws.com/openneuro.org/ds003775/<path> now returns 404, while
the same URL with ?versionId=<version_id> still serves the v1.2.1 file.

Usage:
    python3 fetch_ds003775_manifest.py                 # clones the mirror
    python3 fetch_ds003775_manifest.py /path/to/clone  # a clone with its
                                                       # git-annex branch
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'ds003775_epochs_manifest.tsv'
MIRROR = 'https://github.com/OpenNeuroDatasets/ds003775.git'
TAG = '1.2.1'
COMMIT = 'cb354b249827d17fa0575b2b42521ba25982c16b'
KEY = re.compile(r'MD5E-s(\d+)--([0-9a-f]{32})\.set$')


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo)] + list(args), check=True,
                          capture_output=True, text=True).stdout


def main():
    if len(sys.argv) > 1:
        repo = Path(sys.argv[1])
    else:
        repo = ROOT / '_ds003775_meta'
        if not repo.exists():
            subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout',
                            MIRROR, str(repo)], check=True)
    git(repo, 'fetch', '-q', 'origin', 'git-annex:git-annex', 'tag', TAG)
    commit = git(repo, 'rev-parse', TAG + '^{commit}').strip()
    if commit != COMMIT:
        print('warning: tag %s is %s, expected %s' % (TAG, commit, COMMIT))

    annex = {}
    for name in git(repo, 'ls-tree', '-r', '--name-only', 'git-annex').split():
        if name.endswith('.set.log.rmet'):
            annex[Path(name).name[:-len('.log.rmet')]] = name

    rows = []
    listing = git(repo, 'ls-tree', '-r', TAG, 'derivatives/cleaned_epochs')
    for line in listing.splitlines():
        meta, path = line.split('\t')
        if not path.endswith('_desc-epochs_eeg.set'):
            continue
        target = git(repo, 'cat-file', '-p', meta.split()[2]).strip()
        key = Path(target).name
        m = KEY.match(key)
        assert m, (path, key)
        rmet = git(repo, 'show', 'git-annex:' + annex[key])
        vids = set()
        for entry in rmet.splitlines():
            v = re.search(r':V \+([^#]+)#(.+)$', entry)
            if v and v.group(2) == 'ds003775/' + path:
                vids.add(v.group(1))
        assert len(vids) == 1, (path, vids)
        rows.append((path, m.group(1), m.group(2), vids.pop()))

    rows.sort()
    with open(OUT, 'w', newline='') as fh:
        fh.write('# OpenNeuro ds003775 v1.2.1 cleaned epochs: size in bytes and MD5 from the '
                 'git-annex keys of mirror commit %s; version_id is the S3 object version '
                 'recorded for each key on the git-annex branch\n' % COMMIT)
        fh.write('path\tsize\tmd5\tversion_id\n')
        for r in rows:
            fh.write('\t'.join(r) + '\n')
    print('%d files, %.1f GB, written to %s' % (
        len(rows), sum(int(r[1]) for r in rows) / 1e9, OUT))


if __name__ == '__main__':
    main()
