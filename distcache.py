"""
distcache.py -- the geodesic distance matrices of run_cortex.py, in whichever
form is on disk.

One Dijkstra run per contact over a 175409-vertex mesh gives a matrix of
shape (contacts, vertices).  Held as raw float32 the two matrices come to
65 MB, which is awkward in a git repository; almost every entry is +inf,
because Dijkstra is run with a distance limit and only a few thousand
vertices lie within it, so the same data compresses to about 0.4 MB.

save() writes the compressed form and, unless asked not to, the raw form
beside it.  load() takes either, preferring the raw one when both exist.
Nothing else in the package needs to know which is present.
"""
from pathlib import Path
import numpy as np


def _stem(results_dir, name):
    return Path(results_dir) / ('cortex_dist_%s' % name)


def save(results_dir, name, dist, raw=True):
    """Store a distance matrix under results/cortex_dist_<name>.{npz,npy}."""
    s = _stem(results_dir, name)
    a = np.asarray(dist, dtype=np.float32)
    np.savez_compressed(str(s) + '.npz', dist=a)
    if raw:
        np.save(str(s) + '.npy', a)
    return s


def load(results_dir, name, dtype=float):
    """Read it back, from the raw file if present and the compressed one
    otherwise.  Raises FileNotFoundError naming the script that writes it."""
    s = _stem(results_dir, name)
    npy, npz = Path(str(s) + '.npy'), Path(str(s) + '.npz')
    if npy.exists():
        return np.load(npy).astype(dtype)
    if npz.exists():
        with np.load(npz) as z:
            return z['dist'].astype(dtype)
    raise FileNotFoundError(
        '%s not found as .npy or .npz; run run_cortex.py to build it' % s)
