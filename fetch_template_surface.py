"""
fetch_template_surface.py -- extract the left pial surface of the FieldTrip
template anatomy (the colin27 single-subject brain, FreeSurfer reconstruction,
MNI coordinates in mm) from a checkout of the FieldTrip repository and store
it as data/colin27_lh_pial.npz for run_cortex.py.

The source file is template/anatomy/surface_pial_left.mat of
https://github.com/fieldtrip/fieldtrip (commit recorded in the npz).

Usage: python3 fetch_template_surface.py /path/to/fieldtrip [commit]
"""
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio

root = Path(sys.argv[1])
commit = sys.argv[2] if len(sys.argv) > 2 else 'unknown'
m = sio.loadmat(root / 'template' / 'anatomy' / 'surface_pial_left.mat')['mesh']
pos = np.asarray(m['pos'][0, 0], float)
tri = np.asarray(m['tri'][0, 0], np.int64) - 1          # MATLAB is 1-based
sulc = np.asarray(m['sulc'][0, 0], float).ravel()       # FreeSurfer sulcal depth (positive in sulci)
curv = np.asarray(m['curv'][0, 0], float).ravel()
unit = str(m['unit'][0, 0][0])
coordsys = str(m['coordsys'][0, 0][0])
assert unit == 'mm' and tri.min() == 0 and tri.max() == len(pos) - 1
out = Path(__file__).resolve().parent / 'data'
out.mkdir(exist_ok=True)
np.savez_compressed(out / 'colin27_lh_pial.npz', pos=pos, tri=tri, sulc=sulc, curv=curv,
                    unit=unit, coordsys=coordsys, source_commit=commit,
                    source='fieldtrip/template/anatomy/surface_pial_left.mat')
print('vertices', len(pos), 'triangles', len(tri), unit, coordsys, commit)
