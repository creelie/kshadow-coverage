"""
run_montages.py -- the k-shadow certificate for the electrode layouts of three
public EEG studies, on the same template cortical surface as run_cortex.py.

What this does and does not claim
---------------------------------
The certificate of this paper is geometric: it reads the combinatorics of a
family of footprints on a surface.  For subdural electrocorticography a
footprint is a physical contact area on the pia and the geodesic-disk model
is the natural one.  Scalp EEG is different.  A scalp electrode does not have
a sharp footprint on the cortex at all; what it records is a volume-conducted
mixture whose cortical sensitivity is broad and overlapping.  Nothing below
is a claim about electrical sensitivity or about source localisation.

What is computed is the geometry of the montage: each electrode is projected
to its nearest point on the pial surface, given the same geodesic-disk
footprint of radius r as in run_cortex.py, and the resulting cover is put
through the same machinery.  The statement that results is of the form "at
footprint radius r, the projected layout of this montage covers the left
lateral cortex k times over, with these Betti numbers", which is a statement
about where the electrodes are, not about what they measure.  Read that way
the radius r is a stand-in for the spatial reach one is willing to credit an
electrode with, and the sweep over r is the useful output.

Montages
--------
stroke_mi_30    Liu et al., Sci. Data 11:131 (2024), doi:10.1038/s41597-023-02787-8
                50 acute stroke patients, left- and right-hand motor imagery,
                30 EEG channels on the international 10-10 system, the channel
                list of their Table 2.  Distributed on figshare, 21679035.
srm_rest_64     Hatlestad-Hall et al., Data in Brief 45:108647 (2022),
                doi:10.1016/j.dib.2022.108647.  111 healthy subjects, eyes
                closed resting state, BioSemi ActiveTwo 64 channels at 10-20
                positions.  OpenNeuro ds003775.
eegmmidb_64     Schalk et al., IEEE Trans. Biomed. Eng. 51:1034 (2004),
                doi:10.1109/TBME.2004.827072; PhysioNet eegmmidb 1.0.0.
                109 healthy subjects, motor movement and imagery, 64 channels
                on the international 10-10 system.

Electrode coordinates come from MNE-Python's colin27_1005 montage, which is
defined on the colin27 head, the same head whose pial surface this paper
already uses, so no inter-subject warp is involved.

Only the left hemisphere is analysed, because the surface in data/ is the
left pial surface; midline electrodes are kept and right-hemisphere ones are
dropped, and the counts below say how many of each montage survive.

Writes results/montages.json.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from kshadow import nerve_from_sets, shadow_betti, dropout_margin

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- the mesh
D = np.load(ROOT / 'data' / 'colin27_lh_pial.npz')
POS = D['pos'].astype(float)
TRI = D['tri'].astype(np.int64)
NV = len(POS)
E = np.vstack([TRI[:, [0, 1]], TRI[:, [1, 2]], TRI[:, [2, 0]]])
E = np.unique(np.sort(E, axis=1), axis=0)
NE = len(E)
LEN = np.linalg.norm(POS[E[:, 0]] - POS[E[:, 1]], axis=1)
G = csr_matrix((np.r_[LEN, LEN],
                (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(NV, NV))

_key = E[:, 0] * NV + E[:, 1]
_order = np.argsort(_key)
_keys = _key[_order]


def edge_ids(a, b):
    k = np.minimum(a, b) * NV + np.maximum(a, b)
    return _order[np.searchsorted(_keys, k)]


TE = np.stack([edge_ids(TRI[:, 0], TRI[:, 1]),
               edge_ids(TRI[:, 1], TRI[:, 2]),
               edge_ids(TRI[:, 2], TRI[:, 0])], axis=1)


def subcomplex_topology(vm, em, fm):
    """Components, Euler characteristic and b1 over GF(2) of the subcomplex
    carried by the given vertex, edge and triangle masks."""
    v = int(vm.sum())
    if v == 0:
        return dict(components=0, chi=0, closed=0, b1=0, vertices=0)
    e = int(em.sum())
    f = int(fm.sum())
    loc = np.full(NV, -1)
    loc[np.flatnonzero(vm)] = np.arange(v)
    sub = E[em]
    A = coo_matrix((np.ones(len(sub)), (loc[sub[:, 0]], loc[sub[:, 1]])), shape=(v, v))
    c, lab = connected_components(A, directed=False)
    closed = 0
    if f:
        cnt = np.bincount(TE[fm].ravel(), minlength=NE)
        bnd = np.flatnonzero(cnt == 1)
        has_face = np.zeros(c, bool)
        has_face[lab[loc[TRI[fm][:, 0]]]] = True
        has_bnd = np.zeros(c, bool)
        if len(bnd):
            has_bnd[lab[loc[E[bnd][:, 0]]]] = True
        closed = int((has_face & ~has_bnd).sum())
    chi = v - e + f
    return dict(components=int(c), chi=int(chi), closed=closed,
                b1=int(c - chi + closed), vertices=v)


# ---------------------------------------------------------------- montages
STROKE_30 = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4',
             'FT7', 'FT8', 'Cz', 'C3', 'C4', 'T7', 'T8', 'CPz', 'CP3', 'CP4',
             'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P7', 'P8', 'Oz', 'O1', 'O2']

EEGMMIDB_64 = [
    'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz',
    'C2', 'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1',
    'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1',
    'Fz', 'F2', 'F4', 'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10',
    'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2', 'Iz']

SRM_64 = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5',
    'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz',
    'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6',
    'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4',
    'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

MONTAGES = {
    'stroke_mi_30': dict(names=STROKE_30, n_nominal=30,
                         study='Liu et al. 2024, figshare 21679035'),
    'srm_rest_64': dict(names=SRM_64, n_nominal=64,
                        study='Hatlestad-Hall et al. 2022, OpenNeuro ds003775'),
    'eegmmidb_64': dict(names=EEGMMIDB_64, n_nominal=64,
                        study='Schalk et al. 2004, PhysioNet eegmmidb 1.0.0'),
}

# a scalp electrode sits well outside the pia; the sweep is over the cortical
# reach one is prepared to credit it with, not over a measured contact area
RADII = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
KMAX = 6
LEFT_X = 5.0          # keep midline electrodes, drop the right hemisphere


def montage_positions():
    import mne
    m = mne.channels.make_standard_montage('colin27_1005')
    p = m.get_positions()
    assert p['coord_frame'] == 'mri', p['coord_frame']
    return {k.lower(): np.asarray(v) * 1000.0 for k, v in p['ch_pos'].items()}


def main():
    ch = montage_positions()
    out = {
        'surface': dict(source=str(D['source']), commit=str(D['source_commit']),
                        vertices=NV, coordsys=str(D['coordsys']), unit=str(D['unit'])),
        'electrode_frame': 'MNE colin27_1005, colin27 MRI space, mm',
        'radii_mm': RADII, 'kmax': KMAX, 'left_x_cutoff_mm': LEFT_X,
        'montages': {},
    }
    for mname, spec in MONTAGES.items():
        print('==', mname, flush=True)
        missing = [n for n in spec['names'] if n.lower() not in ch]
        kept, pos = [], []
        for n in spec['names']:
            if n.lower() not in ch:
                continue
            xyz = ch[n.lower()]
            if xyz[0] > LEFT_X:
                continue
            kept.append(n)
            pos.append(xyz)
        pos = np.asarray(pos)
        # project each electrode to its nearest pial vertex
        vtx, standoff = [], []
        for x in pos:
            d = np.linalg.norm(POS - x, axis=1)
            j = int(np.argmin(d))
            vtx.append(j)
            standoff.append(float(d[j]))
        vtx = np.asarray(vtx)
        uniq = len(set(vtx.tolist()))
        dist = dijkstra(G, directed=False, indices=vtx, limit=max(RADII) + 1e-9)

        entry = dict(study=spec['study'], nominal_channels=spec['n_nominal'],
                     missing_from_montage=missing,
                     kept_left=len(kept), kept_names=kept,
                     distinct_vertices=uniq,
                     standoff_mm=dict(median=float(np.median(standoff)),
                                      min=float(np.min(standoff)),
                                      max=float(np.max(standoff))),
                     radii={})
        for r in RADII:
            within = dist <= r + 1e-9
            # nerve_from_sets takes bitsets, one Python int per footprint
            sets = [int.from_bytes(np.packbits(within[i], bitorder='little').tobytes(),
                                   'little') for i in range(len(vtx))]
            N = nerve_from_sets(sets, max_size=KMAX + 1)
            mult = within.sum(axis=0)
            per_k = {}
            for k in range(1, KMAX + 1):
                vm = mult >= k
                em = vm[E[:, 0]] & vm[E[:, 1]]
                fm = vm[TRI[:, 0]] & vm[TRI[:, 1]] & vm[TRI[:, 2]]
                mesh = subcomplex_topology(vm, em, fm)
                sb = shadow_betti(N, k)
                per_k[str(k)] = dict(mesh=mesh, shadow_b0=int(sb[0]), shadow_b1=int(sb[1]),
                                     faces_k=int(sum(1 for s in N if len(s) == k)))
            kmaxcov = max([k for k in range(1, KMAX + 1)
                           if per_k[str(k)]['mesh']['vertices'] > 0] or [0])
            entry['radii'][str(r)] = dict(
                nerve_faces=len(N),
                covered_vertices=int((mult >= 1).sum()),
                covered_fraction=float((mult >= 1).sum() / NV),
                max_multiplicity=int(mult.max()),
                largest_k_with_cover=kmaxcov,
                dropout_margin=int(dropout_margin(N, len(vtx), kmax=KMAX)),
                k=per_k)
            print('   r=%4.1f  covered %.3f  maxmult %2d  b0/b1 at k=1: %d/%d'
                  % (r, (mult >= 1).sum() / NV, mult.max(),
                     per_k['1']['mesh']['components'], per_k['1']['mesh']['b1']),
                  flush=True)
        out['montages'][mname] = entry

    (RES / 'montages.json').write_text(json.dumps(out, indent=1))
    print('wrote results/montages.json')


if __name__ == '__main__':
    t = time.time()
    main()
    print('%.1f s' % (time.time() - t))
