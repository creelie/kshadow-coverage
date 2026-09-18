# k-shadow coverage certificates for cortical electrode arrays

Code and cached results for **Topological Certificates of Redundant Coverage
for Cortical Electrode Arrays**, by Aditi Bose, Deep Bhattacharjee and
Ushashi Bhattacharya.

An electrode array should keep a region of cortex covered after individual
contacts fail, so what decides its quality is not covered area but the shape
of the region seen by at least *k* contacts at once. This repository computes
that shape exactly, from intersection data alone, and then uses the same
machinery to design arrays that leave nothing out.

## What is in here

The theory, in `kshadow.py`:

- exact nerve of a family of equal disks, by a Helly membership oracle;
- the *k*-shadow complex Δ_k(N) and the subdivision complex Sub_k(N), with
  Betti numbers over GF(2);
- the redundancy barcode, one persistence diagram for the whole filtration
  R_1 ⊇ R_2 ⊇ ... ;
- the dropout margin.

The experiments, one script each, all writing JSON into `results/`: five
planar sensor fields, regular grids with radius and pitch sweeps, anisotropic
footprints, spherical caps with exact spherical geometry, a real cortical
surface, private territory and contact failure, the two stability bounds,
three published scalp montages, and the design search.

## Headline numbers

On a 3777 mm² patch of the colin27 parietal surface, at an 8 mm geodesic
footprint radius:

| placement | patch unseen | (b₀, b₁) of Δ₁(N) | certified level |
|---|---|---|---|
| documented 8×8 grid, 64 contacts | 2082.0 mm² (55.1%) | (9, 2) | 0 |
| 64 contacts placed by covering radius | 1.0 mm² (0.03%) | (1, 1) | 0 |
| 66 contacts placed by covering radius | 0.0 mm² | (1, 0) | 1 |

Same target, same footprint radius, nearly the same contact count. What
separates them is where the contacts sit.

Contacts confined to gyral crowns cannot reach this at any count. Take every
one of the 3016 gyral vertices of the patch as a contact and some point of
cortex is still 12.219 mm from all of them, so no array on the envelope of
the hemisphere with a smaller footprint covers the patch at all.

112 contacts at a 9 mm radius see every point of the patch twice, with
Δ₁(N) and Δ₂(N) both (1, 0): certified level 2, dropout margin 2.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD

python3 run_fields.py          # the five planar fields
python3 run_grid.py            # 8x8 grid, radius and pitch sweeps
python3 make_merged_figures.py # the composite figures of the paper
```

`results/` ships populated, so any figure script runs immediately without
rerunning its experiment. `figures/` carries the nine figures of the paper
and the six surface renders that need PyVista; every other panel is redrawn
from cached results in seconds by `make_field_figures.py`, `make_figures.py`,
`make_cortex_figures.py`, `make_dropout_figure.py` and
`make_stability_figure.py`, so those are left out of the repository rather
than stored. `SCRIPTS.md` maps every script to the results and
figures it produces, and gives the full reproduce sequence in order.

Only four scripts need anything beyond NumPy, SciPy, NetworkX and
Matplotlib: `make_cortex_figures_pv.py`, `make_sphere3d_pv.py` and
`make_optimal_figure.py` render off screen with PyVista, and
`run_montages.py` reads standard montages from MNE-Python.

## Layout

```
kshadow.py              the library
distcache.py            geodesic distance caches, raw or compressed
fields_v2.py            the five planar sensor fields, from their seeds
shading.py              shared camera and lighting for the surface renders
run_*.py                experiments, each writing results/*.json
make_*.py               figure generators
data/                   the template cortical surface, with its source commit
results/                the JSON the paper quotes, and the distance caches
figures/                the paper's nine figures, and the six off-screen
                        renders that need PyVista to rebuild
legacy_v2/              the earlier brute-force pipeline, kept for provenance
SCRIPTS.md              the full script map and reproduce sequence
upload_kshadow.sh       one-shot push of this folder to GitHub, then a tag
```

## The cortical surface

`data/colin27_lh_pial.npz` is the left pial surface of the colin27
single-subject template anatomy, extracted from the FieldTrip toolbox at
commit `8e2307d7e7284c6870a5d12e244d9dc95a1faae3` and stored as a derived
NumPy array: 175409 vertices, 350814 triangles, 526221 edges, every edge in
exactly two triangles, Euler characteristic 2. It is a de-identified,
long-published reference brain, not patient data, and no patient-level data
of any kind is used anywhere in this work. `fetch_template_surface.py`
rebuilds the file from a FieldTrip checkout.

The geodesic distance matrices in `results/` are held compressed. One
Dijkstra run per contact over 175409 vertices gives 65 MB of float32, almost
all of it +inf because the runs are distance limited, which compresses to
about 0.4 MB. `run_cortex.py` writes both forms and `distcache.py` reads
whichever is present, so nothing else needs to know.

## A note on the certified level

The certified level is the largest *k* such that (b₀, b₁)(Δ_j(N)) = (1, 0) for
**every** j up to *k*, not the largest level at which it happens to hold. The
property is not monotone in *k*. On the 8×8 grid at pitch 0.8, Δ₁ and Δ₅ are
both (1, 0) while Δ₂ = (1, 25), Δ₃ = (26, 14) and Δ₄ = (74, 0), so the
certified level is 1 and not 5. `run_grid.py` records both, as `k_certified`
and `k_highest_hit`.

## License

MIT for the code. `data/colin27_lh_pial.npz` carries the terms of the
FieldTrip toolbox it is derived from; see `LICENSE`.

## Citing

`CITATION.cff` has the machine-readable record. Please cite the paper
alongside this archive.
