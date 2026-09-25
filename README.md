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

Contacts confined to gyral crowns cannot reach this at any count. Counting
every gyral crown within 25 mm of the patch (7279 crown vertices), some point
of the patch is 10.94 mm from its nearest crown, and on the triangle rule some
triangle has no crown within 11.24 mm of all three of its vertices, so no
crown-only array covers the patch at a footprint radius of 11.24 mm or less,
at any contact count. (Counting only the 3016 gyral vertices inside the patch
gives 12.219 mm, the figure in `paper/kshadow_natphys.tex`; a sheet's contacts
are not confined to those.)

112 contacts at a 9 mm radius see every point of the patch twice, by direct
computation on the mesh. The certificate gives Δ₁(N) and Δ₂(N) both (1, 0),
but the mesh reference finds one hole in R₂ outside the patch (b₁ = 1) that the
nerve misses: geodesic footprints on a folded surface are not convex, and 961
of the 16835 nerve faces fail the discrete disk test, so the nerve theorem
does not hold there. With 96 contacts at 10 mm the certificate and the mesh
agree at every level (certified level 2), yet 1.7 mm² of the patch is seen only
once: a certificate of (1, 0) does not by itself put the target inside R₂.
Both are in `results/optimal.json`, under `curves.k2`.

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
fetch_ds003775_status.py  builds the real-failure tables from OpenNeuro ds003775
run_eeg_failures.py     real contact failures: pre-specified tests (staged)
run_eeg_design.py       exploratory reach analysis and layout comparison
arrangement.py          exact topology of k-fold cap regions, no nerve
make_eeg_figure.py      figures/fig_eeg.png
run_signal_rebuild.py   signal-level test (pre-specified plan in its docstring)
run_signal_rebuild.ps1  Windows launcher for it
fetch_ds003775_manifest.py  rebuilds the epochs manifest with S3 version ids
run_signal_explore.py   exploratory follow-up of the signal-level test
run_failure_margins.py  exploratory: failure clustering beyond a rate gradient
run_failure_correlation.py  exploratory: pair correlation of failure, correlation length
make_signal_figure.py   figures/fig_signal.png
make_paper_numbers.py   paper/numbers.tex, every number the manuscript quotes
paper/                  the amsart manuscript
run_ecog_designs.py     triangle-rule covering radii, designs, crown floor (ECoG paper)
run_soz_capture.py      onset-zone capture and random-failure analysis (ECoG paper)
make_ecog_paper_data.py paper/ecog/numbers_ecog.tex and the TikZ plot tables
paper/ecog/             the ECoG coverage manuscript, TikZ figure sources, PNGs
upload_kshadow.sh       one-shot push of this folder to GitHub, then a tag
upload_kshadow.ps1      the same thing for Windows PowerShell
```

## Real contact failures (v1.3.0)

`data/ds003775_channel_status.tsv` lists, for each of the 153 recordings of
OpenNeuro ds003775 v1.2.1 (doi:10.18112/openneuro.ds003775.v1.2.1, CC0), the
channels its curators' pipeline marked bad. `fetch_ds003775_status.py`
rebuilds it from the dataset's public git mirror and checks it against the two
retention figures of the data descriptor (98/153 and 36/153).

    python3 run_eeg_failures.py main
    python3 run_eeg_failures.py cert
    python3 run_eeg_failures.py b1
    python3 run_eeg_failures.py stress 0 500
    python3 run_eeg_failures.py stress 500 1000
    python3 run_eeg_failures.py merge
    python3 run_eeg_design.py e1
    python3 run_eeg_design.py opt D1      # and opt 70, 75, 80, 85, 92
    python3 run_eeg_design.py eval
    python3 make_eeg_figure.py

The pre-specified plan is in the docstring of `run_eeg_failures.py`; the
analyses in `run_eeg_design.py` were added afterwards and are exploratory.

The signal-level test (does local redundancy predict how well a dropped
channel is rebuilt from the others?) needs the EEG signals, about 12.7 GB for
the first sessions. On Windows:

    powershell -ExecutionPolicy Bypass -File .\run_signal_rebuild.ps1 -DeleteAfter

It verifies every file against the MD5 of version 1.2.1 recorded in
`data/ds003775_epochs_manifest.tsv`, resumes if interrupted, and writes
`results/signal_rebuild.json`. Later versions of ds003775 dropped the
derivatives folder, so the plain S3 URL of each file now returns 404; the
manifest records the S3 object version that still serves the v1.2.1 content,
and `fetch_ds003775_manifest.py` rebuilds it from the mirror's git-annex
branch. A download that stops early resumes from where it stopped.

The shipped `results/signal_rebuild.json` and `results/signal_rebuild_rows.csv`
come from a run on all 153 recordings. The follow-ups, written after that run
and labelled exploratory, are

    python3 run_signal_explore.py     # results/signal_explore.json
    python3 run_failure_margins.py    # results/failure_margins.json (about an hour)
    python3 run_failure_correlation.py  # results/failure_correlation.json
    python3 make_signal_figure.py
    python3 make_paper_numbers.py     # then: cd paper && latexmk -pdf kshadow_natphys.tex

## What the signal-level test found

All four pre-specified tests came out in the planned direction: within each
channel, across the first-session recordings, redundancy depth λ correlates
with how well the channel is rebuilt from the others (positive for all 64
channels, median Spearman 0.198, p = 1.8e-12), private territory in the
opposite direction, λ beyond the nearest distance, and the second sessions
replicate. With rebuild quality centred within each recording the effect
remains (median 0.109, p = 3.2e-10).

The exploratory follow-ups decide how to read it. λ is exactly a sum over
survivors of a lens-area kernel of each survivor's distance (checked on every
row to 0.0022), so it is geometric by construction. The mean distance to the
three nearest survivors predicts at least as well. Out of sample, adding λ and
the non-additive private territory π to the six nearest distances changes R²
by at most 0.006. Redundancy predicts the rebuild, but through distances; the
topology adds nothing measurable here.

Contact failures are correlated in space beyond each channel's own failure
rate: against a null that keeps every recording's failure count and every
channel's failure rate, adjacent failed pairs, uncovered area and the reach
needed at 95% (46.0 against 30.7 degrees) are all above every null cohort, in
both sessions, and co-failure falls to 1/e of its nearest-neighbour excess by
about 36 degrees.

One second-session file (sub-104_ses-t2) holds no epochs in the published
dataset and is not used. The manuscript is `paper/kshadow_natphys.tex`.

## Optimal ECoG coverage (paper/ecog)

`paper/ecog/ecog_coverage.tex` (amsart, short title *Optimal ECoG coverage*)
states electrode placement for epilepsy surgery as a covering-radius problem
on the folded cortex: every point of the territory suspected of generating the
seizures is seen by at least k contacts exactly when the k-th covering radius
is below the footprint radius, and then every onset zone in the territory is
seen whole after any k - 1 contacts fail. On the triangle mesh this stays exact
when the covering radius is taken over triangles, the distance from a contact
to a triangle being the distance to its farthest vertex; that is the rule the
unseen areas above are measured with. On the parietal patch, at an 8 mm
footprint:

| design | covering radius | patch unseen | 5 mm onset zone seen whole |
|---|---|---|---|
| documented 8x8 grid, 64 contacts | 21.96 mm | 2082.0 mm² | 18.5% of positions |
| 64 crown contacts | 11.24 mm | 89.7 mm² | 82.6% |
| 64 contacts by covering radius | 8.12 mm | 1.0 mm² | 98.1% |
| 66 contacts by covering radius | 7.77 mm | 0 | 100% |

The largest onset zone the grid can miss entirely has a radius of 14.5 mm.
Counting every gyral crown within reach of the patch, some part of it is
11.24 mm from its nearest crown, so no crown-only array covers it at a smaller
footprint radius. Seeing the patch twice takes 132 contacts at 8 mm; after 8
random contact failures they still see a 5 mm onset zone whole in 99.8% of
positions on average, where the 66-contact single-coverage design falls to
78.6%. These are template-brain geometry, not patient outcomes; the paper
states the protocol that would test whether better coverage changes seizure
outcome.

Two statements made earlier in this repository are corrected there. The
fewest contacts covering the patch at 12 mm is 29, not 30 (`run_optimal3.py`
scans in steps of two). The crown floor of 12.219 mm counts only the crown
vertices inside the patch; a sheet's contacts are not confined to them (49 of
the 64 grid contacts lie outside it), and with every crown within reach the
floor is 10.94 mm on the vertex rule and 11.24 mm on the triangle rule.

Every figure except the cortex render is a TikZ/pgfplots source in
`paper/ecog/tikz/`, rendered to PNG at 600 dpi:

    python3 run_ecog_designs.py       # results/ecog_designs.json, ~5 min
    python3 run_soz_capture.py        # results/soz_capture.json, ~1 min
    python3 make_ecog_paper_data.py   # macros and plot tables
    cd paper/ecog && ./build.sh       # figures/*.png, then ecog_coverage.pdf

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
