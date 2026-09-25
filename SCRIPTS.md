# The k-Shadow Nerve Theorem -- supplementary code

Python 3, NumPy, SciPy, NetworkX, Matplotlib. All numbers and figures in the
PRE manuscript (`../tex/kshadow_pre.tex`) are produced by the scripts below;
none are hand-edited after generation.

## Library
- `kshadow.py`  exact nerve of equal disks (Helly), HellyNerve membership
  oracle, k-shadow 2-skeleton and Betti numbers over GF(2), subdivision
  filtration Sub_k(N) and its persistence, raster ground truth, dropout margin.

## Data
- `fetch_template_surface.py`  extracts the left pial surface of the
  FieldTrip template anatomy (the colin27 single-subject brain, FreeSurfer
  reconstruction, MNI mm coordinates) from a local checkout of
  https://github.com/fieldtrip/fieldtrip, `template/anatomy/surface_pial_left.mat`,
  and stores it as `data/colin27_lh_pial.npz` together with the source commit
  hash. Usage: `python3 fetch_template_surface.py /path/to/fieldtrip [commit]`.
  This is a de-identified, long-published reference anatomy, not patient data;
  the commit used for the paper is recorded in `data/colin27_lh_pial.npz` and
  quoted in the manuscript's Data Availability statement.

## Experiments (each writes results/*.json and draft figures/*.png)
- `run_fields.py`, `run_fields_rest.py`  the five planar fields A-E of the
  earlier general-sensor-coverage preprint (`fields_v2.py` regenerates them
  from their seeds). Field E: only k = 1 is computed (41-clique, see paper,
  Section IX.A). Writes `results/fields.json`.
- `run_grid.py`  8x8 grid: certificate, raster comparison, barcode,
  exhaustive/random dropout checks, radius sweep, pitch sweep. Writes
  `results/grid.json` and a first-draft `figures/grid_multiplicity.png`,
  `figures/grid_barcode.png`, `figures/grid_radius_sweep.png`,
  `figures/grid_pitch_sweep.png`.
- `run_sphere.py`  spherical-cap arrays with exact spherical geometry and
  raster ground truth in the gnomonic chart at two resolutions. Writes
  `results/sphere.json` and a first-draft `figures/sphere_3d.png` plus
  `figures/sphere_r013.png`, `figures/sphere_r016.png`.
- `run_anisotropic.py`  circular versus elliptical footprints, membership
  certified by an exact convex minimax program (SLSQP), not an approximate
  ellipse test. Writes `results/anisotropic.json` and `figures/anisotropic.png`.
- `run_cortex.py`  the real cortical-surface benchmark (paper Section IX.D):
  loads `data/colin27_lh_pial.npz`, validates the mesh (every edge in exactly
  two triangles, every vertex link a single cycle, Euler characteristic),
  places the 64-contact parietal and 32-contact temporal grids from the
  FieldTrip SubjectUCI29 tutorial layout, computes geodesic (Dijkstra) and
  Euclidean footprints under a strict triangle-inclusion rule, runs the
  discrete disk test face by face, and cross-checks the shadow complex
  Delta_k(N) against a mesh reference R_k computed independently of the
  nerve. Writes `results/cortex.json` and
  `results/cortex_dist_<grid>.npy`. Run in the background
  (`setsid nohup python3 run_cortex.py > logs/run_cortex.log 2>&1 &`);
  takes about two minutes.
- `run_dropout_cortex.py`  private territory on the two cortical arrays: for
  each contact, the triangles its footprint holds and no other footprint
  holds, and the area that leaves R_1 if that contact fails. Takes the radii
  as command-line arguments and defaults to 8 10 12; the paper uses
  `python3 run_dropout_cortex.py 4 5 6 7 8 9 10 12 15`, which is what
  Table IV reports and what the shipped `results/dropout_cortex.json`
  contains. About a minute and a half.
- `run_stability.py`  the two stability bounds measured on the ring-grid
  field: contact loss (six random draws for each m from one to five) and
  localisation (thirty contacts displaced by delta). Bottleneck distances are
  computed exactly, by bipartite matching at each candidate threshold, not
  estimated. Writes `results/stability.json`.
- `run_montages.py`  the three published scalp montages of Section IX.H on
  the same colin27 pial surface: electrode coordinates from MNE-Python's
  `colin27_1005` montage, projected to the nearest pial vertex, geodesic
  footprints, coverage and multiplicity over radii 10-40 mm. Needs MNE-Python
  in addition to the dependencies above. Writes `results/montages.json`.
- `run_design.py`  the lattice sweep behind Section IX.F: pitch and footprint
  radius varied together on the same surface, with the certificate computed at
  each design point. Reproduces run_cortex.py's numbers at the published pitch
  and shows that no grid on the envelope of the hemisphere repairs them. Writes
  `results/design.json`. Arguments: two comma-separated lists, pitches then
  radii.
- `run_optimal.py`  builds the contact sets of Section IX.F by farthest-point
  sampling in the geodesic metric, for two candidate families: any vertex of
  the target patch, and gyral vertices only. Records the covering radius of
  every prefix of each sequence. Arguments: patch radius in mm, then the
  number of contacts. Writes `results/optimal.json`.
- `run_optimal2.py`  adds to that file the exact crown floor (one multi-source
  Dijkstra from every gyral vertex of the patch at once), the k-fold covering
  radii, and certificates at a sample of contact counts and radii.
- `run_optimal3.py`  scans the contact count upward at each footprint radius
  for the smallest design that leaves nothing of the patch uncovered and gives
  b0 = 1, b1 = 0 with the mesh reference agreeing, which is Table V of the
  paper, and locates the redundant design of certified level 2.
- `run_packing.py`  the lower bound of the paper's packing proposition: a
  maximal 2r-separated subset of the patch, whose size no array of footprint
  radius r can undercut.
- `compare_grid.py`  the matched comparison of Table VI: the published 8x8
  parietal layout and the first 64 greedy sites, on the same patch at the same
  radii.
- `real_data_certificate.py`  SubjectUCI29 protocol with the subject's own
  digitized electrode coordinates; runs only when the FieldTrip raw files are
  present locally under their data-use terms. Not executed for the paper; no
  patient-level numbers are reported anywhere in this package.

## Figures
The manuscript carries nine figures, eight of them composites. `make_merged_figures.py`
builds six of them; the two that are already drawn at the width they are
printed at, `six_footprints.png` and `anisotropic.png`, are written by
`make_field_figures.py` and `run_anisotropic.py` and used unchanged.

The scripts below write the individual panels into `figures/`. Some of those
panels are pasted into a composite and some are redrawn inside it; either way
`make_merged_figures.py` and `make_optimal_figure.py` are the last steps, and
`../tex/figures/` holds only the nine files the manuscript includes.

- `make_field_figures.py`  reads `results/fields.json`; writes
  `figures/field_A_k2.png`, `figures/field_D_k2.png`,
  `figures/field_C_tangency.png`, `figures/field_E_clique.png`,
  `figures/nerve_sizes.png`, `figures/six_footprints.png`.
- `make_figures.py`  reads `results/fields.json` and `results/grid.json`;
  writes `figures/field_barcodes.png` (Fields A and D) and regenerates
  `figures/grid_pitch_sweep.png`, superseding the draft of the same name
  written by `run_grid.py`.
- `make_sphere3d.py`, `make_sphere3d_pv.py`  the 3-D view of the spherical
  array, superseding the draft written by `run_sphere.py`; the `_pv` variant
  is the one used for the paper and needs PyVista.
- `make_cortex_figures.py`, `make_cortex_figures_pv.py`  read
  `results/cortex.json` and `results/cortex_dist_*.npy`; write
  `figures/cortex_overview.png`, `figures/cortex_regions.png`,
  `figures/cortex_models.png`, `figures/cortex_sweep.png`,
  `figures/cortex_barcodes.png`. The `_pv` variant renders the surface
  panels off screen with PyVista and is the one used for the paper;
  `shading.py` holds the shared lighting and camera setup.
- `make_dropout_figure.py`  reads `results/dropout_cortex.json`; writes
  `figures/fig_dropout_cortex.png` and the table of private territory that
  Table IV reports.
- `make_stability_figure.py`  reads `results/stability.json`; writes
  `figures/fig_stability.png`.
- `make_optimal_figure.py`  reads `results/optimal.json` and `results/cortex.json`
  and writes `figures/fig_optimal.png`, the figure of the optimal-coverage
  section; its two surface panels are VTK renders and need PyVista.
- `make_merged_figures.py`  the composites the manuscript actually includes:
  `fig_fields.png`, `fig_grid.png`, `fig_sphere.png`, `fig_cortex_a.png`,
  `fig_cortex_b.png`, `fig_robust.png`, each drawn at its printed width so
  that no label falls below seven points, and copied into `../tex/figures/`.
  It needs the PyVista renders above to exist as PNGs but does not run
  PyVista itself.

`make_fig10_anisotropic.py` is an earlier, standalone draft of the isotropic
versus anisotropic figure. It is superseded by `run_anisotropic.py`, which is
what actually produced `results/anisotropic.json` and `figures/anisotropic.png`
for the paper; `make_fig10_anisotropic.py` is kept only for reference and is
not part of the reproduce sequence below.

## legacy_v2/
Unchanged scripts of the earlier general-sensor-coverage preprint
(`preprints202606.1351.v2`, brute-force pipeline), kept for provenance and not
used by the PRE paper.

## Reproduce
    export PYTHONPATH=$PWD
    python3 run_fields.py
    python3 run_fields_rest.py C_poisson_random E_two_cluster
    python3 run_grid.py
    python3 run_sphere.py
    python3 run_anisotropic.py
    python3 fetch_template_surface.py /path/to/fieldtrip   # writes data/colin27_lh_pial.npz
    python3 run_cortex.py
    python3 run_dropout_cortex.py 4 5 6 7 8 9 10 12 15
    python3 run_stability.py
    python3 run_montages.py
    python3 run_design.py
    python3 run_optimal.py 35 160
    python3 run_optimal2.py
    python3 run_optimal3.py 14,12,10,8
    python3 run_packing.py 14,12,10,9,8
    python3 compare_grid.py
    python3 make_field_figures.py
    python3 make_figures.py
    python3 make_sphere3d_pv.py
    python3 make_cortex_figures_pv.py
    python3 make_cortex_figures.py
    python3 make_dropout_figure.py
    python3 make_stability_figure.py
    python3 make_optimal_figure.py
    python3 make_merged_figures.py
    python3 fetch_ds003775_status.py        # needs git and MNE-Python
    python3 run_eeg_failures.py main
    python3 run_eeg_failures.py cert
    python3 run_eeg_failures.py b1
    python3 run_eeg_failures.py stress 0 500
    python3 run_eeg_failures.py stress 500 1000
    python3 run_eeg_failures.py merge
    python3 run_eeg_design.py e1
    python3 run_eeg_design.py opt D1
    python3 run_eeg_design.py opt 70
    python3 run_eeg_design.py opt 75
    python3 run_eeg_design.py opt 80
    python3 run_eeg_design.py opt 85
    python3 run_eeg_design.py opt 92
    python3 run_eeg_design.py eval
    python3 make_eeg_figure.py
    python3 fetch_ds003775_manifest.py      # optional: rebuilds the manifest
    python3 run_signal_rebuild.py --sessions t1,t2 --delete-after   # ~17 GB streamed
    python3 run_signal_explore.py
    python3 run_failure_margins.py
    python3 run_failure_correlation.py
    python3 make_signal_figure.py
    python3 make_paper_numbers.py
    (cd paper && latexmk -pdf kshadow_natphys.tex)

## Signal level and the manuscript
- `run_signal_rebuild.py`  the pre-specified signal-level test: each good
  channel of each recording is removed and rebuilt from the others by
  spherical splines, and rebuild quality is tested against redundancy depth.
  The plan is in its docstring. Writes `results/signal_rebuild_rows.csv` and
  `results/signal_rebuild.json`.
- `fetch_ds003775_manifest.py`  rebuilds `data/ds003775_epochs_manifest.tsv`
  (path, size, MD5 and S3 version id of every cleaned-epoch file of v1.2.1)
  from the git-annex branch of the dataset's public mirror.
- `run_signal_explore.py`  exploratory, written after the test was run: checks
  that redundancy depth is a pairwise kernel sum, repeats the within-channel
  test with metric predictors, and compares nested predictive models out of
  sample. Writes `results/signal_explore.json`.
- `run_failure_margins.py`  exploratory: the clustering statistics of
  `run_eeg_failures.py` against a null that keeps every recording's failure
  count and every channel's failure rate (curveball swaps). Writes
  `results/failure_margins.json`.
- `run_failure_correlation.py`  exploratory: co-failure of channel pairs by
  angular separation against the same margin-preserving null, and the
  correlation length at which the excess falls to 1/e. Writes
  `results/failure_correlation.json`.
- `make_signal_figure.py`  writes `figures/fig_signal.png`.
- `make_paper_numbers.py`  writes `paper/numbers.tex`, one macro per number
  quoted in `paper/kshadow_natphys.tex`.

All scripts are deterministic (fixed random seeds where randomness is used)
and were re-run in full for this version of the paper; every number quoted in
the manuscript is read directly from the `results/*.json` files this
reproduce sequence regenerates.

## ECoG coverage paper (`paper/ecog/`)
- `run_soz_capture.py`  onset-zone capture on the target patch of
  `run_optimal.py`: for eight designs (the documented 8x8 grid, 64 crown
  contacts, and contact sets placed by covering radius at 8, 9 and 10 mm),
  the share of positions in which an onset zone of geodesic radius s is seen
  whole, seen twice, or missed entirely, for s from 0 to 15 mm; the largest
  onset zone that can be missed; the covering radii rho_1 and rho_2; and the
  share still seen whole after q = 1..8 contacts fail at random (100 draws,
  fixed seeds). Reads `results/optimal.json` and `results/cortex.json`, writes
  `results/soz_capture.json`. About twenty seconds.
- `make_ecog_paper_data.py`  writes `paper/ecog/numbers_ecog.tex`, one macro
  per number quoted in `paper/ecog/ecog_coverage.tex`, the plot tables in
  `paper/ecog/tikz/data/`, and `paper/ecog/figures/fig5_cortex.png` (the two
  surface renders of `figures/fig_optimal.png`).
- `paper/ecog/tikz/fold_geometry.py`  the schematic cross-section of Fig. 3,
  exact in arc length; writes `paper/ecog/tikz/data/fold_*`.
- `paper/ecog/tikz/fig*.tex`  the TikZ/pgfplots sources of Figs. 1-4 and 6-9,
  sharing `ecogstyle.tex`; `paper/ecog/build.sh` renders each to
  `paper/ecog/figures/*.png` at 600 dpi and then builds the PDF.

Reproduce:

    python3 run_soz_capture.py
    python3 make_ecog_paper_data.py
    cd paper/ecog && ./build.sh          # figures, then ecog_coverage.pdf
