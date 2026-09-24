# v1.4.0: signal-level test, correlated failures, manuscript

## Summary
The pre-specified signal-level test has now been run on all usable recordings of OpenNeuro ds003775 v1.2.1. All four tests came out in the planned direction: the redundancy left around a channel predicts how well its signal is rebuilt from the other channels. Analyses written afterwards (exploratory) show the effect is carried by distances, not by the topology of the coverage. Separately, contact failures turn out to be correlated in space beyond each channel's own failure rate, with a correlation length of about 36°.

## Pre-specified results (`run_signal_rebuild.py`, plan unchanged)
Data: 152 recordings, 8673 channel rebuilds. The published file for sub-104_ses-t2 holds no epochs and is not used.

| test | result |
|---|---|
| **P** (primary): λ vs rebuild quality, within channel, first sessions | 64/64 channels positive, median ρ 0.198, p = 1.8 × 10⁻¹² |
| **S1**: private territory π | 62/64 channels negative, median ρ −0.158, p = 6.0 × 10⁻¹² |
| **S2**: λ given nearest-neighbour distance | 75/111 recordings positive, median partial ρ 0.092, p = 7.7 × 10⁻⁷ |
| **S3**: replication, second sessions | 60/64 channels positive, median ρ 0.294, p = 3.3 × 10⁻¹² |

## Exploratory results (written after the tests)
- **Recording quality does not explain it.** With each recording's mean removed, λ still predicts rebuild quality (median ρ 0.109, p = 3.2 × 10⁻¹⁰).
- **λ is geometric by construction.** It is exactly a sum over surviving channels of a lens-area term that depends only on each one's distance; this is checked on every row to within 0.0022.
- **Topology adds nothing measurable.** Adding λ and π to the six nearest distances changes out-of-sample R² by at most 0.006.
- **Failures cluster beyond a rate gradient.** The null keeps every recording's failure count and every channel's failure rate. Against it, the real failures still show more adjacent failed pairs (1455 vs 1114), more uncovered target, and a larger reach needed at the 95% level (46.0° vs 30.7°). This holds in both sessions, with p ≤ 0.001 for each.
- **Co-failure has a correlation length.** Channels 10–20° apart fail together 1.7× as often as their rates predict, and the excess falls to 1/e by about 36° (bootstrap 95% interval 20–45°).

## Fixes
- **Dataset download:** the cleaned-epoch files returned 404, because later versions of ds003775 dropped them. Downloads now request the archived v1.2.1 copy of each file, check its MD5, and resume interrupted transfers. The manifest gains a `version_id` column, and `fetch_ds003775_manifest.py` rebuilds it.
- **README correction:** the 112-contact, 9 mm design was described as certified to level 2. The mesh check finds a hole in the 2-fold region that the certificate misses, because geodesic footprints are not convex. The README now says so.

## Authors and metadata
- Author details: Aditi Bose (IIIT Hyderabad), Deep Bhattacharjee (corresponding author; formerly EGSPL, Bhubaneswar) and Ushashi Bhattacharya (formerly National Taiwan University), with affiliations and emails in the manuscript, `CITATION.cff` and `.zenodo.json`.
- The manuscript adds an author contributions statement and cites the preprint the idea grew out of: Bhattacharjee, D.; Bhattacharya, U. *Iterated Nerve Complexes for k-Fold Sensor Coverage*. Preprints 2026, 2026061351. https://doi.org/10.20944/preprints202606.1351.v2. The Zenodo record links to it as a related work.

## New files
- `run_signal_explore.py`, `run_failure_margins.py`, `run_failure_correlation.py`: exploratory analyses.
- `make_signal_figure.py` → `figures/fig_signal.png`.
- `make_paper_numbers.py` → `paper/numbers.tex`: every number in the manuscript, generated from `results/*.json`.
- `paper/`: amsart manuscript (`kshadow_natphys.tex`, PDF, bibliography).
- `RELEASE_NOTES_v1.4.0.md`: these notes.
- `results/`: `signal_rebuild.json`, `signal_rebuild_rows.csv`, `signal_explore.json`, `failure_margins.json`, `failure_correlation.json`.

## Reproduce
```
python3 run_signal_rebuild.py --sessions t1,t2 --delete-after   # streams ~17 GB
python3 run_signal_explore.py
python3 run_failure_margins.py        # about an hour
python3 run_failure_correlation.py
python3 make_signal_figure.py
python3 make_paper_numbers.py
cd paper && latexmk -pdf kshadow_natphys.tex
```
