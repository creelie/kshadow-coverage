# v1.5.0: Optimal ECoG coverage manuscript

Archived on Zenodo: https://doi.org/10.5281/zenodo.22953062 (tag at commit
`b336049`).

## Summary
A second manuscript, **Optimal ECoG Coverage for Epilepsy Surgery**
(`paper/ecog/`, amsart, running title *Optimal ECoG coverage*), states
electrode placement for epilepsy surgery as a covering-radius problem on the
folded cortex. Every point of the territory suspected of generating the
seizures is seen by at least *k* contacts exactly when the *k*-th covering
radius is below the footprint radius *r*. When that holds, every onset zone in
the territory, of any size and position, is still seen whole after any
*k* − 1 contacts fail. On the triangle mesh this stays exact when the covering
radius is taken over triangles. The paper adds one new analysis: how often an
onset zone of radius *s* is seen whole, seen twice or missed entirely, and how
that changes under random contact failure. It uses no patient data, reports no
outcome, and states the protocol that would test clinical relevance.

## Results (colin27 template, 3777 mm² parietal territory, r = 8 mm)

| design | covering radius | unseen | 5 mm onset zone seen whole |
|---|---|---|---|
| documented 8×8 grid, 64 contacts | 21.96 mm | 2082.0 mm² (55.1%) | 18.5% of positions |
| 64 crown contacts | 11.24 mm | 89.7 mm² | 82.6% |
| 64 contacts placed by covering radius | 8.12 mm | 1.0 mm² | 98.1% |
| 66 contacts placed by covering radius | 7.77 mm | 0 | 100% |

- **Hidden onset zones:** the grid can miss an onset zone of radius up to
  14.5 mm entirely.
- **Crown floor:** counting every gyral crown within reach, part of the
  territory is 11.24 mm from its nearest crown, so no crown-only (subdural
  sheet) array covers it at a footprint radius at or below that, whatever its
  contact count.
- **Double coverage** takes 132 contacts at 8 mm (102 at 10 mm). After 8
  random failures it still sees a 5 mm onset zone whole in 99.8% of positions
  on average, where single coverage (66 contacts) falls to 78.6%.

## Corrections to earlier statements
- The fewest contacts that see the patch at 12 mm is 29, not 30:
  `run_optimal3.py` scans the contact count in steps of two. This affects
  `results/optimal.json` and `paper/kshadow_natphys.tex`, which are not
  changed.
- The crown floor of 12.219 mm counts only the crown vertices inside the
  patch. A sheet's contacts are not confined to those (49 of the 64 grid
  contacts lie outside the patch). With every crown within reach the floor is
  10.94 mm on the vertex rule and 11.24 mm on the triangle rule. The README
  headline paragraph is corrected; `paper/kshadow_natphys.tex` is not changed.

## New files
- `run_ecog_designs.py` → `results/ecog_designs.json`: triangle-rule covering
  radii of the placement sequence; the fewest contacts that see the patch once
  and twice at 8–14 mm, scanned one contact at a time, with certificates; the
  crown site floor over every crown within reach; a crown-only placement
  sequence.
- `run_soz_capture.py` → `results/soz_capture.json`: onset-zone capture with
  triangles as the atoms and zones measured between triangle centroids, and
  random contact failure with fixed seeds.
- `make_ecog_paper_data.py` → `paper/ecog/numbers_ecog.tex` and the plot
  tables in `paper/ecog/tikz/data/`: every number in the manuscript is a macro
  generated from the result files.
- `paper/ecog/`: the manuscript, its PDF and bibliography; TikZ/pgfplots
  sources of Figs. 1–4 and 6–9 with one shared style file, rendered to 600-dpi
  PNGs by `paper/ecog/build.sh`.

## Reproduce
```
python3 run_ecog_designs.py       # about five minutes
python3 run_soz_capture.py        # about a minute
python3 make_ecog_paper_data.py
cd paper/ecog && ./build.sh       # figures, then ecog_coverage.pdf
```

## After this release
A Lean 4 formalization of the paper's propositions on the mesh (`lean/`,
`cd lean && lake build`) was merged after v1.5.0, together with the paper's
Methods subsection on it and the references to this release and its DOI.
