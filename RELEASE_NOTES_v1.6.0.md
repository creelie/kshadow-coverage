# v1.6.0: continuous mesh coverage with certified distance bounds

Authors: Aditi Bose, Deep Bhattacharjee and Ushashi Bhattacharya.

The ECoG manuscript is now **Redundant ECoG Coverage on Folded Surfaces:
Design Bounds and Certified Mesh Coverage**. The existing signal-level
paper remains a separate manuscript. This release adds a complete proof and
implementation connecting valid vertex distance upper bounds to every point
of the represented triangular surface. It makes no patient-outcome claim.

## Results and changes

* Exact disk intersection predicates repair a scale-dependent false nerve
  face; both the pair and triple tests use rational squared radii.
* The sufficient whole-face test is m + D/sqrt(3) < r, where m is the largest
  vertex kth-distance upper bound and D the longest face edge. The universal
  triangle constant is sharp. A passing k-certificate survives any k-1
  contact failures, without assuming a good cover for the nerve.
* `certified_geodesic.py` constructs the required upper bounds from explicit
  edge paths. Edge lengths are rounded upward to multiples of 2^-20 mm by
  integer square roots; shortest-path sums are integers; output conversion
  is checked outward. No approximate geodesic is silently treated as exact.
* `run_certified_mesh.py` audits 70 prescribed (contact count, k, radius)
  combinations on the stored 9659-face target. It certifies single coverage
  at 80 contacts/8 mm and double coverage at 160 contacts/8 mm,
  128 contacts/9 mm, and 112 contacts/10 mm. These are sufficient examples
  from the stored sequence; globally minimum counts are not established.
* The manuscript includes the full proof, a bounded-error corollary, exact
  arithmetic details, an input-hashed result table and a vector figure.

## Validation and reproduction

```
python -m unittest discover -s tests -v
python run_certified_mesh.py
python make_certified_figure.py
cd paper/ecog
pdflatex -interaction=nonstopmode -halt-on-error ecog_coverage.tex
bibtex ecog_coverage
pdflatex -interaction=nonstopmode -halt-on-error ecog_coverage.tex
pdflatex -interaction=nonstopmode -halt-on-error ecog_coverage.tex
```

The 17 tests include exact 3-4-5 paths, outward irrational-edge lengths,
truncation, disconnected-input rejection, open/closed equality, refinement,
and powers-of-two scaling. The cortical audit uses the represented binary64
mesh and its fixed target-face list. It does not certify anatomical accuracy
or the exact geodesic shape of the patch-selection boundary. Unresolved
faces are not declared uncovered. All input hashes are in the audit JSON.

The earlier Lean files cover the propositions described in `lean/README.md`.
The new continuous-face theorem and path-arithmetic code are not claimed
to have been formalized in Lean.

## Credit and provenance

All three authors and existing contribution statements are retained.
The extension was prepared with Codex assistance under Deep Bhattacharjee's
direction. It builds on the standard Lipschitz order-statistic argument,
barycentric variance identity and Dijkstra's algorithm; no general open
conjecture or journal acceptance is claimed. See `AI_PROVENANCE.md`.

The earlier v1.5.0 archive remains DOI 10.5281/zenodo.22953062. That DOI
does not identify this revision. The v1.6.0 DOI is reported in the release
record after archival has been confirmed.
