# Provenance of the v1.6.0 extension

Deep Bhattacharjee directed the preparation of this contribution with Codex
assistance. Authorship and historical contributions of Aditi Bose, Deep
Bhattacharjee and Ushashi Bhattacharya are retained. This document does not
assert that coauthors have completed review or approved journal submission
of the new revision.

Codex prepared the exact-predicate integration, whole-face proof extension,
outward path-bound implementation, tests, cortical audit, figure, manuscript
edits and release materials. Earlier Claude assistance remains disclosed in
the manuscripts. AI systems are not listed as authors.

The new theorem is a sufficient geometric certificate on the represented
piecewise-flat mesh under the stated radius-footprint model. Its analytic
proof is in `paper/ecog/mesh_certification.tex` and
`docs/mesh-coverage-bound.md`. Computational verification is documented in
the tests and input-hashed `results/certified_mesh.json`.

The new theorem and integer-path code are not Lean-formalized. Existing
Lean coverage is stated separately in `lean/README.md`. No claim of patient
benefit, general conjecture resolution or guaranteed journal acceptance is
made. No private clinical data or hidden model reasoning is included.
