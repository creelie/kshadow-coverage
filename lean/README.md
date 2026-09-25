# Lean 4 formalization

Machine-checked proofs of the combinatorial propositions of
`paper/ecog/ecog_coverage.tex`, on the triangulated surface, in core Lean 4
(no Mathlib; toolchain `leanprover/lean4:v4.22.0`).

    cd lean && lake build

| paper | Lean (`KShadowECoG/Coverage.lean`) |
|---|---|
| Proposition 2 on the mesh: X is seen k times iff every atom's k-th smallest contact distance is below r | `le_mult_iff`, `cover_iff` |
| Proposition 3: a zone survives every failure of at most q contacts iff it is seen q + 1 times | `failure_proof`, `zone_failure_proof` |
| Proposition 4 on the mesh: capture by erosion | `capture_by_erosion` |
| Proposition 5: site floor, also k-fold | `site_floor`, `site_floor_k` |
| Proposition 6: packing bound | `packing` |
| Proposition 7: the pigeonhole step of the factor-two bound | `farthest_point_core` |
| rho^v <= rho <= rho^v + l, per contact | `le_triDist`, `triDist_le` |

Contacts are elements of a type `V`, atoms (triangles) of a type `A`, and
distances are natural numbers (for instance micrometres); only their order
and, for the metric statements, symmetry and the triangle inequality are
used. `KShadowECoG/Lists.lean` proves the list facts these need: a
duplicate-free sublist bound, pigeonhole for a proximity relation, and the
order-statistic lemma for sorted lists. Every theorem depends only on the
standard axioms `propext`, `Classical.choice` and `Quot.sound`
(`#print axioms`); there is no `sorry`.

Not formalized: the continuous-surface statements (open and closed geodesic
disks, the measure argument of Proposition 4(2) and its lower bound on h),
and the separation property of the farthest-point sequence that Proposition 7
combines with the pigeonhole step.
