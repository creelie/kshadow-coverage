/-
KShadowECoG/Coverage.lean -- the combinatorial propositions of
"Optimal ECoG Coverage for Epilepsy Surgery" (paper/ecog/ecog_coverage.tex),
on the triangulated surface.

Setting.  Contacts are elements of a type `V` (mesh vertices), atoms are
elements of a type `A` (triangles), and `e p a : Nat` is the distance from
contact `p` to atom `a` (in the paper, the distance to the triangle's farthest
vertex).  Atom `a` lies in the footprint of `p` when `e p a < r`.  Distances
are natural numbers, for instance in micrometres; only their order and, for
the packing and farthest-point statements, the triangle inequality are used.

Formalized here, in the paper's numbering:
  * Proposition 2 on the mesh: every atom of X is seen k times exactly when
    the k-th smallest contact distance of every atom of X is below r
    (`cover_iff`), i.e. when the k-th covering radius is below r;
  * Proposition 3: a set of atoms is seen whole after every failure of at most
    q contacts exactly when each of its atoms is seen q+1 times
    (`failure_proof`, `zone_failure_proof`);
  * Proposition 4 on the mesh: an onset zone about a centre lies in a set of
    good atoms exactly when the centre is farther than s from every bad atom
    (`capture_by_erosion`);
  * Proposition 5: contacts restricted to a site set C see an atom no more
    often than C itself does (`site_floor`, `site_floor_k`);
  * Proposition 6: the packing bound (`packing`);
  * Proposition 7: the pigeonhole step of the factor-two bound
    (`farthest_point_core`);
  * the per-contact bounds between the vertex and the triangle distance behind
    rho^v <= rho <= rho^v + l (`le_triDist`, `triDist_le`).
-/
import KShadowECoG.Lists

namespace KShadowECoG

section Footprints

variable {V A : Type} [DecidableEq V]

/-- Contact `p` sees atom `a` when `e p a < r`. -/
def seen (e : V → A → Nat) (r : Nat) (p : V) (a : A) : Bool :=
  decide (e p a < r)

/-- The multiplicity of atom `a`: how many contacts of `P` see it. -/
def mult (e : V → A → Nat) (r : Nat) (P : List V) (a : A) : Nat :=
  P.countP (fun p => seen e r p a)

/-- The distances from atom `a` to the contacts of `P`, in increasing order. -/
def sortedDists (e : V → A → Nat) (P : List V) (a : A) : List Nat :=
  (P.map (fun p => e p a)).mergeSort (fun x y => decide (x ≤ y))

/-- The distance from atom `a` to its `k`-th nearest contact (`k ≥ 1`), if
there are at least `k` contacts. -/
def kthDist (e : V → A → Nat) (P : List V) (a : A) (k : Nat) : Option Nat :=
  (sortedDists e P a)[k - 1]?

omit [DecidableEq V] in
theorem sortedDists_sorted (e : V → A → Nat) (P : List V) (a : A) :
    (sortedDists e P a).Pairwise (· ≤ ·) := by
  have h := List.sorted_mergeSort (le := fun x y : Nat => decide (x ≤ y))
    (by intro x y z hxy hyz; simp only [decide_eq_true_eq] at *; omega)
    (by intro x y; simp only [Bool.or_eq_true, decide_eq_true_eq]; omega)
    (P.map (fun p => e p a))
  exact h.imp (by intro x y hxy; simpa using hxy)

omit [DecidableEq V] in
theorem mult_eq_countP_sorted (e : V → A → Nat) (r : Nat) (P : List V) (a : A) :
    mult e r P a = (sortedDists e P a).countP (fun x => decide (x < r)) := by
  unfold mult sortedDists seen
  rw [(List.mergeSort_perm _ _).countP_eq, List.countP_map]
  rfl

omit [DecidableEq V] in
/-- **Proposition 2 (one atom).**  Atom `a` is seen by at least `k ≥ 1`
contacts exactly when its `k`-th smallest contact distance is below `r`. -/
theorem le_mult_iff (e : V → A → Nat) (r : Nat) (P : List V) (a : A) (k : Nat)
    (hk : 1 ≤ k) :
    k ≤ mult e r P a ↔ ∃ v, kthDist e P a k = some v ∧ v < r := by
  rw [mult_eq_countP_sorted,
    le_countP_lt_iff _ (sortedDists_sorted e P a) k r hk]
  unfold kthDist
  constructor
  · rintro ⟨h, hlt⟩
    exact ⟨_, List.getElem?_eq_getElem h, hlt⟩
  · rintro ⟨v, hv, hlt⟩
    obtain ⟨h, rfl⟩ := List.getElem?_eq_some_iff.1 hv
    exact ⟨h, hlt⟩

omit [DecidableEq V] in
/-- **Proposition 2 (on the mesh).**  Every atom of `Xs` is seen at least
`k ≥ 1` times exactly when the `k`-th smallest contact distance of every atom
of `Xs` is below `r`, that is, when the `k`-th covering radius of `Xs` is
below `r`. -/
theorem cover_iff (e : V → A → Nat) (r : Nat) (P : List V) (Xs : List A)
    (k : Nat) (hk : 1 ≤ k) :
    (∀ a ∈ Xs, k ≤ mult e r P a) ↔
      ∀ a ∈ Xs, ∃ v, kthDist e P a k = some v ∧ v < r := by
  constructor
  · intro h a ha; exact (le_mult_iff e r P a k hk).1 (h a ha)
  · intro h a ha; exact (le_mult_iff e r P a k hk).2 (h a ha)

/-- **Proposition 3 (one atom).**  With distinct contacts, atom `a` is still
seen after every failure of at most `q` contacts exactly when at least `q + 1`
contacts see it.  A failure set is any list `F` of at most `q` contacts. -/
theorem failure_proof (e : V → A → Nat) (r : Nat) (P : List V) (hP : P.Nodup)
    (a : A) (q : Nat) :
    (∀ F : List V, F.length ≤ q → ∃ p ∈ P, p ∉ F ∧ seen e r p a = true) ↔
      q + 1 ≤ mult e r P a := by
  have hC : (P.filter (fun p => seen e r p a)).Nodup := List.Pairwise.filter _ hP
  have hm : mult e r P a = (P.filter (fun p => seen e r p a)).length :=
    List.countP_eq_length_filter
  constructor
  · intro h
    apply Classical.byContradiction
    intro hlt
    have hle : (P.filter (fun p => seen e r p a)).length ≤ q := by omega
    obtain ⟨p, hp, hpF, hs⟩ := h _ hle
    exact hpF (List.mem_filter.2 ⟨hp, hs⟩)
  · intro hq F hF
    apply Classical.byContradiction
    intro hno
    have hsub : ∀ x ∈ P.filter (fun p => seen e r p a), x ∈ F := by
      intro x hx
      obtain ⟨hxP, hxs⟩ := List.mem_filter.1 hx
      apply Classical.byContradiction
      intro hxF
      exact hno ⟨x, hxP, hxF, hxs⟩
    have := length_le_of_nodup_subset hC hsub
    omega

/-- **Proposition 3.**  A zone (a list of atoms) is seen whole after every
failure of at most `q` distinct contacts exactly when every atom of it is seen
at least `q + 1` times; by Proposition 2, when the `(q+1)`-th covering radius
of the zone is below `r`. -/
theorem zone_failure_proof (e : V → A → Nat) (r : Nat) (P : List V)
    (hP : P.Nodup) (Zs : List A) (q : Nat) :
    (∀ F : List V, F.length ≤ q → ∀ a ∈ Zs, ∃ p ∈ P, p ∉ F ∧ seen e r p a = true) ↔
      ∀ a ∈ Zs, q + 1 ≤ mult e r P a := by
  constructor
  · intro h a ha
    exact (failure_proof e r P hP a q).1 (fun F hF => h F hF a ha)
  · intro h F hF a ha
    exact (failure_proof e r P hP a q).2 (h a ha) F hF

/-- **Proposition 5 (site floor).**  Distinct contacts placed at sites of `C`
see every atom no more often than the sites of `C` do. -/
theorem site_floor (e : V → A → Nat) (r : Nat) (P C : List V) (hP : P.Nodup)
    (hPC : ∀ p ∈ P, p ∈ C) (a : A) :
    mult e r P a ≤ mult e r C a := by
  unfold mult
  rw [List.countP_eq_length_filter, List.countP_eq_length_filter]
  apply length_le_of_nodup_subset (List.Pairwise.filter _ hP)
  intro x hx
  obtain ⟨hxP, hxs⟩ := List.mem_filter.1 hx
  exact List.mem_filter.2 ⟨hPC x hxP, hxs⟩

/-- **Proposition 5, k-fold form.**  If fewer than `k` sites of `C` see atom
`a` (the `k`-th site floor at `a` is at least `r`), no array of distinct
contacts on `C` sees `a` `k` times, whatever its number of contacts. -/
theorem site_floor_k (e : V → A → Nat) (r : Nat) (P C : List V) (hP : P.Nodup)
    (hPC : ∀ p ∈ P, p ∈ C) (a : A) (k : Nat) (hC : mult e r C a < k) :
    mult e r P a < k :=
  Nat.lt_of_le_of_lt (site_floor e r P C hP hPC a) hC

end Footprints

section Erosion

variable {A : Type}

/-- **Proposition 4 (on the mesh).**  With `δ c a` the distance between the
centroids of atoms `c` and `a`, the onset zone of radius `s` about `c` (the
atoms of `Xs` within `s` of `c`) consists of good atoms exactly when every bad
atom of `Xs` is farther than `s` from `c`.  Taking "good" to be "seen at least
`k` times" (or "unseen") gives the two capture criteria of the paper. -/
theorem capture_by_erosion (δ : A → A → Nat) (good : A → Bool) (Xs : List A)
    (c : A) (s : Nat) :
    (∀ a ∈ Xs, δ c a ≤ s → good a = true) ↔
      ∀ a ∈ Xs, good a = false → s < δ c a := by
  constructor
  · intro h a ha hg
    apply Classical.byContradiction
    intro hlt
    have := h a ha (by omega)
    rw [hg] at this
    exact Bool.false_ne_true this
  · intro h a ha hle
    cases hg : good a with
    | true => rfl
    | false => have := h a ha hg; omega

end Erosion

section Metric

variable {V : Type} [DecidableEq V]

/-- **Proposition 6 (packing bound).**  If the points of `S` are pairwise at
least `2r` apart and every one of them lies within `r` of some contact of `P`,
then `P` has at least as many contacts as `S` has points. -/
theorem packing (d : V → V → Nat) (hsymm : ∀ x y, d x y = d y x)
    (htri : ∀ x y z, d x z ≤ d x y + d y z) (S P : List V) (r : Nat)
    (hS : S.Nodup) (hsep : ∀ s ∈ S, ∀ s' ∈ S, s ≠ s' → 2 * r ≤ d s s')
    (hcov : ∀ s ∈ S, ∃ p ∈ P, d s p < r) :
    S.length ≤ P.length := by
  apply length_le_of_near (fun s p => d s p < r) S P hS hcov
  intro s hs s' hs' hne p h1 h2
  have := hsep s hs s' hs' hne
  have := htri s p s'
  have := hsymm p s'
  omega

/-- **Proposition 7, the pigeonhole step.**  If `L` holds more points than `Q`,
the points of `L` are pairwise at least `D` apart, and every point of `L` is
within `R` of some point of `Q`, then `D ≤ 2R`.  Applied to the first `n + 1`
farthest-point sites (pairwise at least `ρ(P_n)` apart) and any `n` centres
`Q` (so `R = ρ(Q)`), it gives `ρ(P_n) ≤ 2ρ(Q)`. -/
theorem farthest_point_core (d : V → V → Nat) (hsymm : ∀ x y, d x y = d y x)
    (htri : ∀ x y z, d x z ≤ d x y + d y z) (L Q : List V) (D R : Nat)
    (hL : L.Nodup) (hsep : ∀ x ∈ L, ∀ y ∈ L, x ≠ y → D ≤ d x y)
    (hcov : ∀ x ∈ L, ∃ q ∈ Q, d x q ≤ R) (hlen : Q.length < L.length) :
    D ≤ 2 * R := by
  apply Classical.byContradiction
  intro hlt
  have := length_le_of_near (fun x q => d x q ≤ R) L Q hL hcov (by
    intro x hx y hy hne q h1 h2
    have := hsep x hx y hy hne
    have := htri x q y
    have := hsymm q y
    omega)
  omega

/-- The distance from contact `p` to a triangle with vertex list `vs`: the
distance to its farthest vertex. -/
def triDist (d : V → V → Nat) (p : V) (vs : List V) : Nat :=
  vs.foldr (fun v m => max (d v p) m) 0

omit [DecidableEq V] in
/-- Each vertex is no farther from `p` than the triangle is. -/
theorem le_triDist (d : V → V → Nat) (p : V) :
    ∀ (vs : List V), ∀ v ∈ vs, d v p ≤ triDist d p vs
  | [], v, hv => by simp at hv
  | w :: ws, v, hv => by
    unfold triDist
    simp only [List.foldr_cons]
    rcases List.mem_cons.1 hv with h | h
    · subst h; exact Nat.le_max_left _ _
    · exact Nat.le_trans (le_triDist d p ws v h) (Nat.le_max_right _ _)

omit [DecidableEq V] in
/-- If the vertices of the triangle are pairwise within `l` of each other, the
triangle is at most `l` farther from `p` than any of its vertices: the
per-contact form of `ρ^v ≤ ρ ≤ ρ^v + l`. -/
theorem triDist_le (d : V → V → Nat) (hsymm : ∀ x y, d x y = d y x)
    (htri : ∀ x y z, d x z ≤ d x y + d y z) (p v : V) (l : Nat) :
    ∀ (vs : List V), (∀ w ∈ vs, d w v ≤ l) → triDist d p vs ≤ d v p + l
  | [], _ => by unfold triDist; simp
  | w :: ws, hl => by
    unfold triDist
    simp only [List.foldr_cons]
    have hw := hl w (by simp)
    have ht := htri w v p
    have hrest := triDist_le d hsymm htri p v l ws (fun u hu => hl u (by simp [hu]))
    unfold triDist at hrest
    apply Nat.max_le.2
    constructor <;> omega

end Metric

end KShadowECoG
