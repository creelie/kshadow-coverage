/-
KShadowECoG/Lists.lean -- list facts used by Coverage.lean.

Core Lean only (no Mathlib): a duplicate-free list inside another is no
longer than it; the pigeonhole principle for a proximity relation; and the
statement that in a sorted list of naturals at least `k` entries lie below
`r` exactly when the `k`-th entry does.
-/

namespace KShadowECoG

/-- A duplicate-free list contained in another list is no longer than it. -/
theorem length_le_of_nodup_subset {α : Type} [DecidableEq α] :
    ∀ {l₁ l₂ : List α}, l₁.Nodup → (∀ x ∈ l₁, x ∈ l₂) → l₁.length ≤ l₂.length
  | [], _, _, _ => by simp
  | a :: t, l₂, hnd, hsub => by
    rw [List.nodup_cons] at hnd
    have ha : a ∈ l₂ := hsub a (by simp)
    have ht : ∀ x ∈ t, x ∈ l₂.erase a := by
      intro x hx
      have hxa : x ≠ a := fun h => hnd.1 (h ▸ hx)
      exact (List.mem_erase_of_ne hxa).2 (hsub x (by simp [hx]))
    have ih := length_le_of_nodup_subset hnd.2 ht
    rw [List.length_erase_of_mem ha] at ih
    have := List.length_pos_of_mem ha
    simp only [List.length_cons]
    omega

/-- Pigeonhole with a proximity relation.  If every point of a duplicate-free
list `S` is near some point of `P`, and no point of `P` is near two distinct
points of `S`, then `S` is no longer than `P`. -/
theorem length_le_of_near {α β : Type} [DecidableEq β] (near : α → β → Prop) :
    ∀ (S : List α) (P : List β), S.Nodup →
      (∀ s ∈ S, ∃ p ∈ P, near s p) →
      (∀ s ∈ S, ∀ s' ∈ S, s ≠ s' → ∀ p, near s p → near s' p → False) →
      S.length ≤ P.length
  | [], _, _, _, _ => by simp
  | s :: t, P, hnd, hcov, hsep => by
    rw [List.nodup_cons] at hnd
    obtain ⟨p, hp, hnp⟩ := hcov s (by simp)
    have ht : ∀ s' ∈ t, ∃ p' ∈ P.erase p, near s' p' := by
      intro s' hs'
      obtain ⟨p', hp', hn'⟩ := hcov s' (by simp [hs'])
      refine ⟨p', ?_, hn'⟩
      have hne : p' ≠ p := by
        intro h
        subst h
        have hss : s ≠ s' := fun h => hnd.1 (h ▸ hs')
        exact hsep s (by simp) s' (by simp [hs']) hss p' hnp hn'
      exact (List.mem_erase_of_ne hne).2 hp'
    have hsep' : ∀ a ∈ t, ∀ b ∈ t, a ≠ b → ∀ q, near a q → near b q → False :=
      fun a ha b hb hab q h1 h2 =>
        hsep a (by simp [ha]) b (by simp [hb]) hab q h1 h2
    have ih := length_le_of_near near t (P.erase p) hnd.2 ht hsep'
    rw [List.length_erase_of_mem hp] at ih
    have := List.length_pos_of_mem hp
    simp only [List.length_cons]
    omega

/-- In a sorted list of naturals, at least `k ≥ 1` entries lie below `r`
exactly when the `k`-th entry exists and lies below `r`. -/
theorem le_countP_lt_iff (S : List Nat) (hS : S.Pairwise (· ≤ ·)) (k r : Nat)
    (hk : 1 ≤ k) :
    k ≤ S.countP (fun x => decide (x < r)) ↔ ∃ h : k - 1 < S.length, S[k - 1] < r := by
  have hmono : ∀ (i j : Nat) (hi : i < S.length) (hj : j < S.length),
      i ≤ j → S[i] ≤ S[j] := by
    intro i j hi hj hij
    rcases Nat.lt_or_eq_of_le hij with h | h
    · exact List.pairwise_iff_getElem.1 hS i j hi hj h
    · subst h; exact Nat.le_refl _
  have hsplit := fun m => congrArg (List.countP (fun x => decide (x < r)))
    (List.take_append_drop m S)
  constructor
  · intro hc
    have hlen : k ≤ S.length := Nat.le_trans hc List.countP_le_length
    have h1 : k - 1 < S.length := by omega
    refine ⟨h1, ?_⟩
    apply Classical.byContradiction
    intro hge
    have hge' : r ≤ S[k - 1] := by omega
    have hdrop : (S.drop (k - 1)).countP (fun x => decide (x < r)) = 0 := by
      rw [List.countP_eq_zero]
      intro a ha
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 ha
      rw [List.getElem_drop]
      have hlt : k - 1 + i < S.length := by
        simp only [List.length_drop] at hi; omega
      have := hmono (k - 1) (k - 1 + i) h1 hlt (by omega)
      simp only [decide_eq_true_eq]
      omega
    have htake : (S.take (k - 1)).countP (fun x => decide (x < r)) ≤ k - 1 := by
      have := @List.countP_le_length _ (fun x => decide (x < r)) (S.take (k - 1))
      rw [List.length_take] at this
      omega
    have := hsplit (k - 1)
    rw [List.countP_append] at this
    omega
  · rintro ⟨h1, hlt⟩
    have htake : (S.take k).countP (fun x => decide (x < r)) = (S.take k).length := by
      rw [List.countP_eq_length]
      intro a ha
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 ha
      rw [List.getElem_take]
      have hi' : i < S.length := by simp only [List.length_take] at hi; omega
      have hik : i ≤ k - 1 := by simp only [List.length_take] at hi; omega
      have := hmono i (k - 1) hi' h1 hik
      simp only [decide_eq_true_eq]
      omega
    have hlen : (S.take k).length = k := by rw [List.length_take]; omega
    have := hsplit k
    rw [List.countP_append] at this
    omega

end KShadowECoG
