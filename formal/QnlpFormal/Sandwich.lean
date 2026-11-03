/-
doc 141: the formal sandwich. Doc 140's real strategy S30 is an instance of the general qubit model of
`Upper.lean`, so classical optimum < q30 = value of S30 ≤ Ubound, and every qubit strategy is ≤ Ubound.
-/
import Mathlib
import QnlpFormal.Basic
import QnlpFormal.Upper

open Matrix ComplexConjugate ComplexOrder

set_option linter.style.longLine false

namespace QnlpFormal

/-- a rational matrix viewed as a complex one. -/
def cplx (A : Matrix (Fin 2) (Fin 2) ℚ) : Matrix (Fin 2) (Fin 2) ℂ := fun i j => ((A i j : ℚ) : ℂ)

/-- a symmetric rational 2×2 matrix with nonnegative diagonal and A₀₁² ≤ A₀₀A₁₁ is PSD over ℂ. -/
theorem psd_of_real (A : Matrix (Fin 2) (Fin 2) ℚ) (hs : A 1 0 = A 0 1) (h0 : 0 ≤ A 0 0) (h1 : 0 ≤ A 1 1)
    (hd : A 0 1 ^ 2 ≤ A 0 0 * A 1 1) : (cplx A).PosSemidef := by
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun x => ?_
  · ext i j; fin_cases i <;> fin_cases j <;> simp [cplx, hs]
  · rw [Complex.nonneg_iff]
    simp only [dotProduct, mulVec, Fin.sum_univ_two, cplx, Pi.star_apply, Complex.add_re, Complex.mul_re,
      Complex.add_im, Complex.mul_im, Complex.star_def, Complex.conj_re, Complex.conj_im,
      Complex.ratCast_re, Complex.ratCast_im, hs]
    have a0 : (0 : ℝ) ≤ (A 0 0 : ℝ) := by exact_mod_cast h0
    have a1 : (0 : ℝ) ≤ (A 1 1 : ℝ) := by exact_mod_cast h1
    have ad : ((A 0 1 : ℝ)) ^ 2 ≤ (A 0 0 : ℝ) * (A 1 1 : ℝ) := by exact_mod_cast hd
    set p : ℝ := ((A 0 0 : ℚ) : ℝ); set q : ℝ := ((A 1 1 : ℚ) : ℝ); set c : ℝ := ((A 0 1 : ℚ) : ℝ)
    set a := (x 0).re; set b := (x 0).im; set u := (x 1).re; set v := (x 1).im
    refine ⟨?_, by ring⟩
    rcases a0.lt_or_eq with hp | hp
    · have : 0 ≤ p * (p * (a * a + b * b) + q * (u * u + v * v) + 2 * c * (a * u + b * v)) := by
        nlinarith [sq_nonneg (p * a + c * u), sq_nonneg (p * b + c * v), sq_nonneg u, sq_nonneg v]
      have := (mul_nonneg_iff_of_pos_left hp).mp this
      nlinarith
    · have hc : c = 0 := by rw [← hp] at ad; nlinarith [sq_nonneg c]
      rw [hc, ← hp]; nlinarith [sq_nonneg u, sq_nonneg v]

/-- the pure state ψψᵀ/(ψ·ψ) as a rational matrix. -/
def rhoQ (ψ : Fin 2 → ℚ) : Matrix (Fin 2) (Fin 2) ℚ := fun i j => ψ i * ψ j / dot ψ ψ

theorem rank1_psd (m : Fin 2 → ℚ) (h : dot m m ≠ 0) : (cplx (proj m)).PosSemidef := by
  have hn : 0 < dot m m :=
    lt_of_le_of_ne (by unfold dot; nlinarith [mul_self_nonneg (m 0), mul_self_nonneg (m 1)]) (Ne.symm h)
  apply psd_of_real <;> simp only [proj]
  · ring
  · exact div_nonneg (mul_self_nonneg _) hn.le
  · exact div_nonneg (mul_self_nonneg _) hn.le
  · apply le_of_eq; field_simp

theorem rhoQ_eq (ψ : Fin 2 → ℚ) : rhoQ ψ = proj ψ := rfl

theorem compl_psd (m : Fin 2 → ℚ) (h : dot m m ≠ 0) : (1 - cplx (proj m)).PosSemidef := by
  have hq : 1 - proj m = proj ![-m 1, m 0] := by
    unfold dot at h
    have h' : m 0 ^ 2 + m 1 ^ 2 ≠ 0 := by simpa [sq] using h
    have h'' : m 1 ^ 2 + m 0 ^ 2 ≠ 0 := by rwa [add_comm]
    ext i j; fin_cases i <;> fin_cases j <;> simp [proj, dot] <;> field_simp <;> ring
  have : 1 - cplx (proj m) = cplx (proj ![-m 1, m 0]) := by
    rw [← hq]; ext i j; fin_cases i <;> fin_cases j <;> simp [cplx, Matrix.one_apply]
  rw [this]; apply rank1_psd; simp only [dot] at h ⊢; simp; intro h'; apply h; linarith

theorem trace_rho (ψ : Fin 2 → ℚ) (h : dot ψ ψ ≠ 0) : (cplx (rhoQ ψ)).trace = 1 := by
  simp only [Matrix.trace, Fin.sum_univ_two, Matrix.diag, cplx, rhoQ]
  unfold dot at h ⊢; push_cast; rw [← add_div]; exact div_self (by exact_mod_cast h)

theorem trace_cplx (A B : Matrix (Fin 2) (Fin 2) ℚ) : (cplx A * cplx B).trace = (((A * B).trace : ℚ) : ℂ) := by
  simp only [Matrix.trace, Fin.sum_univ_two, Matrix.diag, Matrix.mul_apply, cplx]; push_cast; ring

theorem one_sub_cplx (P : Matrix (Fin 2) (Fin 2) ℚ) : 1 - cplx P = cplx (1 - P) := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [cplx, Matrix.one_apply]

/-- for a symmetric E, tr(ρ_ψ E) is the Born value of doc 140. -/
theorem trace_rho_born (ψ : Fin 2 → ℚ) (E : Matrix (Fin 2) (Fin 2) ℚ) (hE : E 1 0 = E 0 1) (h : dot ψ ψ ≠ 0) :
    (rhoQ ψ * E).trace = born E ψ := by
  simp only [Matrix.trace, Fin.sum_univ_two, Matrix.diag, Matrix.mul_apply, rhoQ, born, dot, mulVec, dotProduct,
    hE]
  unfold dot at h; field_simp

/-- the payoff of a real strategy equals the Born-rule payoff of doc 140. -/
theorem payoff_real (ψ ma mb : Fin 2 → ℚ) (h : dot ψ ψ ≠ 0) :
    ((cplx (rhoQ ψ) * cplx (proj ma)).trace + (cplx (rhoQ ψ) * (1 - cplx (proj mb))).trace).re =
      ((born (proj ma) ψ + born (1 - proj mb) ψ : ℚ) : ℝ) := by
  rw [one_sub_cplx, trace_cplx, trace_cplx, trace_rho_born ψ _ (by simp [proj]; ring) h,
    trace_rho_born ψ _ (by simp [proj, Matrix.one_apply]; ring) h]
  push_cast; simp

/-- doc 140's strategy S30, written in the general qubit model. -/
def rho30 (a b : Fin 6) : Matrix (Fin 2) (Fin 2) ℂ := cplx (rhoQ (psi30 a b))
def E30 (v : Fin 6) : Matrix (Fin 2) (Fin 2) ℂ := cplx (proj (m30 v))

theorem m30_ne (v : Fin 6) : dot (m30 v) (m30 v) ≠ 0 := by
  fin_cases v <;> simp [dot, m30] <;> norm_num

theorem qval_eq_qvalC : ((qval psi30 m30 : ℚ) : ℝ) = qvalC rho30 E30 := by
  unfold qval qvalC
  push_cast
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
  split_ifs
  · simp
  · rw [rho30, E30, E30, payoff_real _ _ _ (psi30_ne a b), wR, twoZ_eq]; push_cast; ring

/-- S30 satisfies the hypotheses of T4. -/
theorem s30_valid : (∀ a b, a ≠ b → (rho30 a b).PosSemidef ∧ (rho30 a b).trace = 1) ∧
    ∀ v, (E30 v).PosSemidef ∧ (1 - E30 v).PosSemidef :=
  ⟨fun a b _ => ⟨rank1_psd _ (psi30_ne a b), trace_rho _ (psi30_ne a b)⟩,
    fun v => ⟨rank1_psd _ (m30_ne v), compl_psd _ (m30_ne v)⟩⟩

/-- Headline (doc 141): the formal sandwich.
every one-bit strategy (incl. finite mixtures) ≤ 871675/937862 < q30 = value of S30 ≤ Ubound,
and every qubit strategy (mixed states, general two-outcome POVMs) scores ≤ Ubound. -/
theorem sandwich :
    cOpt < q30 ∧ ((q30 : ℚ) : ℝ) ≤ Ubound ∧
    ∀ (ρ : Fin 6 → Fin 6 → Matrix (Fin 2) (Fin 2) ℂ) (E : Fin 6 → Matrix (Fin 2) (Fin 2) ℂ),
      (∀ a b, a ≠ b → (ρ a b).PosSemidef ∧ (ρ a b).trace = 1) →
      (∀ v, (E v).PosSemidef ∧ (1 - E v).PosSemidef) → qvalC ρ E ≤ Ubound := by
  refine ⟨q30_gt, ?_, fun ρ E hρ hE => qubit_le ρ E hρ hE⟩
  rw [← qval_s30, qval_eq_qvalC]
  exact qubit_le _ _ s30_valid.1 s30_valid.2

/-- the bound itself, numerically: Ubound < 0.9583915. -/
theorem Ubound_lt : Ubound < 9583915 / 10000000 := by unfold Ubound; norm_num

/-- T4 with shared randomness: a finite mixture of qubit strategies also scores ≤ Ubound. -/
theorem qubit_mixture_le {ι : Type*} (s : Finset ι) (q : ι → ℝ) (hq : ∀ i ∈ s, 0 ≤ q i) (h1 : ∑ i ∈ s, q i = 1)
    (ρ : ι → Fin 6 → Fin 6 → Matrix (Fin 2) (Fin 2) ℂ) (E : ι → Fin 6 → Matrix (Fin 2) (Fin 2) ℂ)
    (hρ : ∀ i, ∀ a b, a ≠ b → (ρ i a b).PosSemidef ∧ (ρ i a b).trace = 1)
    (hE : ∀ i v, (E i v).PosSemidef ∧ (1 - E i v).PosSemidef) :
    ∑ i ∈ s, q i * qvalC (ρ i) (E i) ≤ Ubound := by
  calc ∑ i ∈ s, q i * qvalC (ρ i) (E i) ≤ ∑ i ∈ s, q i * Ubound :=
        Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (qubit_le _ _ (hρ i) (hE i)) (hq i hi)
    _ = Ubound := by rw [← Finset.sum_mul, h1, one_mul]

end QnlpFormal
