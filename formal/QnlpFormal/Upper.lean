/-
(formal/README.md) every qubit strategy on the K = 2 real-count game scores ≤ Ubound.
-/
import Mathlib
import QnlpFormal.Basic
import QnlpFormal.UbData

open Matrix ComplexConjugate ComplexOrder

set_option linter.style.longLine false

namespace QnlpFormal

/-- per-query weight of input (a, b) as a real number. -/
noncomputable def wR (a b : Fin 6) : ℝ := ((cnt a * cnt b : ℕ) : ℝ) / 11254344

/-- the quadratic form Σ_x w_x λ_x (y_a − y_b)². -/
noncomputable def Qf (y : Fin 6 → ℝ) : ℝ :=
  ∑ a : Fin 6, ∑ b : Fin 6, if a = b then 0 else wR a b * lamT a b * (y a - y b) ^ 2

theorem cert_id (y : Fin 6 → ℝ) : (∑ v : Fin 6, muT v * y v ^ 2) - Qf y = ldlSq y := by
  simp only [Qf, wR, lamT, muT, ldlSq, cnt, Fin.sum_univ_succ, Fin.sum_univ_zero]
  norm_num
  ring
theorem quad_nonneg {M : Matrix (Fin 2) (Fin 2) ℂ} (h : M.PosSemidef) (x : Fin 2 → ℂ) :
    0 ≤ (star x ⬝ᵥ (M *ᵥ x)).re ∧ (star x ⬝ᵥ (M *ᵥ x)).im = 0 := by
  have := Complex.nonneg_iff.mp (h.dotProduct_mulVec_nonneg x)
  exact ⟨this.1, this.2.symm⟩

/-- entry facts of a 2×2 positive semidefinite complex matrix. -/
theorem psd2 {M : Matrix (Fin 2) (Fin 2) ℂ} (h : M.PosSemidef) :
    (M 0 0).im = 0 ∧ (M 1 1).im = 0 ∧ 0 ≤ (M 0 0).re ∧ 0 ≤ (M 1 1).re ∧
    M 1 0 = conj (M 0 1) ∧
    (M 0 1).re ^ 2 + (M 0 1).im ^ 2 ≤ (M 0 0).re * (M 1 1).re := by
  have hH : M 1 0 = conj (M 0 1) := (h.1.apply 1 0).symm
  have e0 := quad_nonneg h ![1, 0]
  have e1 := quad_nonneg h ![0, 1]
  simp [dotProduct, mulVec, Fin.sum_univ_two] at e0 e1
  set p := (M 0 0).re; set q := (M 1 1).re
  set u := (M 0 1).re; set v := (M 0 1).im
  have hq := quad_nonneg h ![(q : ℂ), -conj (M 0 1)]
  have hp := quad_nonneg h ![-(M 0 1), (p : ℂ)]
  have h1 := quad_nonneg h ![1, -conj (M 0 1)]
  simp [dotProduct, mulVec, Fin.sum_univ_two, hH] at hq hp h1
  refine ⟨e0.2, e1.2, e0.1, e1.1, hH, ?_⟩
  obtain ⟨hp0, hi0⟩ := e0; obtain ⟨hq0, hi1⟩ := e1
  simp only [hi0, hi1] at hq hp h1
  have Hq : 0 ≤ q * (p * q - (u ^ 2 + v ^ 2)) := by linarith [hq.1]
  have Hp : 0 ≤ p * (p * q - (u ^ 2 + v ^ 2)) := by linarith [hp.1]
  have H1 : 0 ≤ p + q * (u ^ 2 + v ^ 2) - 2 * (u ^ 2 + v ^ 2) := by linarith [h1.1]
  rcases hq0.lt_or_eq with hqpos | hqz
  · nlinarith
  rcases hp0.lt_or_eq with hppos | hpz
  · nlinarith
  rw [← hqz, ← hpz] at H1 ⊢; nlinarith [sq_nonneg u, sq_nonneg v]


/-- scalar part e₀ of a two-outcome effect E = e₀ I + e·σ. -/
noncomputable def escal (E : Matrix (Fin 2) (Fin 2) ℂ) : ℝ := ((E 0 0).re + (E 1 1).re) / 2

/-- Bloch part e of an effect (coordinates z, x, y up to sign conventions). -/
noncomputable def ev (E : Matrix (Fin 2) (Fin 2) ℂ) (k : Fin 3) : ℝ :=
  ![((E 0 0).re - (E 1 1).re) / 2, (E 0 1).re, (E 0 1).im] k

/-- 0 ⪯ E ⪯ I forces |e|² ≤ 1/4. -/
theorem effect_norm {E : Matrix (Fin 2) (Fin 2) ℂ} (h1 : E.PosSemidef) (h2 : (1 - E).PosSemidef) :
    ev E 0 ^ 2 + ev E 1 ^ 2 + ev E 2 ^ 2 ≤ 1 / 4 := by
  obtain ⟨-, -, a0, d0, -, hn⟩ := psd2 h1
  obtain ⟨-, -, a1, d1, -, hn'⟩ := psd2 h2
  simp at a1 d1 hn'
  simp only [ev, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.head_cons,
    Matrix.tail_cons]
  rcases le_total ((E 0 0).re + (E 1 1).re) 1 with hs | hs <;> nlinarith

/-- trace of a state against an effect, in coordinates. -/
theorem tr_coord {ρ E : Matrix (Fin 2) (Fin 2) ℂ} (hρ : ρ.PosSemidef) (htr : ρ.trace = 1)
    (hE : E.PosSemidef) :
    (ρ * E).trace.re = escal E + ((2 * (ρ 0 0).re - 1) * ev E 0 + 2 * (ρ 0 1).re * ev E 1 +
      2 * (ρ 0 1).im * ev E 2) := by
  obtain ⟨i0, i1, -, -, h10, -⟩ := psd2 hρ
  obtain ⟨j0, j1, -, -, k10, -⟩ := psd2 hE
  have ht : (ρ 0 0).re + (ρ 1 1).re = 1 := by
    have := congrArg Complex.re htr; simpa [Matrix.trace, Fin.sum_univ_two] using this
  simp only [Matrix.trace, Fin.sum_univ_two, Matrix.diag, Matrix.mul_apply, h10, k10, escal, ev,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons]
  simp only [Complex.add_re, Complex.mul_re, Complex.conj_re, Complex.conj_im, i0, i1, j0, j1]
  have : (ρ 1 1).re = 1 - (ρ 0 0).re := by linarith
  rw [this]; ring

/-- per-input AM-GM bound, for any λ > 0. -/
theorem payoff_le {ρ Ea Eb : Matrix (Fin 2) (Fin 2) ℂ} (hρ : ρ.PosSemidef) (htr : ρ.trace = 1)
    (hEa : Ea.PosSemidef) (hEb : Eb.PosSemidef) {l : ℝ} (hl : 0 < l) :
    ((ρ * Ea).trace + (ρ * (1 - Eb)).trace).re ≤ 1 + escal Ea - escal Eb +
      (l * ((ev Ea 0 - ev Eb 0) ^ 2 + (ev Ea 1 - ev Eb 1) ^ 2 + (ev Ea 2 - ev Eb 2) ^ 2) + 1 / l) / 2 := by
  have h1 : (ρ * (1 - Eb)).trace = 1 - (ρ * Eb).trace := by
    rw [Matrix.mul_sub, Matrix.mul_one, Matrix.trace_sub, htr]
  rw [Complex.add_re, h1, Complex.sub_re, Complex.one_re, tr_coord hρ htr hEa, tr_coord hρ htr hEb]
  obtain ⟨-, -, p0, q0, -, hn⟩ := psd2 hρ
  have ht : (ρ 0 0).re + (ρ 1 1).re = 1 := by
    have := congrArg Complex.re htr; simpa [Matrix.trace, Fin.sum_univ_two] using this
  set r0 := 2 * (ρ 0 0).re - 1; set r1 := 2 * (ρ 0 1).re; set r2 := 2 * (ρ 0 1).im
  have hr : r0 ^ 2 + r1 ^ 2 + r2 ^ 2 ≤ 1 := by
    have : (ρ 1 1).re = 1 - (ρ 0 0).re := by linarith
    rw [this] at hn; simp only [r0, r1, r2]; nlinarith
  set d0 := ev Ea 0 - ev Eb 0; set d1 := ev Ea 1 - ev Eb 1; set d2 := ev Ea 2 - ev Eb 2
  have key : l * (2 * (r0 * d0 + r1 * d1 + r2 * d2)) ≤ l * (l * (d0 ^ 2 + d1 ^ 2 + d2 ^ 2) + 1 / l) := by
    have e1 : l * (l * (d0 ^ 2 + d1 ^ 2 + d2 ^ 2) + 1 / l) = l ^ 2 * (d0 ^ 2 + d1 ^ 2 + d2 ^ 2) + 1 := by
      field_simp
    rw [e1]
    nlinarith [sq_nonneg (l * d0 - r0), sq_nonneg (l * d1 - r1), sq_nonneg (l * d2 - r2)]
  have := le_of_mul_le_mul_left key hl
  have e : r0 * ev Ea 0 + r1 * ev Ea 1 + r2 * ev Ea 2 - (r0 * ev Eb 0 + r1 * ev Eb 1 + r2 * ev Eb 2) =
      r0 * d0 + r1 * d1 + r2 * d2 := by simp only [d0, d1, d2]; ring
  linarith

/-- value of a general qubit strategy: states ρ a b, effects E v ("first"). -/
noncomputable def qvalC (ρ : Fin 6 → Fin 6 → Matrix (Fin 2) (Fin 2) ℂ) (E : Fin 6 → Matrix (Fin 2) (Fin 2) ℂ) : ℝ :=
  ∑ a : Fin 6, ∑ b : Fin 6, if a = b then 0 else
    wR a b * ((ρ a b * E a).trace + (ρ a b * (1 - E b)).trace).re

theorem lam_pos (a b : Fin 6) (h : a ≠ b) : 0 < lamT a b := by
  fin_cases a <;> fin_cases b <;> simp_all [lamT]

theorem Qf_le (y : Fin 6 → ℝ) : Qf y ≤ ∑ v : Fin 6, muT v * y v ^ 2 := by
  have h := cert_id y
  have : 0 ≤ ldlSq y := by unfold ldlSq; positivity
  linarith

/-- T4: every qubit strategy (mixed states, general two-outcome POVMs) scores at most Ubound. -/
theorem qubit_le (ρ : Fin 6 → Fin 6 → Matrix (Fin 2) (Fin 2) ℂ) (E : Fin 6 → Matrix (Fin 2) (Fin 2) ℂ)
    (hρ : ∀ a b, a ≠ b → (ρ a b).PosSemidef ∧ (ρ a b).trace = 1)
    (hE : ∀ v, (E v).PosSemidef ∧ (1 - E v).PosSemidef) : qvalC ρ E ≤ Ubound := by
  have step : qvalC ρ E ≤ ∑ a : Fin 6, ∑ b : Fin 6, if a = b then 0 else wR a b *
      (1 + escal (E a) - escal (E b) + (lamT a b * ((ev (E a) 0 - ev (E b) 0) ^ 2 +
        (ev (E a) 1 - ev (E b) 1) ^ 2 + (ev (E a) 2 - ev (E b) 2) ^ 2) + 1 / lamT a b) / 2) := by
    unfold qvalC
    refine Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun b _ => ?_
    split_ifs with hab
    · exact le_rfl
    · exact mul_le_mul_of_nonneg_left
        (payoff_le (hρ a b hab).1 (hρ a b hab).2 (hE a).1 (hE b).1 (lam_pos a b hab))
        (by unfold wR; positivity)
  refine step.trans ?_
  have q0 := Qf_le (fun v => ev (E v) 0)
  have q1 := Qf_le (fun v => ev (E v) 1)
  have q2 := Qf_le (fun v => ev (E v) 2)
  have n := fun v => effect_norm (hE v).1 (hE v).2
  have m : ∀ v, muT v * (ev (E v) 0 ^ 2 + ev (E v) 1 ^ 2 + ev (E v) 2 ^ 2) ≤ muT v * (1 / 4) := fun v =>
    mul_le_mul_of_nonneg_left (n v) (by fin_cases v <;> norm_num [muT])
  have m0 := m 0; have m1 := m 1; have m2 := m 2; have m3 := m 3; have m4 := m 4; have m5 := m 5
  simp only [Qf, wR, lamT, muT, cnt, Fin.sum_univ_succ, Fin.sum_univ_zero, Ubound] at q0 q1 q2 m0 m1 m2 m3 m4 m5 ⊢
  norm_num at q0 q1 q2 m0 m1 m2 m3 m4 m5 ⊢
  linear_combination (1 / 2 : ℝ) * (q0 + q1 + q2 + m0 + m1 + m2 + m3 + m4 + m5)

end QnlpFormal
