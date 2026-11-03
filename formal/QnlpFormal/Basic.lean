/-
(formal/README.md) the two-item which-referent game under the PADT pro-drop subject counts.
Classes F-Du, F-Pl, F-Sg, M-Du, M-Pl, M-Sg with counts (9, 3, 1265, 20, 275, 1537).
An input is an ordered pair (a, b) of distinct classes, drawn with probability ∝ cnt a * cnt b; the reader is
told one of the two classes (each with probability 1/2) and must say whether it came first or second.

T1  every one-bit classical strategy, deterministic or a finite mixture, scores ≤ 871675/937862 (attained);
T2  an explicit one-qubit strategy with orthogonal-projector measurements scores q30 (exact rational);
T3  q30 > 871675/937862.
-/
import Mathlib
import QnlpFormal.Data

open Matrix

set_option linter.style.longLine false

namespace QnlpFormal

/-- PADT pro-drop subject counts: F-Du, F-Pl, F-Sg, M-Du, M-Pl, M-Sg. -/
def cnt : Fin 6 → ℕ := ![9, 3, 1265, 20, 275, 1537]

/-- normaliser: twoZ = 2 * Σ_{a ≠ b} cnt a * cnt b; input (a, b) and either query has probability cnt a * cnt b / twoZ. -/
def twoZ : ℕ := 2 * ∑ a : Fin 6, ∑ b : Fin 6, if a = b then 0 else cnt a * cnt b

theorem twoZ_eq : twoZ = 11254344 := by decide +kernel

/-! ## Classical side -/

/-- correct answers of decoder labelling `l` on input (a, b): class a must get `false` (first),
class b must get `true` (second). -/
def hits (l : Fin 6 → Bool) (a b : Fin 6) : ℕ :=
  (if l a = false then 1 else 0) + (if l b = true then 1 else 0)

/-- a deterministic one-bit strategy: encoder `e a b` (the bit sent) and decoders `d s` (one labelling per bit). -/
def cvalN (e : Fin 6 → Fin 6 → Bool) (d : Bool → Fin 6 → Bool) : ℕ :=
  ∑ a : Fin 6, ∑ b : Fin 6, if a = b then 0 else cnt a * cnt b * hits (d (e a b)) a b

/-- success probability of a deterministic one-bit strategy. -/
def cval (e : Fin 6 → Fin 6 → Bool) (d : Bool → Fin 6 → Bool) : ℚ := (cvalN e d : ℚ) / twoZ

/-- best possible encoder for fixed decoders: per input, the better of the two messages. -/
def bestN (l0 l1 : Fin 6 → Bool) : ℕ :=
  ∑ a : Fin 6, ∑ b : Fin 6, if a = b then 0 else cnt a * cnt b * max (hits l0 a b) (hits l1 a b)

/-- Lemma (i): for fixed decoders no encoder beats choosing the better message per input. -/
theorem cvalN_le_bestN (e : Fin 6 → Fin 6 → Bool) (d : Bool → Fin 6 → Bool) :
    cvalN e d ≤ bestN (d false) (d true) := by
  unfold cvalN bestN
  refine Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun b _ => ?_
  split_ifs
  · exact le_rfl
  · apply Nat.mul_le_mul_left
    cases e a b
    · exact le_max_left _ _
    · exact le_max_right _ _

/-- The finite check: all 64 × 64 decoder pairs, evaluated by the kernel. -/
theorem bestN_le_bits : ∀ b0 b1 b2 b3 b4 b5 c0 c1 c2 c3 c4 c5 : Bool,
    bestN ![b0, b1, b2, b3, b4, b5] ![c0, c1, c2, c3, c4, c5] ≤ 10460100 := by
  decide +kernel

theorem vec6 (l : Fin 6 → Bool) : l = ![l 0, l 1, l 2, l 3, l 4, l 5] := by
  funext i; fin_cases i <;> rfl

theorem bestN_le (l0 l1 : Fin 6 → Bool) : bestN l0 l1 ≤ 10460100 := by
  rw [vec6 l0, vec6 l1]; exact bestN_le_bits _ _ _ _ _ _ _ _ _ _ _ _

/-- the classical optimum. -/
def cOpt : ℚ := 871675 / 937862

/-- T1, deterministic part. -/
theorem cval_le (e : Fin 6 → Fin 6 → Bool) (d : Bool → Fin 6 → Bool) : cval e d ≤ cOpt := by
  have h : (cvalN e d : ℚ) ≤ 10460100 := by
    exact_mod_cast (cvalN_le_bestN e d).trans (bestN_le _ _)
  unfold cval cOpt; rw [twoZ_eq, div_le_iff₀ (by norm_num)]
  push_cast; linarith

/-- T1, shared randomness: a finite mixture of deterministic strategies (Lemma (ii)). -/
theorem mixture_le {ι : Type*} (s : Finset ι) (q : ι → ℚ) (hq : ∀ i ∈ s, 0 ≤ q i) (h1 : ∑ i ∈ s, q i = 1)
    (e : ι → Fin 6 → Fin 6 → Bool) (d : ι → Bool → Fin 6 → Bool) :
    ∑ i ∈ s, q i * cval (e i) (d i) ≤ cOpt := by
  calc ∑ i ∈ s, q i * cval (e i) (d i) ≤ ∑ i ∈ s, q i * cOpt :=
        Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (cval_le _ _) (hq i hi)
    _ = cOpt := by rw [← Finset.sum_mul, h1, one_mul]

/-- T1, attainment. -/
theorem cval_opt : cval enc0 dec0 = cOpt := by
  have h : cvalN enc0 dec0 = 10460100 := by decide +kernel
  unfold cval cOpt; rw [h, twoZ_eq]; norm_num

/-! ## Quantum side: one real qubit, two-outcome projective measurements -/

def dot (u v : Fin 2 → ℚ) : ℚ := u 0 * v 0 + u 1 * v 1

/-- orthogonal projector onto the line through `m`. -/
def proj (m : Fin 2 → ℚ) : Matrix (Fin 2) (Fin 2) ℚ := fun i j => m i * m j / dot m m

theorem proj_symm (m : Fin 2 → ℚ) : (proj m)ᵀ = proj m := by
  ext i j; simp [proj, mul_comm]

theorem proj_idem (m : Fin 2 → ℚ) (h : dot m m ≠ 0) : proj m * proj m = proj m := by
  ext i j
  rw [Matrix.mul_apply, Fin.sum_univ_two]
  simp only [proj]
  unfold dot at h ⊢
  field_simp

theorem proj_complete (m : Fin 2 → ℚ) : proj m + (1 - proj m) = 1 := by abel

/-- Born rule for a pure (unnormalised) state ψ and effect E: ⟨ψ|E|ψ⟩ / ⟨ψ|ψ⟩. -/
def born (E : Matrix (Fin 2) (Fin 2) ℚ) (ψ : Fin 2 → ℚ) : ℚ := dot ψ (E *ᵥ ψ) / dot ψ ψ

/-- success probability of a one-qubit strategy: on input (a, b) the state is ψ a b; class v is read by the
measurement {proj (m v) ↦ "first", 1 - proj (m v) ↦ "second"}. -/
def qval (ψ : Fin 6 → Fin 6 → Fin 2 → ℚ) (m : Fin 6 → Fin 2 → ℚ) : ℚ :=
  ∑ a : Fin 6, ∑ b : Fin 6, if a = b then 0 else
    ((cnt a * cnt b : ℕ) : ℚ) / twoZ * (born (proj (m a)) (ψ a b) + born (1 - proj (m b)) (ψ a b))

/-- the explicit strategy's measurements are genuine orthogonal projectors. -/
theorem m30_proj (v : Fin 6) :
    (proj (m30 v))ᵀ = proj (m30 v) ∧ proj (m30 v) * proj (m30 v) = proj (m30 v) ∧
    proj (m30 v) + (1 - proj (m30 v)) = 1 := by
  refine ⟨proj_symm _, proj_idem _ ?_, proj_complete _⟩
  fin_cases v <;> simp [dot, m30] <;> norm_num

/-- the states are non-zero vectors (diagonal entries are unused but non-zero too). -/
theorem psi30_ne (a b : Fin 6) : dot (psi30 a b) (psi30 a b) ≠ 0 := by
  revert a b; decide +kernel

def q30 : ℚ := 7688863504347166840759984627 / 8025740657507907428642070900

/-- T2: the exact value of S30. -/
theorem qval_s30 : qval psi30 m30 = q30 := by decide +kernel

/-- T3. -/
theorem q30_gt : cOpt < q30 := by unfold cOpt q30; norm_num

/-- Headline: the explicit one-qubit strategy beats every one-bit classical strategy, including any finite
mixture (shared randomness), under the real pro-drop counts. -/
theorem quantum_beats_classical {ι : Type*} (s : Finset ι) (q : ι → ℚ) (hq : ∀ i ∈ s, 0 ≤ q i)
    (h1 : ∑ i ∈ s, q i = 1) (e : ι → Fin 6 → Fin 6 → Bool) (d : ι → Bool → Fin 6 → Bool) :
    ∑ i ∈ s, q i * cval (e i) (d i) < qval psi30 m30 := by
  rw [qval_s30]; exact (mixture_le s q hq h1 e d).trans_lt q30_gt

end QnlpFormal
