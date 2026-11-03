# K = 2 which-referent game: machine-checked bounds (Lean 4)

This project proves, in Lean 4 with Mathlib, three bounds on the two-referent which-referent game under the real Arabic pro-drop subject counts (9, 3, 1265, 20, 275, 1537 for F-Du, F-Pl, F-Sg, M-Du, M-Pl, M-Sg). The game is described in the top-level README (Arabic axis, exp101 and exp137).

| file | theorem | statement |
|---|---|---|
| `QnlpFormal/Basic.lean` | `mixture_le`, `cval_opt` | every one-bit strategy, including finite mixtures, scores at most 871675/937862 (≈ 0.929428), and one strategy attains it |
| `QnlpFormal/Basic.lean` | `qval_s30`, `m30_proj` | the explicit qubit strategy S30 scores exactly q30 ≈ 0.9580254, with genuine projective measurements |
| `QnlpFormal/Upper.lean` | `qubit_le` | every one-qubit strategy (any state per input, any two-outcome measurement per class, complex entries) scores at most Ubound < 0.9583915 |
| `QnlpFormal/Sandwich.lean` | `sandwich`, `qubit_mixture_le` | the three bounds together, and the quantum bound for finite mixtures |

Every finite computation is checked by the kernel (`decide +kernel`, `norm_num`). There is no `sorry` and no `native_decide`.

## Rebuilding

    lake exe cache get
    lake build

`QnlpFormal/Axioms.lean` prints the axioms of each main theorem; expect only `propext`, `Classical.choice` and `Quot.sound`.

The Python scripts in `scripts/` regenerate the explicit strategy (`gen_s30.py`) and the upper-bound certificate (`gen_ub.py`, which needs cvxpy with the Clarabel solver), and emit the Lean data files. Lean recomputes every value from those files and does not trust the Python output.
