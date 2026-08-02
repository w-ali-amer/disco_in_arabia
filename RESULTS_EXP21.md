# Exp21 series — trained semantic geometry: final accounting (2026-08-03)

## Question
Can composition fixed by grammar + a tiny trained lexical encoder produce
sentence-level semantic geometry that generalizes from very little data?

## Designs
- **21a**: 546 free symbol values, SPSA, 36/12/12 pair split.
- **21b-lite**: 91-parameter linear encoder (per-type PCA-6 features → circuit
  parameters), initialized by least squares to the exact-AE point, same split
  protocol, SPSA 6000 iters, model selection on val. Five independent splits
  (seeds 42–46). Classical controls at matched data (C1 bag-of-vectors,
  C2 144-parameter linear metric on order-sensitive concatenation, same loss,
  same optimizer, same splits).

## Results

### 21a (capacity control)
Train AUC 0.9986 (p=1e-4) — the composition CAN hold near-perfect twin
geometry. Held-out AUC 0.496 — unconstrained training memorizes.
Noun-semantics correlation destroyed (rho 0.083 → −0.005).

### 21b-lite, held-out test AUC (twin vs same-order), five splits
| split | PRE | POST | perm p (POST) |
|-------|------|------|----------------|
| 42 | 0.504 | 0.693 | 0.017 |
| 43 | 0.606 | 0.575 | 0.203 |
| 44 | 0.608 | 0.733 | 0.004 |
| 45 | 0.619 | 0.529 | 0.377 |
| 46 | 0.515 | 0.551 | 0.287 |
| **mean ± sd** | 0.570 ± 0.052 | **0.616 ± 0.077** | 2/5 significant |

Mean improvement +0.046 AUC. Train AUC 0.85–0.96 on all splits (learning
happens); transfer is small, split-dependent, and not consistently
significant. Order-discrimination (CV) drifts 0.642 → 0.525–0.650 depending
on split (mild degradation on some). Word-level cosine alignment is not
preserved on any split.

### Classical controls (same 36 training pairs, same loss, 144 params)
C2 held-out meaning AUC: **0.960 / 0.994 / 0.989 / 0.982** (seeds 43–46).
C1 bag-of-vectors: ~1.0 by construction (order-blind).

## Verdict (honest)
1. Split 42 was a favourable draw. Across five splits the trained quantum
   encoder shows at best a **small, fragile transfer effect** (mean +0.05
   AUC, 2/5 splits significant) — not a robust result, not publishable as a
   positive on its own.
2. **The matched classical control decisively wins the meaning task**
   (≈0.98 vs ≈0.62). At 3–4 qubits with post-selected sentence states, the
   quantum pipeline offers no advantage on semantic geometry. This is now
   measured, not suspected.
3. Combined with exp15c/17/18/19/20: the semantic-geometry program closes
   with a complete characterization — untrained: nothing (all ansatze, all
   encoders incl. provably exact AE); unconstrained trained: memorization;
   constrained trained: marginal fragile transfer, classically dominated.

## What survives untouched
The structural/topological program (L0 theorem, 64.9% entanglement result,
depth-stabilization effect with shuffle control), the analog resonance
window (V/Ω≈1, AUC 0.815 in device-model simulation), hardware access
(IBM Herons; Pasqal FRESNEL_SA1 via region="sa"), and the paper. None of
these depended on semantic geometry.

## Where this points
- Journal version gains a rigorous "limits of semantic geometry in
  near-term compositional QNLP" section — anti-hype, fully controlled,
  first of its kind for any language.
- Hardware stories proceed on the ROBUST results: L0/L1 architectural
  theorem on IBM; analog geometry experiment toward FRESNEL_SA1.
- The open theoretical door: derive which compositional structures provably
  transport word similarity (transport-theory track) — a math question,
  not a data question.
