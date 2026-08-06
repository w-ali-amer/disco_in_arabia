# S2 — Story-Mode (DisCoCirc) Port: Design Specification

**Status:** spec approved for build · 2026-08-05
**Depends on:** exp38 (analog-native zxx4 encoder), exp37b (Rydberg compilation),
exp34b (dense DM simulation wall), exp36b/exp36c (volume-law entanglement,
arbitrary-pair; exp36 is the adjacent-pair variant and does not show saturation —
must not be cited for this claim), exp35 (QPU toolchain),
S1 dial bank (in progress).

## Goal

A post-selection-free architecture that reads an Arabic *text* (not a sentence),
tracks its referents on persistent qubit wires, applies frame-invariant composition
blocks as the story unfolds, and answers questions by measurement — runnable on
15–40 qubits of real hardware, in the regime where exact dense density-matrix
simulation is bounded (~14 wires on 16 GB, results_exp34b.json); the honest
classical wall for the C-variant is near ~30 referents and the advantage argument
is an open analysis (trajectory count × cost at target fidelity — see C-variant
note below).

## Why post-selection-free

DisCoCat sentence circuits implement grammar cups by post-selection; acceptance
decays exponentially with cup count, which cancels any hardware scaling story.
The DisCoCirc formulation (precedent: Duneau et al. 2024, trapped-ion QA) compiles
text to *unitaries on referent wires* with a small-readout measurement — shot cost
flat in text length. Everything in this port is unitary + optional reset.

## Architecture

**Registers.** One qubit per referent (cast member). Introductions prepare the
wire with Ry(2·atan2(v₁,v₀)) from the introducing noun's 2D encoded embedding
(same enc() basis as the dial bank). Optional later: +1 sense qubit per
ambiguous referent (exp26/27 machinery).

**Sentences.** A sentence with verb v, subject referent i, object referent j
applies the frame-invariant composition block U_v = CRz(θ₁)·XX(φ₁)
⊗-positioned on (i, j), preceded by the verb's single-qubit word rotations on
the acting wires (from the same solve; **open spec item:** these single-qubit
rotations are currently not fitted by s1c — they must either be added to
solve_idx in s1c or explicitly stated to remain at warmstart values before the
port proceeds to M1). Intransitives: solved 1-qubit block (Rz·Rx, 2 dials).
Adjectives/PPs (later): 1-qubit blocks on their head's wire — the fusion parser
already extracts these.

**Exp39 verdict (results_exp39.json):** The "verb encoder" label has been retired.
Median cross-verb alignment = 98.9% of matched across both zx4 and zxx4 gate
families (verb_label_retired = TRUE; pre-registered threshold 0.95 tripped in
both families). The composition block is frame-invariant, not verb-specific.
Verb-level distinction at the task level remains undemonstrated — see Honest
Risks and the M3 precondition below.

**Order.** Unitary open-wire composition is order-sensitive (exp31's non-conformal
branch) — linguistically correct for stories: reading order now matters, as
rhetoric says it should (التقديم والتأخير).

**Forgetting / evidence channels.** Two variants, built in this order:
 - **U-variant (pure unitary):** no channels. Classically 2^k statevector;
   MPS-attackable in principle, but exp36b/exp36c measured volume-law entanglement
   saturation under arbitrary-pair co-reference with solved dials (S_200 ≈ 7.27
   nats at k=16, growing with k; results_exp36b.json, results_exp36c.json).
   Note: exp36 is the adjacent-pair variant; it does not saturate and must not
   be cited for volume-law claims. This is the first hardware target.
 - **C-variant (channels):** mid-text discard/reset of exited referents and
   exp26-style sense-evidence accumulation (measure-and-forget). Exact dense
   density-matrix simulation cost 4^k: measured wall ~14 wires on 16 GB
   (results_exp34b.json; mixed-state slope 2.67, last measured point k=13 at
   ~7.7 s per step). However, C-variant channels are quantum-trajectory-simulable
   at ~2^k statevector cost per trajectory; the honest classical wall is near
   ~30 referents and depends on trajectory count × cost at target fidelity —
   **this is an open analysis, not a settled fact.** The advantage argument must
   rest on this ratio, and any external write-up must present it as such.
   On hardware, discard = qubit reset — free. Simulate exactly only ≤13 wires.

**Readout.** Questions as measurements:
 - sense question ("which قطع is this?") — measure the sense/referent wire in the
   solved sense basis (rotate-then-Z);
 - agent/plausibility question — measure along the animacy axis;
 - who-did-what QA — compare basis-measurement statistics between candidate
   referent wires (Duneau-style yes/no framing to keep readout 1–2 qubits).

**QA decision protocol.** Decision statistic: same post-selected evidence-alignment
score as exp34a. Threshold chosen on a held-out calibration set of 20 stories.
Shot budget: 10k per circuit (matching exp35). Success criterion: QA accuracy >
0.5, binomial p < 0.05 on ≥ 50 test items.

## Task & data

Synthetic Arabic story generator (M2): K referents (K = 4…30), T sentences
(T = 5…60), drawn from the S1 dial bank's verb inventory with role-consistent
casting; each story paired with ground-truth QA whose answer requires
accumulating ≥2 sentences of evidence (single-sentence-answerable items are
discarded — that's the control class). Balanced yes/no answers. The generator
also emits the co-reference density knob (path-graph ↔ dense), because
exp36b/exp36c demonstrated that knob controls classical approximability via
entanglement saturation under arbitrary-pair co-reference.

## Validation plan

1. **M0.5 single-block gate (precondition for M1):** Single-block U_v =
   CRz(θ₁)·XX(φ₁) must reproduce the two-block solve's evidence direction on all
   frames (alignment ≥ 0.99) before the port proceeds — otherwise both blocks are
   compiled. This gate is directly motivated by exp37b block-2 fidelities
   (F≈1, results_exp37b.json; F_block2 ≥ 0.999963 for all six verbs).
2. **M1 compiler correctness:** text → circuit via the fusion parser; every
   compiled story ≤14 wires cross-checked against exact statevector semantics
   (same ~1e-9 bar as the actual exp35 verification threshold).
3. **M3 accuracy:** QA accuracy vs (a) chance, (b) classical bag-of-vectors
   (blind to structure), (c) classical structured baseline given the same
   parse + dial bank (parity expected — the honest ceiling), at K = 4…12.
   Decision statistic, shot budget, and success criterion: see QA decision
   protocol above. **Precondition (blocked until resolved):** this milestone is
   blocked until a mechanism distinguishes verbs beyond the shared-frame
   machinery. Exp39 verdict: dials interchangeable at 98.9% median cross-verb
   alignment (both gate families); S1 dry-run swap gap = 0.034;
   dials_carry_task_signal_strict = false (right-tail p = 0.083; a derangement
   where no verb receives its own dial ties the original assignment exactly).
4. **Entanglement audit:** measured bond-dimension growth per story config —
   confirms which configs are genuinely MPS-hard before hardware money is spent.
   Cite exp36b/exp36c for saturation evidence; do NOT cite exp36 (adjacent-pair
   variant, does not saturate).
5. **M4 hardware:** U-variant at K = 15–20 on IBM Heron (free tier),
   ratio-scored readout as exp35; C-variant / higher K on Quantinuum H2
   (the Ilyas ask) and Pasqal analog (the Reem ask — verb blocks already compile
   at F≈1, results_exp37b.json; whole-story analog needs the multi-pair geometric
   packing solved, which is an optimization problem, not new physics).

## Milestones

- **M0** this spec (done).
- **M0.5** single-block validation gate: U_v single-block alignment ≥ 0.99 on all
  frames before proceeding (see Validation plan §1; motivated by exp37b
  F_block2 ≥ 0.999963).
- **M1** compiler: story text → referent circuit (fusion parser + exact-match
  coreference linker first; pronoun linking later). Resolve verb single-qubit
  rotation source (solve_idx vs warmstart) before starting.
- **M2** story generator + QA datasets (needs S1 bank for vocabulary).
- **M3** simulation campaign ≤14 wires: accuracy + entanglement audit.
  **Blocked** pending demonstration of verb-level signal beyond the frame-invariant
  composition block (exp39 precondition; see Validation plan §3 and Honest Risks).
- **M4** first hardware story run (15–20 qubits, IBM free tier).

## Honest risks

- **Coreference** in real text is hard; the generator sidesteps it, real-text
  demos will need CAMeL-based linking — scope-boxed to exact-match initially.
- **Parity ceiling stands:** at simulable sizes the classical structured baseline
  will match QA accuracy; the claim is native execution + the scaling wall, and
  every write-up must say so.
- **U-variant simulability:** 2^k statevector reaches k≈30 on HPC; the past-the-
  wall claim belongs to the C-variant and to entanglement-certified U-configs
  only. Do not blur this line in any external material.
- **S1 dependency:** vocabulary breadth and dial quality gate story realism;
  if S1's real-corpus LOO validation fails (<0.8), frame-invariance on real text
  is falsified and the port pauses for diagnosis.
- **Exp39 verdict — frame-invariant block (results_exp39.json):** The "verb
  encoder" label has been retired (verb_label_retired = TRUE; median cross-verb
  alignment = 98.9% of matched, both zx4 and zxx4 gate families, pre-registered
  threshold 0.95). The composition block U_v is frame-invariant, not
  verb-specific. Additionally, dials_carry_task_signal_strict = false: under the
  strict companion criterion (right-tail p = 0.083), a derangement where no verb
  receives its own dial ties the original task score exactly, meaning the
  pre-registered permissive pass (dials_carry_task_signal = true) holds only at
  the boundary via tie-at-max. The M3/M-QA milestones are blocked until a
  mechanism distinguishes verbs beyond the shared-frame machinery (exp39: dials
  interchangeable at 98.9%; S1 dry-run swap gap 0.034).
