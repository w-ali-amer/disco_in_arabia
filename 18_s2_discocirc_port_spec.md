# S2 — Story-Mode (DisCoCirc) Port: Design Specification

**Status:** spec approved for build · 2026-08-05
**Depends on:** exp38 (analog-native zxx4 encoder), exp37b (Rydberg compilation),
exp34b (4^k wall), exp36 (volume-law entanglement), exp35 (QPU toolchain),
S1 dial bank (in progress).

## Goal

A post-selection-free architecture that reads an Arabic *text* (not a sentence),
tracks its referents on persistent qubit wires, applies solved verb blocks as the
story unfolds, and answers questions by measurement — runnable on 15–40 qubits of
real hardware, in the regime where exact classical simulation of the
channel-bearing variant is measurably impossible (>~15 referents).

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
applies the solved block U_v = CRz(θ₁)·XX(φ₁) ⊗-positioned on (i, j), preceded by
the verb's single-qubit word rotations on the acting wires (from the same solve).
Intransitives: solved 1-qubit block (Rz·Rx, 2 dials). Adjectives/PPs (later):
1-qubit blocks on their head's wire — the fusion parser already extracts these.

**Order.** Unitary open-wire composition is order-sensitive (exp31's non-conformal
branch) — linguistically correct for stories: reading order now matters, as
rhetoric says it should (التقديم والتأخير).

**Forgetting / evidence channels.** Two variants, built in this order:
 - **U-variant (pure unitary):** no channels. Classically 2^k statevector;
   MPS-attackable in principle, but exp36 measured volume-law entanglement under
   arbitrary-pair co-reference with solved dials (bond ≈155 at k=16 → ≈10⁴–10⁶
   at k=30–40). This is the first hardware target.
 - **C-variant (channels):** mid-text discard/reset of exited referents and
   exp26-style sense-evidence accumulation (measure-and-forget). Classical cost
   4^k (the measured wall at ~15 referents). On hardware, discard = qubit reset —
   free. This is the advantage-argument variant; simulate exactly only ≤13 wires.

**Readout.** Questions as measurements:
 - sense question ("which قطع is this?") — measure the sense/referent wire in the
   solved sense basis (rotate-then-Z);
 - agent/plausibility question — measure along the animacy axis;
 - who-did-what QA — compare basis-measurement statistics between candidate
   referent wires (Duneau-style yes/no framing to keep readout 1–2 qubits).

## Task & data

Synthetic Arabic story generator (M2): K referents (K = 4…30), T sentences
(T = 5…60), drawn from the S1 dial bank's verb inventory with role-consistent
casting; each story paired with ground-truth QA whose answer requires
accumulating ≥2 sentences of evidence (single-sentence-answerable items are
discarded — that's the control class). Balanced yes/no answers. The generator
also emits the co-reference density knob (path-graph ↔ dense), because exp36
proved that knob controls classical approximability.

## Validation plan

1. **M1 compiler correctness:** text → circuit via the fusion parser; every
   compiled story ≤14 wires cross-checked against exact statevector semantics
   (same 1e-16 bar as exp35 export).
2. **M3 accuracy:** QA accuracy vs (a) chance, (b) classical bag-of-vectors
   (blind to structure), (c) classical structured baseline given the same
   parse + dial bank (parity expected — the honest ceiling), at K = 4…12.
3. **Entanglement audit:** measured bond-dimension growth per story config —
   confirms which configs are genuinely MPS-hard before hardware money is spent.
4. **M4 hardware:** U-variant at K = 15–20 on IBM Heron (free tier),
   ratio-scored readout as exp35; C-variant / higher K on Quantinuum H2
   (the Ilyas ask) and Pasqal analog (the Reem ask — verb blocks already
   compile at F = 1.000000; whole-story analog needs the multi-pair geometric
   packing solved, which is an optimization problem, not new physics).

## Milestones

- **M0** this spec (done).
- **M1** compiler: story text → referent circuit (fusion parser + exact-match
  coreference linker first; pronoun linking later).
- **M2** story generator + QA datasets (needs S1 bank for vocabulary).
- **M3** simulation campaign ≤14 wires: accuracy + entanglement audit.
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
