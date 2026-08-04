# Exp27: DisCoCirc referent-wire skeleton (design, 2026-08-04)

## What it is
The missing layer of the original vision: text = referent wires persisting
across sentences; each sentence = a FILTERED GATE on the wires of the
referents it mentions (s-ancilla prepared |+>, referent–s coupling, H,
post-select — the exp1/exp24b mechanism as a sentence-update map). Word/sense
states come from exp23 Phase-2 anchors. Reading registers persist alongside
wires; conditioning accumulates across sentences. Readout of the final text
state = MAP joint interpretation (registers) + per-referent residual
ambiguity (entropy) — the "one state for the whole text, measured for what
is meant (= the intended reading)".

## Sentence gate (verb-directional filter)
gate_v(R): Rx(v_axis) on R → CRz(v_strength) R→s|+> → H(s) → postselect s=0
→ Rx(-v_axis). Algebra: diag(1, cos(pi*v_strength)) in the verb-chosen
basis — the verb defines WHICH referent-state direction passes and how
strongly. Sense-selectivity therefore EMERGES from overlap between sense
states and the verb's filter axis; it is not a hand-coded compatibility
table.

## Dataset (texts_exp27.json, author-generated, native review pending)
Controlled mini-texts over ambiguous referents with existing sense anchors:
رجل (man/leg), جمل (camel/sentence), جمل (camel/beauty), عين (eye/spring —
anchors added). Per referent: S1 introduces it ambiguously; S2_disambig uses
a verb whose MEASURED filter axis selects the target sense; S2_neutral uses
a verb with near-zero measured selectivity at matched pass-rate. Plus: one
3-sentence text (entropy trajectory), one 2-referent text (joint
conditioning), order-swapped variants. Arabic surfaces recorded for review;
physics runs on anchors. External corpora (SALMA sense-annotated, in-house
WSD_v2) are Phase-2 validation targets — real corpus sentences exceed both
parser coverage and skeleton scale today; said plainly.

## Pre-registered
D1 purification: entropy(register | S1+S2_disambig) < entropy(S1)
D2 control: entropy(S1+S2_neutral) ≈ entropy(S1) (matched pass-rates!)
D3 MAP flips to the verb-selected sense after S2_disambig
D4 p_post stacks ~multiplicatively across sentences (cost law, real gates)
D5 order swap (S2 before S1): exploratory, report as found
D6 two-referent joint conditioning runs end-to-end (architecture demo)

## Limitations designed-in, not discovered later
1. Verb selectivity is engineered via anchor geometry + verb CHOICE from a
   measured spectrum (selection step reported). Deployment needs
   sense-selectional preferences from a lexicon — classical knowledge
   entering as parameters; consistent with the structure-first thesis, but
   it means this experiment demonstrates the MECHANISM, not lexical
   discovery. The "true sense" is planted by the author via verb choice.
2. Referent wire = 1 qubit: carries sense-discrimination (2 senses), NOT
   full semantics — deliberately inside what Rung-2 proved a qubit can hold.
3. Each sentence adds a post-selection: cost stacks; measured (D4);
   amplitude amplification applies as in exp26.
4. Neutral-vs-disambig comparison is only valid at matched pass-rates
   (else entropy differences are evidence-strength artifacts) — matching is
   part of the protocol, both numbers reported.
5. No parser in the loop for Phase 1 (templates only) — isolates the physics
   from Stanza noise; parser integration is the subsequent step and depends
   on the extended reader (enrichment mode).
6. Filtered maps need not commute; order effects are measured, not assumed.
7. Gates here are the minimal IQP-derived family, not compiled DisCoCirc
   frames; richer sentence types (transitives updating two wires) are the
   next increment (D6 is the first step).
