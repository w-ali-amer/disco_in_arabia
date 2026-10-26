# Correction to "Quantum Compositional NLP for Arabic: Grammar, Morphology, and Word Sense in Circuit Topology"

**Wajahath Mohammed** · arXiv:2607.14100 · Zenodo doi:10.5281/zenodo.19316164

After publishing this paper I re-examined its pipeline, data and evaluation. Several claims do not hold up, and I am correcting them here. The first section lists statements in the paper that are wrong, with what is actually true and the evidence. The second section lists what still stands. The third section reports findings from later experiments that were **not** in the paper and are given for context only. A revised arXiv version will follow.

## 1. Corrections

| # | Where in the paper | What the paper says | What is true |
|---|---|---|---|
| C1 | §4.1–4.2, Fig. 1 ("Why the Swap is not an engineering choice") | The Swap for VSO order is the unique minimal operation, with no other valid reduction path. | **False.** The VSO type s·nˡ·nˡ followed by n·n also reduces with nested planar cups, with no Swap. The Swap is a convention that fixes which verb leg binds the subject. Crossed and planar diagrams with the same bindings give identical feature vectors, and the word-order signal survives under the planar convention on a fresh draw (0.630 against a 0.620 bar). |
| C2 | §4.2 / §6, and the topology-only classifier | VSO sentences produce fewer controlled gates, and that is the Swap's signature. | Only 17 of the 60 VSO sentences were typed as transitive VSO, and those have **more** gates (3 vs 2). The "fewer gates" pattern comes from 43 sentences the parser typed with a single nominal argument. The topology-only classifier was counting how diagrams were presented, not computed structure. |
| C3 | Exp. 1 table and text: "AraVec 12.8%" | A below-chance baseline showing vocabulary-blindness. | **Withdrawn.** The data contain matched twin sentences with opposite labels. The score was 1.4% when an item's twin was in training and 50.0% otherwise, so the number measures twin anti-learning and is not a baseline. |
| C4 | "Topology-only 35.8% / symmetrised 64.2%" | Symmetrised topology-only accuracy roughly matches QFM L1. | **Withdrawn.** 64.2% has no computed source (it is 1 − 0.358 of an in-sample classifier), and the classifier depends on presentation (C2). |
| C5 | Fig. 3 learning curve | AraVec degrades with more data; quantum and classical have opposite trajectories. | **Withdrawn.** The curve tracks the rate at which twins land in training (it reproduces 0.5 × (1 − N/60)). With twin-grouped sampling, AraVec is at 0.500 for every N. |
| C6 | QFM parameters "fixed to word embedding values" | The circuits are warm-started from AraVec embeddings. | **The description is false.** The lookup never matched, because symbol names carry morphology tags, so every parameter fell back to an md5 hash of the symbol name. The reported numbers are what that configuration produced, but they are hash-parameter results, not embedding-initialised ones. |
| C7 | Exp. 3 word sense, Table 6 (رفع 68.9%, قطع 79.5%) | Recovered discriminability after correcting orientation. | **Void for these two verbs.** 80% of قطع items and 56% of رفع items compile to a circuit identical to an item with the opposite sense, and the رفع set includes 8 identical sentences with opposite labels. No model can separate these. |
| C8 | Contribution 4: SPSA "label inversion", 99% → 23% | An optimiser orientation phenomenon. | **Largely void.** On the hardest task the inversion is forced by identical circuits with opposite labels (C7). |
| C9 | Exp. 2 tense; the statement that the parser correctly tags morphological tense | Tense information is carried by the circuits. | **False.** The parser typed 18 of the 50 past-tense items as verbless nominal sentences. With a rule that recovers the verb, all 100 items are verbal. Under cross-validation that keeps each past/present pair together, QFM scores 0.53 and bag-of-words 0.50, both at chance. The only above-chance score (0.60) appears with the original parse, where the mistyping itself separates the classes. |
| C10 | Related work | — | Missing prior work: D. Bargelli and J. Lambek, "An algebraic approach to Arabic sentence structure", *Linguistic Analysis* 31(3–4), 2001, pp. 301–315 (preprint 2003), an Arabic pregroup grammar with case, agreement, pro-drop and form-restricted valency; and M. Abd Al-Aziz, "Parsing Arabic Verb Phrases Using Pregroup Grammars", *International Journal on Natural Language Computing* 7(3), 2018 (doi:10.5121/ijnlc.2018.7303). |
| C11 | Reference [22] | Srivastava, Babu H, Mishra, Tripathi | The correct authors are Srivastava, Belekar, Saumya, Babu H. |
| C12 | "No prior Arabic NLP paper has applied minimal-pair evaluation to word order classification" | — | Should be narrowed. Expert-built Arabic diagnostic sets now exist (e.g. ALPS, arXiv 2602.17054, which probes morpho-syntax and pragmatics). The claim should be read only as "matched-pair evaluation of word-order classification", and my search for such prior work was not exhaustive. |

## 2. What stands
- **Word order with one entangling layer:** 64.9% (64.85% on the 116 deduplicated sentences). Its source needs reframing (§3).
- **The L0 result:** grammar topology without entanglement gives exactly 50% on matched word-order pairs, by construction.
- **The IBM hardware run.**
- **The word-order signal survives the planar (no-Swap) convention.**
- **The scope of the "first" claim:** this is, to my knowledge, the first pregroup-based quantum NLP study for Arabic, meaning the first to compile Arabic pregroup derivations into circuits. The Arabic pregroup *grammar* itself predates it (C10).

## 3. Notes added after publication (later experiments, not part of the paper)
- **What the word-order signal is.** A compositional semantic model is designed to give SVO and VSO sentences with the same words the same meaning, so word-order classification is outside what such a model should do. The signal comes from the difference in type assignment between the two orders. A classifier that only reads each diagram's structure reaches 0.90 on the paper's own parses (0.97 with a rebuilt parser), and trained circuit models score at chance once test words do not appear in training. The 64.9% is a structural signal: not specific to the Swap, and not semantic.
- **Word sense, with the data errors removed.**
  - Under the original pipeline, after dropping identical-label conflicts and colliding items, a QFM classifier scores 0.48 on 24 رفع items. Only 10 قطع items survive, too few to interpret.
  - A bag-of-words classifier is at least as accurate on every verb with enough items.
  - With a rebuilt parser, the identical-circuit collisions fall to zero for two of the verbs (قطع 40 → 0, رفع 10 → 0; for حمل they go from 3 to 4). Trained circuit models then reach roughly bag-of-words accuracy, and a frozen AraBERT classifier is higher on every task.
- **Root and pattern.** Representing Arabic words as root + pattern (using CAMeL Tools' contextual disambiguator) lets a standard classical model generalise to unseen word forms on a constructed test set, and recover tense from the verb template on this paper's tense items (0.82 on sentences whose verb root is unseen in training). The diagram-based circuit models did not use the verb template.
- **Reproducibility.** The parses depend on the Stanza version: 1.10.1 reproduces the published structure exactly, and later versions change it.

## 4. Data and code
The sentence set, the parser and the type rules are in the public repository: https://github.com/w-ali-amer/disco_in_arabia. The sentence set is released as it was used, including the defects described above, so that the failure cases can be inspected.
