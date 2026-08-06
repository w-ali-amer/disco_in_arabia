# Quantum Compositional NLP for Arabic

**Paper:** [Quantum Compositional NLP for Arabic: Grammar, Morphology, and Word Sense in Circuit Topology](https://doi.org/10.5281/zenodo.19316164)
**Author:** Wajahath Mohammed — wajahath123@gmail.com

First application of pregroup grammar-based QNLP to Arabic. Converts Arabic sentences into quantum circuits whose topology mirrors grammatical structure: subjects, verbs, and objects become quantum gates; the pregroup grammar determines how those gates are wired together. Central finding: a zero-variance L0/L1 entanglement ablation on matched-pair Arabic word order — grammar topology without entanglement gives exactly 50% (by construction); adding one entangling layer gives 64.9%.

---

## Repository structure

This repository has two levels:

```
/                        ← working code only — everything needed to reproduce the paper
└── dev_history/         ← development record — earlier iterations and abandoned approaches
```

The root contains only the files that actually run the experiments reported in the paper. `dev_history/` preserves the full development journey — the pipeline went through many iterations before converging on the current design. Those files are not needed to reproduce results but are kept for transparency. See `dev_history/README.md` for a description of each.

---

## Root files — what each does

### Pipeline (load order matters)

The experiments depend on a chain of modules. `arabic_dep_reader.py` is the entry point; the others are its dependencies:

| File | Role |
|------|------|
| `arabic_dep_reader.py` | **Entry point.** Takes Arabic sentences, runs dependency parsing via Stanza + CAMeL Tools, assigns pregroup grammar types (SVO, VSO, Nominal), and returns lambeq diagrams. VSO verb type assignment (`s ⊗ n.l ⊗ n.l`) and the Swap derivation are implemented here. |
| `camel_test2.py` | **Analysis backend.** Arabic morphological analysis, CAMeL Tools integration, sentence structure detection. Imported directly by `arabic_dep_reader`. This file has a long development history (see `dev_history/camel_test*.py`) — `camel_test2.py` is the version that works. |
| `common_qnlp_types.py` | **Shared type definitions.** Pregroup types, lambeq box functors, and type utilities shared across the pipeline. Imported by `camel_test2.py`. |
| `arabic_discocirc_pipeline.py` | **DisCoCirc enrichment layer.** Adds feature-enriched diagrams for discourse-level composition. Loaded at runtime by `camel_test2.py` with a fallback if unavailable. |
| `arabic_morpho_lex_core.py` | **Morphological lexicon.** Stanza and CAMeL pipeline initialisation, morphological feature extraction. Imported by `arabic_discocirc_pipeline.py`. |

### Data

| File | Contents |
|------|---------|
| `sentences.json` | All experiment data. Keys: `WordOrder` (120 sentences, 3 classes), `WordOrderMatched` (120, matched pairs), `LexicalAmbiguity` (210), `Morphology` (230), `TenseBinary` (100), `WordSenseDisambiguation_v2` (200, 4 verbs × 2 senses × 25). |
| `generate_exp13_data.py` | Adds `WordOrderMatched` and `TenseBinary` to `sentences.json`. Run this before exp13. |
| `generate_exp14_data_v2.py` | Adds `WordSenseDisambiguation_v2` to `sentences.json`. Run this before exp14. |

### Experiments (paper sections in brackets)

| File | Paper | Description |
|------|-------|-------------|
| `exp8.py` | not in paper | Binary lexical ambiguity per polysemous pair. 7 pairs, 300 epochs, AraVec warm-start. Informed the WSD design in §8. |
| `exp9_tense_deep.py` | not in paper | Tense ablation across n_layers=1/2/3, 3 seeds, 500 epochs. Informed the tense experiment in §7. |
| `exp10_wordorder.py` | not in paper | Word order 3-class (SVO/VSO/Nominal). Established why 3-class is too hard for SPSA at N=40; motivated the matched-pair binary design in §6. |
| `exp11_sense_switch.py` | not in paper | Sense-switch: polysemous words get two parameter sets. Exploratory. |
| `exp12_quantum_advantage.py` | not in paper | Earlier framing of structural encoding vs. bag-of-words. See note at top of file — "quantum advantage" here means advantage over AraVec, not computational quantum advantage. |
| `exp13_arabert_comparison.py` | §6 §7 §8 | **Main paper experiment.** Word order L0/L1 ablation, tense, WSD. QFM vs. SPSA vs. AraBERT. |
| `exp14_ancillary_wsd.py` | §8 | Ancilla qubit WSD with density-matrix label encoding. SPSA inversion analysis. |
| `reprocess_exp14_symmetric.py` | §8.3 | Post-processes exp14 results: symmetric SPSA correction (max(acc, 1−acc) per fold). |
| `baseline_binary.py` | §6 | Classical AraVec + SVM/RF/MLP baselines for binary tasks. |
| `baseline_classical.py` | §6 | Additional classical baselines. |
| `visualize_results_v2.py` | — | Generates all figures from exp8–exp12 results. |
| `visualize_exp13.py` | — | Generates figures for exp13 results. |

### Results and figures

| Path | Contents |
|------|---------|
| `qnlp_experiment_outputs_per_set_v2/exp13_arabert/` | Main results JSON (taskA_wordorder.json, learning_curves.json, arabert_finetuned_results.json) |
| `qnlp_experiment_outputs_per_set_v2/exp14_ancillary_wsd_v2/` | WSD results (exp14_v2_summary.json) |
| `figures/` | All generated figures |

---

## Environment

**Python 3.10.** Other versions are untested.

```bash
python3.10 -m venv qiskit_lambeq_env
source qiskit_lambeq_env/bin/activate
pip install -r requirements.txt
```

> **Important:** `lambeq==0.5.0` requires `numpy<2.0`. Do not upgrade numpy independently — it will break silently.

**Stanza Arabic model** (required before first run):
```bash
python3 -c "import stanza; stanza.download('ar')"
```

**CAMeL Tools Arabic models** (required before first run):
```bash
camel_data -i defaults
```

**AraVec** (required for AraVec baseline experiments only):
Download from [github.com/bakrianoo/aravec](https://github.com/bakrianoo/aravec) and place the model files in an `aravec/` directory at the project root. The experiments use the Twitter CBOW model. `aravec/` is gitignored due to size.

**AraBERT** (`aubmindlab/bert-base-arabertv02`) downloads automatically from HuggingFace on first run.

---

## Reproducing the main result

`sentences.json` already contains all datasets including `WordOrderMatched` and `WordSenseDisambiguation_v2`. The generate scripts only need to be run if you modify the raw data or start from scratch.

```bash
# Main experiment — word order L0/L1 ablation + AraBERT comparison
python exp13_arabert_comparison.py
# outputs → qnlp_experiment_outputs_per_set_v2/exp13_arabert/

# WSD experiment — ancilla qubit + SPSA inversion analysis
python exp14_ancillary_wsd.py
python reprocess_exp14_symmetric.py   # apply symmetric SPSA correction
# outputs → qnlp_experiment_outputs_per_set_v2/exp14_ancillary_wsd_v2/

# Regenerate datasets from scratch (optional)
python generate_exp13_data.py         # rebuilds WordOrderMatched + TenseBinary
python generate_exp14_data_v2.py      # rebuilds WordSenseDisambiguation_v2
```

---

## Questions and issues

If you run into problems setting up the environment or running the experiments, feel free to open a GitHub issue or email wajahath123@gmail.com directly. Happy to help.

## Citation

```
Mohammed, W. (2026). Quantum Compositional NLP for Arabic: Grammar, Morphology,
and Word Sense in Circuit Topology. Zenodo.
https://doi.org/10.5281/zenodo.19316164
```

## Recent results (August 2026) — beyond the paper

The paper (arXiv:2607.14100) covers sentence-level word order, tense, and WSD.
See [ERRATUM.md](ERRATUM.md) for corrections to the paper's methods description (published numbers unchanged) and [RESULTS_EXP21_v2.md](RESULTS_EXP21_v2.md) for a measured negative: on leakage-free splits 0/5 seeds show significant transfer (quantum POST mean AUC 0.563 vs matched classical control 0.975 on all five splits); v1's apparent positive was train→test leakage (see RESULTS_EXP21.md, superseded).
Since then, committed in this repo:

- **Frame-invariant 4-parameter verb blocks** — `exp33_mixed_ansatz.py`, `exp38_analog_native.py`:
  4 numerically solved parameters per verb (no gradient training; targets from pre-trained AraVec
  centroids over this repo's own corpus) reproduce the same evidence direction on held-out syntactic
  frames (mean LOO alignment 0.996, 6 verbs / 5 lemmas). Caveats, measured: cross-verb parameters are
  87–99.9% interchangeable, and both mixed variants failed the pre-registered verb-specificity
  criterion (0.036 < 0.05) — the block is frame-stable but only weakly verb-specific. A follow-up control with criteria fixed before the run (exp39, results_exp39.json) retired the ‘verb encoder’ reading outright: other verbs’ parameters reproduce a verb’s target at 98.9% of matched alignment (median, both gate families), and a dial-permutation test found no verb-specific task signal beyond a tie-at-max (right-tail p = 0.083; a full derangement ties the original assignment).
- **Gate-to-analog compilation** — `exp37_analog_compile.py`, `exp37b_analog_xx.py`,
  `results_exp37b.json`: the original CRz·CRx blocks do NOT all compile (F 0.79–0.997,
  `results_exp37.json`); the re-solved exchange-symmetric CRz·XX family compiles at F≈1 on a
  decay-free pulse model — a compiler-correctness check for a family chosen to be expressible.
- **Measured simulation-cost wall** — `exp34b_scaling.py`: exact dense density-matrix simulation of
  the discourse register costs ×4 memory per referent (measured step-time ×6.4 on one machine,
  cache-transition regime; asymptotic ratio ≈4); 16 GB wall ≈14 referents. Bounds exact simulation
  of THIS architecture only; no task-level quantum advantage is claimed.
- **Entanglement scaling** — `exp36*.py`, `results_exp36*.json`: half-chain entropy saturates toward
  the k/2 ceiling under dense random co-reference coupling (surrogate random schedules; solved
  entangler angles in the 36c variant). Adjacent-pair-only coupling (results_exp36.json) does not saturate — dense cross-references are what force the growth.
- **Argument-swap plausibility** — `exp34a_swap_plausibility.py`: 36/54 vs 50% chance (p=0.0099,
  uncorrected). By construction equivalent to the 2D classical cosine it was solved against
  (decisions agree 53/54; the classical computation scores 37/54).
- **First Arabic QNLP on quantum hardware** — `exp35_*.py`, `results_exp35_hardware.json`:
  ibm_kingston (Heron; 4-qubit circuits), the 19 highest-margin of 54 pairs, 114 circuits, 10k
  shots, no error mitigation — hardware reproduced 19/19 noiseless decisions (12 of 19 semantically
  correct; the same 7 misses as the noiseless model).
- **Parser fix** — CAMeL-POS fusion in `arabic_dep_reader.py` (flag-gated, off by default;
  published numbers unchanged).
