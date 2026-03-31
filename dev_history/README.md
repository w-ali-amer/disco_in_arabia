# Development History

This folder contains earlier iterations of the pipeline and abandoned approaches. None of these files are needed to reproduce the paper results — everything required is in the root directory. They are kept here because they document the path taken to reach the working design.

The root `arabic_dep_reader.py → camel_test2.py → common_qnlp_types.py` pipeline did not arrive fully formed. The files below show what was tried, what broke, and what was replaced. Researchers interested in how the VSO type assignment and Swap derivation evolved, or in why certain approaches were abandoned, will find that history here.

---

## Pipeline iterations

### `camel_test.py`, `camel_test1.9.py`, `camel_test1.9.1.py`, `camel_test1.9.2.py`, `camel_test3.py`
Earlier versions of what eventually became `camel_test2.py`. Each version attempted a different approach to integrating CAMeL Tools morphological analysis with lambeq diagram construction:

- `camel_test.py` — first working Arabic morphological parser connected to lambeq types. No Swap, no VSO handling, SVO only.
- `camel_test1.9.py` / `1.9.1` / `1.9.2` — progressive refinements to the type assignment logic. VSO handling was attempted here but the verb type used was incorrect (`n.r ⊗ s ⊗ n.l` applied to VSO positions rather than the correct `s ⊗ n.l ⊗ n.l`). Diagrams were structurally wrong but the code ran without errors.
- `camel_test3.py` — an attempt to unify SVO and VSO handling through a single functor. Abandoned because the functor produced identical circuit topologies for both word orders, defeating the purpose.
- `camel_test2 copy.py` — working copy made during a debugging session. No intentional changes from `camel_test2.py`.

**What changed in `camel_test2.py` (the working version):** correct `s ⊗ n.l ⊗ n.l` type for VSO verbs, Swap properly inserted by wire sequence analysis, CAMeL/Stanza token mismatch handled via fallback.

### `common_qnlp_types_backup.py`
Snapshot of `common_qnlp_types.py` taken before a refactor. Kept in case the refactor broke something. The current `common_qnlp_types.py` in the root is the correct version.

### `v4.py`, `v6.py`, `v8.py`
Monolithic early versions of the full pipeline — parsing, type assignment, diagram construction, and training all in one file. The progression shows the architecture becoming more modular:

- `v4.py` — all logic in one script, hardcoded sentences, SVO only.
- `v6.py` — first attempt at VSO. Used a naive wire permutation (explicit Box with manual output type) rather than a principled pregroup type. Results were not reproducible across runs.
- `v8.py` — closest predecessor to the current architecture. Introduces the idea of separating the analysis backend from the diagram builder. The type assignment is still not fully correct (mixed use of `n.r ⊗ s ⊗ n.l` and `s ⊗ n.l ⊗ n.l`) but the structural separation here is what the current modular design grew from.

### `debug_type.py`
Utility script used during development to print pregroup type sequences and verify that cups fire correctly. Useful for diagnosing type mismatch errors. Not part of any experiment.

---

## Early experiments

### `exp3.py`, `exp4.py`, `exp5_training.py`, `exp6.py`, `exp7.py`
Experiments run before the dataset and methodology were finalised. Problems with these:

- `exp3.py` / `exp4.py` — used the monolithic v4/v6 pipeline. Results not reproducible; vocabulary not controlled between word order classes.
- `exp5_training.py` — first attempt at SPSA training end-to-end. Discovered that SPSA converges to inverted minima on symmetric binary tasks — this observation eventually became the inversion-rate analysis in the paper (Section 8.3).
- `exp6.py` / `exp7.py` — attempted multi-class extensions. Failed due to `n_s_qubits` mismatch with lambeq's NumpyModel output shape. Not rerun; the issue is explained in the paper.

The experiments in the paper (exp8–exp14) were designed from scratch with controlled vocabulary, proper train/test splits, and correct type assignment.

---

## Abandoned approaches

### `quantum_kernel.py`, `quantum_kernel_v2.py`, `quantum_kernel_v3.py`, `kernel_v5.py`
An attempt to use quantum kernel methods (fidelity-based kernel SVM) as an alternative to the IQP ansatz + SPSA approach. The idea was to use quantum circuits as a feature kernel rather than a trainable model. Abandoned because:
- Kernel computation scaled quadratically with dataset size (infeasible at N=200)
- The kernel gave no insight into which structural features drove classification
- The IQP ansatz + QFM approach provided the same structural encoding more efficiently and with interpretable ablation properties

### `quantum_experiment1.py`
First attempt to run a complete experiment from raw sentences to accuracy number. Uses `v4.py` as backend. Kept as a historical baseline showing the starting point of the project.

### `lambeq_min_test.py`, `lambeq_test.py`, `lambeq_type_test.py`
Scripts written while learning the lambeq API. Used to verify that `IQPAnsatz`, `NumpyModel`, and `RemoveCupsRewriter` behaved as expected. Not part of any experiment.

---

## Test scripts

### `test_amb_H3_v2.py`, `test_ambiguity_H3.py`, `test_core_features.py`, `test_discocirc.py`, `test_unified_binding_h3.py`
Unit tests written during development to verify specific pipeline behaviours — diagram reduction, type cancellation, word sense encoding. All tests were written against intermediate pipeline versions. They may not pass against the current code without modification.

---

## Old data files

### `sentences_original.json`, `sentences_augmented.json`, `sentences.txt`
Earlier versions of the sentence dataset, superseded by `sentences.json` in the root.

- `sentences_original.json` — dataset before WordOrderMatched and TenseBinary were added.
- `sentences_augmented.json` — an augmented version with generated sentence variants. Augmentation was dropped because it introduced vocabulary overlap that undermined the matched-pair design.
- `sentences.txt` — plain text export used for a CAMeL Tools preprocessing step that was later integrated into the pipeline directly.

### `generate_exp14_data.py`
First version of the WSD dataset generator. Used only 3 verbs and had vocabulary leakage (AraVec achieved 90–97% because subject/object words were not shared across senses). Replaced by `generate_exp14_data_v2.py` in the root, which uses 4 verbs with strict matched-pair and shared-pool vocabulary control.

---

## Sanity check directories

### `discocat_sanity_check/`, `oldest_mvp_sancheck/`
Ad hoc directories created at different stages to run quick sanity checks on diagram structure, type reduction, and circuit compilation. Each contains a frozen copy of the pipeline at that point in time. Not maintained.

---

## Miscellaneous

### `path.py`, `file_process.py`, `nominal_diags_2_7.py`
Small utility scripts from early development. `nominal_diags_2_7.py` was a prototype for handling Nominal (subjectless) sentence diagrams; the logic was later absorbed into `arabic_dep_reader.py`.

### `visualize_results.py`
First version of the visualisation script. Superseded by `visualize_results_v2.py` in the root, which adds exp13/exp14 figures and corrects axis scaling.

### `modifier_box/`, `.vscode/`
`modifier_box/` contains prototype code for box-level morphological modifiers (never integrated). `.vscode/` contains IDE launch and settings configuration from the development environment.
