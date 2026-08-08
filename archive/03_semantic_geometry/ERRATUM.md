# Erratum and Dataset Corrections — 2026-08-01

These issues were discovered by the author during a follow-up experiment
(exp15/exp15c, sentence-state geometry) through a pre-registered symbol audit,
and were fixed before any external report. The journal version of the paper
(arXiv:2607.14100) will carry the corrected methods text.

**Summary: no published number changes.** Both findings concern how the
pipeline was described and bookkept, not what it measured.

## 1. QFM warm-start parameters were hash-derived, not AraVec-derived (exp13)

**What the paper says:** circuit parameters were "fixed to word embedding
values" (AraVec warm start).

**What the code did:** lambeq symbol names carry morphological tags
(e.g. `الولد_NUM-s_GEN-m__n_0`). `warmstart_weights` in
`exp13_arabert_comparison.py` passed the tagged form to the AraVec lookup,
which therefore never matched, and **every** parameter silently fell back to
the deterministic md5-hash branch. `exp14_ancillary_wsd.py` is unaffected —
its `_vec_for_word` strips the tags before lookup.

**Impact on results: none.** Parameters were fixed and deterministic under
either scheme; no trainable parameters were introduced; the L0 = 50.0%
result is architectural and parameter-independent. If anything the
interpretation is sharpened: with content-free parameters, the only
systematic difference between matched SVO and VSO circuits is wiring
topology, consistent with QFM L1 (64.9%) landing next to the topology-only
control (64.2%).

**Fix:** the lookup now strips morph tags before querying AraVec (~81% of
symbols then match; the remainder still use the hash fallback). The exact
published behaviour is reproducible with `QFM_WARMSTART=legacy`.

## 2. WordOrderMatched `pair_id` column was scrambled

The `pair_id` column in `sentences.json` did not link the matched twins. The
construction claim itself holds and was verified programmatically: every one
of the 120 sentences has a word-multiset twin with the opposite label — 60
true pairs. `fix_dataset_pair_ids.py` rebuilt the column by word-multiset
grouping; the previous values are preserved as `pair_id_legacy`.

No published experiment used `pair_id` (exp13 stratifies on labels only), so
no reported number is affected. exp15's condition A initially used the
scrambled column; exp15c re-ran it with true multiset twins (conclusions
unchanged; see `results_exp15c.json`).

## 3. Two duplicated sentence pairs in WordOrderMatched

Two twin pairs appear twice verbatim (116 unique sentences of 120):
`السائق يوقف السيارة / يوقف السائق السيارة` and
`البنت رسمت الصورة / رسمت البنت الصورة`. Duplicates that straddle
cross-validation folds create mild train/test leakage; with 4 of 120
sentences affected, the effect on reported accuracy is bounded above by
~3 percentage points and is in practice far smaller. Entries are retained
for reproducibility (identical word multisets make them easy to exclude);
this will be noted in the journal version and fixed in any standalone
benchmark release.

## Verification re-run (2026-08-02, exp13b)

Task A word-order QFM re-run under both warmstart modes, identical protocol
(10 seeds x 5-fold stratified CV, SVM-RBF on NumpyModel probability features):

| condition | legacy (published behaviour) | fixed (repaired AraVec) |
|-----------|------------------------------|--------------------------|
| L0        | 50.00% (exact)               | 50.00% (exact)           |
| L1        | 64.92% [62.4, 67.3]          | 63.58% [61.7, 65.4]      |
| L2        | 61.83% [58.9, 64.7]          | 64.50% [62.4, 66.6]      |

The legacy path reproduces the published numbers (64.9 / 61.8 / 50.0) exactly.
The repaired embedding-derived parameters land within ~1pp: word content adds
essentially nothing to this task, and the discriminative signal is topological,
as the paper argues. Fold-level data in results_exp13b.json.

One further correction to the exp15 symbol-audit note: that audit ran on a
pair_id-scrambled (non-twin) pair, which is why it reported zero shared
symbols. True multiset twins DO share parameters (twin_audit.py: the subject
noun contributes 3 identical symbols to both circuits). The corrected
mechanism analysis is in results_exp17.json.
