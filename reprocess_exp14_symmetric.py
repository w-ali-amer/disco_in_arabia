"""
Re-report exp14 v2 SPSA results using symmetric evaluation.
For each fold-seed combo: report max(acc, 1-acc).
Rationale: SPSA on binary tasks can converge to the inverted minimum
(same loss magnitude, opposite orientation). The symmetric result
recovers the true discriminability regardless of inversion direction.
"""
import re
import numpy as np
from collections import defaultdict

LOG = "/home/waj/discocat_arabic_v2/exp14_ancillary_wsd_v2.log"

# Pattern: [verb/method] s=X f=Y/15 acc=Z
FOLD_PAT = re.compile(
    r"\[([^/\]]+)/SPSA_(base|ancilla)\] s=(\d+) f=(\d+)/15 acc=([\d.]+)"
)

# Also grab original summary lines for QFM and AraVec/AraBERT (unchanged)
SUMMARY_PAT = re.compile(
    r"\[([^\]]+)\] (AraVec_SVM_rbf|AraVec_RF|AraBERT_frozen|QFM n_anc=\d+ discard=\w+|SPSA n_anc=\d+ discard=\w+)\s+([\d.]+) ± ([\d.]+)"
)

# Collect fold-level SPSA accuracies
fold_accs = defaultdict(list)  # (verb, method) -> list of accs

with open(LOG) as f:
    for line in f:
        m = FOLD_PAT.search(line)
        if m:
            verb, method, seed, fold, acc = m.groups()
            key = (verb, f"SPSA_{method}")
            fold_accs[key].append(float(acc))

# Collect original summary results (for non-SPSA methods)
original = {}
with open(LOG) as f:
    for line in f:
        m = SUMMARY_PAT.search(line)
        if m:
            ctx, method, mean, std = m.groups()
            original[(ctx, method)] = (float(mean), float(std))

# Verbs in order
VERBS = ["رفع", "حمل", "قطع", "ضرب", "POOLED"]

# Symmetric correction
sym_results = {}
for (verb, method), accs in fold_accs.items():
    sym = [max(a, 1-a) for a in accs]
    sym_results[(verb, method)] = (np.mean(sym), np.std(sym), accs, sym)

# Print comparison
print("=" * 80)
print("EXP14 v2 — SPSA Symmetric Re-evaluation (max(acc, 1-acc) per fold)")
print("=" * 80)
print()
print("RATIONALE: SPSA on binary tasks can converge to the inverted minimum.")
print("Symmetric eval recovers true discriminability regardless of orientation.")
print()

header = f"{'Task':<8}  {'Raw_base':>10}  {'Sym_base':>10}  {'Δ':>6}  {'Raw_anc':>10}  {'Sym_anc':>10}  {'Δ':>6}"
print(header)
print("-" * 75)

for verb in VERBS:
    base_raw_accs = fold_accs.get((verb, "SPSA_base"), [])
    anc_raw_accs  = fold_accs.get((verb, "SPSA_ancilla"), [])

    if not base_raw_accs:
        continue

    raw_base = np.mean(base_raw_accs)
    sym_base_vals = [max(a, 1-a) for a in base_raw_accs]
    sym_base = np.mean(sym_base_vals)

    raw_anc = np.mean(anc_raw_accs)
    sym_anc_vals = [max(a, 1-a) for a in anc_raw_accs]
    sym_anc = np.mean(sym_anc_vals)

    delta_base = sym_base - raw_base
    delta_anc  = sym_anc  - raw_anc

    print(f"{verb:<8}  {raw_base:>10.4f}  {sym_base:>10.4f}  {delta_base:>+6.4f}  {raw_anc:>10.4f}  {sym_anc:>10.4f}  {delta_anc:>+6.4f}")

print()
print("=" * 80)
print("FULL RESULTS TABLE (symmetric SPSA)")
print("=" * 80)
print()

# Get AraVec/AraBERT/QFM from log summary lines
# These keys appear in the log as [verb] method_string mean ± std
arabert_results = {}
aravec_results  = {}
qfm_base_results = {}
qfm_anc_results  = {}

for (ctx, method), (mean, std) in original.items():
    if "AraBERT" in method:
        arabert_results[ctx] = mean
    elif "AraVec_SVM_rbf" in method:
        aravec_results[ctx] = mean
    elif "QFM n_anc=0" in method:
        qfm_base_results[ctx] = mean
    elif "QFM n_anc=1" in method:
        qfm_anc_results[ctx] = mean

# Map verb names to log context strings
VERB_MAP = {
    "رفع": "رفع",
    "حمل": "حمل",
    "قطع": "قطع",
    "ضرب": "ضرب",
    "POOLED": "POOLED",
}

hdr = f"{'Task':<8}  {'AraVec':>8}  {'AraBERT':>8}  {'QFM_base':>9}  {'QFM_anc':>8}  {'SPSA_base(sym)':>14}  {'SPSA_anc(sym)':>13}"
print(hdr)
print("-" * 85)

for verb in VERBS:
    ctx = VERB_MAP[verb]
    av  = aravec_results.get(ctx,   aravec_results.get(f"{ctx}/QFM_base", float('nan')))
    ab  = arabert_results.get(ctx,  float('nan'))
    qb  = qfm_base_results.get(f"{ctx}/QFM_base", qfm_base_results.get(ctx, float('nan')))
    qa  = qfm_anc_results.get(f"{ctx}/QFM_ancilla", qfm_anc_results.get(ctx, float('nan')))

    base_raw_accs = fold_accs.get((verb, "SPSA_base"), [])
    anc_raw_accs  = fold_accs.get((verb, "SPSA_ancilla"), [])

    sym_base = np.mean([max(a, 1-a) for a in base_raw_accs]) if base_raw_accs else float('nan')
    sym_anc  = np.mean([max(a, 1-a) for a in anc_raw_accs])  if anc_raw_accs  else float('nan')

    print(f"{verb:<8}  {av:>8.4f}  {ab:>8.4f}  {qb:>9.4f}  {qa:>8.4f}  {sym_base:>14.4f}  {sym_anc:>13.4f}")

print()
print("Chance level (binary): 50.0%")
print()
print("NOTE: Symmetric SPSA = max(acc, 1-acc) per fold-seed, then averaged.")
print("      This is a discriminability measure, not raw accuracy.")
print("      QFM and classical results are unchanged (no orientation issue).")
print()

# Per-verb inversion analysis
print("=" * 80)
print("INVERSION ANALYSIS — Fraction of folds that inverted (acc < 0.45)")
print("=" * 80)
print()
for verb in VERBS:
    base_raw_accs = fold_accs.get((verb, "SPSA_base"), [])
    anc_raw_accs  = fold_accs.get((verb, "SPSA_ancilla"), [])
    if not base_raw_accs:
        continue
    inv_base = sum(1 for a in base_raw_accs if a < 0.45) / len(base_raw_accs)
    inv_anc  = sum(1 for a in anc_raw_accs  if a < 0.45) / len(anc_raw_accs)
    print(f"  {verb:<8}  SPSA_base inverted: {inv_base:.1%} of folds    SPSA_anc inverted: {inv_anc:.1%}")
