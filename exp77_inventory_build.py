#!/usr/bin/env python3
"""exp77_inventory_build.py -- §2+§3: assemble DRAFT, compute density+char-overlap.

Reads:
    exp77_tier2_verdicts.json   (tier-2 gemma verdicts, produced by exp77_tier2_gemma.py)

Writes:
    exp77_inventory_DRAFT.json  (full per-word provenance, ready for tier-3 orchestration)
    exp77_inventory_audit_sample.json  (~50-word stratified spot-audit sample)
    exp77_section3_stats.json   (density table + char-overlap baseline)

CHAR-OVERLAP BASELINE METHOD (§3):
    Positives = word pairs sharing a root, drawn from DIFFERENT columns.
    Negatives = word pairs NOT sharing a root, sampled uniformly from the
        cross-root pair pool (same cardinality as positives by default).
    Metric = Jaccard on LETTER SETS (not multisets): |A∩B| / |A∪B|
        where A, B are the sets of distinct Arabic letters in each word.
    ال-stripping: ال is stripped before computing letter sets (leaving it in
        inflates every pair's jaccard since every word starts with ال).
    Threshold sweep: 0.0 to 1.0 in steps of 0.05 — at each threshold T,
        a pair is classified as "same root" if jaccard >= T.
    Accuracy metric: balanced accuracy = (TPR + TNR) / 2.
    Stratification: strong/weak separately, where weak = root has '#' radical.
    Negative sampling: seeded (seed=42) for reproducibility; sample is
        WITHOUT replacement from cross-root pairs.

Run:  python exp77_inventory_build.py
"""
import hashlib
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TIER2 = os.path.join(HERE, "exp77_tier2_verdicts.json")
DRAFT = os.path.join(HERE, "exp77_inventory_DRAFT.json")
AUDIT = os.path.join(HERE, "exp77_inventory_audit_sample.json")
STATS = os.path.join(HERE, "exp77_section3_stats.json")
ENUM_GRID = os.path.join(HERE, "exp77_enum_grid.json")

AL_RE = re.compile("^ال")


def strip_al(w):
    return AL_RE.sub("", w)


def letter_set(w):
    """Set of distinct Arabic letters in w, after stripping ال prefix."""
    w = strip_al(w)
    return set(c for c in w if "؀" <= c <= "ۿ" and ord(c) not in range(0x064B, 0x0653))


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / float(union) if union else 0.0


def canonical_sha256(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ------------------------------------------------------------------ main
def main():
    # Load tier-2 verdicts
    t2 = json.load(open(TIER2, encoding="utf-8"))
    words = t2["words"]
    print("[build] loaded %d words from tier-2" % len(words))

    # Accept/reject classification
    # Accepted: aravec_attested=True AND gemma_verdict=True
    # Tier3_pending: gemma_verdict=False or None (awaiting haiku)
    accepted = [w for w in words if w.get("aravec_attested") and w.get("gemma_verdict") is True]
    tier3 = [w for w in words if w.get("tier3_pending")]
    print("[build] accepted=%d tier3_pending=%d" % (len(accepted), len(tier3)))

    # Load enum grid for root_metadata
    enum_doc = json.load(open(ENUM_GRID, encoding="utf-8"))
    root_meta = enum_doc.get("root_metadata", {})

    # ---- §3-i: GRID DENSITY TABLE ----------------------------------------
    def density_table(word_list, label):
        cells = {}           # (root, col) -> [words]
        roots_set = set()
        cols_set = set()
        for w in word_list:
            root = w["root"]
            col = w["column"]
            cells.setdefault((root, col), []).append(w["word"])
            roots_set.add(root)
            cols_set.add(col)

        n_roots = len(roots_set)
        n_cols = len(cols_set)
        n_cells = len(cells)
        n_words = len(word_list)

        # cross-template share-root pairs
        from_root = {}
        for (root, col), ws in cells.items():
            from_root.setdefault(root, {})[col] = ws

        share_root_pairs = []     # (word_a, word_b, root, col_a, col_b)
        for root, cols_d in from_root.items():
            cols_list = sorted(cols_d.keys())
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    ca, cb = cols_list[i], cols_list[j]
                    for wa in cols_d[ca]:
                        for wb in cols_d[cb]:
                            share_root_pairs.append((wa, wb, root, ca, cb))

        wm = root_meta if root_meta else {}
        sr_strong = [(wa, wb, r, ca, cb)
                     for (wa, wb, r, ca, cb) in share_root_pairs
                     if not wm.get(r, {}).get("weak_or_geminate", True)]
        sr_weak = [(wa, wb, r, ca, cb)
                   for (wa, wb, r, ca, cb) in share_root_pairs
                   if wm.get(r, {}).get("weak_or_geminate", False)]

        per_dir_N = {}
        for (wa, wb, root, ca, cb) in share_root_pairs:
            pair = tuple(sorted([ca, cb]))
            per_dir_N[pair] = per_dir_N.get(pair, 0) + 1

        return {
            "label": label,
            "n_roots": n_roots,
            "n_columns": n_cols,
            "n_cells_occupied": n_cells,
            "n_words_total": n_words,
            "grid_density": round(n_cells / float(n_roots * n_cols), 4) if n_roots and n_cols else 0,
            "cross_template_share_root_pairs_total": len(share_root_pairs),
            "cross_template_share_root_pairs_strong": len(sr_strong),
            "cross_template_share_root_pairs_weak": len(sr_weak),
            "per_direction_pair_counts": {str(k): v for k, v in sorted(per_dir_N.items())},
        }

    dens_accepted = density_table(accepted, "accepted (gemma=True)")
    dens_draft = density_table(words, "all_enumerated (pre-tier2)")
    print("[build] density table done: roots=%d cols=%d cells=%d pairs=%d" %
          (dens_accepted["n_roots"], dens_accepted["n_columns"],
           dens_accepted["n_cells_occupied"],
           dens_accepted["cross_template_share_root_pairs_total"]))

    # ---- §3-ii: CHAR-OVERLAP BASELINE ------------------------------------
    def char_overlap_baseline(word_list, label, seed=42, n_neg=None):
        """Jaccard baseline: positives = cross-template same-root pairs,
        negatives = sampled cross-root pairs."""
        from_root = {}
        for w in word_list:
            from_root.setdefault(w["root"], {})\
                      .setdefault(w["column"], []).append(w["word"])

        # Build positives
        positives = []   # (word_a, word_b, root, col_a, col_b, is_weak)
        wm = root_meta if root_meta else {}
        for root, cols_d in from_root.items():
            is_weak = bool(wm.get(root, {}).get("weak_or_geminate", False))
            cols_list = sorted(cols_d.keys())
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    ca, cb = cols_list[i], cols_list[j]
                    for wa in cols_d[ca]:
                        for wb in cols_d[cb]:
                            positives.append((wa, wb, root, ca, cb, is_weak))

        # Build negatives: pairs from different roots
        all_roots = sorted(from_root.keys())
        neg_pool = []
        for i in range(len(all_roots)):
            for j in range(i + 1, len(all_roots)):
                ra, rb = all_roots[i], all_roots[j]
                is_weak = bool(wm.get(ra, {}).get("weak_or_geminate", False) or
                               wm.get(rb, {}).get("weak_or_geminate", False))
                for cola, words_a in from_root[ra].items():
                    for colb, words_b in from_root[rb].items():
                        for wa in words_a:
                            for wb in words_b:
                                neg_pool.append((wa, wb, None, cola, colb, is_weak))

        # Sample negatives
        rng = random.Random(seed)
        n_neg_target = n_neg if n_neg else min(len(positives) * 5, len(neg_pool))
        negatives = rng.sample(neg_pool, min(n_neg_target, len(neg_pool)))

        def run_sweep(pos_subset, neg_subset, sublabel):
            thresholds = [t / 20.0 for t in range(0, 21)]
            best_ba, best_t = 0.0, 0.0
            rows = []
            for T in thresholds:
                tp = sum(1 for (wa, wb, *_) in pos_subset
                         if jaccard(letter_set(wa), letter_set(wb)) >= T)
                fn = len(pos_subset) - tp
                tn = sum(1 for (wa, wb, *_) in neg_subset
                         if jaccard(letter_set(wa), letter_set(wb)) < T)
                fp = len(neg_subset) - tn
                tpr = tp / float(tp + fn) if (tp + fn) else 0.0
                tnr = tn / float(tn + fp) if (tn + fp) else 0.0
                ba = (tpr + tnr) / 2.0
                rows.append({"T": round(T, 2), "TPR": round(tpr, 4),
                             "TNR": round(tnr, 4), "BA": round(ba, 4)})
                if ba > best_ba:
                    best_ba, best_t = ba, T
            return {
                "sublabel": sublabel,
                "n_positives": len(pos_subset),
                "n_negatives": len(neg_subset),
                "best_balanced_accuracy": round(best_ba, 4),
                "best_threshold": round(best_t, 2),
                "threshold_sweep": rows,
            }

        results = {}
        results["all"] = run_sweep(positives, negatives, label + " all")

        pos_strong = [p for p in positives if not p[5]]
        neg_strong = [n for n in negatives if not n[5]]
        neg_strong_samp = rng.sample(neg_strong, min(len(pos_strong) * 5, len(neg_strong)))
        results["strong"] = run_sweep(pos_strong, neg_strong_samp, label + " strong")

        pos_weak = [p for p in positives if p[5]]
        neg_weak = [n for n in negatives if n[5]]
        neg_weak_samp = rng.sample(neg_weak, min(len(pos_weak) * 5, len(neg_weak)))
        if pos_weak:
            results["weak"] = run_sweep(pos_weak, neg_weak_samp, label + " weak")
        else:
            results["weak"] = {"sublabel": label + " weak", "n_positives": 0,
                               "n_negatives": 0, "note": "no weak positives"}

        return results

    print("[build] computing char-overlap baseline ...", flush=True)
    overlap_accepted = char_overlap_baseline(accepted, "accepted")
    # Also compute on all enumerated words (pre-tier2) for comparison
    overlap_all = char_overlap_baseline(words, "all_enumerated")
    print("[build] char-overlap done: best_BA_accepted=%.4f best_BA_all=%.4f" %
          (overlap_accepted["all"]["best_balanced_accuracy"],
           overlap_all["all"]["best_balanced_accuracy"]))

    # ---- §3-iii: STRATIFIED SPOT-AUDIT SAMPLE (~50 words) ---------------
    def draw_audit_sample(word_list, n=50, seed=77):
        """Stratified sample: proportional to (column, weak/strong)."""
        from collections import defaultdict
        strata = defaultdict(list)
        wm = root_meta if root_meta else {}
        for w in word_list:
            col = w["column"]
            is_weak = bool(wm.get(w["root"], {}).get("weak_or_geminate", False))
            strata[(col, is_weak)].append(w)

        rng = random.Random(seed)
        total = len(word_list)
        sample = []
        for (col, is_weak), group in sorted(strata.items()):
            quota = max(1, round(n * len(group) / total))
            sample.extend(rng.sample(group, min(quota, len(group))))
        # Trim or pad to ~50
        rng.shuffle(sample)
        sample = sample[:n]
        return sample

    audit_sample = draw_audit_sample(accepted, n=50)
    print("[build] audit sample: %d words" % len(audit_sample))

    # ---- WRITE OUTPUTS -------------------------------------------------------
    # §3 stats
    s3 = {
        "meta": {
            "purpose": "exp77 §3 post-build statistics",
            "protocol_ref": "67_exp77_inventory_protocol.md §3",
            "char_overlap_method": {
                "letter_set": ("distinct Arabic letters per word after stripping ال; "
                               "Arabic diacritics (U+064B-U+0652) excluded from set"),
                "jaccard": "|A∩B| / |A∪B|",
                "positives": "cross-template pairs sharing root (different columns, same root)",
                "negatives": "sampled cross-root pairs (seed=42, up to 5x positive count)",
                "metric": "balanced accuracy = (TPR + TNR) / 2",
                "threshold_grid": "0.00 to 1.00 step 0.05",
                "stratification": "strong/weak by root_metadata.weak_or_geminate",
            },
            "tier2_file_sha256": hashlib.sha256(open(TIER2, "rb").read()).hexdigest(),
            "enum_grid_file_sha256": hashlib.sha256(open(ENUM_GRID, "rb").read()).hexdigest(),
        },
        "density_accepted": dens_accepted,
        "density_all_enumerated": dens_draft,
        "char_overlap_accepted": overlap_accepted,
        "char_overlap_all_enumerated": overlap_all,
    }
    json.dump(s3, open(STATS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[build] wrote %s" % STATS)

    # Audit sample
    json.dump({
        "meta": {
            "purpose": "exp77 §3 stratified spot-audit sample (~50 words); NON-BLOCKING",
            "n": len(audit_sample),
            "stratification": "proportional by (column, weak_or_geminate); seed=77",
            "note": "For morning desk review: verify each word is plausible MSA with its stated root and column",
        },
        "sample": audit_sample,
    }, open(AUDIT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[build] wrote %s" % AUDIT)

    # DRAFT inventory
    draft_words = []
    for w in words:
        dw = dict(w)
        # Ensure tier3_pending is correctly set
        dw["tier3_pending"] = bool(w.get("tier3_pending"))
        dw["aravec_attested"] = True
        dw["aravec_attested_basis"] = "enumerated from AraVec vocabulary scan"
        # Add root metadata
        rm = root_meta.get(w["root"], {})
        dw["root_weak"] = rm.get("weak")
        dw["root_geminate"] = rm.get("geminate")
        dw["root_weak_or_geminate"] = rm.get("weak_or_geminate")
        draft_words.append(dw)

    # Meta stats
    n_accepted_draft = sum(1 for w in draft_words
                           if w.get("gemma_verdict") is True)
    n_tier3 = sum(1 for w in draft_words if w.get("tier3_pending"))
    roots_accepted = {w["root"] for w in draft_words if w.get("gemma_verdict") is True}
    roots_weak = {r for r in roots_accepted if root_meta.get(r, {}).get("weak_or_geminate")}

    draft_doc = {
        "meta": {
            "purpose": "exp77_inventory_DRAFT: exp77 root×template grid lexicon (pre-tier-3)",
            "status": ("DRAFT -- tier-3 haiku pass not yet run; "
                       "tier3_pending=True words may be admitted after orchestrator review"),
            "protocol_ref": "67_exp77_inventory_protocol.md",
            "route": "AraVec vocabulary scan (fused enumeration + tier-1 attestation)",
            "n_words_total": len(draft_words),
            "n_words_accepted": n_accepted_draft,
            "n_words_tier3_pending": n_tier3,
            "n_roots_accepted": len(roots_accepted),
            "n_roots_weak_or_geminate_accepted": len(roots_weak),
            "weak_pct_accepted": round(len(roots_weak) / len(roots_accepted), 4)
                                  if roots_accepted else 0.0,
            "weak_quota_met": (len(roots_weak) / len(roots_accepted) >= 0.5)
                              if roots_accepted else False,
            "acceptance_rule": ("aravec_attested=True (all, by construction) AND "
                                "gemma_verdict=True; tier3_pending=True words require "
                                "haiku orchestrator pass to finalize"),
            "shas": {
                "tier2_file": hashlib.sha256(open(TIER2, "rb").read()).hexdigest(),
                "enum_grid_file": hashlib.sha256(open(ENUM_GRID, "rb").read()).hexdigest(),
            },
            "tier3_disagreement_file": "exp77_tier3_disagreements.json",
            "tier3_note": ("DO NOT run haiku pass yourself; "
                           "leave exp77_tier3_disagreements.json for the orchestrator"),
        },
        "words": draft_words,
    }
    json.dump(draft_doc, open(DRAFT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[build] wrote %s" % DRAFT)
    print("[build] FINAL SUMMARY:")
    print("  total words: %d" % len(draft_words))
    print("  accepted (gemma=True): %d" % n_accepted_draft)
    print("  tier3_pending: %d" % n_tier3)
    print("  roots_accepted: %d" % len(roots_accepted))
    print("  roots_weak_or_geminate: %d (%.1f%%)" %
          (len(roots_weak), 100 * len(roots_weak) / len(roots_accepted)
           if roots_accepted else 0))
    print("  weak_quota_met: %s" % (len(roots_weak) / len(roots_accepted) >= 0.5
                                    if roots_accepted else False))
    print("  char_overlap_best_BA_accepted: %.4f" %
          overlap_accepted["all"]["best_balanced_accuracy"])
    print("  char_overlap_best_BA_accepted_strong: %.4f" %
          overlap_accepted["strong"]["best_balanced_accuracy"])
    weak_ba = overlap_accepted.get("weak", {}).get("best_balanced_accuracy")
    if weak_ba is not None:
        print("  char_overlap_best_BA_accepted_weak: %.4f" % weak_ba)


if __name__ == "__main__":
    main()
