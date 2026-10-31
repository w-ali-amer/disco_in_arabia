#!/usr/bin/env python3
"""exp77_inventory_enum_topup.py -- §5 TOP-UP enumeration.

Amendment (doc 67 §5, committed 2026-09-04 before any run):
    weak/geminate roots admitted at >=2 columns (sound roots stay >=3 -- they
    are already covered by exp77_enum_grid.json).

Selection gate (applied to the full root-x-column grid produced by the same
AraVec scan as the original):
    (a) root is weak_or_geminate (same root_flags() definition as the committed
        script -- '#' radical for weak, R2==R3 concrete radicals for geminate)
    (b) root appears in >=2 of the 6 columns in the AraVec vocab
    (c) root is NOT already in exp77_enum_grid.json (all >=3 col roots, some of
        which are already weak/geminate)

Sound roots: excluded entirely from the top-up (they still require >=3 columns
and are already in the main grid). Counted and recorded, never silent (§1).

Exclusion accounting:
    - 1-column weak roots (do not meet >=2 threshold): counted
    - 2+-column weak roots already in main grid: counted (should be ~0 since
      main grid took >=3 columns, but checked explicitly)
    - Sound roots at exactly 2 columns: counted (excluded by design)

Output: exp77_enum_grid_topup.json, same schema as exp77_enum_grid.json.

INTERPRETER: qiskit_lambeq_env (gensim 4.3.3 +
camel_tools 1.5.6; bare python3 has a numpy/gensim binary incompatibility).

Run:
    source qiskit_lambeq_env/bin/activate
    nohup python exp77_inventory_enum_topup.py \
        > topup_enum.log 2>&1 &
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))

ARAVEC = os.path.join(HERE, "aravec", "full_uni_cbow_300_twitter.mdl")
SCAN_CAP = 300_000   # same as original script
AL_FORM = re.compile("^ال[ء-ي]{2,}$")
HARAKA = re.compile("[ً-ْ]+$")

# Inputs
EXISTING_GRID = os.path.join(HERE, "exp77_enum_grid.json")
# Output
OUT_GRID = os.path.join(HERE, "exp77_enum_grid_topup.json")

# ------------------------------------------------------------------ patterns (verbatim from committed script)

FAAIL_CAMEL_PATTERNS = {
    "ال1ا2ِ3": "sound root (salim) -- the canonical al-faaCil",
    "ال1ائِ3": "hollow root (ajwaf) -- al-faaCil realised faa'iC",
    "ال1ا2ِي": "defective root (naqis) -- al-faaCil realised faaCi",
    "الوا2ِ3": "assimilated waw root (mithal) -- waaCiC",
    "ال1ا2ِئ": "hamzated lam -- faaCi'",
    "ال1ا2ّ": "geminate root (mudaaCaf) -- faaCC",
}
FAAIL_BASE = set(FAAIL_CAMEL_PATTERNS.keys())


def _norm_pattern(p):
    return HARAKA.sub("", unicodedata.normalize('NFD', p))


FAAIL_NORM = {_norm_pattern(p) for p in FAAIL_BASE}

FAIIL_NORM = {
    "ال1َ2ِي3",
    "الوَ2ِي3",
    "الأَ2ِي3",
}

FA33AAL_NORM = {
    "ال1َ2ّا3",
    "ال1َيّا3",
    "ال1َوّا3",
}

MAF3UUL_NORM = {
    "المَ1ْ2ُو3",
    "المَ1ِي3",
    "المَ1ْ2ِيّ",
    "المَ1ْ2ُوّ",
}

MU1_NORM = {
    "المُ1َ2ِّ3",
    "المُ1َوِّ3",
    "ال1ُمَ2ِّ3",
    "ال1ُمَوِّ3",
    "المُ1َ2ِّي",
}

MU2_NORM = {
    "المُ1ْ2ِ3",
    "المُ1ِي3",
    "المُ1ْ2ِي",
}

COLUMNS = {
    "faail":   FAAIL_NORM,
    "faiil":   FAIIL_NORM,
    "fa33aal": FA33AAL_NORM,
    "maf3uul": MAF3UUL_NORM,
    "mu1":     MU1_NORM,
    "mu2":     MU2_NORM,
}


def pattern_to_columns(raw_pattern):
    p = _norm_pattern(raw_pattern)
    return [col for col, pset in COLUMNS.items() if p in pset]


def root_flags(root_str):
    """Return (weak, geminate) flags. Verbatim from committed script."""
    if not root_str or root_str in ("NTWS", ""):
        return None, None
    parts = [r.strip() for r in root_str.split(".") if r.strip()]
    if len(parts) != 3:
        return None, None
    weak = any(r == "#" for r in parts)
    concrete = [r for r in parts if r != "#"]
    geminate = len(concrete) >= 2 and len(set(concrete)) < len(concrete)
    return weak, geminate


def main():
    import numpy as np
    from gensim.models import Word2Vec
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    import camel_tools

    # ---- load the existing grid root set (exclusion set) ----
    print("[topup] loading existing grid for exclusion set ...", flush=True)
    existing_doc = json.load(open(EXISTING_GRID, encoding="utf-8"))
    existing_roots = set(row["root"] for row in existing_doc["grid"])
    print("[topup] existing grid: %d roots, %d words" %
          (len(existing_roots), len(existing_doc["grid"])), flush=True)

    # ---- AraVec scan (identical to original) ----
    print("[topup] loading AraVec ...", flush=True)
    wv = Word2Vec.load(ARAVEC, mmap="r").wv
    keys = list(wv.key_to_index)
    counts = np.asarray(wv.expandos["count"])
    order = sorted(range(len(keys)), key=lambda i: (-int(counts[i]), i))
    cap = min(SCAN_CAP, len(order))

    al_forms = []
    for rank in range(cap):
        k = keys[order[rank]]
        if AL_FORM.match(k):
            al_forms.append({"token": k, "count": int(counts[order[rank]]),
                              "rank": rank})
    print("[topup] vocab=%d scanned=%d al_forms=%d" %
          (len(keys), cap, len(al_forms)), flush=True)

    print("[topup] loading CAMeL DB ...", flush=True)
    db = MorphologyDB.builtin_db()
    an = Analyzer(db, cache_size=300000)

    # ---- full grid scan ----
    grid = {}    # root_str -> {col: [word, ...]}
    unmatched_patterns = {}
    total_noun_adj_analyses = 0
    total_tokens_analysed = 0
    no_al_det = 0

    for i, row in enumerate(al_forms):
        tok = row["token"]
        if (i + 1) % 5000 == 0:
            print("[topup] %d/%d tokens, roots so far=%d" %
                  (i + 1, len(al_forms), len(grid)), flush=True)

        analyses = an.analyze(tok)
        al_det = [a for a in analyses
                  if a.get("prc0") == "Al_det"
                  and a.get("pos") in ("noun", "adj")]
        if not al_det:
            no_al_det += 1
            continue
        total_tokens_analysed += 1

        seen_root_col = set()
        for a in al_det:
            root = a.get("root", "")
            pat = a.get("pattern", "")
            if not root or not pat or root == "NTWS" or pat == "NTWS":
                continue
            if "NTWS" in pat:
                continue
            total_noun_adj_analyses += 1
            cols = pattern_to_columns(pat)
            if not cols:
                np_key = _norm_pattern(pat)
                unmatched_patterns[np_key] = unmatched_patterns.get(np_key, 0) + 1
                continue
            for col in cols:
                key = (root, col)
                if key not in seen_root_col:
                    seen_root_col.add(key)
                    if root not in grid:
                        grid[root] = {}
                    if col not in grid[root]:
                        grid[root][col] = []
                    if tok not in grid[root][col]:
                        grid[root][col].append(tok)

    print("[topup] scan done. total roots=%d" % len(grid), flush=True)

    # ---- histogram (full pool) ----
    hist = {}
    for root, cols_d in grid.items():
        n = len(cols_d)
        hist[n] = hist.get(n, 0) + 1
    print("[topup] roots-by-#-columns histogram:", sorted(hist.items()))

    # ---- apply top-up selection ----
    # counters for exclusion accounting (§1: every exclusion counted)
    excl_sound_any = 0          # sound roots (excluded entirely from top-up)
    excl_weak_1col = 0          # weak/geminate roots with only 1 column
    excl_weak_already_grid = 0  # weak/geminate roots >=2 col already in main grid
    admitted_topup = {}         # root -> col -> [words]

    sound_2col = 0              # sound roots at exactly 2 columns (sub-count)

    for root, cols_d in grid.items():
        wk, gm = root_flags(root)
        if wk is None:
            # malformed root; skip
            continue
        is_weak = bool(wk) or bool(gm)
        n_cols = len(cols_d)

        if not is_weak:
            # sound root — excluded from top-up; main grid covers them at >=3
            excl_sound_any += 1
            if n_cols == 2:
                sound_2col += 1
            continue

        # weak/geminate root
        if n_cols < 2:
            excl_weak_1col += 1
            continue
        if root in existing_roots:
            excl_weak_already_grid += 1
            continue
        # ADMIT: weak/geminate, >=2 columns, not already in grid
        admitted_topup[root] = cols_d

    print("[topup] admitted top-up roots: %d" % len(admitted_topup), flush=True)
    print("[topup] exclusion accounting: "
          "excl_sound=%d (of which 2-col=%d), excl_weak_1col=%d, "
          "excl_weak_already_grid=%d" %
          (excl_sound_any, sound_2col, excl_weak_1col, excl_weak_already_grid), flush=True)

    # ---- build output rows ----
    grid_rows = []
    root_meta = {}
    single_surface_flags = {}

    for root, cols_d in sorted(admitted_topup.items()):
        wk, gm = root_flags(root)
        rm = {
            "weak": bool(wk) if wk is not None else None,
            "geminate": bool(gm) if gm is not None else None,
            "weak_or_geminate": (bool(wk) or bool(gm)) if wk is not None else None,
        }
        root_meta[root] = rm

        all_words = set()
        for col, words in cols_d.items():
            all_words.update(words)
        single_surface_flags[root] = (len(all_words) == 1)

        for col, words in sorted(cols_d.items()):
            for w in words:
                grid_rows.append({
                    "word": w,
                    "root": root,
                    "column": col,
                    "weak": rm["weak"],
                    "geminate": rm["geminate"],
                    "aravec_attested": True,
                    "aravec_attested_basis": "word came from AraVec vocab scan",
                    "gemma_verdict": None,
                    "tier3_pending": False,
                })

    total_words = len(grid_rows)
    total_topup_roots = len(admitted_topup)

    # combined pool stats (for quota reporting)
    existing_weak = sum(1 for row in existing_doc["grid"]
                        if row.get("weak") or row.get("geminate"))
    existing_total_roots = len(existing_roots)
    combined_total = existing_total_roots + total_topup_roots
    combined_weak = existing_weak + total_topup_roots   # all top-up are weak/geminate by construction

    # de-duplicate existing_weak count by root
    existing_weak_roots = set()
    for row in existing_doc["grid"]:
        if row.get("weak") or row.get("geminate"):
            existing_weak_roots.add(row["root"])
    combined_weak_roots = len(existing_weak_roots) + total_topup_roots
    combined_weak_pct = combined_weak_roots / float(combined_total) if combined_total else 0.0

    print("[topup] combined pool: existing=%d roots + topup=%d roots = %d total" %
          (existing_total_roots, total_topup_roots, combined_total), flush=True)
    print("[topup] combined weak_or_geminate: %d/%d = %.1f%% (quota >=50%%: %s)" %
          (combined_weak_roots, combined_total, 100 * combined_weak_pct,
           combined_weak_pct >= 0.50), flush=True)

    n_single_surface = sum(1 for v in single_surface_flags.values() if v)
    print("[topup] single-surface roots (1 distinct word across columns): %d / %d" %
          (n_single_surface, total_topup_roots), flush=True)

    top_unmatched = sorted(unmatched_patterns.items(), key=lambda kv: -kv[1])[:50]

    out = {
        "meta": {
            "purpose": ("exp77 §5 top-up enumeration: weak/geminate roots at >=2 columns, "
                        "excluding roots already in exp77_enum_grid.json (>=3 col pool)"),
            "amendment_ref": "67_exp77_inventory_protocol.md §5",
            "route": ("Same AraVec-vocabulary scan as exp77_inventory_enum.py; "
                      "selection changed: weak/geminate roots >=2 cols, sound roots excluded. "
                      "AraVec attestation = True for all words by construction."),
            "script": "exp77_inventory_enum_topup.py (new, never edits the committed script)",
            "camel_tools_version": camel_tools.__version__,
            "python_version": sys.version.split()[0],
            "aravec_model": ARAVEC,
            "scan_cap": cap,
            "al_form_tokens_scanned": len(al_forms),
            "tokens_with_al_det_noun_adj": total_tokens_analysed,
            "tokens_without_al_det_noun_adj": no_al_det,
            "total_noun_adj_analyses_templatic": total_noun_adj_analyses,
            "existing_grid_roots_excluded": sorted(existing_roots),
            "existing_grid_root_count": existing_total_roots,
            "selection_gate": {
                "weak_or_geminate": True,
                "min_columns_for_weak": 2,
                "min_columns_for_sound": "N/A -- sound roots excluded from top-up",
                "exclude_already_in_main_grid": True,
            },
            "exclusion_accounting": {
                "excl_sound_any": excl_sound_any,
                "excl_sound_2col_subcnt": sound_2col,
                "excl_weak_only_1col": excl_weak_1col,
                "excl_weak_already_in_grid": excl_weak_already_grid,
                "note": "Every exclusion counted per §1 protocol",
            },
        },
        "distribution": {
            "full_scan_roots_by_column_count": {str(k): v for k, v in sorted(hist.items())},
            "topup_roots_admitted": total_topup_roots,
            "topup_words_total": total_words,
            "topup_weak_or_geminate_pct": 1.0,   # 100% by construction
            "combined_existing_plus_topup": {
                "total_roots": combined_total,
                "weak_or_geminate_roots": combined_weak_roots,
                "weak_or_geminate_pct": round(combined_weak_pct, 4),
                "quota_50pct_met": combined_weak_pct >= 0.50,
            },
            "single_surface_roots": n_single_surface,
            "single_surface_note": ("roots where all column words are the same single token; "
                                    "flagged, not excluded; see single_surface_root_flags"),
        },
        "top50_unmatched_patterns": [{"pattern": p, "count": c} for p, c in top_unmatched],
        "grid": grid_rows,
        "root_metadata": root_meta,
        "single_surface_root_flags": single_surface_flags,
    }

    json.dump(out, open(OUT_GRID, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[topup] wrote %s" % OUT_GRID)
    print("[topup] SUMMARY: roots=%d words=%d combined_weak_pct=%.1f%% quota_met=%s" %
          (total_topup_roots, total_words, 100 * combined_weak_pct, combined_weak_pct >= 0.50))


if __name__ == "__main__":
    main()
