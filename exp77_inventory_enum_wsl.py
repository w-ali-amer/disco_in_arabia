#!/usr/bin/env python3
"""exp77_inventory_enum_wsl.py -- §1 ENUMERATION: root x template grid from AraVec.

ROUTE CHOSEN (recorded per protocol):
    "analyze the AraVec vocabulary itself (every vocab token through the
    analyzer, collect (root, pattern) hits for the 6 columns) -- that fuses
    enumeration and tier-1 attestation in one pass and is the most
    token/compute-efficient route."

    Every word in the output IS in AraVec by construction (it came FROM the
    vocab scan), so AraVec-attested = True for all enumerated words. The
    tier-1 attestation column in the DRAFT carries this flag explicitly with
    its basis recorded.

DIVERGENCE FROM extract_aravec_candidates.py::_classify (stated here, not
discovered later):
    D4's _classify filters to rat='r', gen='m', num='s' (masculine singular
    rational human noun convention).
    exp77's protocol DROPS that filter: "No POS/semantic restriction to
    human nouns -- nouns and participles admitted; verbs excluded."
    We therefore collect ALL pos in {noun, adj} with prc0=Al_det and
    accept any root regardless of rat, gen, num.

WEAK FLAGS (per d4_roots_of_record.json conventions):
    weak = root carries at least one '#' radical.
    CAMeL's builtin DB writes و/ي/ء as '#' when absorbed; '#' is the ONLY
    weak-radical marker.
    geminate = the three consonant radicals include R2==R3 (when neither is '#').

INTERPRETER: ~/discocat_arabic_v2/qiskit_lambeq_env (gensim 4.3.3 +
camel_tools 1.5.6; bare python3 has a numpy/gensim binary incompatibility).

Run:
    source ~/discocat_arabic_v2/qiskit_lambeq_env/bin/activate
    python ~/discocat_arabic_v2/exp77_inventory_enum_wsl.py
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
import datetime
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))

ARAVEC = os.path.join(HERE, "aravec", "full_uni_cbow_300_twitter.mdl")
SCAN_CAP = 300_000   # same as extract_aravec_candidates.py (doc population anchor)
AL_FORM = re.compile("^ال[ء-ي]{2,}$")   # same regex as the committed script
HARAKA = re.compile("[ً-ْ]+$")  # strip all trailing harakat (case-marker chain)

OUT_GRID = os.path.join(HERE, "exp77_enum_grid.json")


# ------------------------------------------------------------------ load helpers
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# FAAIL_CAMEL_PATTERNS -- copied verbatim from instrument_scaling_bounds.py
# (instrument_scaling_bounds.FAAIL_CAMEL_PATTERNS, lines 139-146).
# Inlined here rather than imported to avoid the heavy dependency chain
# (instrument_scaling_bounds.py -> instrument_sb_bounds.py -> camel_tools
# at MODULE level), but the values are the committed set per doc 63 SS3.2.
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
    """NFD-normalize then strip all trailing harakat.

    CAMeL combines shadda+kasra (U+0651 U+0650) in a different order from
    the Unicode canonical (kasra ccc=32 before shadda ccc=33 in NFD).
    Applying NFD before comparison ensures matching regardless of which
    order the diacritics were stored. Stripping all trailing harakat handles
    the case-marker chain (e.g. ال1ا2ِّ = ال1ا2ّ with genitive marker).
    This normalization is applied to BOTH the query pattern AND the set keys.
    """
    return HARAKA.sub("", unicodedata.normalize('NFD', p))


# ------------------------------------------------------------------ 6 COLUMNS
# Adviser-reviewed, derived empirically by probing CAMeL against known words.
# ONLY sound root and attested weak-root realizations; broken plurals and
# non-participating forms are stated in the exclusion lists below.

FAAIL_NORM = {_norm_pattern(p) for p in FAAIL_BASE}   # from FAAIL_CAMEL_PATTERNS

FAIIL_NORM = {
    "ال1َ2ِي3",     # sound / geminate (الكفيل, الرفيق)
    "الوَ2ِي3",     # assimilated waw root (الوكيل: #.ك.ل)
    "الأَ2ِي3",     # hamzated 1st radical (الأمير: #.م.ر)
}

FA33AAL_NORM = {
    "ال1َ2ّا3",     # sound root (الرسّام, الصرّاف, الكذّاب)
    "ال1َيّا3",     # hollow root w->y (الخيّاط: خ.#.ط)
    "ال1َوّا3",     # hollow root waw (الجوّاب)
}
# EXCLUDED from fa33aal:
#   ال1ُ2ّا3 = fu33aal (broken plural of فاعل, e.g. كُتّاب "scribes")
#   ال1ِ2ا3 / ال1ُ2ا3 = fi3aal/fu3aal occupational (النجّار: CAMeL gives ال1ُ2ا3)
#   ال1ُ2اء = fu3aa' (collective/intensives like الدعّاء)

MAF3UUL_NORM = {
    "المَ1ْ2ُو3",   # sound root (المكتوب, المحبوب)
    "المَ1ِي3",     # hollow root passive participle (المبيع: ب.#.ع)
    "المَ1ْ2ِيّ",   # defective 3rd passive adj (المقضي: ق.ض.#)
    "المَ1ْ2ُوّ",   # defective 3rd with waw (المدعو: د.ع.#)
}
# EXCLUDED from maf3uul:
#   المَ1ْ2َى = maf3al (place/time noun, e.g. المرمى "goal", المبنى "building")
#   المُ1ا3 / المَ1ا3 = muf3al/maf3al of hollow (not مفعول)

MU1_NORM = {
    "المُ1َ2ِّ3",   # standard Form II active مُفعِّل (المدرّس, المفكّر)
    "المُ1َوِّ3",   # hollow Form II active (المقوّم: ق.#.م, المحوّل: ح.#.ل)
    "ال1ُمَ2ِّ3",   # م-initial roots sound Form II active (الممثّل: م.ث.ل)
    "ال1ُمَوِّ3",   # م-initial roots hollow Form II active (المموّل: م.#.ل)
    "المُ1َ2ِّي",   # defective 3rd Form II active (المربّي: ر.ب.#)
}
# EXCLUDED from mu1:
#   المُ1َ2َّ3 = Form II PASSIVE (مُفعَّل, e.g. المقدَّم) -- pinned to active only
#   ال1ُمَوَّ3 = م-initial hollow Form II PASSIVE

MU2_NORM = {
    "المُ1ْ2ِ3",    # standard Form IV active مُفعِل (المعلن, المرسل)
    "المُ1ِي3",     # hollow Form IV active (المقيم: ق.#.م, المبين: ب.#.ن)
    "المُ1ْ2ِي",    # defective 3rd Form IV active (المعطي: ع.ط.#)
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
    """Return list of column names this pattern belongs to (0, 1, or 2)."""
    p = _norm_pattern(raw_pattern)
    return [col for col, pset in COLUMNS.items() if p in pset]


# ------------------------------------------------------------------ root flags
def root_flags(root_str):
    """Return (weak, geminate) flags for a '1.2.3' style root string."""
    if not root_str or root_str in ("NTWS", ""):
        return None, None
    parts = [r.strip() for r in root_str.split(".") if r.strip()]
    if len(parts) != 3:
        return None, None    # non-triliteral or malformed
    weak = any(r == "#" for r in parts)
    # geminate: any two non-# radicals are equal (most commonly R2==R3)
    concrete = [r for r in parts if r != "#"]
    geminate = len(concrete) >= 2 and len(set(concrete)) < len(concrete)
    return weak, geminate


# ------------------------------------------------------------------ main
def main():
    import numpy as np
    from gensim.models import Word2Vec
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    import camel_tools

    print("[enum] loading AraVec ...", flush=True)
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
    print("[enum] vocab=%d scanned=%d al_forms=%d" %
          (len(keys), cap, len(al_forms)), flush=True)

    print("[enum] loading CAMeL DB ...", flush=True)
    db = MorphologyDB.builtin_db()
    an = Analyzer(db, cache_size=300000)

    # root -> {col -> [word, ...]}
    grid = {}         # root_str -> {col: [word, ...]}
    unmatched_patterns = {}  # for audit: pattern -> count (unmatched tokens)
    total_noun_adj_analyses = 0
    total_tokens_analysed = 0
    no_al_det = 0

    for i, row in enumerate(al_forms):
        tok = row["token"]
        if (i + 1) % 5000 == 0:
            print("[enum] %d/%d tokens analysed, roots so far=%d" %
                  (i + 1, len(al_forms), len(grid)), flush=True)

        analyses = an.analyze(tok)
        al_det = [a for a in analyses
                  if a.get("prc0") == "Al_det"
                  and a.get("pos") in ("noun", "adj")]
        if not al_det:
            no_al_det += 1
            continue
        total_tokens_analysed += 1

        # collect distinct (root, pattern, col) triples from this token
        seen_root_col = set()
        for a in al_det:
            root = a.get("root", "")
            pat = a.get("pattern", "")
            if not root or not pat or root == "NTWS" or pat == "NTWS":
                continue
            # skip the 'الNTWS' non-templatic pattern
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

    print("[enum] done scanning. roots=%d total_noun_adj=%d unmatched=%d" %
          (len(grid), total_noun_adj_analyses,
           sum(unmatched_patterns.values())), flush=True)

    # histogram: roots by number of columns
    hist = {}
    for root, cols_d in grid.items():
        n = len(cols_d)
        hist[n] = hist.get(n, 0) + 1
    print("[enum] roots-by-#-columns histogram:", sorted(hist.items()))

    # gate: how many roots have >=3 columns?
    roots_ge3 = {r: d for r, d in grid.items() if len(d) >= 3}
    print("[enum] roots with >=3 columns: %d" % len(roots_ge3))

    if len(roots_ge3) < 60:
        print("[BLOCKER] fewer than 60 roots meet the >=3-column rule (%d found)."
              % len(roots_ge3))
        print("[BLOCKER] Full distribution:", sorted(hist.items()))
        # Still write the partial data for reporting
        _write_output(grid, hist, roots_ge3, al_forms, unmatched_patterns,
                      camel_tools, cap, total_tokens_analysed, no_al_det,
                      total_noun_adj_analyses, blocker=True)
        sys.exit(0)

    _write_output(grid, hist, roots_ge3, al_forms, unmatched_patterns,
                  camel_tools, cap, total_tokens_analysed, no_al_det,
                  total_noun_adj_analyses, blocker=False)


def _write_output(grid, hist, roots_ge3, al_forms, unmatched_patterns,
                  camel_tools, cap, total_tokens_analysed, no_al_det,
                  total_noun_adj_analyses, blocker):
    import importlib.util, sys

    # ---- weak / geminate flags -----------------------------------------
    def annotate(root):
        w, g = root_flags(root)
        return {
            "weak": bool(w) if w is not None else None,
            "geminate": bool(g) if g is not None else None,
            "weak_or_geminate": (bool(w) or bool(g)) if w is not None else None,
        }

    root_meta = {r: annotate(r) for r in roots_ge3}

    n_weak = sum(1 for m in root_meta.values()
                 if m["weak_or_geminate"] is True)
    n_total = len(root_meta)
    weak_pct = n_weak / float(n_total) if n_total else 0.0
    quota_met = weak_pct >= 0.50

    print("[enum] weak_or_geminate roots: %d / %d = %.1f%% (quota>=50%%: %s)"
          % (n_weak, n_total, 100 * weak_pct, quota_met))

    # ---- weak-quota-aware cap: prefer weak roots if needed ----------------
    # Sort: first by #-columns (desc), then weak_or_geminate (True first),
    # then alphabetical on root. Apply cap 100-150.
    # The cap is intended to give 400-800 words; count words first.
    sorted_roots = sorted(
        roots_ge3.keys(),
        key=lambda r: (
            -len(roots_ge3[r]),                             # more columns first
            0 if root_meta[r]["weak_or_geminate"] else 1,  # weak/geminate first
            r                                               # alphabetical
        )
    )

    # Count words at each cap level
    def word_count_at_cap(roots, cap_n):
        total = 0
        for r in roots[:cap_n]:
            for col, words in roots_ge3[r].items():
                total += len(words)
        return total

    chosen_cap = None
    for cap_n in range(100, 201):
        if cap_n > len(sorted_roots):
            break
        wc = word_count_at_cap(sorted_roots, cap_n)
        if 400 <= wc <= 800:
            chosen_cap = cap_n
            break
    if chosen_cap is None:
        # fallback: use 150 or max available
        chosen_cap = min(150, len(sorted_roots))

    selected_roots = sorted_roots[:chosen_cap]
    total_words = sum(len(words) for r in selected_roots
                      for words in roots_ge3[r].values())
    print("[enum] selected %d roots, %d words (cap=%d)"
          % (len(selected_roots), total_words, chosen_cap))

    # ---- recheck weak quota on selected set ----------------------------
    sel_weak = sum(1 for r in selected_roots
                   if root_meta[r]["weak_or_geminate"] is True)
    sel_pct = sel_weak / float(len(selected_roots)) if selected_roots else 0.0
    sel_quota_met = sel_pct >= 0.50
    print("[enum] selected set: weak_or_geminate %d/%d=%.1f%% (quota: %s)"
          % (sel_weak, len(selected_roots), 100 * sel_pct, sel_quota_met))

    # ---- flag single-surface roots (%3 column status from one token) ----
    single_surface_flags = {}
    for r in selected_roots:
        all_words = set()
        for col_words in roots_ge3[r].values():
            all_words.update(col_words)
        single_surface_flags[r] = (len(all_words) == 1)

    # ---- top unmatched patterns ----------------------------------------
    top_unmatched = sorted(unmatched_patterns.items(),
                           key=lambda kv: -kv[1])[:50]

    # ---- build the grid rows for output --------------------------------
    grid_rows = []
    for r in selected_roots:
        rm = root_meta[r]
        for col, words in roots_ge3[r].items():
            for w in words:
                wk, gm = root_flags(r)
                grid_rows.append({
                    "word": w,
                    "root": r,
                    "column": col,
                    "weak": bool(wk) if wk is not None else None,
                    "geminate": bool(gm) if gm is not None else None,
                    "aravec_attested": True,   # by construction (scanned from vocab)
                    "aravec_attested_basis": "word came from AraVec vocab scan",
                    "gemma_verdict": None,       # filled by tier-2
                    "tier3_pending": False,      # filled by tier-2
                })

    out = {
        "meta": {
            "purpose": "exp77 §1 enumeration: root x template grid from AraVec vocabulary",
            "route": ("AraVec-vocabulary scan: every ال-form token analyzed through "
                      "CAMeL builtin DB; (root, pattern) collected for pos in "
                      "{noun, adj} with prc0=Al_det; pattern mapped to 6 columns. "
                      "AraVec attestation = True for all words by construction."),
            "protocol_ref": "67_exp77_inventory_protocol.md §1",
            "camel_tools_version": camel_tools.__version__,
            "python_version": sys.version.split()[0],
            "aravec_model": ARAVEC,
            "scan_cap": cap,
            "al_form_tokens_scanned": len(al_forms),
            "tokens_with_al_det_noun_adj": total_tokens_analysed,
            "tokens_without_al_det_noun_adj": no_al_det,
            "total_noun_adj_analyses_templatic": total_noun_adj_analyses,
            "divergence_from_d4_classify": (
                "D4's _classify requires rat='r', gen='m', num='s' (human noun "
                "convention). exp77 drops that: all pos in {noun, adj} with "
                "prc0=Al_det admitted. Verbs (pos='verb') excluded."),
            "column_pattern_sets": {
                "faail": sorted(FAAIL_NORM),
                "faiil": sorted(FAIIL_NORM),
                "fa33aal": sorted(FA33AAL_NORM),
                "maf3uul": sorted(MAF3UUL_NORM),
                "mu1": sorted(MU1_NORM),
                "mu2": sorted(MU2_NORM),
            },
            "column_exclusions": {
                "fa33aal_excluded": ["ال1ُ2ّا3 (fu33aal, broken plural)", "ال1ِ2ا3/ال1ُ2ا3 (fi3aal/fu3aal occ.)", "ال1ُ2اء (fu3aa')"],
                "maf3uul_excluded": ["المَ1ْ2َى (maf3al place/time noun)", "المُ1ا3/المَ1ا3 (hollow maf3al)"],
                "mu1_excluded": ["المُ1َ2َّ3 (Form II PASSIVE mufaccel)"],
            },
            "pattern_normalization": "trailing single haraka stripped before lookup (handles construct/case twins)",
            "weak_flag_definition": "root carries at least one '#' radical (CAMeL convention per d4_roots_of_record.json)",
            "geminate_flag_definition": "two or more non-'#' radicals are identical (most commonly R2==R3)",
            "ge3_col_rule": "roots attested in >=3 of the 6 columns",
            "weak_quota": ">=50% of selected roots must be weak_or_geminate",
            "size_cap": "100-150 roots, 400-800 words",
            "blocker": blocker,
        },
        "distribution": {
            "roots_by_column_count_histogram": {str(k): v for k, v in sorted(hist.items())},
            "roots_ge3_columns_total": len(roots_ge3),
            "selected_cap": chosen_cap,
            "selected_roots": len(selected_roots),
            "selected_words_total": total_words,
            "selected_weak_or_geminate": sel_weak,
            "selected_weak_pct": round(sel_pct, 4),
            "weak_quota_met": sel_quota_met,
        },
        "top50_unmatched_patterns": [{"pattern": p, "count": c}
                                     for p, c in top_unmatched],
        "grid": grid_rows,
        "root_metadata": {r: root_meta[r] for r in selected_roots},
        "single_surface_root_flags": single_surface_flags,
    }
    json.dump(out, open(OUT_GRID, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[enum] wrote %s" % OUT_GRID)
    print("[enum] SUMMARY: roots=%d words=%d weak_quota=%s blocker=%s" %
          (len(selected_roots), total_words, sel_quota_met, blocker))


if __name__ == "__main__":
    main()
