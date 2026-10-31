#!/usr/bin/env python3
"""exp77 pre-freeze roots pass: PASS 2 (independent) + reconciliation.

THREE ROUTES, none of them allowed to overwrite another.

  PASS 1 -- the pipeline of record. `extract_aravec_candidates.py`'s admitting
    analysis (first Al_det definite NOUN reading with rat='r', gen='m',
    num='s'), whose `pattern` field is what doc 63 SS3.2 made D4's wazn
    membership of record and whose `root` field rides on the same analysis.
    Reproduced bit-identically on WSL (exp77_roots_pass1.py): 0/519 root,
    pattern and lemma mismatch, under camel_tools 1.5.5 AND 1.5.6.

  PASS 2 -- INDEPENDENT DERIVATION, this file. The analyser's stored `root` is
    not consulted. CAMeL's `pattern` is a templatic skeleton in which digits
    1..n mark radical slots and everything else is literal; aligning it against
    the surface RECOVERS the radicals. Different field, different computation.
    What it can check: whether the template and the root tell the same story
    about this surface. What it CANNOT check: whether a different analysis
    should have been chosen -- pass 2 rides the same admitting analysis.

  PASS 2b -- DIFFERENT SELECTOR (exp77_roots_pass2b.py). CAMeL's MLE
    disambiguator replaces `_classify`'s selection rule. This is the axis pass 2
    cannot probe, and it is reported separately rather than merged.

  (non-record third column: d4_roots_camel.json, the scaffolding single-pass
   majority vote. Carried for diagnosis. It is NOT of record and it is measurably
   worse -- see meta.agreement.scaffolding_majority_vote.)

Every route disagreement is recorded with BOTH candidates and agree=false.
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

DIAC = re.compile("[ً-ْـٰ]")   # == extract_aravec_candidates.strip_diac
DIGITS = "1234567"

# Orthographic equivalence classes. Needed only so a LITERAL character of the
# pattern may match its spelling variants in the surface, and so a recovered
# radical may be compared with the analyser's root radical: the root writes bare
# hamza, the surface writes a carrier (أ إ ئ ؤ آ).
EQ = [set("اأإآٱى"), set("ءأإئؤآا"), set("يىئ"), set("وؤ"), set("ةه")]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def variants(ch):
    v = {ch}
    for g in EQ:
        if ch in g:
            v |= g
    return v


def norm_rad(ch):
    if ch in set("ءأإئؤآ"):
        return "ء"
    if ch in set("اٱ"):
        return "ا"
    if ch in set("ىي"):
        return "ي"
    if ch == "ؤ":
        return "و"
    if ch == "ة":
        return "ه"
    return ch


def strip_diac(s):
    return DIAC.sub("", s or "")


def align(pattern, surface):
    """(pattern x surface) -> {slot: radical}. See module docstring."""
    p, s = strip_diac(pattern), strip_diac(surface)
    if not p or "NTWS" in p:
        return {}, False, "pattern_non_templatic"
    if not any(c in DIGITS for c in p):
        return {}, False, "pattern_has_no_radical_slot"
    rx, order = [], []
    for c in p:
        if c in DIGITS:
            rx.append("(.)")
            order.append(c)
        else:
            rx.append("[" + re.escape("".join(sorted(variants(c)))) + "]")
    m = re.fullmatch("".join(rx), s)
    if m is None:
        return {}, False, "alignment_failed"
    slots = {}
    for i, d in enumerate(order):
        ch = m.group(i + 1)
        if d in slots and norm_rad(slots[d]) != norm_rad(ch):
            return {}, False, "repeated_slot_inconsistent"
        slots.setdefault(d, ch)
    return slots, True, None


def split_root(root):
    if not root or root == "NTWS":
        return None
    return root.split(".")


def analyse_one(pattern, root1, lemma_bare, noun):
    """Run pass 2 for one noun and reconcile it against pass 1."""
    lb = lemma_bare or ""
    target = lb if lb.startswith("ال") else "ال" + lb
    slots, ok, why = align(pattern or "", target)
    if not ok and target != noun:
        s2, ok2, why2 = align(pattern or "", noun)
        if ok2:
            slots, ok, why = s2, True, "aligned_on_surface_not_lemma"
    parts1 = split_root(root1)
    n_pos = len(parts1) if parts1 else 0
    if ok and n_pos:
        root2 = ".".join(slots.get(str(i), "?") for i in range(1, n_pos + 1))
    elif ok:
        mx = max(int(d) for d in slots)
        root2 = ".".join(slots.get(str(i), "?") for i in range(1, mx + 1))
    else:
        root2 = None
    mism, hashes, unseen, seen = [], [], [], 0
    if ok and parts1:
        for i in range(1, n_pos + 1):
            pr, got = parts1[i - 1], slots.get(str(i))
            if got is None:
                unseen.append(i)
                continue
            seen += 1
            if pr == "#":
                hashes.append({"slot": i, "pass1": "#", "pass2": got})
            elif norm_rad(pr) != norm_rad(got):
                mism.append({"slot": i, "pass1": pr, "pass2": got})
    both_non_templatic = (not ok and why == "pattern_non_templatic"
                          and parts1 is None)
    agree = bool(both_non_templatic
                 or (ok and parts1 and not mism and not hashes and seen > 0))
    agree_strict = bool(agree and ok and parts1 and not unseen and seen == n_pos)
    return dict(root2=root2, ok=ok, why=why, mism=mism, hashes=hashes,
                unseen=unseen, seen=seen, n_pos=n_pos, parts1=parts1,
                agree=agree, agree_strict=agree_strict,
                both_non_templatic=both_non_templatic)


def main():
    ISB = _load("instrument_scaling_bounds",
                os.path.join(HERE, "instrument_scaling_bounds.py"))
    FAAIL_CAMEL = set(ISB.FAAIL_CAMEL_PATTERNS)

    p1_p = os.path.join(HERE, "exp77_pass1_camel.json")
    inv_p = os.path.join(HERE, "d4_confirmed_inventory.json")
    p2b_p = os.path.join(HERE, "exp77_pass2b_mle.json")
    scaf_p = os.path.join(HERE, "d4_roots_camel.json")

    p1 = json.load(open(p1_p, encoding="utf-8"))
    inv = json.load(open(inv_p, encoding="utf-8"))
    p2b_doc = json.load(open(p2b_p, encoding="utf-8"))
    p2b = {r["noun"]: r for r in p2b_doc["rows"] if r["set"] == "d4_519"}
    p2b_h = {r["noun"]: r for r in p2b_doc["rows"] if r["set"] == "humans_24"}
    scaf = {r["noun"]: r for r in json.load(open(scaf_p, encoding="utf-8"))}
    rows1 = {r["noun"]: r for r in p1["rows"] if r["set"] == "d4_519"}

    out_rows = []
    for n in inv["nouns"]:
        noun, tag = n["noun"], n["wazn_tag"]
        r = rows1[noun]
        pattern, root1 = r["pattern_committed"], r["root_committed"]
        A = analyse_one(pattern, root1, r["lemma_bare_committed"], noun)
        parts1 = A["parts1"]

        # ---- pass 2b (different selector) ---------------------------------
        mle_root = (p2b.get(noun) or {}).get("root_mle_top")
        agree_2b = (mle_root == root1)

        # ---- weak / geminate ----------------------------------------------
        # MEASURED FACT, recorded because it drives the definition below: across
        # all 600 candidate roots CAMeL's builtin DB never spells و, ي or ء as a
        # radical -- an absorbed/weak radical is always written '#'. So "weak"
        # cannot be read off the letters; '#' IS the weak-radical marker.
        hash_pos = [i + 1 for i, p in enumerate(parts1 or []) if p == "#"]
        weak = bool(hash_pos)
        geminate = bool(parts1 and len(parts1) >= 3
                        and parts1[-1] == parts1[-2] and parts1[-1] != "#")

        # ---- nisba ----------------------------------------------------------
        # Keyed on the TRUE PATTERN, not the surface tag. The nisba template
        # ends in ya+shadda (ال1ِ2ْ3ِيّ); the DEFECTIVE participle ends in a
        # plain ya (ال1ا2ِي) and the surface regex confuses the two -- that
        # confusion is 6 of doc 63 SS3.3's 19 missed faail forms, and keying the
        # root-nulling rule on the tag would destroy their legitimate roots.
        nisba_by_pattern = bool(pattern and pattern.endswith("يّ"))
        nisba_by_surface_tag = (tag == "nisba_yaa")
        nisba_flag = nisba_by_pattern
        # "genuine triliteral decomposition of the surface": exactly three root
        # positions, none of them '#', and pass 2 recovers all three FROM THE
        # SURFACE. Place-name and loanword bases fail this.
        genuine_tri = bool(parts1 and len(parts1) == 3 and "#" not in parts1
                           and A["agree_strict"])
        nisba_not_decomposable = bool(nisba_flag and not genuine_tri)

        # ---- root of record --------------------------------------------------
        root_null_reason = None
        if parts1 is None:
            root_null_reason = ("camel_root_NTWS_non_templatic"
                                if root1 == "NTWS" else "camel_root_absent")
        elif nisba_not_decomposable:
            root_null_reason = "nisba_root_not_a_triliteral_decomposition"
        root = None if root_null_reason else root1

        reasons = []
        if tag == "other":
            reasons.append("wazn_tag_other")
        if root_null_reason:
            reasons.append(root_null_reason)

        faail_camel = pattern in FAAIL_CAMEL
        out_rows.append({
            "noun": noun,
            "wazn_tag": tag,
            "pattern": pattern,
            "lemma": r["lemma_committed"],
            "root": root,
            "root_complete_triliteral": genuine_tri,
            "root_pass1_camel_raw": root1,
            "root_second_pass": A["root2"],
            "root_second_pass_route": "pattern x surface alignment",
            "root_third_pass_mle": mle_root,
            "root_third_pass_route": "MLEDisambiguator top analysis",
            "root_majority_vote_scaffolding_NOT_OF_RECORD":
                (scaf.get(noun) or {}).get("root"),
            "agree": A["agree"],
            "agree_strict": A["agree_strict"],
            "agree_third_pass": agree_2b,
            "agree_all_routes": bool(A["agree"] and agree_2b),
            "agree_detail": {
                "slots_compared": A["seen"],
                "root_positions": A["n_pos"],
                "slots_not_exposed_by_pattern": A["unseen"],
                "mismatches": A["mism"],
                "pass1_hash_slots_resolved_by_pass2": A["hashes"],
                "both_passes_say_non_templatic": A["both_non_templatic"],
                "pass2_status": A["why"] if not A["ok"] else "ok",
                "third_pass_candidates": (None if agree_2b else
                                          {"pass1": root1, "pass2b_mle": mle_root,
                                           "pass2b_pattern":
                                               (p2b.get(noun) or {}).get("pattern_mle_top"),
                                           "pass2b_pos":
                                               (p2b.get(noun) or {}).get("pos_mle_top")}),
            },
            "weak": weak,
            "weak_hash_positions": hash_pos,
            "geminate": geminate,
            "faail_by_camel_pattern": faail_camel,
            "weak_faail_missed_by_surface_tag": bool(faail_camel and tag != "faail"),
            "nisba_flag": nisba_flag,
            "nisba_by_true_pattern": nisba_by_pattern,
            "nisba_by_surface_tag": nisba_by_surface_tag,
            "nisba_tag_pattern_disagree": (nisba_by_pattern != nisba_by_surface_tag),
            "nisba_root_not_decomposable": nisba_not_decomposable,
            "excluded_reason": ("; ".join(reasons) if reasons else None),
            "excluded_reasons": reasons,
            "note_committed": n.get("note"),
        })

    # ---------------- the 24 frozen HUMANS, same treatment ------------------
    humans = []
    for r in [x for x in p1["rows"] if x["set"] == "humans_24"]:
        A = analyse_one(r["pattern_reproduced"], r["root_reproduced"],
                        r["lemma_bare_committed"], r["noun"])
        parts1 = A["parts1"]
        mle_root = (p2b_h.get(r["noun"]) or {}).get("root_mle_top")
        humans.append({
            "noun": r["noun"], "gloss_en": r.get("gloss_en"),
            "wazn_tag": r["wazn_tag_committed"],
            "pattern": r["pattern_reproduced"],
            "lemma": r["lemma_reproduced"],
            "root": r["root_reproduced"],
            "root_second_pass": A["root2"],
            "root_third_pass_mle": mle_root,
            "agree": A["agree"], "agree_third_pass": mle_root == r["root_reproduced"],
            "weak": bool(parts1 and "#" in parts1),
            "pass2_status": A["why"] if not A["ok"] else "ok",
            "wazn_tag_provenance": r.get("wazn_tag_provenance"),
        })

    # ------------------------------ meta ------------------------------------
    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    canon = hashlib.sha256(
        json.dumps([n["noun"] for n in inv["nouns"]],
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    n = len(out_rows)
    a2 = sum(1 for r in out_rows if r["agree"])
    a2s = sum(1 for r in out_rows if r["agree_strict"])
    a2b = sum(1 for r in out_rows if r["agree_third_pass"])
    aall = sum(1 for r in out_rows if r["agree_all_routes"])
    dis2 = [r["noun"] for r in out_rows if not r["agree"]]
    dis2b = [r["noun"] for r in out_rows if not r["agree_third_pass"]]
    scaf_same = sum(1 for r in out_rows
                    if r["root_majority_vote_scaffolding_NOT_OF_RECORD"]
                    == r["root_pass1_camel_raw"])

    meta = {
        "purpose": ("exp77 pre-freeze roots pass over the D4 519-noun inventory "
                    "of record: CAMeL root + the TRUE pattern field, with an "
                    "independent second pass (pattern x surface) and a "
                    "third pass under a different analysis selector"),
        "status": ("PRE-FREEZE DATA PASS, not a registration. No experiment "
                   "decision is licensed by this file. The 19 weak-root faail "
                   "forms remain a USER GATE (doc 66 SS5/SS9 item 1); they are "
                   "flagged here, not adjudicated."),
        "pipeline_provenance": {
            "pass1_producer_of_record": "extract_aravec_candidates.py",
            "pass1_functions_reused_by_import": ["_readings", "_classify",
                                                 "wazn_tag", "norm_ta_marbuta",
                                                 "strip_diac"],
            "pass1_field_source": ("root and pattern of the ADMITTING analysis = "
                                   "first Al_det definite NOUN reading with "
                                   "rat='r', gen='m', num='s'; "
                                   "extract_aravec_candidates.py:321-322. The "
                                   "pattern field is what doc 63 SS3.2 made D4's "
                                   "wazn membership of record."),
            "pass1_reproduction": ("exp77_roots_pass1.py on WSL: 0/519 root, "
                                   "0/519 pattern, 0/519 lemma mismatch vs the "
                                   "committed candidate_human_nouns.json, under "
                                   "camel_tools 1.5.5 AND 1.5.6"),
            "pass2_route": ("exp77_roots_pass2_reconcile.py::align -- the stored "
                            "root field is never consulted; radicals are derived "
                            "from (CAMeL pattern x surface)"),
            "pass2_limitation": ("pass 2 rides the SAME admitting analysis as "
                                 "pass 1, so it tests the internal consistency of "
                                 "that analysis, not the choice of analysis. "
                                 "Pass 2b exists because of this."),
            "pass2b_route": ("exp77_roots_pass2b.py -- "
                             "MLEDisambiguator.pretrained() top analysis replaces "
                             "_classify's selection rule; same DB, different "
                             "SELECTOR"),
            "also_reused": ("instrument_scaling_bounds.FAAIL_CAMEL_PATTERNS, "
                            "doc 63 SS3.2's enumerated participle realisations"),
            "third_column_not_of_record": ("d4_roots_camel.json, scaffolding "
                                           "single-pass majority vote"),
            "camel_env_pass1": p1["meta"]["env"],
            "camel_env_pass2b": p2b_doc["meta"],
        },
        "inputs_sha256": dict(
            p1["meta"]["inputs_sha256"],
            **{"exp77_pass1_camel.json": sha(p1_p),
               "exp77_pass2b_mle.json": sha(p2b_p),
               "d4_roots_camel.json": sha(scaf_p),
               "instrument_scaling_bounds.py": sha(
                   os.path.join(HERE, "instrument_scaling_bounds.py"))}),
        "inventory_nouns_sha256": {
            "claimed_in_inventory_meta": inv["meta"]["nouns_sha256"],
            "recomputed_here": canon,
            "match": inv["meta"]["nouns_sha256"] == canon,
            "definition": ("sha256 of json.dumps(list_of_noun_strings, "
                           "ensure_ascii=False, sort_keys=True) -- the sha doc 66 "
                           "SS3 cites as '34e95fc6...'"),
        },
        "agreement": {
            "n": n,
            "pass2_agree": a2, "pass2_agreement_rate": round(a2 / float(n), 4),
            "pass2_disagreements": dis2,
            "pass2_agree_strict": a2s,
            "pass2_agreement_rate_strict": round(a2s / float(n), 4),
            "pass2b_agree": a2b, "pass2b_agreement_rate": round(a2b / float(n), 4),
            "pass2b_disagreements": dis2b,
            "agree_all_routes": aall,
            "agreement_rate_all_routes": round(aall / float(n), 4),
            "definition_pass2_agree": ("every root position the pattern exposes as "
                                       "a radical slot is recovered from the "
                                       "surface and matches pass 1 "
                                       "(hamza-normalised), with no '#' at an "
                                       "exposed slot; OR both passes independently "
                                       "return non-templatic"),
            "definition_pass2_agree_strict": ("agree AND every root position was "
                                              "recovered -- fails wherever a weak "
                                              "radical is absorbed by the template "
                                              "and never surfaces"),
            "scaffolding_majority_vote": {
                "same_as_pass1": scaf_same,
                "rate": round(scaf_same / float(n), 4),
                "verdict": ("the scaffolding differs on %d/%d and is measurably "
                            "worse (e.g. الانسان -> 'ن.س.#' vs '#.ن.س'); it is "
                            "NOT of record and is superseded by this file"
                            % (n - scaf_same, n)),
            },
        },
        "counts": {
            "n_nouns": n,
            "by_wazn_tag": dict(Counter(r["wazn_tag"] for r in out_rows)),
            "weak_hash_radical": sum(1 for r in out_rows if r["weak"]),
            "geminate": sum(1 for r in out_rows if r["geminate"]),
            "faail_by_camel_pattern": sum(1 for r in out_rows
                                          if r["faail_by_camel_pattern"]),
            "weak_faail_missed_by_surface_tag": sum(
                1 for r in out_rows if r["weak_faail_missed_by_surface_tag"]),
            "nisba_by_true_pattern": sum(1 for r in out_rows
                                         if r["nisba_by_true_pattern"]),
            "nisba_by_surface_tag": sum(1 for r in out_rows
                                        if r["nisba_by_surface_tag"]),
            "nisba_tag_pattern_disagree": sum(1 for r in out_rows
                                              if r["nisba_tag_pattern_disagree"]),
            "nisba_root_not_decomposable": sum(
                1 for r in out_rows if r["nisba_root_not_decomposable"]),
            "root_is_null": sum(1 for r in out_rows if r["root"] is None),
            "root_present": sum(1 for r in out_rows if r["root"]),
            "root_complete_triliteral": sum(1 for r in out_rows
                                            if r["root_complete_triliteral"]),
            "excluded_any_reason": sum(1 for r in out_rows if r["excluded_reason"]),
            "exclusion_reason_histogram": dict(Counter(
                x for r in out_rows for x in r["excluded_reasons"])),
            "weak_radical_note": ("across all 600 candidate roots CAMeL's builtin "
                                  "DB never spells و, ي or ء as a radical; an "
                                  "absorbed/weak radical is always written '#'. "
                                  "'weak' here therefore means \"root carries at "
                                  "least one '#'\"."),
        },
        "humans_24_note": ("the 24 frozen HUMANS were excluded from the D4 "
                           "extraction by construction (candidate_human_nouns."
                           "json meta.frozen_44_excluded_from_extraction). They "
                           "are analysed here with the same imported pipeline so "
                           "the committed category task can be joined to a "
                           "wazn/root. Their wazn_tag is COMPUTED here by the "
                           "committed regex; it is not an inventory field."),
    }

    out = {"meta": meta, "nouns": out_rows, "humans_24": humans}
    op = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1
                      else "exp77_roots_working.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[pass2] wrote %s" % op)
    print(json.dumps({k: v for k, v in meta["agreement"].items()
                      if k not in ("pass2_disagreements",)},
                     ensure_ascii=False, indent=1))
    print(json.dumps(meta["counts"], ensure_ascii=False, indent=1))
    print("nouns_sha match:", meta["inventory_nouns_sha256"]["match"])


if __name__ == "__main__":
    main()
