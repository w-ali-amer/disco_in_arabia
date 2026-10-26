#!/usr/bin/env python3
"""exp77 pre-freeze pass 1 -- REPRODUCTION of the committed CAMeL root/pattern
extraction over the D4 inventory of record, plus the 24 frozen HUMANS.

PROVENANCE. This script RE-USES, by import and never by retyping, the exact
functions of the committed producer of `candidate_human_nouns.json`:

    extract_aravec_candidates.py::_readings   (Al_det definite readings)
    extract_aravec_candidates.py::_classify   (admission tiers; the admitting
                                               analysis is the first rat='r',
                                               gen='m', num='s' NOUN reading)
    extract_aravec_candidates.py::wazn_tag    (the shipped surface regex tag)
    extract_aravec_candidates.py::norm_ta_marbuta / strip_diac

`root` and `pattern` are taken off the SAME admitting analysis that
`stage_candidates` (lines 321-322) took them off. That is what makes D4's wazn
membership "of record" per doc 63 SS3.2, and it is what this pass reproduces.

Nothing here is written as *_of_record. Output is a working file consumed by
the Mac-side pass-2 reconciler.
"""
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EX = _load("extract_aravec_candidates",
           os.path.join(HERE, "extract_aravec_candidates.py"))
INS = _load("instrument_sb_bounds", os.path.join(HERE, "instrument_sb_bounds.py"))


def file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    import camel_tools

    inv_p = os.path.join(HERE, "d4_confirmed_inventory.json")
    cand_p = os.path.join(HERE, "candidate_human_nouns.json")
    inv = json.load(open(inv_p, encoding="utf-8"))
    cand_doc = json.load(open(cand_p, encoding="utf-8"))
    cand = {c["noun"]: c for c in cand_doc["candidates"]}

    db = MorphologyDB.builtin_db()
    an = Analyzer(db)

    rows = []

    def analyse(noun, token):
        readings = EX._readings(an, token)
        tier, admit, st = EX._classify(readings)
        # every distinct root over ALL definite noun readings -- the material
        # the scaffolding majority vote used; kept for disagreement diagnosis.
        alt = {}
        for a in readings:
            if a.get("pos") == "noun" and a.get("root"):
                alt[a["root"]] = alt.get(a["root"], 0) + 1
        return {
            "noun": noun,
            "token_analysed": token,
            "tier_reproduced": tier,
            "root_reproduced": (admit or {}).get("root"),
            "pattern_reproduced": (admit or {}).get("pattern"),
            "lemma_reproduced": str((admit or {}).get("lex", "")) or None,
            "stemgloss_reproduced": (admit or {}).get("stemgloss"),
            "n_definite_readings": st["n_definite_readings"],
            "n_noun_readings": st["n_noun_readings"],
            "root_multiset_all_noun_readings": alt,
        }

    # ---- the 519 inventory of record -------------------------------------
    for n in inv["nouns"]:
        c = cand[n["noun"]]
        r = analyse(n["noun"], c["aravec_key"])
        r["set"] = "d4_519"
        r["wazn_tag_committed"] = n["wazn_tag"]
        r["root_committed"] = c["root"]
        r["pattern_committed"] = c["pattern"]
        r["lemma_committed"] = c["lemma"]
        r["lemma_bare_committed"] = c["lemma_bare"]
        r["stemgloss_committed"] = c.get("camel_stemgloss")
        r["note_committed"] = n.get("note")
        rows.append(r)

    # ---- the 24 frozen HUMANS (excluded from D4 extraction by design) -----
    for h in INS.HUMANS:
        r = analyse(h, EX.norm_ta_marbuta(h))
        r["set"] = "humans_24"
        lb = EX.strip_diac(r["lemma_reproduced"] or "")
        r["wazn_tag_committed"] = EX.wazn_tag(lb) if lb else None
        r["wazn_tag_provenance"] = ("computed here by the committed wazn_tag() "
                                    "regex over the reproduced lemma; the 24 "
                                    "were excluded from the D4 extraction")
        r["root_committed"] = None
        r["pattern_committed"] = None
        r["lemma_bare_committed"] = lb or None
        r["gloss_en"] = INS.GLOSS[h]
        rows.append(r)

    out = {
        "meta": {
            "purpose": ("exp77 pre-freeze PASS 1 (working file, NOT of record): "
                        "reproduction of the committed CAMeL root/pattern "
                        "extraction over the 519 D4 inventory + the 24 HUMANS"),
            "pipeline_reused": {
                "producer_of_record": "extract_aravec_candidates.py",
                "functions_imported": ["_readings", "_classify", "wazn_tag",
                                       "norm_ta_marbuta", "strip_diac"],
                "root_pattern_source": ("the admitting analysis, i.e. the first "
                                        "Al_det definite NOUN reading with "
                                        "rat='r', gen='m', num='s' -- "
                                        "extract_aravec_candidates.py:321-322"),
                "camel_db": "camel_tools MorphologyDB.builtin_db()",
            },
            "env": {
                "host": "wsl",
                "python": sys.version.split()[0],
                "camel_tools": camel_tools.__version__,
                "camel_data_version": getattr(db, "version", None),
            },
            "inputs_sha256": {
                "d4_confirmed_inventory.json": file_sha(inv_p),
                "candidate_human_nouns.json": file_sha(cand_p),
                "extract_aravec_candidates.py": file_sha(
                    os.path.join(HERE, "extract_aravec_candidates.py")),
                "instrument_sb_bounds.py": file_sha(
                    os.path.join(HERE, "instrument_sb_bounds.py")),
            },
            "inventory_nouns_sha256_claimed": inv["meta"]["nouns_sha256"],
            "n_519": sum(1 for r in rows if r["set"] == "d4_519"),
            "n_humans_24": sum(1 for r in rows if r["set"] == "humans_24"),
        },
        "rows": rows,
    }
    op = os.path.join(HERE, "exp77_pass1_camel.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # --- reproducibility report, printed ---------------------------------
    d = [r for r in rows if r["set"] == "d4_519"]
    rmis = [r["noun"] for r in d if r["root_reproduced"] != r["root_committed"]]
    pmis = [r["noun"] for r in d if r["pattern_reproduced"] != r["pattern_committed"]]
    lmis = [r["noun"] for r in d if r["lemma_reproduced"] != r["lemma_committed"]]
    print("[pass1] wrote %s" % op)
    print("[pass1] camel_tools %s  db %s  python %s"
          % (camel_tools.__version__, getattr(db, "version", None),
             sys.version.split()[0]))
    print("[pass1] 519 root mismatch vs committed: %d %s" % (len(rmis), rmis[:15]))
    print("[pass1] 519 pattern mismatch vs committed: %d %s" % (len(pmis), pmis[:15]))
    print("[pass1] 519 lemma mismatch vs committed: %d %s" % (len(lmis), lmis[:15]))
    print("[pass1] 24 HUMANS analysed: %d, no admitting analysis: %d"
          % (len([r for r in rows if r["set"] == "humans_24"]),
             len([r for r in rows if r["set"] == "humans_24"
                  and r["root_reproduced"] is None
                  and r["pattern_reproduced"] is None])))


if __name__ == "__main__":
    main()
