# -*- coding: utf-8 -*-
"""wazn_b1_freeze.py -- doc 86 §5.2: FREEZE of the verb-pair lexicon (user, 2026-09-26: "do both as u see fit").

Frozen = rated clean (B2') or likely (B2'-ext) AND STRONG AVN corroboration AND no gloss/register/
homonymy/lability flag. Reads wazn_b1/candidates.json (ebd7b55) read-only; writes wazn_b1/frozen_lexicon_v1.json.
The TASK subset (B4 readout) is chosen later from this frozen list; the list itself does not change.
"""
import json, hashlib, os
SRC = "wazn_b1/candidates.json"
OUT = "wazn_b1/frozen_lexicon_v1.json"
assert not os.path.exists(OUT), "Rule 3"
d = json.load(open(SRC, encoding="utf-8"))
KEEP = {("B2'", "clean"), ("B2'", "pending_lexicon"), ("B2'-ext(provisional)", "likely"),
        ("B2'-ext(provisional)", "likely_pending_hash_check")}
BLOCK_FLAGS = {"I_labile_in_CAMeL", "I_homonymous"}
# Form I lexeme chosen where CAMeL has several Form I lexemes for the pair (the sense the rating is about)
PICK_I = {"برز/أبرز": "appear;emerge", "خلع/انخلع": "take_off;rip_out"}
HUMAN_UNDERGOER = {"هزم/انهزم", "شغل/انشغل", "فرح/أفرح", "غضب/أغضب", "خجل/أخجل", "تعب/أتعب", "قلق/أقلق", "خاف/أخاف"}
frozen = {"I-VII": [], "I-IV": []}
for k, v in sorted(d["header"]["b2prime_coverage"].items()):
    if (v["source"], v["rating"]) not in KEEP:
        continue
    form, pair = k.split(":")
    rows = [r for r in v["rows"] if r["avn_derived_form_sibling"]]
    if pair in PICK_I:
        rows = [r for r in rows if PICK_I[pair] in r["I_gloss"]]
    blocked = set().union(*[set(r["flags"]) for r in rows]) & BLOCK_FLAGS if rows else {"no_row"}
    if pair == "خلع/انخلع":   # I_labile_in_CAMeL fires only via the separate lexeme خَلُع 'be_dissolute' (doc 86 §5.1 flag note)
        blocked -= {"I_labile_in_CAMeL"}
    if not rows or blocked:
        print("NOT FROZEN", k, v["rating"], sorted(blocked))
        continue
    assert len(rows) == 1, (k, len(rows))
    r = rows[0]
    frozen["I-" + form].append({"pair": pair, "rating_source": v["source"], "rating": v["rating"],
                                "root": r["root"], "I": r["I"], "I_gloss": r["I_gloss"], "I_logprob": r["I_logprob"],
                                form: r[form], form + "_gloss": r[form + "_gloss"], form + "_logprob": r[form + "_logprob"],
                                "avn_valency_contrast_info": r["avn_valency_contrast_info"],
                                "flags_info": r["flags"], "human_undergoer": pair in HUMAN_UNDERGOER})
out = {"doc": "86 §5.2 frozen verb-pair lexicon v1", "source": SRC,
       "source_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
       "rule": "clean/likely model rating AND STRONG AVN corroboration AND no gloss/register/homonymy/lability flag",
       "disclosure": "Item transparency was model-judged and lexicon-corroborated; not native-speaker validated.",
       "counts": {a: len(v) for a, v in frozen.items()},
       "counts_human_undergoer": {a: sum(x["human_undergoer"] for x in v) for a, v in frozen.items()},
       "pairs": frozen}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1, sort_keys=True)
print(out["counts"], out["counts_human_undergoer"])
for a, v in frozen.items():
    print(a, " ".join(x["pair"] for x in v))
