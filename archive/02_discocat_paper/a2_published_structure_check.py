# -*- coding: utf-8 -*-
"""a2_published_structure_check.py -- doc 82 §13 A2 (registered 4b5a75d before running).
For every published task in archive/02_discocat_paper/sentences.json: parse with the
published reader (arabic_dep_reader.sentences_to_diagrams, default flags, the Swap
convention as published), record the word-box type signature per label, and count items
whose compiled DIAGRAM is identical to an item with a DIFFERENT label (the circuit cannot
separate them). WSD tasks: per verb. Pre-committed WSD rule: >=30% of a verb's items in
opposite-label identical-diagram collisions => that verb's sub-chance WSD is parser
collapse (or identical strings) and its symmetric correction is VOID."""
import os, sys, json, hashlib, logging
from collections import Counter, defaultdict
logging.disable(logging.WARNING)
ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, "archive", "02_discocat_paper"); PIPE = os.path.join(ROOT, "archive", "01_parser_pipeline")
sys.path.insert(0, PIPE)
from lambeq.backend.grammar import Word
from arabic_dep_reader import sentences_to_diagrams
data = json.load(open(os.path.join(PAPER, "sentences.json"), encoding="utf-8"))
def sig(d): return "|".join(str(b.cod) for b in d.boxes if isinstance(b, Word))
def key(d): return repr(d)
out = {"purpose": "doc 82 §13 A2", "sentences_sha256": hashlib.sha256(open(os.path.join(PAPER, "sentences.json"), "rb").read()).hexdigest(), "tasks": {}}
for task, items in data.items():
    sents = [x["sentence"] for x in items]; labs = [x["label"] for x in items]
    diags = sentences_to_diagrams(sents, log_interval=0)
    k = [key(d) for d in diags]
    bykey = defaultdict(set)
    for kk, l in zip(k, labs): bykey[kk].add(l)
    collide = [len(bykey[kk]) > 1 for kk in k]
    same_string_conflict = defaultdict(set)
    for s, l in zip(sents, labs): same_string_conflict[s].add(l)
    str_coll = [len(same_string_conflict[s]) > 1 for s in sents]
    t = {"n": len(items),
         "structure_by_label": {l: dict(Counter(sig(d) for d, ll in zip(diags, labs) if ll == l)) for l in sorted(set(labs))},
         "n_diagram_collision_opposite_label": sum(collide),
         "n_identical_string_opposite_label": sum(str_coll),
         "single_box_fallbacks": sum(1 for d in diags if len(d.boxes) == 1)}
    if "verb" in items[0]:
        per = {}
        for v in sorted({x["verb"] for x in items}):
            idx = [i for i, x in enumerate(items) if x["verb"] == v]
            per[v] = {"n": len(idx), "diagram_collision": sum(collide[i] for i in idx),
                      "frac_diagram_collision": round(sum(collide[i] for i in idx) / len(idx), 4),
                      "identical_string_conflict": sum(str_coll[i] for i in idx),
                      "structure": dict(Counter(sig(diags[i]) for i in idx)),
                      "collapse_rule_fires_ge_0.30": sum(collide[i] for i in idx) / len(idx) >= 0.30}
        t["per_verb"] = per
    out["tasks"][task] = t
    print(task, "n", t["n"], "diag-collisions", t["n_diagram_collision_opposite_label"], "string-conflicts", t["n_identical_string_opposite_label"], "fallbacks", t["single_box_fallbacks"])
    if "per_verb" in t:
        for v, r in t["per_verb"].items(): print("   ", v, r["n"], "collide", r["diagram_collision"], "(%.2f)" % r["frac_diagram_collision"], "strconf", r["identical_string_conflict"], "RULE" if r["collapse_rule_fires_ge_0.30"] else "")
OUT = os.path.join(ROOT, "results_a2_published_structure_check.json")
assert not os.path.exists(OUT), "Rule 3"
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT)
