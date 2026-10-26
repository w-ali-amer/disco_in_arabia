# -*- coding: utf-8 -*-
"""make_failure_catalogue.py -- builds FAILURE_CASES.md (shareable) from the archived pipeline (Stanza 1.10.1,
reproduces results_a2 exactly) and committed results (results_exp117_v2.json fix log).  Original vs rebuilt parser."""
import json, os, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import exp119_run as R
data = json.load(open(os.path.join(R.PAPER, "sentences.json"), encoding="utf-8"))
out = ["# Known failure cases in the Arabic → pregroup → circuit pipeline", "",
       "Generated from the archived pipeline (`archive/01_parser_pipeline/arabic_dep_reader.py`, Stanza 1.10.1, lambeq 0.5.0),",
       "which reproduces the published structure exactly. \"Original\" is the pipeline as used in the paper. \"Rebuilt\" adds the",
       "opt-in POS-fusion rules (`ARABIC_POS_FUSION=1`) and a verb-recovery rule: if no token is tagged VERB, the first",
       "non-ال token with a CAMeL verb analysis becomes the verb. Sentences are from `archive/02_discocat_paper/sentences.json`.", ""]
# 1. identical strings with opposite labels
out += ["## 1. Identical sentences with opposite labels (data errors)", ""]
for task, items in data.items():
    by = defaultdict(set)
    for x in items:
        by[x["sentence"]].add(x["label"])
    bad = {s: sorted(l) for s, l in by.items() if len(l) > 1}
    for s, l in bad.items():
        out.append("- `%s` — %s: %s" % (task, s, " / ".join(l)))
out.append("")
# 2. identical circuits, opposite labels (original vs rebuilt)
out += ["## 2. Different sentences that compile to the identical diagram with opposite labels", ""]
for task in ("WordSenseDisambiguation_v2", "TenseBinary", "WordOrderMatched", "LexicalAmbiguity"):
    items = data[task]; sents = [x["sentence"] for x in items]
    for imp in (False, True):
        ds = R.parse(sents, imp); groups = defaultdict(list)
        for d, x in zip(ds, items):
            groups[repr(d)].append(x)
        coll = [g for g in groups.values() if len({x["label"] for x in g}) > 1 and len({x["sentence"] for x in g}) > 1]
        n_items = sum(len(g) for g in coll)
        out.append("**%s, %s parser:** %d collision groups covering %d items." % (task, "rebuilt" if imp else "original", len(coll), n_items))
        if not imp:
            for g in coll[:6]:
                sig = R.sig(ds[sents.index(g[0]["sentence"])])
                out.append("- diagram `%s`: " % sig + "; ".join("%s (%s)" % (x["sentence"], x["label"]) for x in g[:4]))
        out.append("")
# 2b. per-verb WSD breakdown, counting rule of the rebuilt-pipeline experiment (exp119)
out += ["### Word-sense collisions per verb", "",
        "Counting rule: first drop identical sentences with opposite labels (section 1), then count, per verb, the items whose diagram",
        "is identical to an item with the opposite sense. (The pooled counts above keep identical sentences and pool all verbs, so they differ.)", "",
        "| verb | items after dropping identical sentences | original parser | rebuilt parser |", "|---|---|---|---|"]
W = data["WordSenseDisambiguation_v2"]
for v in sorted({x["verb"] for x in W}):
    its = [x for x in W if x["verb"] == v]; ss = defaultdict(set)
    for x in its:
        ss[x["sentence"]].add(x["label"])
    its = [x for x in its if len(ss[x["sentence"]]) == 1]; sents = [x["sentence"] for x in its]; row = []
    for imp in (False, True):
        ds = R.parse(sents, imp); byk = defaultdict(set)
        for d, x in zip(ds, its):
            byk[repr(d)].add(x["label"])
        row.append(sum(len(byk[repr(d)]) > 1 for d in ds))
    out.append("| %s | %d | %d | %d |" % (v, len(its), row[0], row[1]))
out.append("")
# 3. tense items typed verbless
v2 = json.load(open(os.path.join(ROOT, "results_exp117_v2.json"), encoding="utf-8"))
tc = v2["tense_check"]
out += ["## 3. Past-tense verbs parsed as nouns (sentence typed as verbless)", "",
        "Verb-typed items before the fix: past %d/50, present %d/50. After the verb-recovery rule: past %d/50, present %d/50." % (
            tc["verb_typed_before"]["Tense_Past"], tc["verb_typed_before"]["Tense_Pres"], tc["verb_typed_after"]["Tense_Past"], tc["verb_typed_after"]["Tense_Pres"]), ""]
for f in tc["fix_log"]:
    out.append("- %s — verb token `%s` was tagged %s" % (f["sentence"], f["token"], f["old_upos"]))
out.append("")
# 4. dropped words and mis-tagged verbs (examples)
out += ["## 4. Other examples", ""]
ex = ["رفعت المرأة الدعوى الجديدة", "كتب الطالب الدرس"]
for s, d in zip(ex, R.parse(ex, False)):
    out.append("- `%s` → word boxes: %s" % (s, ", ".join("%s : %s" % (b.name, b.cod) for b in d.boxes if isinstance(b, R.Word))))
out += ["", "In the first, the adjective الجديدة is silently dropped. In the second, the verb كتب carries the noun-plural tag `NUM-p` (\"books\").", "",
        "## 5. The VSO Swap is a convention", "",
        "The VSO type s·nˡ·nˡ applied to two nouns also reduces with nested planar cups, without a Swap. The pipeline's Swap fixes which",
        "verb leg binds the subject. Crossed and planar diagrams with the same bindings give identical circuit features.", "",
        "## 6. Reproducibility", "", "The parses depend on the Stanza version: 1.10.1 reproduces the published structure; 1.15 changes the collision counts",
        "(e.g. 99 vs 75 in the word-sense set)."]
p = os.path.join(ROOT, "FAILURE_CASES.md"); assert not os.path.exists(p)
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n"); print("wrote", p, len(out))
