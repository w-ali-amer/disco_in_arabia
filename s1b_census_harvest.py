# -*- coding: utf-8 -*-
"""S1b: verb census + subject/object harvest from arwiki_sents.txt.

1. Census: frequent tokens with a CAMeL perfective-verb reading → candidate
   verb list (top 250 by corpus frequency).
2. Sample: up to 150 corpus sentences per verb (global cap 35k parses).
3. Parse: Stanza+CAMeL roles (verb/subject/object indices); keep triples
   where the target verb is the clause verb.
Output: s1_harvest.json — verb → {subjects, objects, n_sampled, n_hit}.

Run in the lambeq env with ARABIC_POS_FUSION=1.
"""
import json, os
from collections import Counter, defaultdict

MAX_PER_VERB = 150
GLOBAL_CAP = 35000
TOP_VERBS = 250
MIN_FREQ = 80

def hn(w):
    return w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

print("[s1b] census pass…", flush=True)
freq = Counter()
sents = []
with open("arwiki_sents.txt", encoding="utf-8") as f:
    for line in f:
        t = line.split()
        sents.append(t)
        for w in t:
            freq[hn(w)] += 1
print(f"[s1b] {len(sents)} sentences, {len(freq)} token types", flush=True)

import camel_test2
CA = camel_test2.CAMEL_ANALYZER

def is_verb_form(w):
    if len(w) < 3 or len(w) > 6:
        return False
    if w.startswith(("ال", "لل", "بال", "كال", "وال")):
        return False
    try:
        for a in CA.analyze(w) or []:
            if isinstance(a, dict) and a.get("pos") == "verb" \
                    and a.get("asp") in ("p", "i"):
                return True
    except Exception:
        return False
    return False

cands = []
for w, c in freq.most_common(6000):
    if c < MIN_FREQ:
        break
    if is_verb_form(w):
        cands.append((w, c))
    if len(cands) >= TOP_VERBS:
        break
print(f"[s1b] candidate verbs: {len(cands)} "
      f"(top: {[w for w, _ in cands[:15]]})", flush=True)

by_verb = defaultdict(list)
total = 0
for t in sents:
    if total >= GLOBAL_CAP:
        break
    hset = {hn(w) for w in t}
    for v, _ in cands:
        if v in hset and len(by_verb[v]) < MAX_PER_VERB:
            by_verb[v].append(" ".join(t))
            total += 1
            break
print(f"[s1b] sampled {total} sentences across {len(by_verb)} verbs",
      flush=True)

from camel_test2 import analyze_arabic_sentence_with_morph
harvest = {v: {"subjects": [], "objects": [], "n_sampled": len(ss),
               "n_hit": 0} for v, ss in by_verb.items()}
done = 0
for v, ss in by_verb.items():
    for s in ss:
        done += 1
        if done % 500 == 0:
            print(f"[s1b] parsed {done}/{total}", flush=True)
        try:
            toks, ana, struct, roles = analyze_arabic_sentence_with_morph(s)
        except Exception:
            continue
        vi, si, oi = roles.get("verb"), roles.get("subject"), roles.get("object")
        if vi is None or si is None:
            continue
        if hn(ana[vi]["text"]) != v:
            continue
        rec = harvest[v]
        rec["subjects"].append(ana[si]["text"])
        if oi is not None:
            rec["objects"].append(ana[oi]["text"])
        rec["n_hit"] += 1
json.dump(harvest, open("s1_harvest.json", "w"), indent=1,
          ensure_ascii=False)
ok = sorted(((v, r["n_hit"]) for v, r in harvest.items()),
            key=lambda x: -x[1])
print(f"[s1b] verbs with ≥15 subject hits: "
      f"{sum(1 for _, n in ok if n >= 15)}", flush=True)
print(f"[s1b] top harvests: {ok[:15]}", flush=True)
print("[s1b] DONE", flush=True)
