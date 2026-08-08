# -*- coding: utf-8 -*-
"""Full-corpus audit: which rule fires for every sentence in every split.
Usage: python audit.py OUTPUT.json
Captures the '→ XXX' rule log from arabic_dep_reader as the rule identity.
"""
import sys, json, logging, traceback

OUT = sys.argv[1] if len(sys.argv) > 1 else "audit_before.json"

CAPTURED = []
class RuleCatcher(logging.Handler):
    def emit(self, record):
        m = record.getMessage()
        if m.startswith("→"):   # → arrow
            CAPTURED.append(m.replace("→", "").strip())

import arabic_dep_reader as adr
adr.logger.setLevel(logging.DEBUG)
adr.logger.addHandler(RuleCatcher())
# silence camel_test2 chatter to keep log clean
logging.getLogger("camel_test2").setLevel(logging.CRITICAL)

from camel_test2 import analyze_arabic_sentence_with_morph

data = json.load(open("sentences.json"))

result = {}
total = sum(len(v) for v in data.values())
done = 0
for split, items in data.items():
    counts = {}
    per = []
    fallbacks = []
    for idx, item in enumerate(items):
        sent = item.get("sentence", "")
        label = item.get("label", "")
        rule = "ERROR"
        diagram = ""
        toks = []
        try:
            CAPTURED.clear()
            tokens, analyses, structure, roles = analyze_arabic_sentence_with_morph(sent)
            diag = adr.sentence_to_diagram_from_parse(tokens, analyses, structure, roles)
            rule = CAPTURED[-1] if CAPTURED else "NONE"
            diagram = str(diag)
            toks = [{"t": a["text"], "u": a["upos"], "d": a["deprel"], "h": a["head"]}
                    for a in analyses]
        except Exception as e:
            rule = "EXC"
            diagram = "EXC:" + repr(e)
            traceback.print_exc()
        counts[rule] = counts.get(rule, 0) + 1
        rec = {"idx": idx, "sentence": sent, "label": label,
               "rule": rule, "structure": structure if 'structure' in dir() else "",
               "diagram": diagram, "tokens": toks}
        per.append(rec)
        if rule == "fallback":
            fallbacks.append({"idx": idx, "sentence": sent, "label": label,
                              "structure": rec["structure"], "diagram": diagram})
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{total} ...", flush=True)
    result[split] = {"counts": counts, "n": len(items),
                     "fallback_sentences": fallbacks, "per_sentence": per}
    print(f"[{split}] counts={counts}", flush=True)

json.dump(result, open(OUT, "w"), ensure_ascii=False, indent=1)
print("WROTE", OUT, flush=True)
