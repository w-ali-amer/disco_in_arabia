# -*- coding: utf-8 -*-
"""exp110_chains.py -- doc 110 (REGISTERED 5886cfc): idafa chains + adjective attachment from PADT.

Chain: idafa links = NOUN/PROPN child with deprel nmod* and Case=Gen under a NOUN/PROPN head; maximal chain through the
adjective's head (walk up, then down following the closest idafa child).  Candidates = chain nouns preceding the ADJ.
Noun agreement class: CAMeL (gen, rat) [lemma-matched if possible; else unique over noun analyses of the form; else
AMBIGUOUS], number from PADT; deflection plural+irrational -> (f, s).  ADJ class: its PADT Gender/Number.
Split (a) distinct / (b) tied / (c) no-match.  Writes chains_exp110.json (Rule 3).  Run with ~/camel_mac_env.
"""
import collections, json, os
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
import exp94_common as C94
import exp102_census as C102

FILES = ["data/UD_Arabic-PADT/ar_padt-ud-%s.conllu" % s for s in ("train", "dev", "test")]
NUM = {"Sing": "s", "Dual": "d", "Plur": "p"}; GEN = {"Masc": "m", "Fem": "f"}
NOUN = ("NOUN", "PROPN")


def sentences():
    for f in FILES:
        cur = []
        for l in open(f, encoding="utf-8"):
            l = l.rstrip("\n")
            if not l:
                if cur:
                    yield cur
                cur = []; continue
            c = l.split("\t")
            if len(c) != 10 or "-" in c[0] or "." in c[0]:
                continue
            feats = dict(kv.split("=", 1) for kv in c[5].split("|")) if c[5] != "_" else {}
            cur.append({"id": int(c[0]), "form": c[1], "lemma": c[2], "up": c[3], "f": feats, "h": int(c[6]), "dr": c[7]})
        if cur:
            yield cur


def is_idafa(child, parent):
    return child["up"] in NOUN and parent["up"] in NOUN and child["dr"].startswith("nmod") and child["f"].get("Case") == "Gen"


def main():
    an = Analyzer(MorphologyDB.builtin_db()); cache = {}
    def noun_class(t):
        k = (t["form"], t["lemma"])
        if k not in cache:
            al = [a for a in an.analyze(t["form"]) if a.get("pos") in ("noun", "noun_prop")]
            lm = [a for a in al if C102.norm(a.get("lex")) == C102.norm(t["lemma"])] or al
            gr = {(a.get("gen"), a.get("rat")) for a in lm}
            cache[k] = gr.pop() if len(gr) == 1 else None
        gr = cache[k]; num = NUM.get(t["f"].get("Number"))
        if gr is None or gr[0] not in ("m", "f") or num is None:
            return None
        g, rat = gr
        return ("f", "s") if (num == "p" and rat == "i") else (g, num)
    st = collections.Counter(); Kd = collections.Counter(); rows = []
    for s in sentences():
        idx = {t["id"]: t for t in s}
        kids = collections.defaultdict(list)
        for t in s:
            kids[t["h"]].append(t)
        for a in s:
            if a["up"] != "ADJ" or not a["dr"].startswith("amod") or a["h"] not in idx or idx[a["h"]]["up"] not in NOUN:
                continue
            st["adj"] += 1
            head = idx[a["h"]]; top = head
            while top["h"] in idx and is_idafa(top, idx[top["h"]]):
                top = idx[top["h"]]
            chain = [top]; cur = top
            while True:
                ch = sorted([k for k in kids[cur["id"]] if is_idafa(k, cur)], key=lambda k: abs(k["id"] - cur["id"]))
                if not ch:
                    break
                cur = ch[0]; chain.append(cur)
            if head not in chain:
                st["head_off_main_path"] += 1; chain.append(head)
            cands = sorted([n for n in chain if n["id"] < a["id"]], key=lambda n: n["id"])
            if head not in cands:
                st["head_after_adj"] += 1; continue
            K = len(cands); Kd[K] += 1
            if K < 2:
                continue
            cls = [noun_class(n) for n in cands]
            if any(c is None for c in cls):
                st["ambiguous_noun_class"] += 1; continue
            ac = (GEN.get(a["f"].get("Gender")), NUM.get(a["f"].get("Number")))
            if None in ac:
                st["adj_class_missing"] += 1; continue
            gold = cands.index(head)
            match = [i for i, c in enumerate(cls) if c == ac]
            kind = "a" if len(match) == 1 else ("b" if len(match) >= 2 else "c")
            st["split_" + kind] += 1
            row = {"K": K, "x": ["%s%s" % c for c in cls], "r": "%s%s" % ac, "gold": gold, "kind": kind}
            if kind == "a":
                st["a_agreement_correct"] += int(match[0] == gold)
            if kind == "b":
                row["proximity_correct"] = int(max(match) == gold)
                cases = [cands[i]["f"].get("Case") for i in match]
                row["case_breaks_tie"] = int(sum(1 for c in cases if c == a["f"].get("Case")) == 1)
            rows.append(row)
    out = {"doc": "110 (5886cfc)", "stats": dict(st), "K_dist": dict(Kd), "rows": rows, "provenance": C94.provenance()}
    C94.write_new_json("chains_exp110.json", out)
    tot = sum(Kd.values())
    print(json.dumps({"stats": dict(st), "K_dist": dict(sorted(Kd.items())), "share_K>=2": round(sum(v for k, v in Kd.items() if k >= 2) / max(1, tot), 4)}, indent=1))


if __name__ == "__main__":
    main()
