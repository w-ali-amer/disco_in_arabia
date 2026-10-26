# -*- coding: utf-8 -*-
"""exp106_roots.py -- doc 104 §1 step 1 (REGISTERED 8c7e3bd): list the 50 most frequent roots among PADT NOUN tokens
in the 4 frozen template classes.  OUTPUT = root strings + total frequency ONLY (no template breakdown), so the field
assignment (step 2) is blind to templates.  Writes roots_exp106.json (Rule 3).  Run with ~/camel_mac_env.
"""
import collections, json, os
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
import exp94_common as C94
import exp100_census as CEN
import exp102_census as C102

TEMPLATES = {
    "AGENT": {"1ا2ِ3", "1ا2ِ3َة", "1ائِ3", "1ائِ3َة"},
    "PATIENT": {"مَ1ْ2ُو3", "مَ1ْ2ُو3َة", "مَ1ْؤُو3"},
    "PLACE": {"مَ1ْ2َ3", "مَ1ْ2ِ3", "مَ1ْ2َ3َة", "مَ1ْ2ِ3َة", "مِ1ْ2َ3َة", "مَ1ا3", "مَ1ا3َة"},
    "MASDAR": set(C102.MASDAR),
}


def template_of(p):
    for k, s in TEMPLATES.items():
        if p in s:
            return k
    return None


def analysed_nouns():
    an = Analyzer(MorphologyDB.builtin_db()); cache = {}
    for fn in CEN.FILES:
        for sent in CEN.sentences(os.path.join(CEN.DATA, fn)):
            pass
    for fn in CEN.FILES:
        for line in open(os.path.join(CEN.DATA, fn), encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if len(c) != 10 or "-" in c[0] or "." in c[0] or c[3] != "NOUN":
                continue
            k = (c[1], c[2])
            if k not in cache:
                keep = [a for a in an.analyze(c[1]) if a.get("pos") == "noun" and C102.norm(a.get("lex")) == C102.norm(c[2])]
                roots = {a.get("root") for a in keep}
                cache[k] = (keep[0].get("root"), C102.pat(keep[0].get("pattern"))) if keep and len(roots) == 1 else None
            if cache[k]:
                yield c, cache[k]


def main():
    cnt = collections.Counter()
    for c, (root, p) in analysed_nouns():
        if template_of(p):
            cnt[root] += 1
    N = int(os.environ.get("TOPN", "50")); top = cnt.most_common(N)
    C94.write_new_json("roots_exp106.json" if N == 50 else "roots_exp106_top%d.json" % N, {"doc": "104 §1 step 1 (8c7e3bd)", "top50": [{"root": r, "freq": n} for r, n in top],
                                              "n_roots_total": len(cnt), "provenance": C94.provenance()})
    print(" ".join(r for r, _ in top))


if __name__ == "__main__":
    main()
