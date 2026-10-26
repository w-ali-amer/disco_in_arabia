#!/usr/bin/env python3
"""exp77 pass 2b -- a DIFFERENT ANALYSIS-SELECTION ROUTE.

Pass 2 (pattern x surface) shares the *admitting analysis* with pass 1: it
tests whether the analyser's pattern and root fields decompose the surface
consistently, not whether a different analysis would have been chosen. The
degree of freedom pass 2 cannot probe is exactly the one
`extract_aravec_candidates._classify` exercises: "first Al_det definite NOUN
reading with rat='r', gen='m', num='s'".

This pass replaces that selection rule with CAMeL's MLE disambiguator
(`MLEDisambiguator.pretrained('calima-msa-r13')`), scoring the token in
isolation and taking the highest-scoring analysis, with a fallback to the
highest-scoring definite-noun analysis when the top analysis is not one.
Same database, different SELECTOR -- which is the axis that matters.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    from camel_tools.disambig.mle import MLEDisambiguator
    import camel_tools

    mle = MLEDisambiguator.pretrained()
    p1 = json.load(open(os.path.join(HERE, "exp77_pass1_camel.json"),
                        encoding="utf-8"))
    out = []
    for r in p1["rows"]:
        tok = r["token_analysed"]
        d = mle.disambiguate([tok])[0]
        top = d.analyses[0].analysis if d.analyses else {}
        # fallback: the best-scoring Al_det definite NOUN reading
        best_dn = None
        for sa in d.analyses:
            a = sa.analysis
            definite = (a.get("prc0") == "Al_det"
                        or str(a.get("gloss", "")).startswith("the+"))
            if definite and a.get("pos") == "noun":
                best_dn = a
                break
        out.append({
            "noun": r["noun"], "set": r["set"],
            "root_mle_top": top.get("root"),
            "pattern_mle_top": top.get("pattern"),
            "pos_mle_top": top.get("pos"),
            "root_mle_best_definite_noun": (best_dn or {}).get("root"),
            "pattern_mle_best_definite_noun": (best_dn or {}).get("pattern"),
            "n_analyses_scored": len(d.analyses),
        })
    doc = {"meta": {"purpose": ("exp77 pass 2b (working file, NOT of record): "
                                "root under a DIFFERENT analysis selector"),
                    "selector": ("camel_tools.disambig.mle.MLEDisambiguator."
                                 "pretrained(), token in isolation"),
                    "camel_tools": camel_tools.__version__,
                    "python": sys.version.split()[0]},
           "rows": out}
    op = os.path.join(HERE, "exp77_pass2b_mle.json")
    json.dump(doc, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    d519 = [x for x in out if x["set"] == "d4_519"]
    base = {r["noun"]: r["root_committed"] for r in p1["rows"]
            if r["set"] == "d4_519"}
    tm = sum(1 for x in d519 if x["root_mle_top"] == base[x["noun"]])
    dm = sum(1 for x in d519 if x["root_mle_best_definite_noun"] == base[x["noun"]])
    print("[pass2b] wrote %s  camel %s" % (op, camel_tools.__version__))
    print("[pass2b] MLE top analysis root == committed root: %d/519" % tm)
    print("[pass2b] MLE best definite-noun root == committed root: %d/519" % dm)


if __name__ == "__main__":
    main()
