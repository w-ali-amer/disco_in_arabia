# -*- coding: utf-8 -*-
"""exp106_joint.py -- doc 104 (REGISTERED 8c7e3bd; A1 04d163c; lexicon FROZEN 9ed7d59): the PADT joint p(t, f) over
NOUN tokens in the 4 template classes whose root has a field in the frozen top-150 lexicon; the G-TEMPLATE sample
(10 tokens per class, drawn from TOKENS, seed 106); the split descriptive.  Run with ~/camel_mac_env.
Writes joint_exp106.json + gtemplate_sample_exp106.json (Rule 3).
"""
import collections, json, random
import exp94_common as C94
import exp106_roots as R

FIELDS = ("POL", "ECON", "SEC", "SOC")


def main():
    lex = json.load(open("field_lexicon_exp106.json"))
    assign = lex["assignment"]; weak = set(lex["weak"])
    by_class = collections.defaultdict(list); joint = collections.Counter(); joint_noweak = collections.Counter()
    tokens_in_classes = 0
    for c, (root, p) in R.analysed_nouns():
        t = R.template_of(p)
        if not t:
            continue
        tokens_in_classes += 1
        by_class[t].append({"form": c[1], "lemma": c[2], "root": root, "pattern": p})
        f = assign.get(root)
        if f in FIELDS:
            joint[(t, f)] += 1
            if root not in weak:
                joint_noweak[(t, f)] += 1
    rng = random.Random(106)
    sample = {t: rng.sample(v, 10) for t, v in sorted(by_class.items())}
    C94.write_new_json("gtemplate_sample_exp106.json", {"doc": "104 G-TEMPLATE sample (tokens, 10/class, seed 106)", "items": sample})
    n = sum(joint.values())
    out = {"doc": "104 joint (lexicon 9ed7d59)", "tokens_in_template_classes": tokens_in_classes,
           "tokens_in_joint": n, "coverage": n / tokens_in_classes,
           "joint": {"%s|%s" % k: v for k, v in joint.items()}, "joint_noweak": {"%s|%s" % k: v for k, v in joint_noweak.items()},
           "split_descriptive": {"assigned_roots": lex["gate_A1"]["assigned"], "split_roots": lex["split_count"],
                                 "split_fraction_of_assigned_plus_split": lex["split_count"] / (lex["gate_A1"]["assigned"] + lex["split_count"])},
           "provenance": C94.provenance()}
    C94.write_new_json("joint_exp106.json", out)
    print(json.dumps({k: out[k] for k in ("tokens_in_template_classes", "tokens_in_joint", "coverage", "split_descriptive")}, indent=1))
    for t, v in sample.items():
        print("==", t)
        for i, it in enumerate(v):
            print(" ", i + 1, it["form"], it["lemma"], it["pattern"])


if __name__ == "__main__":
    main()
