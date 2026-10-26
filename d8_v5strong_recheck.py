# -*- coding: utf-8 -*-
"""d8_v5strong_recheck.py -- doc 82 §11.8 (registered 0fdb9f3 before running).
Runs the doc-65 STRONG single-sentence oracle (validate_exp41r_data.run_v5_strong)
on stories_exp43b_L2.json, whose original validation used the weak, degenerate V5
(D8). Criterion: the validator's own chance bound on the TWO-HOP test subset."""
import json, hashlib, os
import validate_exp41r_data as v41r

DATA, OUT = "stories_exp43b_L2.json", "results_exp43b_L2_v5strong_recheck.json"
d = json.load(open(DATA))
items = d["items"]
tr = [i for i in items if i["split"] == "train"]
va = [i for i in items if i["split"] == "val"]
te = [i for i in items if i["split"] == "test"]
te2 = [i for i in te if i.get("hop_count") == 2]
full = v41r.run_v5_strong(tr, va, te)
two = v41r.run_v5_strong(tr, va, te2)
res = {"purpose": "doc 82 §11.8 D8 re-check", "data": DATA,
       "data_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest(),
       "validator_sha256": hashlib.sha256(open("validate_exp41r_data.py", "rb").read()).hexdigest(),
       "n": {"train": len(tr), "val": len(va), "test": len(te), "test_two_hop": len(te2)},
       "two_hop_subset": two, "full_test_descriptive": full,
       "l2_twohop_v5strong_passed": two["passed"]}
assert not os.path.exists(OUT), "Rule 3"
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
for k, r in (("two-hop", two), ("full", full)):
    print(k, "acc_max=%.4f ci=%s acc_best=%.4f ci=%s passed=%s" % (
        r["test_accuracy_max_aggregation"], r["ci95_max_aggregation"],
        r["test_accuracy_best_sentence"], r["ci95_best_sentence"], r["passed"]))
print("l2_twohop_v5strong_passed =", two["passed"], "->", OUT)
