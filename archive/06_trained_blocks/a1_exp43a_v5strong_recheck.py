# -*- coding: utf-8 -*-
"""a1_exp43a_v5strong_recheck.py -- doc 82 §13 A1 (registered 4b5a75d before running).
Strong single-sentence oracle (validate_exp41r_data.run_v5_strong) on stories_exp43a_family.json
(804 Q1 two-hop items, weak-V5-validated only). Criterion: the validator's own chance bound."""
import json, hashlib, os
from collections import Counter
import validate_exp41r_data as v41r
DATA, OUT = "stories_exp43a_family.json", "results_exp43a_family_v5strong_recheck.json"
d = json.load(open(DATA)); items = d["items"] if "items" in d else d
splits = Counter(i["split"] for i in items); print("splits", dict(splits))
tr = [i for i in items if i["split"] == "train"]; va = [i for i in items if i["split"] == "val"]
te = [i for i in items if i["split"] not in ("train", "val")]
r = v41r.run_v5_strong(tr, va, te)
res = {"purpose": "doc 82 §13 A1", "data": DATA, "data_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest(),
       "validator_sha256": hashlib.sha256(open("validate_exp41r_data.py", "rb").read()).hexdigest(),
       "n": {"train": len(tr), "val": len(va), "test": len(te)}, "strong_v5": r, "a1_family_v5strong_passed": r["passed"]}
assert not os.path.exists(OUT), "Rule 3"
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("acc_max=%.4f ci=%s acc_best=%.4f passed=%s" % (r["test_accuracy_max_aggregation"], r["ci95_max_aggregation"], r["test_accuracy_best_sentence"], r["passed"]))
