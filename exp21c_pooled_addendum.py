"""Derived pooled statistics for RESULTS_EXP21_v2.md.

results_exp21c.json stores the POST pooled t-test but not two comparisons that
the write-up needs to keep the pooled number honest:
  (1) the SAME one-sided pooled t applied to the PRE (untrained) test AUCs, and
  (2) the paired POST-vs-PRE test.
Both are computed here from the arrays already stored in results_exp21c.json
(`pooled.pre_test_auc_A_vs_B`, `pooled.post_test_auc_A_vs_B`).  Written to a
NEW file so results_exp21c.json is never overwritten (hygiene rule #3) while
every public number still traces to a committed JSON (hygiene rule #6).
"""
import json

import numpy as np
from scipy import stats

PROJ = "/home/waj/discocat_arabic_v2"
src = json.load(open(f"{PROJ}/results_exp21c.json"))
pre = np.array(src["pooled"]["pre_test_auc_A_vs_B"])
post = np.array(src["pooled"]["post_test_auc_A_vs_B"])

t_pre, p2_pre = stats.ttest_1samp(pre, 0.5)
t_post, p2_post = stats.ttest_1samp(post, 0.5)
t_pair, p2_pair = stats.ttest_rel(post, pre)

out = {
    "derived_from": "results_exp21c.json :: pooled.{pre,post}_test_auc_A_vs_B",
    "source_script": "exp21c_pooled_addendum.py",
    "note": "purely derived; no new model runs. Exists because "
            "results_exp21c.json must not be overwritten.",
    "inputs": {"pre_test_auc_A_vs_B": [float(x) for x in pre],
               "post_test_auc_A_vs_B": [float(x) for x in post]},
    "pre_one_sample_t_vs_0.5": {
        "t": float(t_pre), "df": len(pre) - 1,
        "p_one_sided_greater": float(p2_pre / 2 if t_pre > 0 else 1 - p2_pre / 2),
        "p_two_sided": float(p2_pre),
        "interpretation": "the UNTRAINED (exact-AE init) encoder passes the "
                          "pooled test at least as strongly as the trained one; "
                          "the pooled POST result therefore reflects a property "
                          "of the initialisation, not of training"},
    "post_one_sample_t_vs_0.5": {
        "t": float(t_post), "df": len(post) - 1,
        "p_one_sided_greater": float(p2_post / 2 if t_post > 0 else 1 - p2_post / 2),
        "p_two_sided": float(p2_post),
        "matches_results_exp21c": abs(
            float(p2_post / 2) -
            src["pooled"]["one_sample_t_vs_0.5"]["p_one_sided_greater"]) < 1e-12},
    "paired_post_minus_pre": {
        "t": float(t_pair), "df": len(post) - 1, "p_two_sided": float(p2_pair),
        "mean_difference": float(post.mean() - pre.mean()),
        "interpretation": "training has no detectable effect on held-out AUC"},
    "caveat": "all three share the non-independence caveat of "
              "results_exp21c.json::pooled.one_sample_t_vs_0.5 (the 5 splits "
              "re-sample the same 58 couples / 116 sentences)",
}
assert out["post_one_sample_t_vs_0.5"]["matches_results_exp21c"], \
    "recomputed POST pooled p disagrees with results_exp21c.json"
path = f"{PROJ}/results_exp21c_pooled_addendum.json"
json.dump(out, open(path, "w"), indent=2)
print(json.dumps(out, indent=2))
print("wrote", path)
