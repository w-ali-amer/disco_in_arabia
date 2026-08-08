"""Derived pooled statistics for RESULTS_EXP21_v2.md.

results_exp21c.json stores the POST pooled t-test against 0.5, but the
write-up needs three things it does not store:

  (1) RE-CENTRED pooled test.  The pooled t vs 0.5 and the per-seed
      permutation test answer DIFFERENT questions.  The permutation null is
      not centred at 0.5: its mean is 0.484-0.615 (mean 0.546), because
      cross-order twin pairs and within-order pairs differ in mean fidelity
      for reasons that carry no twin information (a main effect of pair
      TYPE, not of twin identity).  Testing against 0.5 therefore charges
      that main effect to "transfer".  The correct pooled analogue of the
      permutation test is a one-sample t on the per-seed shift
      (observed AUC - that seed's own permutation-null mean) against 0.
  (2) the same one-sided pooled t applied to the PRE (untrained) AUCs, and
  (3) the paired POST-vs-PRE test.
  (4) the A-vs-C (twin vs cross-order non-twin) pooled summary, which
      results_exp21c.json stores per seed but not in aggregate.

All are computed from values already stored in results_exp21c.json.  Written
to a NEW file so results_exp21c.json is never overwritten (hygiene rule #3)
while every public number still traces to a committed JSON (hygiene rule #6).
"""
import json

import numpy as np
from scipy import stats

PROJ = "/home/waj/discocat_arabic_v2"
SEEDS = ["42", "43", "44", "45", "46"]
src = json.load(open(f"{PROJ}/results_exp21c.json"))
S = src["seeds"]

pre = np.array(src["pooled"]["pre_test_auc_A_vs_B"])
post = np.array(src["pooled"]["post_test_auc_A_vs_B"])


def one_sided(x, mu=0.0):
    t, p2 = stats.ttest_1samp(x, mu)
    return {"t": float(t), "df": len(x) - 1,
            "p_one_sided_greater": float(p2 / 2 if t > 0 else 1 - p2 / 2),
            "p_two_sided": float(p2), "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1))}


def null_means(phase, endpoint, key="perm_p_sentence_null_mean"):
    return np.array([S[s][phase]["test"][endpoint][key] for s in SEEDS])


def aucs(phase, endpoint):
    return np.array([S[s][phase]["test"][endpoint]["auc"] for s in SEEDS])


# ── (1) re-centred pooled test on A_vs_B ────────────────────────────────────
nm_post = null_means("post", "A_vs_B")
nm_pre = null_means("pre", "A_vs_B")
nm_post_der = null_means("post", "A_vs_B", "perm_p_sentence_derangement_null_mean")
shift_post = post - nm_post
shift_pre = pre - nm_pre

recentred = {
    "what": "one-sample t of (observed test AUC - that seed's own "
            "sentence-level permutation-null mean) against 0",
    "why": "the permutation null is not centred at 0.5 (its mean is "
           "0.484-0.615), so a t-test against 0.5 charges a pair-TYPE main "
           "effect -- which carries no twin information -- to 'transfer'. "
           "Re-centring on each seed's own null mean makes the pooled test "
           "ask the same question as the per-seed permutation test.",
    "null_means_post": [float(x) for x in nm_post],
    "null_means_pre": [float(x) for x in nm_pre],
    "null_mean_of_null_means_post": float(nm_post.mean()),
    "null_mean_range_post": [float(nm_post.min()), float(nm_post.max())],
    "shifts_post": [float(x) for x in shift_post],
    "shifts_pre": [float(x) for x in shift_pre],
    "post": one_sided(shift_post, 0.0),
    "pre": one_sided(shift_pre, 0.0),
    "post_vs_derangement_null": one_sided(post - nm_post_der, 0.0),
}

# ── (2)(3) uncentred and paired, on A_vs_B ──────────────────────────────────
t_pair, p2_pair = stats.ttest_rel(post, pre)

# ── (4) A_vs_C pooled summary ───────────────────────────────────────────────
ac_post = aucs("post", "A_vs_C")
ac_pre = aucs("pre", "A_vs_C")
ac_p = np.array([S[s]["post"]["test"]["A_vs_C"]["perm_p_sentence"] for s in SEEDS])
ac_pd = np.array([S[s]["post"]["test"]["A_vs_C"]["perm_p_sentence_derangement"]
                  for s in SEEDS])
a_vs_c = {
    "endpoint": "twin pairs vs cross-order NON-twin pairs (controls for "
                "word-order composition, unlike A_vs_B)",
    "post_auc_per_seed": {s: float(v) for s, v in zip(SEEDS, ac_post)},
    "pre_auc_per_seed": {s: float(v) for s, v in zip(SEEDS, ac_pre)},
    "post_mean": float(ac_post.mean()), "post_sd": float(ac_post.std(ddof=1)),
    "pre_mean": float(ac_pre.mean()), "pre_sd": float(ac_pre.std(ddof=1)),
    "mean_change_pre_to_post": float(ac_post.mean() - ac_pre.mean()),
    "p_sentence_per_seed": {s: float(v) for s, v in zip(SEEDS, ac_p)},
    "p_derangement_per_seed": {s: float(v) for s, v in zip(SEEDS, ac_pd)},
    "n_seeds_sig_sentence_level": int((ac_p < 0.05).sum()),
    "n_seeds_sig_derangement": int((ac_pd < 0.05).sum()),
    "min_p_sentence": float(ac_p.min()),
    "interpretation": "training moves the twin-vs-non-twin-cross-order "
                      "separation DOWN by 0.027; 0/5 significant",
}

out = {
    "derived_from": "results_exp21c.json :: pooled.{pre,post}_test_auc_A_vs_B "
                    "and seeds.*.{pre,post}.test.A_vs_{B,C}",
    "source_script": "exp21c_pooled_addendum.py",
    "note": "purely derived; no new model runs. Exists because "
            "results_exp21c.json must not be overwritten.",
    "inputs": {"pre_test_auc_A_vs_B": [float(x) for x in pre],
               "post_test_auc_A_vs_B": [float(x) for x in post]},
    "recentred_pooled_t_vs_permutation_null": recentred,
    "pre_one_sample_t_vs_0.5": {
        **one_sided(pre, 0.5),
        "interpretation": "the UNTRAINED (exact-AE init) encoder passes the "
                          "uncentred pooled test at least as strongly as the "
                          "trained one; colour only -- comparing two "
                          "non-independent p-values is not itself a test"},
    "post_one_sample_t_vs_0.5": {
        **one_sided(post, 0.5),
        "matches_results_exp21c": abs(
            one_sided(post, 0.5)["p_one_sided_greater"] -
            src["pooled"]["one_sample_t_vs_0.5"]["p_one_sided_greater"]) < 1e-12},
    "paired_post_minus_pre": {
        "t": float(t_pair), "df": len(post) - 1, "p_two_sided": float(p2_pair),
        "mean_difference": float(post.mean() - pre.mean()),
        "interpretation": "training has no detectable effect on held-out AUC; "
                          "this is the primary within-split comparison"},
    "a_vs_c_pooled": a_vs_c,
    "caveat": "all of these share the non-independence caveat of "
              "results_exp21c.json::pooled.one_sample_t_vs_0.5 (the 5 splits "
              "re-sample the same 58 couples / 116 sentences)",
}
assert out["post_one_sample_t_vs_0.5"]["matches_results_exp21c"], \
    "recomputed POST pooled p disagrees with results_exp21c.json"
path = f"{PROJ}/results_exp21c_pooled_addendum.json"
json.dump(out, open(path, "w"), indent=2)
print(json.dumps({k: v for k, v in out.items()
                  if k in ("recentred_pooled_t_vs_permutation_null",
                           "a_vs_c_pooled")}, indent=2))
print("wrote", path)
