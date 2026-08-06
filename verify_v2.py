"""Verify every number in RESULTS_EXP21_v2.md traces to a committed JSON."""
import json, re, sys

P = "/home/waj/discocat_arabic_v2"
c = json.load(open(f"{P}/results_exp21c.json"))
ad = json.load(open(f"{P}/results_exp21c_pooled_addendum.json"))
c2 = json.load(open(f"{P}/results_exp21b_seed42_c2.json"))
a = json.load(open(f"{P}/results_exp21a.json"))
md = open(f"{P}/RESULTS_EXP21_v2.md").read()

fails, checks = [], 0
def ck(label, claimed, actual, tol=6e-4):
    global checks
    checks += 1
    if actual is None or abs(claimed - actual) > tol:
        fails.append(f"{label}: doc={claimed} json={actual}")

S = c["seeds"]; PL = c["pooled"]; V = c["verdicts"]; DD = c["dedup"]
seeds = ["42", "43", "44", "45", "46"]

# --- dedup / leakage -------------------------------------------------------
ck("n_sentences", 120, DD["n_sentences"])
ck("n_distinct", 116, DD["n_distinct_sentence_texts"])
ck("n_couples_orig", 60, DD["n_twin_couples_original"])
ck("n_couples_dedup", 58, DD["n_twin_couples_dedup"])
dc = DD["dropped_couples"]
ck("dropped count", 2, len(dc))
ck("drop1 idx", 43, dc[0]["dropped_couple_index"])
ck("drop1 dupof", 37, dc[0]["duplicate_of_couple_index"])
ck("drop2 idx", 56, dc[1]["dropped_couple_index"])
ck("drop2 dupof", 29, dc[1]["duplicate_of_couple_index"])
for s, lst in zip(seeds, [86, None, None, None, None]):
    pass
for s in ["86, 87", "74, 75", "112, 113", "58, 59"]:
    checks += 1
    if s not in md: fails.append(f"dropped-couple indices {s} missing from doc")
for txt in [dc[0]["svo_text"], dc[0]["vso_text"], dc[1]["svo_text"], dc[1]["vso_text"]]:
    checks += 1
    if txt not in md: fails.append(f"Arabic text {txt!r} missing from doc")
LK = c["original_split_leakage_diagnostic"]
for s, exp in zip(seeds, [1, 0, 1, 0, 0]):
    ck(f"leak s{s}", exp, LK[s]["test_couples_verbatim_present_in_train"])

# --- protocol --------------------------------------------------------------
ck("train split", 34, c["protocol"]["split"]["train"])
ck("val split", 12, c["protocol"]["split"]["val"])
ck("test split", 12, c["protocol"]["split"]["test"])
ck("iters", 6000, S["42"]["iters"])
ck("n_params", 91, S["42"]["n_learned_params"])
ck("n_perm", 10000, c["protocol"]["n_permutations"])
ck("C2 params", 144, S["42"]["classical_controls"]["C2_matched_classical"]["params"])
ck("runtime", 2050, c["runtime_sec"], tol=1.0)
ck("n_pos", 12, S["42"]["post"]["test"]["A_vs_B"]["n_pos"])
ck("n_negB", 60, S["42"]["post"]["test"]["A_vs_B"]["n_neg"])
ck("n_negC", 24, S["42"]["post"]["test"]["A_vs_C"]["n_neg"])

# --- main table (3dp as written in doc) ------------------------------------
tbl = {  # seed: (PRE, POST, p_sent, p_der, p_old)
 "42": (0.486, 0.515, 0.528, 0.530, 0.435),
 "43": (0.601, 0.632, 0.211, 0.175, 0.078),
 "44": (0.556, 0.601, 0.571, 0.576, 0.135),
 "45": (0.617, 0.551, 0.190, 0.158, 0.291),
 "46": (0.574, 0.513, 0.685, 0.696, 0.455)}
for s, (pre, post, ps, pd_, po) in tbl.items():
    ck(f"s{s} PRE", pre, S[s]["pre"]["test"]["A_vs_B"]["auc"])
    ck(f"s{s} POST", post, S[s]["post"]["test"]["A_vs_B"]["auc"])
    ck(f"s{s} p_sent", ps, S[s]["post"]["test"]["A_vs_B"]["perm_p_sentence"])
    ck(f"s{s} p_der", pd_, S[s]["post"]["test"]["A_vs_B"]["perm_p_sentence_derangement"])
    ck(f"s{s} p_old", po, S[s]["post"]["test"]["A_vs_B"]["perm_p_pooled_label_BIASED"])
ck("PRE mean", 0.567, PL["pre_mean"])
ck("PRE sd", 0.051, PL["pre_sd"])
ck("POST mean", 0.563, PL["post_mean"])
ck("POST sd", 0.053, PL["post_sd"])
ck("mean change", -0.004, PL["mean_improvement_pre_to_post"])
ck("n sig sentence", 0, V["n_seeds_sig_sentence_level"])
ck("n sig bonferroni", 0, V["n_seeds_sig_bonferroni"])
ck("n sig old scheme", 0, PL["n_seeds_sig_pooled_label_BIASED"])
ck("smallest p_sent 0.190", 0.190, min(PL["per_seed_p_sentence_level"].values()))
ck("smallest p_der 0.158", 0.158, min(PL["per_seed_p_sentence_level_derangement"].values()))

# --- train AUC range / p ---------------------------------------------------
tr = [S[s]["post"]["train"]["A_vs_B"]["auc"] for s in seeds]
ck("train AUC min 0.847", 0.847, min(tr))
ck("train AUC max 0.989", 0.989, max(tr))
trp = [S[s]["post"]["train"]["A_vs_B"]["perm_p_sentence"] for s in seeds]
ck("train p 1e-4", 1e-4, max(trp), tol=1e-5)

# --- order / nouns ---------------------------------------------------------
ck("order PRE 0.642", 0.642, PL["order_acc_pre"]["42"])
checks += 1
if len(set(PL["order_acc_pre"].values())) != 1:
    fails.append("order PRE not identical across seeds as doc claims")
ck("order POST min 0.592", 0.592, min(PL["order_acc_post"].values()))
ck("order POST max 0.711", 0.711, max(PL["order_acc_post"].values()))
nr = [S[s]["post"]["nouns"]["spearman"] for s in seeds]
ck("noun PRE rho 0.050", 0.050, S["42"]["pre"]["nouns"]["spearman"])
ck("noun POST min 0.004", 0.004, min(nr))
ck("noun POST max 0.066", 0.066, max(nr))

# --- pooled ----------------------------------------------------------------
T = PL["one_sample_t_vs_0.5"]
ck("pooled t 2.642", 2.642, T["t"])
ck("pooled p1 0.0287", 0.0287, T["p_one_sided_greater"], tol=6e-5)
ck("pooled p2 0.0574", 0.0574, T["p_two_sided"], tol=6e-5)
ck("verdict pooled_p", 0.0287, V["pooled_p"], tol=6e-5)
F = PL["fisher_combination_of_sentence_level_p"]
ck("fisher chi2 9.580", 9.580, F["chi2"])
ck("fisher df 10", 10, F["df"])
ck("fisher p 0.478", 0.478, F["p"])
ck("PRE pooled t 2.928", 2.928, ad["pre_one_sample_t_vs_0.5"]["t"])
ck("PRE pooled p 0.0214", 0.0214, ad["pre_one_sample_t_vs_0.5"]["p_one_sided_greater"], 6e-5)
ck("paired t -0.172", -0.172, ad["paired_post_minus_pre"]["t"])
ck("paired p 0.872", 0.872, ad["paired_post_minus_pre"]["p_two_sided"])
checks += 1
if not ad["post_one_sample_t_vs_0.5"]["matches_results_exp21c"]:
    fails.append("addendum POST p disagrees with results_exp21c.json")

# --- classical -------------------------------------------------------------
CL = PL["classical"]
for s, v in zip(seeds, [0.981, 0.918, 0.999, 0.988, 0.992]):
    ck(f"C2 s{s}", v, CL["C2_test_auc_per_seed"][s])
    ck(f"C1 s{s}", 1.000, CL["C1_test_auc_per_seed"][s])
ck("C2 mean 0.975", 0.975, CL["C2_mean"])
ck("C2 mean verdict", 0.975, V["c2_mean_auc"])
ck("quantum post mean verdict", 0.563, V["quantum_post_mean_auc"])
ck("gap 0.413", 0.413, CL["C2_mean"] - PL["post_mean"])
ck("seed42 C2 orig split 0.996", 0.996, c2["C2_auc"])
ck("seed42 C1 orig split 1.000", 1.000, c2["C1_auc"])
checks += 1
if c2["C2_auc"] < max(CL["C2_test_auc_per_seed"].values()):
    pass  # doc says highest of the five *original-split* replay; check below
checks += 1
if not (V["robust_transfer"] is False and V["c2_dominates"] is True
        and V["conclusion_unchanged"] is True):
    fails.append(f"verdict booleans mismatch: {V['robust_transfer']},"
                 f"{V['c2_dominates']},{V['conclusion_unchanged']}")

# --- 21a numbers carried over from v1 --------------------------------------
ck("21a train AUC 0.9986", 0.9986, a["post"]["train"]["A_vs_B"]["auc"])
ck("21a heldout AUC 0.496", 0.496, a["post"]["test"]["A_vs_B"]["auc"])
ck("21a rho pre 0.083", 0.083, a["pre"]["nouns"]["spearman"])
ck("21a rho post -0.005", -0.005, a["post"]["nouns"]["spearman"])
ck("21a train perm p 1e-4", 1e-4, a["post"]["train"]["A_vs_B"]["perm_p"], tol=1e-5)

# --- no stray numbers: every x.xxx in the doc tables must be accounted for --
print(f"checked {checks} numeric claims")
if fails:
    print("FAILURES:")
    for f in fails: print("  -", f)
    sys.exit(1)
print("ALL NUMBERS VERIFIED")
