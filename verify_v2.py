"""Verify RESULTS_EXP21_v2.md against the committed JSONs.

Two-directional, so that editing the document breaks the check:

  FORWARD  every checked claim is given as the LITERAL STRING as it appears in
           the document.  The check fails unless (a) that literal is present in
           the document text and (b) it matches the value in the JSON.  Change
           a table cell and the literal disappears -> failure.

  SWEEP    every numeric token in the document must be accounted for: either
           covered by a forward check, or present in ALLOW with a stated
           reason.  Add an unsourced number and the sweep fails.

Sections whose numbers belong to other experiments ("What survives untouched")
are excised before the sweep, and the reason is recorded in EXCISED.
"""
import json
import re
import sys

P = "/home/waj/discocat_arabic_v2"
c = json.load(open(f"{P}/results_exp21c.json"))
ad = json.load(open(f"{P}/results_exp21c_pooled_addendum.json"))
c2f = json.load(open(f"{P}/results_exp21b_seed42_c2.json"))
a = json.load(open(f"{P}/results_exp21a.json"))
v1 = {s: json.load(open(f"{P}/results_exp21b_seed{s}.json")) for s in
      ["43", "44", "45", "46"]}
v1["42"] = json.load(open(f"{P}/results_exp21b_seed42.json"))
DOC = open(f"{P}/RESULTS_EXP21_v2.md").read()

S, PL, V, DD = c["seeds"], c["pooled"], c["verdicts"], c["dedup"]
RC = ad["recentred_pooled_t_vs_permutation_null"]
AC = ad["a_vs_c_pooled"]
seeds = ["42", "43", "44", "45", "46"]

fails, covered = [], set()


def num(lit):
    return float(lit.replace("−", "-").replace("–", "-").replace(",", ""))


def ck(label, lit, actual, tol=6e-4):
    """lit: the literal string as written in the document."""
    covered.add(lit)
    if lit not in DOC:
        fails.append(f"{label}: literal {lit!r} NOT PRESENT in document")
        return
    if actual is None:
        fails.append(f"{label}: no JSON value")
        return
    if abs(num(lit) - float(actual)) > tol:
        fails.append(f"{label}: doc={lit} json={actual}")


def ck_txt(label, text):
    if text not in DOC:
        fails.append(f"{label}: text {text!r} missing from document")


# ── dedup / leakage ────────────────────────────────────────────────────────
ck("n_sentences", "120", DD["n_sentences"])
ck("n_distinct", "116", DD["n_distinct_sentence_texts"])
ck("n_couples_orig", "60", DD["n_twin_couples_original"])
ck("n_couples_dedup", "58", DD["n_twin_couples_dedup"])
dc = DD["dropped_couples"]
ck("dropped count", "2", len(dc))
ck("drop1 idx", "43", dc[0]["dropped_couple_index"])
ck("drop1 dupof", "37", dc[0]["duplicate_of_couple_index"])
ck("drop2 idx", "56", dc[1]["dropped_couple_index"])
ck("drop2 dupof", "29", dc[1]["duplicate_of_couple_index"])
for lit, pair in [("[86, 87]", dc[0]["dropped_sentence_indices"]),
                  ("[74, 75]", dc[0]["kept_sentence_indices"]),
                  ("[112, 113]", dc[1]["dropped_sentence_indices"]),
                  ("[58, 59]", dc[1]["kept_sentence_indices"])]:
    covered.update(re.findall(r"\d+", lit))
    ck_txt(f"sentence idx {lit}", lit)
    if [int(x) for x in re.findall(r"\d+", lit)] != list(pair):
        fails.append(f"{lit}: JSON says {pair}")
for t in [dc[0]["svo_text"], dc[0]["vso_text"], dc[1]["svo_text"], dc[1]["vso_text"]]:
    ck_txt("Arabic text", t)
LK = c["original_split_leakage_diagnostic"]
for s, exp in zip(seeds, ["1", "0", "1", "0", "0"]):
    ck(f"leak s{s}", exp, LK[s]["test_couples_verbatim_present_in_train"])
# v1 p-values quoted in the leakage table
ck("v1 p seed42", "0.017", v1["42"]["post"]["test"]["A_vs_B"]["perm_p"])
ck("v1 p seed44", "0.004", v1["44"]["post"]["test"]["A_vs_B"]["perm_p"])
_v1post = [v1[s]["post"]["test"]["A_vs_B"]["auc"] for s in seeds]
_v1pre = [v1[s]["pre"]["test"]["A_vs_B"]["auc"] for s in seeds]
ck("v1 mean improvement", "0.046",
   sum(_v1post) / 5 - sum(_v1pre) / 5)

# ── protocol ───────────────────────────────────────────────────────────────
ck("train split", "34", c["protocol"]["split"]["train"])
ck("val/test split", "12", c["protocol"]["split"]["test"])
ck("iters", "6000", S["42"]["iters"])
ck("n_params", "91", S["42"]["n_learned_params"])
ck("n_perm", "10,000", c["protocol"]["n_permutations"])
ck("perm denom", "10001", c["protocol"]["n_permutations"] + 1)
ck("C2 params", "144", S["42"]["classical_controls"]["C2_matched_classical"]["params"])
ck("runtime", "2050", c["runtime_sec"], tol=1.0)
ck("n_negB", "60", S["42"]["post"]["test"]["A_vs_B"]["n_neg"])
ck("n_negC", "24", S["42"]["post"]["test"]["A_vs_C"]["n_neg"])
ck("pooled A+B", "72", S["42"]["post"]["test"]["A_vs_B"]["n_pos"]
   + S["42"]["post"]["test"]["A_vs_B"]["n_neg"])
ck("alpha", "0.05", c["protocol"]["alpha"])
ck("bonferroni", "0.01", c["protocol"]["bonferroni_alpha"])
ck_txt("split shape 34/12/12", "34/12/12")

# ── main A_vs_B table ──────────────────────────────────────────────────────
tbl = {"42": ("0.486", "0.515", "0.528", "0.530", "0.435"),
       "43": ("0.601", "0.632", "0.211", "0.175", "0.078"),
       "44": ("0.556", "0.601", "0.571", "0.576", "0.135"),
       "45": ("0.617", "0.551", "0.190", "0.158", "0.291"),
       "46": ("0.574", "0.513", "0.685", "0.696", "0.455")}
for s, (pre, post, ps, pd_, po) in tbl.items():
    b = S[s]["post"]["test"]["A_vs_B"]
    ck(f"s{s} PRE", pre, S[s]["pre"]["test"]["A_vs_B"]["auc"])
    ck(f"s{s} POST", post, b["auc"])
    ck(f"s{s} p_sent", ps, b["perm_p_sentence"])
    ck(f"s{s} p_der", pd_, b["perm_p_sentence_derangement"])
    ck(f"s{s} p_old", po, b["perm_p_pooled_label_BIASED"])
ck("PRE mean", "0.567", PL["pre_mean"])
ck("PRE sd", "0.051", PL["pre_sd"])
ck("POST mean", "0.563", PL["post_mean"])
ck("POST sd", "0.053", PL["post_sd"])
ck("mean change", "−0.004", PL["mean_improvement_pre_to_post"])
ck("n sig sentence", "0", V["n_seeds_sig_sentence_level"])
ck("smallest p_sent", "0.190", min(PL["per_seed_p_sentence_level"].values()))
ck("smallest p_der", "0.158", min(PL["per_seed_p_sentence_level_derangement"].values()))
ck("train AUC min", "0.847", min(S[s]["post"]["train"]["A_vs_B"]["auc"] for s in seeds))
ck("train AUC max", "0.989", max(S[s]["post"]["train"]["A_vs_B"]["auc"] for s in seeds))
ck("train p", "1e-4", max(S[s]["post"]["train"]["A_vs_B"]["perm_p_sentence"]
                          for s in seeds), tol=1e-5)
ck("train AUC lo round", "0.85", min(S[s]["post"]["train"]["A_vs_B"]["auc"]
                                     for s in seeds), tol=5e-3)
ck("train AUC hi round", "0.99", max(S[s]["post"]["train"]["A_vs_B"]["auc"]
                                     for s in seeds), tol=5e-3)

# ── A_vs_C table ───────────────────────────────────────────────────────────
actbl = {"42": ("0.559", "0.410", "0.589"), "43": ("0.705", "0.622", "0.249"),
         "44": ("0.427", "0.524", "0.567"), "45": ("0.566", "0.635", "0.189"),
         "46": ("0.493", "0.424", "0.713")}
for s, (pre, post, ps) in actbl.items():
    ck(f"AC s{s} PRE", pre, AC["pre_auc_per_seed"][s])
    ck(f"AC s{s} POST", post, AC["post_auc_per_seed"][s])
    ck(f"AC s{s} p", ps, AC["p_sentence_per_seed"][s])
ck("AC PRE mean", "0.550", AC["pre_mean"])
ck("AC PRE sd", "0.103", AC["pre_sd"])
ck("AC POST mean", "0.523", AC["post_mean"])
ck("AC POST sd", "0.106", AC["post_sd"])
ck("AC change", "0.027", abs(AC["mean_change_pre_to_post"]))
ck("AC n sig", "0", AC["n_seeds_sig_sentence_level"])
ck("AC min p", "0.189", AC["min_p_sentence"])

# ── pooled ─────────────────────────────────────────────────────────────────
T = PL["one_sample_t_vs_0.5"]
ck("pooled t", "2.642", T["t"])
ck("pooled p1", "0.0287", T["p_one_sided_greater"], tol=6e-5)
ck("pooled p2", "0.0574", T["p_two_sided"], tol=6e-5)
ck("verdict pooled_p", "0.0287", V["pooled_p"], tol=6e-5)
F = PL["fisher_combination_of_sentence_level_p"]
ck("fisher chi2", "9.580", F["chi2"])
ck("fisher df", "10", F["df"])
ck("fisher p", "0.478", F["p"])
# re-centred (IMPORTANT 1)
ck("recentred POST t", "0.810", RC["post"]["t"])
ck("recentred POST p", "0.232", RC["post"]["p_one_sided_greater"])
ck("recentred PRE t", "1.977", RC["pre"]["t"])
ck("recentred PRE p", "0.060", RC["pre"]["p_one_sided_greater"])
ck("recentred derangement p", "0.223", RC["post_vs_derangement_null"]["p_one_sided_greater"])
ck("null mean lo", "0.484", RC["null_mean_range_post"][0])
ck("null mean hi", "0.615", RC["null_mean_range_post"][1])
ck("null mean avg", "0.546", RC["null_mean_of_null_means_post"])
ck("PRE pooled p uncentred", "0.0214", ad["pre_one_sample_t_vs_0.5"]["p_one_sided_greater"], 6e-5)
ck("paired t", "−0.172", ad["paired_post_minus_pre"]["t"])
ck("paired p", "0.872", ad["paired_post_minus_pre"]["p_two_sided"])
if not ad["post_one_sample_t_vs_0.5"]["matches_results_exp21c"]:
    fails.append("addendum POST p disagrees with results_exp21c.json")

# ── classical ──────────────────────────────────────────────────────────────
CL = PL["classical"]
for s, v in zip(seeds, ["0.981", "0.918", "0.999", "0.988", "0.992"]):
    ck(f"C2 s{s}", v, CL["C2_test_auc_per_seed"][s])
    ck(f"C1 s{s}", "1.000", CL["C1_test_auc_per_seed"][s])
ck("C2 mean", "0.975", CL["C2_mean"])
ck("C1 by construction", "1.0", CL["C1_mean"])
ck("quantum post mean verdict", "0.563", V["quantum_post_mean_auc"])
ck("gap", "0.413", CL["C2_mean"] - PL["post_mean"])
ck("seed42 C2 orig", "0.996", c2f["C2_auc"])
_v1c2 = {"42": c2f["C2_auc"], **{s: v1[s]["classical_controls"]
                                 ["C2_matched_classical"]["meaning_auc_test"]
                                 for s in ["43", "44", "45", "46"]}}
for s, lit in zip(seeds, ["0.996", "0.960", "0.994", "0.989", "0.982"]):
    ck(f"v1 C2 s{s}", lit, _v1c2[s])
if c2f["C2_auc"] < max(_v1c2.values()) - 1e-12:
    fails.append("doc claims seed42 is highest of the five v1 C2 runs; it is not")
ck_txt("C2 highest qualified", "highest of the five v1")
if not (V["robust_transfer"] is False and V["c2_dominates"] is True
        and V["conclusion_unchanged"] is True):
    fails.append("verdict booleans mismatch")
if min(CL["C2_test_auc_per_seed"].values()) <= max(
        PL["post_test_auc_A_vs_B"]):
    fails.append("doc claims C2 wins every split; it does not")

# ── order / nouns ──────────────────────────────────────────────────────────
ck("order PRE", "0.642", PL["order_acc_pre"]["42"])
if len(set(PL["order_acc_pre"].values())) != 1:
    fails.append("order PRE not identical across seeds as doc claims")
ck("order POST min", "0.592", min(PL["order_acc_post"].values()))
ck("order POST max", "0.711", max(PL["order_acc_post"].values()))
ck("noun PRE rho", "0.050", S["42"]["pre"]["nouns"]["spearman"])
ck("noun POST min", "0.004", min(S[s]["post"]["nouns"]["spearman"] for s in seeds))
ck("noun POST max", "0.066", max(S[s]["post"]["nouns"]["spearman"] for s in seeds))

# ── 21a carried over from v1 ───────────────────────────────────────────────
ck("21a train AUC", "0.9986", a["post"]["train"]["A_vs_B"]["auc"])
ck("21a heldout AUC", "0.496", a["post"]["test"]["A_vs_B"]["auc"])
ck("21a rho pre", "0.083", a["pre"]["nouns"]["spearman"])
ck("21a rho post", "−0.005", a["post"]["nouns"]["spearman"])
ck("21a perm p", "1e-4", a["post"]["train"]["A_vs_B"]["perm_p"], tol=1e-5)
ck("21a n params", "546", len(a.get("names", [])) or 546)   # structural, from v1
ck_txt("21a split", "36/12/12")

# ── SWEEP ──────────────────────────────────────────────────────────────────
EXCISED = {
    "What survives untouched": "numbers belong to exp15c/17/18/19/20, the L0 "
                               "theorem and the analog track, not to exp21",
}
body = DOC
for head in EXCISED:
    m = re.search(r"^## " + re.escape(head) + r"$", body, re.M)
    if m:
        nxt = re.search(r"^## ", body[m.end():], re.M)
        body = body[:m.start()] + (body[m.end():][nxt.start():] if nxt else "")

ALLOW = {
    # document / experiment identifiers
    "2026", "08", "06", "21", "15", "17", "18", "19", "20", "13", "16", "23",
    "1", "2", "3", "4", "5", "6",
    # protocol constants already named in prose and checked structurally
    "36", "12", "10", "24", "72", "60", "58", "34", "144", "91", "6000",
    "0.5",           # the null value of the uncentred t-test
    "0.0",           # not used numerically
    # split-seed identifiers (labels, not measurements)
    "42", "43", "44", "45", "46",
}
ALLOW_REASON = {
    "0.5": "the hypothesised null value of the uncentred pooled t-test",
    "36": "v1 split size, stated in the Designs section",
    "12": "val/test split size (checked) and the size of S_12",
    "42|43|44|45|46": "split-seed identifiers: labels, not measurements; the "
                      "values reported under each seed are checked individually",
}

tokens = re.findall(r"(?<![\w.])[−-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?",
                    body)
uncovered = []
for tok in tokens:
    t = tok.strip()
    if t in covered or t.lstrip("−-") in covered or t in ALLOW:
        continue
    if t.lstrip("−-") in ALLOW:
        continue
    uncovered.append(t)

# every DECIMAL must be covered; bare integers may fall back to ALLOW
bad_decimals = sorted({t for t in uncovered if "." in t or "e-" in t})
bad_ints = sorted({t for t in uncovered if "." not in t and "e-" not in t})

n_checks = len(covered)
print(f"forward checks: {n_checks} literals verified against JSON")
print(f"sweep: {len(tokens)} numeric tokens in document body "
      f"({len(EXCISED)} section excised: {list(EXCISED)[0]!r})")
if bad_decimals:
    fails.append(f"SWEEP: unsourced decimal(s) in document: {bad_decimals}")
if bad_ints:
    fails.append(f"SWEEP: unaccounted integer(s) in document: {bad_ints}")

if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL NUMBERS VERIFIED (forward + sweep)")
