"""Exp 21c: de-duplicated twin re-run of the exp21b-lite protocol, with a
sentence-level permutation null and one pooled across-split test.

WHY THIS RE-RUN EXISTS (three defects found in the exp21b analysis)
------------------------------------------------------------------
D1  Duplicate twins.  `sentences.json:WordOrderMatched` contains 120 sentences
    but only 116 distinct sentence texts.  The exp21b twin construction
    (multiset grouping of SVO against VSO) therefore produced 60 twin couples
    of which 2 are VERBATIM duplicates of 2 other couples.  Because the split
    is over COUPLES, a duplicated couple can land in train while its identical
    twin lands in test -> the test "held-out" pair is literally a training
    pair.  This inflates POST test AUC on exactly the splits where it happens.

D2  Wrong permutation null.  `analyse()` in exp21b_robust.py pools the twin
    fidelities (A) and the same-order fidelities (B) into one score vector and
    permutes the A/B LABEL over that pooled vector.  That null assumes all
    72 fidelities are exchangeable.  They are not: B is built by sampling
    ~60 within-order sentence pairs from only 24 test sentences, so each
    sentence appears in ~5 B-pairs and the B fidelities are strongly
    dependent.  Label permutation over a dependent pool understates the null
    variance of the AUC and therefore understates p.

D3  No pooled inference.  "2/5 splits significant" is a vote count, not a
    test.  There was no single statistic for "does the effect exist at all".

All three defects bias the analysis TOWARD the positive.  Correcting them can
only weaken the (already fragile) positive, so this re-run is the honest
version of the negative result.  The numbers are measured here, not assumed.

WHAT THIS SCRIPT DOES
---------------------
1.  Rebuilds the twin couples exactly as exp21b_robust.py does, then removes
    the verbatim-duplicated couples (keeping the FIRST occurrence of each),
    leaving 58 unique couples over 116 distinct sentences.
2.  Re-runs the identical exp21b protocol on 5 split seeds (42-46):
    same encoder class, same least-squares init to the exact-AE point,
    same SPSA (6000 iters, same gain schedule, same srng seed), same
    val-margin model selection, same evaluation, same classical controls.
3.  Adds a SENTENCE-LEVEL permutation test (see PERMUTATION SCHEME below) and
    keeps the original pooled-label p side by side so the size of the D2 bias
    is visible.
4.  Adds a pooled across-split one-sample t-test of the 5 POST test AUCs
    against 0.5.
5.  Writes results_exp21c.json including a machine-checkable `verdicts` block.

SPLIT SIZE CHOICE
-----------------
exp21b used 36/12/12 out of 60 couples (0.60 / 0.20 / 0.20).  On 58 couples
the exact proportions give 34.8 / 11.6 / 11.6.  We use 34 / 12 / 12.
Rationale: the evaluation sets are held at 12 couples so that every test-set
statistic (12 A-pairs, 60 B-pairs, 24 C-pairs, 24 test sentences) has exactly
the same shape as in exp21b and the AUCs are directly comparable across the
two analyses; the 2 removed couples are taken out of TRAIN, which is the only
place where losing 2 couples is a power question rather than a comparability
question.  The alternative floor-based 34/11/11 would leave 2 couples unused
and would change the test-set geometry, breaking comparability with v1.

PERMUTATION SCHEME (the D2 fix) -- read this before believing any p-value
------------------------------------------------------------------------
Setup.  A test split contains 12 twin couples = 12 SVO sentences
s_1..s_12 and 12 VSO sentences v_1..v_12, with the TRUE twin map being the
identity (s_m is the twin of v_m).  The trained encoder produces one fixed
quantum state per sentence.  The statistic is
        AUC_obs = AUC( {fid(s_m, v_m)}_m=1..12   vs   negatives ),
where the negatives are either B (same-order pairs) or C (cross-order
non-twin pairs).

Null hypothesis H0.  "The trained encoder transports no twin-specific
information": the fidelity between a sentence-state and a VSO state carries
no information about whether that VSO sentence is the meaning-twin.

Randomisation.  Under H0 the twin map is uninformative, so any bijection
pi: {1..12} -> {1..12} is as good a "twin map" as the true one.  We therefore
draw pi uniformly from S_12 and recompute
        AUC_pi = AUC( {fid(s_m, v_pi(m))}_m   vs   the SAME negatives ),
10,000 times.  p = (#{AUC_pi >= AUC_obs} + 1) / (10,000 + 1).

Why exchangeability holds under THIS null and not the old one.
  * The randomisation acts on the twin ASSIGNMENT, not on the fidelity
    values.  The 24 sentence states are held fixed, so every dependence
    induced by sentences being reused across many pairs is carried
    identically into every permuted replicate.  Nothing is assumed about the
    fidelities being i.i.d.
  * For the primary endpoint A vs B the negatives are within-order pairs,
    which the permutation does not touch at all: B is exactly ancillary, and
    the null distribution is generated purely by the group action on the
    twin map.  This is a valid randomisation test conditional on the observed
    24 states and the observed B set.
  * The old scheme instead permuted the A/B label across a pooled vector of
    72 fidelities, which requires the 72 values to be exchangeable under H0.
    They are not (each sentence contributes ~5 B-values), so its null was too
    tight.

Two caveats, both reported.
  (a) Uniform draws from S_12 include permutations with fixed points (on
      average one true twin survives), which mildly inflates the null under
      H1 and so costs power.  It is exactly valid under H0.  Because a
      LOW-POWER test is the obvious hostile reading of a negative result, we
      ALSO report a derangement-restricted p-value (pi sampled with no fixed
      points), which removes that conservatism.  If both are non-significant
      the negative is not a power artefact.
  (b) For the A vs C endpoint the negatives are themselves cross-order pairs,
      so a permuted A-pair can coincide with a C-pair.  That makes the A vs C
      randomisation slightly conservative too; the primary endpoint (A vs B)
      is unaffected.

POOLED TEST (the D3 fix)
------------------------
One-sample t of the 5 POST test AUCs against 0.5, df = 4, one-sided
(direction pre-registered in exp21b: "test AUC > 0.5").  CAVEAT, stated in
the output and in the write-up: the 5 splits are re-samples of the SAME 58
couples and 116 sentences, so they are not independent replicates; the t-test
treats them as such and its p is therefore optimistic (too small).  It is
reported as an upper bound on the evidence, which is the conservative
direction for a negative claim.  Fisher's combination of the 5 sentence-level
p-values is reported as a secondary statistic under the same caveat.

PRE-REGISTERED VERDICT RULE (computed, never hardcoded)
-------------------------------------------------------
robust_transfer  := (pooled one-sided p < 0.05) AND (>= 3 of 5 seeds
                    significant at the Bonferroni level alpha = 0.05/5)
c2_dominates     := (mean C2 test AUC - mean POST quantum test AUC > 0.15)
                    AND (mean C2 test AUC >= 0.90)
conclusion_unchanged := (NOT robust_transfer) AND c2_dominates

CODE REUSE / WHAT IS REIMPLEMENTED
----------------------------------
Reused verbatim from exp21b_robust.py via the exec-slice idiom (same pattern
as exp21b_c2_seed42.py): the whole header (data load, diagram/circuit build,
interpreter calibration, encoder class + least-squares init, pair_sets,
states, margin, loss), the SPSA training block, and the evaluation helpers
`analyse`, `order_eval`, `noun_semantics`, `sent_z`, `classical_controls`.
exp21b_robust.py is NOT modified and NOT imported as a module (it is a
script); its own globals dict is reused across seeds by rebinding tr_p/va_p/
te_p, which is safe because every downstream function resolves those names
from that same globals dict at call time.

Reimplemented here (flagged in the report):
  * `_build_C` -- a line-for-line copy of the C-set construction inside
    `analyse` (that code is inline and not exposed as a function).  It is
    verified at runtime: the AUCs recomputed from the locally rebuilt A/B/C
    are asserted equal to the AUCs returned by the original `analyse`.
  * `_auc_perm_sentence` -- the new sentence-level randomisation.
  * the pooled t-test and the verdict rule.

Outputs: results_exp21c.json, exp21c_encoder_seed4*.npz.  Nothing existing is
overwritten.
"""

import json
import os
import time
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJ = "/home/waj/discocat_arabic_v2"
os.chdir(PROJ)
os.environ.setdefault("SPLIT_SEED", "42")   # header only; splits are overridden

SEEDS = [42, 43, 44, 45, 46]
N_TR, N_VA, N_TE = 34, 12, 12
N_PERM = 10000
ALPHA = 0.05
OUT_PATH = f"{PROJ}/results_exp21c.json"

# EXP21C_SMOKE=1 exercises the whole data path (dedup -> split -> SPSA ->
# permutation -> pooled -> verdicts) in ~2 min with a throwaway output file.
# It never touches results_exp21c.json.  Used only for the dry run.
SMOKE = bool(os.environ.get("EXP21C_SMOKE"))
SMOKE_ITERS = 200
if SMOKE:
    SEEDS = [42, 43]
    N_PERM = 500
    OUT_PATH = f"{PROJ}/results_exp21c_smoke.json"

T0 = time.time()


def log(msg):
    print(f"[21c] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. exec the exp21b header (data load, circuits, encoder init)  -- ONCE
# ═══════════════════════════════════════════════════════════════════════════
src = open(f"{PROJ}/exp21b_robust.py").read()
HEAD_END = src.index("ITERS = 6000")
SPSA_END = src.index("# ── evaluation")
CC_START = src.index("# classical controls")
CC_END = src.index("\nw_init = weights_from(theta0)")

head_src = src[:HEAD_END]
spsa_src = src[HEAD_END:SPSA_END]
if SMOKE:
    assert spsa_src.startswith("ITERS = 6000")
    spsa_src = spsa_src.replace("ITERS = 6000", f"ITERS = {SMOKE_ITERS}", 1)
    spsa_src = spsa_src.replace("(k + 1) % 200 == 0",
                                f"(k + 1) % {max(SMOKE_ITERS // 5, 1)} == 0", 1)
eval_src = src[SPSA_END:CC_START]          # analyse / order_eval / noun_semantics
cc_src = src[CC_START:CC_END]              # sent_z / classical_controls

log("exec-ing exp21b_robust.py header (data load + circuit build)...")
t_head = time.time()
ns: dict = {}
exec(compile(head_src, "exp21b_robust.py:head", "exec"), ns)      # noqa: S102
exec(compile(eval_src, "exp21b_robust.py:eval", "exec"), ns)      # noqa: S102
exec(compile(cc_src, "exp21b_robust.py:cc", "exec"), ns)          # noqa: S102
log(f"header + helpers ready in {time.time() - t_head:.1f}s")

sents = ns["sents"]
labels = ns["labels"]
pos = ns["pos"]
vidx = ns["vidx"]
pair_sets = ns["pair_sets"]
states = ns["states"]
fid = ns["fid"]
margin = ns["margin"]
weights_from = ns["weights_from"]
analyse = ns["analyse"]

# ═══════════════════════════════════════════════════════════════════════════
# 2. de-duplication
# ═══════════════════════════════════════════════════════════════════════════
svo_i = [i for i, l in enumerate(labels) if l.endswith("SVO")]
vso_i = [i for i, l in enumerate(labels) if l.endswith("VSO")]
_vpool = defaultdict(list)
for i in vso_i:
    _vpool[tuple(sorted(sents[i].split()))].append(i)
twins_all = []
for i in svo_i:
    k = tuple(sorted(sents[i].split()))
    if _vpool[k]:
        twins_all.append((i, _vpool[k].pop(0)))
assert twins_all == [tuple(t) for t in ns["twins"]], \
    "twin reconstruction does not match exp21b header"

_seen, dropped, twins = {}, [], []
for ci, (i, j) in enumerate(twins_all):
    sig = (sents[i], sents[j])
    if sig in _seen:
        dropped.append({
            "dropped_couple_index": ci,
            "dropped_sentence_indices": [i, j],
            "duplicate_of_couple_index": _seen[sig],
            "kept_sentence_indices": list(twins_all[_seen[sig]]),
            "svo_text": sents[i], "vso_text": sents[j]})
    else:
        _seen[sig] = ci
        twins.append((i, j))

log(f"twins before dedup: {len(twins_all)}   after dedup: {len(twins)}")
for d in dropped:
    log(f"  DROPPED couple #{d['dropped_couple_index']} "
        f"{d['dropped_sentence_indices']} (verbatim duplicate of couple "
        f"#{d['duplicate_of_couple_index']} {d['kept_sentence_indices']}): "
        f"SVO={d['svo_text']!r} VSO={d['vso_text']!r}")
assert len(twins) == 58, f"expected 58 unique twin couples, got {len(twins)}"
assert len({(sents[i], sents[j]) for i, j in twins}) == 58
assert len(dropped) == 2, f"expected 2 duplicated couples, got {len(dropped)}"
_missing = [x for c in twins for x in c if x not in pos]
assert not _missing, f"twin sentences without a compiled circuit: {_missing}"

dup_texts = defaultdict(list)
for i, s in enumerate(sents):
    dup_texts[s].append(i)
dup_texts = {s: v for s, v in dup_texts.items() if len(v) > 1}

DEDUP = {
    "n_sentences": len(sents),
    "n_distinct_sentence_texts": len(set(sents)),
    "n_twin_couples_original": len(twins_all),
    "n_twin_couples_dedup": len(twins),
    "rule": "couple signature = (svo text, vso text); keep first occurrence",
    "dropped_couples": dropped,
    "duplicate_sentence_texts": {s: v for s, v in dup_texts.items()},
}


def make_split(seed, pairs, n_tr=N_TR, n_va=N_VA, n_te=N_TE):
    """Identical protocol to exp21b: fresh default_rng(seed).permutation."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    return ([pairs[i] for i in perm[:n_tr]],
            [pairs[i] for i in perm[n_tr:n_tr + n_va]],
            [pairs[i] for i in perm[n_tr + n_va:n_tr + n_va + n_te]])


# ── diagnostic: how much train->test leakage did the ORIGINAL 60-twin split
#    actually have, per seed?  (this is the size of defect D1) ───────────────
LEAK = {}
for sd in SEEDS:
    o_tr, o_va, o_te = make_split(sd, twins_all, 36, 12, 12)
    t_tr = {sents[x] for c in o_tr for x in c}
    t_va = {sents[x] for c in o_va for x in c}
    t_te = {sents[x] for c in o_te for x in c}
    leaked_couples = [list(c) for c in o_te
                      if (sents[c[0]], sents[c[1]]) in
                      {(sents[a], sents[b]) for a, b in o_tr}]
    LEAK[str(sd)] = {
        "test_sentences_also_in_train_verbatim": len(t_tr & t_te),
        "test_sentences_also_in_val_verbatim": len(t_va & t_te),
        "test_couples_verbatim_present_in_train": len(leaked_couples),
        "leaked_test_couples": leaked_couples,
    }
    log(f"ORIGINAL 60-twin split seed {sd}: "
        f"{LEAK[str(sd)]['test_couples_verbatim_present_in_train']} of 12 test "
        f"couples were verbatim already in train "
        f"({LEAK[str(sd)]['test_sentences_also_in_train_verbatim']} sentences)")

# dedup splits must be leak-free
for sd in SEEDS:
    a, b, c = make_split(sd, twins)
    assert len(a) == N_TR and len(b) == N_VA and len(c) == N_TE
    used = [x for p in (a, b, c) for cc in p for x in cc]
    assert len(used) == 2 * (N_TR + N_VA + N_TE) == 116
    assert len(set(used)) == 116, "sentence reused across dedup split"
    t_tr = {sents[x] for cc in a for x in cc}
    t_va = {sents[x] for cc in b for x in cc}
    t_te = {sents[x] for cc in c for x in cc}
    assert not (t_tr & t_te) and not (t_va & t_te), \
        f"verbatim leak survived dedup on seed {sd}"
log("dedup splits verified leak-free for all 5 seeds (34/12/12, 116 sentences)")


# ═══════════════════════════════════════════════════════════════════════════
# 3. sentence-level permutation machinery
# ═══════════════════════════════════════════════════════════════════════════
def _build_C(A, idxs, svo_k, vso_k):
    """Verbatim copy of the C-set construction inside exp21b_robust.analyse."""
    twinset = set((min(a, b), max(a, b)) for a, b in A)
    rngl = np.random.default_rng(321)
    C, seen = [], set()
    while len(C) < len(A) * 2 and len(seen) < 500:
        i = int(rngl.choice(svo_k))
        j = int(rngl.choice(vso_k))
        key = (min(i, j), max(i, j))
        seen.add(key)
        if key in twinset or (i, j) in C:
            continue
        C.append((i, j))
    return C


def _rank_matrix(F, neg):
    """W[m,n] = #{neg < F[m,n]} + 0.5*#{neg == F[m,n]}; AUC = mean_m W[m,pi(m)]/n_neg."""
    sn = np.sort(np.asarray(neg, dtype=float))
    lo = np.searchsorted(sn, F, side="left")
    hi = np.searchsorted(sn, F, side="right")
    return lo + 0.5 * (hi - lo)


def _uniform_perms(n, n_perm, rng, derangement=False):
    P = np.argsort(rng.random((n_perm, n)), axis=1)
    if derangement:
        bad = (P == np.arange(n)[None, :]).any(axis=1)
        guard = 0
        while bad.any() and guard < 200:
            k = int(bad.sum())
            P[bad] = np.argsort(rng.random((k, n)), axis=1)
            bad = (P == np.arange(n)[None, :]).any(axis=1)
            guard += 1
    return P


def _auc_perm_sentence(F, neg, n_perm, rng):
    """Randomisation over the twin map pi; states and negatives held fixed."""
    n = F.shape[0]
    n_neg = len(neg)
    W = _rank_matrix(F, neg)
    obs = float(np.trace(W) / (n * n_neg))
    rows = np.arange(n)
    out = {"auc": obs}
    for tag, der in (("perm_p_sentence", False),
                     ("perm_p_sentence_derangement", True)):
        P = _uniform_perms(n, n_perm, rng, derangement=der)
        aucs = W[rows[None, :], P].sum(axis=1) / (n * n_neg)
        out[tag] = float((np.sum(aucs >= obs) + 1) / (n_perm + 1))
        out[tag + "_null_mean"] = float(aucs.mean())
        out[tag + "_null_sd"] = float(aucs.std())
    return out


def analyse_c(w, pairs, tag, rng):
    """Original analyse() (medians, AUCs, pooled-label p) + sentence-level p."""
    out = analyse(w, pairs, tag)                       # exp21b code, unmodified
    A, B, idxs = pair_sets(pairs)
    assert len(A) == len(pairs), f"{len(pairs)-len(A)} couples lost in pair_sets"
    svo_k = [k for k in idxs if str(labels[vidx[k]]).endswith("SVO")]
    vso_k = [k for k in idxs if str(labels[vidx[k]]).endswith("VSO")]
    C = _build_C(A, idxs, svo_k, vso_k)
    S = states(w, idxs)
    a_svo = [a for a, _ in A]
    a_vso = [b for _, b in A]
    assert all(str(labels[vidx[k]]).endswith("SVO") for k in a_svo)
    assert all(str(labels[vidx[k]]).endswith("VSO") for k in a_vso)
    F = np.array([[fid(S[i], S[j]) for j in a_vso] for i in a_svo])
    fB = np.array([fid(S[i], S[j]) for i, j in B])
    fC = np.array([fid(S[i], S[j]) for i, j in C])
    for nm, neg in (("A_vs_B", fB), ("A_vs_C", fC)):
        res = _auc_perm_sentence(F, neg, N_PERM, rng)
        # faithfulness check: our rebuilt A/B/C must reproduce exp21b's AUC
        assert abs(res["auc"] - out[nm]["auc"]) < 1e-9, \
            f"{tag} {nm}: rebuilt AUC {res['auc']} != analyse AUC {out[nm]['auc']}"
        out[nm]["perm_p_pooled_label_BIASED"] = out[nm].pop("perm_p")
        out[nm].update({k: v for k, v in res.items() if k != "auc"})
        out[nm]["n_pos"] = int(F.shape[0])
        out[nm]["n_neg"] = int(len(neg))
    log(f"{tag}: AUC_AB={out['A_vs_B']['auc']:.4f} "
        f"p_sent={out['A_vs_B']['perm_p_sentence']:.4f} "
        f"p_der={out['A_vs_B']['perm_p_sentence_derangement']:.4f} "
        f"p_pooled(biased)={out['A_vs_B']['perm_p_pooled_label_BIASED']:.4f} | "
        f"AUC_AC={out['A_vs_C']['auc']:.4f} "
        f"p_sent={out['A_vs_C']['perm_p_sentence']:.4f}")
    return out


# self-test of the permutation machinery against sklearn
_rng = np.random.default_rng(0)
_F = _rng.random((6, 6))
_neg = _rng.random(20)
_W = _rank_matrix(_F, _neg)
_ref = roc_auc_score(np.r_[np.ones(6), np.zeros(20)],
                     np.r_[np.diag(_F), _neg])
assert abs(np.trace(_W) / (6 * 20) - _ref) < 1e-12, "rank-matrix AUC self-test"
_P = _uniform_perms(9, 500, np.random.default_rng(1), derangement=True)
assert not (_P == np.arange(9)[None, :]).any(), "derangement sampler self-test"
log("permutation machinery self-tests passed")


# ═══════════════════════════════════════════════════════════════════════════
# 4. per-seed re-run
# ═══════════════════════════════════════════════════════════════════════════
RESULTS = {
    "experiment": "exp21c",
    "description": "de-duplicated (58-twin) re-run of exp21b-lite with "
                   "sentence-level permutation null and pooled across-split test",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_script": "exp21c_dedup_rerun.py",
    "reused_from": "exp21b_robust.py (exec-slice; unmodified)",
    "protocol": {
        "n_twin_couples": len(twins),
        "split": {"train": N_TR, "val": N_VA, "test": N_TE,
                  "original_exp21b": {"train": 36, "val": 12, "test": 12,
                                      "n_couples": 60},
                  "rationale": "exp21b proportions 0.60/0.20/0.20 of 60 give "
                               "34.8/11.6/11.6 on 58; val and test are held at "
                               "12 couples so every test statistic keeps the "
                               "exact shape it had in exp21b and the AUCs stay "
                               "comparable; the 2 removed couples come out of "
                               "train."},
        "seeds": SEEDS,
        "spsa_iters": 6000,
        "model_selection": "max val margin, checked every 200 iters (exp21b rule)",
        "n_permutations": N_PERM,
        "alpha": ALPHA,
        "bonferroni_alpha": ALPHA / len(SEEDS),
    },
    "dedup": DEDUP,
    "original_split_leakage_diagnostic": LEAK,
    "permutation_scheme": {
        "primary": "A_vs_B",
        "null": "uniform bijection pi over the 12 test twin couples reassigns "
                "which VSO sentence-state is the twin of each SVO "
                "sentence-state; the 24 sentence states and the negative pair "
                "set are held fixed; AUC recomputed per draw",
        "why_valid": "randomisation acts on the twin assignment, not on the "
                     "fidelity values, so the dependence induced by each "
                     "sentence appearing in ~5 within-order (B) pairs is "
                     "carried identically into every replicate; B is ancillary "
                     "(untouched by pi) for the primary endpoint",
        "old_scheme_defect": "exp21b permuted the A/B label over the pooled "
                             "72 fidelities, which requires those 72 dependent "
                             "values to be exchangeable; they are not, so the "
                             "null was too tight and p too small",
        "derangement_variant": "pi restricted to have no fixed points; removes "
                               "the conservatism of retaining ~1 true twin per "
                               "draw, so a non-significant result cannot be "
                               "dismissed as low power",
        "a_vs_c_caveat": "for A_vs_C the negatives are themselves cross-order "
                         "pairs, so a permuted A-pair can coincide with a "
                         "C-pair; this makes A_vs_C mildly conservative",
    },
    "seeds": {},
}


def dump():
    tmp = OUT_PATH + ".tmp"
    json.dump(RESULTS, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, OUT_PATH)


THETA0_REF = np.asarray(ns["theta0"]).copy()
W0_REF = np.asarray(ns["w0"]).copy()

for sd in SEEDS:
    t_seed = time.time()
    log("=" * 70)
    log(f"SEED {sd} starting")
    # the encoder init point must be identical for every seed (exp21b ran each
    # seed in a fresh process; here one process is reused across seeds)
    assert np.array_equal(np.asarray(ns["theta0"]), THETA0_REF), "theta0 mutated"
    assert np.array_equal(np.asarray(ns["w0"]), W0_REF), "w0 mutated"
    tr_p, va_p, te_p = make_split(sd, twins)
    ns["tr_p"], ns["va_p"], ns["te_p"] = tr_p, va_p, te_p
    ns["SPLIT_SEED"] = sd
    # rebuild the training/val pair sets that loss() and the SPSA block read
    ns["A_tr"], ns["B_tr"], ns["k_tr"] = pair_sets(tr_p)
    ns["A_va"], ns["B_va"], ns["k_va"] = pair_sets(va_p)
    assert len(ns["A_tr"]) == N_TR and len(ns["A_va"]) == N_VA
    theta0 = ns["theta0"]
    m0_tr = margin(weights_from(theta0), ns["A_tr"], ns["B_tr"], ns["k_tr"])
    m0_va = margin(weights_from(theta0), ns["A_va"], ns["B_va"], ns["k_va"])
    log(f"seed {sd} init margins: train {m0_tr:+.4f} val {m0_va:+.4f} "
        f"| |A_tr|={len(ns['A_tr'])} |B_tr|={len(ns['B_tr'])} "
        f"|A_va|={len(ns['A_va'])} |B_va|={len(ns['B_va'])}")

    # --- SPSA training block, executed verbatim from exp21b_robust.py --------
    t_spsa = time.time()
    exec(compile(spsa_src, "exp21b_robust.py:spsa", "exec"), ns)   # noqa: S102
    log(f"seed {sd} SPSA done in {time.time() - t_spsa:.1f}s "
        f"(best val margin {ns['best_val']:+.4f})")

    rng = np.random.default_rng(100000 + sd)
    w_init = weights_from(theta0)
    w_best = weights_from(ns["best_theta"])
    entry = {
        "split_seed": sd,
        "n_learned_params": int(ns["n_params"]),
        "iters": int(ns["ITERS"]),
        "init_margins": {"train": float(m0_tr), "val": float(m0_va)},
        "history": ns["history"],
        "pre": {
            "test": analyse_c(w_init, te_p, f"s{sd} PRE test", rng),
            "train": analyse_c(w_init, tr_p, f"s{sd} PRE train", rng),
            "order_test_acc": ns["order_eval"](w_init, f"s{sd} PRE"),
            "nouns": ns["noun_semantics"](w_init, f"s{sd} PRE"),
        },
        "post": {
            "test": analyse_c(w_best, te_p, f"s{sd} POST test", rng),
            "train": analyse_c(w_best, tr_p, f"s{sd} POST train", rng),
            "order_test_acc": ns["order_eval"](w_best, f"s{sd} POST"),
            "nouns": ns["noun_semantics"](w_best, f"s{sd} POST"),
            "best_val_margin": float(ns["best_val"]),
        },
    }
    np.savez(f"{PROJ}/exp21c_encoder_seed{sd}.npz", theta0=theta0,
             theta_best=ns["best_theta"], names=np.array(ns["names"]))
    try:
        entry["classical_controls"] = ns["classical_controls"]()
    except Exception as e:                                     # noqa: BLE001
        entry["classical_controls"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"seed {sd} classical controls FAILED: {type(e).__name__}: {e}")
    entry["runtime_sec"] = round(time.time() - t_seed, 1)
    RESULTS["seeds"][str(sd)] = entry
    log(f"SEED {sd} DONE in {entry['runtime_sec']}s | "
        f"POST test AUC_AB {entry['post']['test']['A_vs_B']['auc']:.4f} "
        f"p_sent {entry['post']['test']['A_vs_B']['perm_p_sentence']:.4f}")
    dump()


# ═══════════════════════════════════════════════════════════════════════════
# 5. pooled inference + verdicts
# ═══════════════════════════════════════════════════════════════════════════
ok = [s for s in map(str, SEEDS) if s in RESULTS["seeds"]]
post_auc = np.array([RESULTS["seeds"][s]["post"]["test"]["A_vs_B"]["auc"]
                     for s in ok])
pre_auc = np.array([RESULTS["seeds"][s]["pre"]["test"]["A_vs_B"]["auc"]
                    for s in ok])
p_sent = np.array([RESULTS["seeds"][s]["post"]["test"]["A_vs_B"]
                   ["perm_p_sentence"] for s in ok])
p_der = np.array([RESULTS["seeds"][s]["post"]["test"]["A_vs_B"]
                  ["perm_p_sentence_derangement"] for s in ok])
p_old = np.array([RESULTS["seeds"][s]["post"]["test"]["A_vs_B"]
                  ["perm_p_pooled_label_BIASED"] for s in ok])
c2 = [RESULTS["seeds"][s]["classical_controls"].get(
        "C2_matched_classical", {}).get("meaning_auc_test")
      for s in ok]
c2_ok = [x for x in c2 if x is not None]
c1 = [RESULTS["seeds"][s]["classical_controls"].get(
        "C1_bow", {}).get("meaning_auc") for s in ok]
c1_ok = [x for x in c1 if x is not None]

t_stat, p_two = stats.ttest_1samp(post_auc, 0.5)
p_one = float(p_two / 2 if t_stat > 0 else 1 - p_two / 2)
fisher_stat = float(-2 * np.sum(np.log(np.clip(p_sent, 1e-300, None))))
fisher_p = float(stats.chi2.sf(fisher_stat, 2 * len(p_sent)))

n_sig = int(np.sum(p_sent < ALPHA))
n_sig_bonf = int(np.sum(p_sent < ALPHA / len(SEEDS)))
n_sig_old = int(np.sum(p_old < ALPHA))

RESULTS["pooled"] = {
    "seeds_included": ok,
    "post_test_auc_A_vs_B": [float(x) for x in post_auc],
    "pre_test_auc_A_vs_B": [float(x) for x in pre_auc],
    "post_mean": float(post_auc.mean()), "post_sd": float(post_auc.std(ddof=1)),
    "pre_mean": float(pre_auc.mean()), "pre_sd": float(pre_auc.std(ddof=1)),
    "mean_improvement_pre_to_post": float(post_auc.mean() - pre_auc.mean()),
    "one_sample_t_vs_0.5": {
        "t": float(t_stat), "df": len(post_auc) - 1,
        "p_one_sided_greater": p_one, "p_two_sided": float(p_two),
        "caveat": "the 5 splits re-sample the SAME 58 couples / 116 sentences, "
                  "so they are not independent replicates; this t-test treats "
                  "them as independent and its p is therefore optimistic "
                  "(an upper bound on the evidence for transfer)"},
    "fisher_combination_of_sentence_level_p": {
        "chi2": fisher_stat, "df": 2 * len(p_sent), "p": fisher_p,
        "caveat": "same non-independence caveat as the t-test"},
    "per_seed_p_sentence_level": {s: float(p) for s, p in zip(ok, p_sent)},
    "per_seed_p_sentence_level_derangement": {s: float(p)
                                              for s, p in zip(ok, p_der)},
    "per_seed_p_pooled_label_BIASED": {s: float(p) for s, p in zip(ok, p_old)},
    "n_seeds_sig_pooled_label_BIASED": n_sig_old,
    "classical": {
        "C2_test_auc_per_seed": {s: v for s, v in zip(ok, c2)},
        "C2_mean": float(np.mean(c2_ok)) if c2_ok else None,
        "C1_test_auc_per_seed": {s: v for s, v in zip(ok, c1)},
        "C1_mean": float(np.mean(c1_ok)) if c1_ok else None,
    },
    "order_acc_pre": {s: RESULTS["seeds"][s]["pre"]["order_test_acc"] for s in ok},
    "order_acc_post": {s: RESULTS["seeds"][s]["post"]["order_test_acc"] for s in ok},
}

c2_mean = float(np.mean(c2_ok)) if c2_ok else float("nan")
robust_transfer = bool(p_one < ALPHA and n_sig_bonf >= 3)
c2_dominates = bool(len(c2_ok) == len(ok)
                    and (c2_mean - float(post_auc.mean())) > 0.15
                    and c2_mean >= 0.90)
RESULTS["verdicts"] = {
    "pooled_p": p_one,
    "n_seeds_sig_sentence_level": n_sig,
    "n_seeds_sig_bonferroni": n_sig_bonf,
    "n_seeds_sig_corrected": n_sig_bonf,          # brief's name for the above
    "robust_transfer": robust_transfer,
    "no_robust_transfer": (not robust_transfer),
    "c2_dominates": c2_dominates,
    "c2_mean_auc": c2_mean,
    "quantum_post_mean_auc": float(post_auc.mean()),
    "conclusion_unchanged": bool((not robust_transfer) and c2_dominates),
    "rule": {
        "robust_transfer": "pooled one-sided p < 0.05 AND >=3/5 seeds "
                           "significant at Bonferroni alpha = 0.05/5 = 0.01",
        "c2_dominates": "mean C2 test AUC - mean POST quantum test AUC > 0.15 "
                        "AND mean C2 test AUC >= 0.90",
        "conclusion_unchanged": "(NOT robust_transfer) AND c2_dominates",
        "pooled_p_definition": "one-sided (greater) one-sample t of the 5 POST "
                               "test A_vs_B AUCs against 0.5, df=4",
    },
}
RESULTS["runtime_sec"] = round(time.time() - T0, 1)
dump()

log("=" * 70)
log(f"POST test AUCs: {[round(float(x), 4) for x in post_auc]}")
log(f"mean POST {post_auc.mean():.4f} +/- {post_auc.std(ddof=1):.4f} "
    f"(PRE {pre_auc.mean():.4f})")
log(f"sentence-level p: {[round(float(x), 4) for x in p_sent]}  "
    f"-> {n_sig}/{len(ok)} at a=0.05, {n_sig_bonf}/{len(ok)} at Bonferroni")
log(f"derangement p:    {[round(float(x), 4) for x in p_der]}")
log(f"OLD pooled-label p (biased): {[round(float(x), 4) for x in p_old]} "
    f"-> {n_sig_old}/{len(ok)} at a=0.05")
log(f"pooled t({len(post_auc)-1}) = {t_stat:.3f}, one-sided p = {p_one:.4f}")
log(f"C2 mean test AUC {c2_mean:.4f} vs quantum POST {post_auc.mean():.4f}")
log(f"VERDICTS: {json.dumps({k: v for k, v in RESULTS['verdicts'].items() if k != 'rule'})}")
log(f"wrote {OUT_PATH}")
log(f"ALL DONE in {RESULTS['runtime_sec']}s")
