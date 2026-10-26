# -*- coding: utf-8 -*-
"""exp86_aggregate.py -- SOLE WRITER of results_exp86.json.

REGISTRATION: qnlp_private_docs/78_exp86_identity_mechanism.md (REGISTERED 2026-09-17)

Reads the 30 per-unit JSONs from exp86_ckpt/ and computes the three discriminants
exactly as frozen in doc 78 SS0:

  H-PURITY:   paired one-sided t (identity higher mean readout purity at each arm's
               selected epoch), threshold p < 0.05.
  H-DRIFT:    paired one-sided t on final Frobenius distance-from-I (identity nearer),
               threshold p < 0.05. Path-length-from-own-init reported alongside.
  H-COUPLING: Spearman rho between per-seed delta-accuracy (from results_exp82.json)
               and delta-purity, fires iff rho >= 0.5 AND permutation p < 0.05
               (10,000 permutations, seed 86999 -- the ONE allowed RNG use here).

Each hypothesis: SUPPORTED / REFUTED / UNDECIDED (three outcomes, named, no lever
language). Verdict rule (stated explicitly per doc 78):
  SUPPORTED  = test statistic meets the pre-registered threshold in the expected direction
  REFUTED    = opposite direction fires at p < 0.05 (one-sided, other direction)
  UNDECIDED  = neither (including p >= 0.05 without directional reversal)

Usage:
  python exp86_aggregate.py           # check completeness; write results_exp86.json
  python exp86_aggregate.py --check   # completeness check only (no write)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

import exp82_run as R82
import exp73_run as R73

EXP86_CKPT = "exp86_ckpt"
RESULTS_EXP86 = "results_exp86.json"
RESULTS_EXP82 = "results_exp82.json"
SPEC = ("qnlp_private_docs/78_exp86_identity_mechanism.md (REGISTERED 2026-09-17)")

ARMS = R82.ARMS
SEEDS = R82.SEEDS
N_SEEDS = R82.N_SEEDS_82  # 15

PERM_SEED = 86999        # the ONE allowed RNG use in the aggregate
N_PERM = 10_000
RHO_THRESHOLD = 0.5
P_THRESHOLD = 0.05


# ============================================================= data loading
def load_unit(arm, seed):
    path = os.path.join(EXP86_CKPT, "%s_s%d.json" % (arm, seed))
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def load_all_units():
    units = {}
    missing = []
    for arm in ARMS:
        for seed in SEEDS:
            r = load_unit(arm, seed)
            if r is None:
                missing.append("%s_s%d" % (arm, seed))
            else:
                units[(arm, seed)] = r
    return units, missing


def load_exp82_ref():
    ref = json.load(open(RESULTS_EXP82, encoding="utf-8"))
    lookup = {}
    for row in ref["per_seed"]:
        s = row["seed"]
        lookup[(s, "identity_acc")] = row["identity_acc"]
        lookup[(s, "uniform_acc")] = row["uniform_acc"]
        lookup[(s, "identity_selected_epoch")] = row["identity_selected_epoch"]
        lookup[(s, "uniform_selected_epoch")] = row["uniform_selected_epoch"]
    return lookup


# ========================================================= per-seed summaries
def readout_purity_at_epoch(unit, epoch):
    """Mean readout purity at epoch `epoch` (1-based) from the unit's trace.
    Epoch 0 is the init snapshot (index 0 in traces["epoch"]).
    """
    trace_epochs = unit["traces"]["epoch"]
    try:
        idx = trace_epochs.index(epoch)
    except ValueError:
        raise KeyError("epoch %d not found in trace (available: %r...)"
                       % (epoch, trace_epochs[:5]))
    return unit["traces"]["mean_readout_purity"][idx]


def final_frob_from_I(unit, verbs):
    """Mean per-verb Frobenius distance from I at the FINAL epoch."""
    last_idx = -1   # last element is the final epoch
    vals = [unit["traces"]["per_verb_frob_from_I"][v][last_idx] for v in verbs]
    return float(np.mean(vals))


def path_length_from_init(unit, verbs):
    """Cumulative path length = sum of per-epoch Frobenius steps from epoch 1 onwards,
    averaged over verbs. (Epoch-0 step is 0 by construction.)
    """
    # per_verb_frob_step[v] includes epoch-0 step (=0) at index 0
    total = 0.0
    for v in verbs:
        steps = unit["traces"]["per_verb_frob_step"][v]
        # steps[0] = epoch-0 step = 0; steps[1:] = epoch-1 through epoch-N steps
        total += sum(steps[1:])  # skip the epoch-0 entry
    return float(total / len(verbs))


# ======================================================= hypothesis tests
def h_purity(units, ref, verbs):
    """H-PURITY: paired one-sided t (identity higher readout purity at each arm's
    selected epoch). p < 0.05 -> SUPPORTED.
    """
    ident_purities = []
    unif_purities = []
    detail = []
    for seed in SEEDS:
        ep_id = ref[(seed, "identity_selected_epoch")]
        ep_un = ref[(seed, "uniform_selected_epoch")]
        u_id = units.get(("A3-T2-identity", seed))
        u_un = units.get(("A3-T2-uniform", seed))
        if u_id is None or u_un is None:
            continue
        p_id = readout_purity_at_epoch(u_id, ep_id)
        p_un = readout_purity_at_epoch(u_un, ep_un)
        ident_purities.append(p_id)
        unif_purities.append(p_un)
        detail.append({
            "seed": seed,
            "identity_selected_epoch": ep_id,
            "uniform_selected_epoch": ep_un,
            "identity_purity_at_sel": p_id,
            "uniform_purity_at_sel": p_un,
            "diff_identity_minus_uniform": p_id - p_un,
        })

    n = len(ident_purities)
    if n < 2:
        return {"verdict": "UNDECIDED", "reason": "too few paired units",
                "n": n, "detail": detail}

    ia = np.array(ident_purities)
    ua = np.array(unif_purities)
    diffs = ia - ua
    mean_diff = float(np.mean(diffs))
    t_stat, p_val_two = stats.ttest_rel(ia, ua)
    # one-sided: H1 identity > uniform
    p_one = float(stats.ttest_rel(ia, ua, alternative="greater").pvalue)

    if p_one < P_THRESHOLD and mean_diff > 0:
        verdict = "SUPPORTED"
    elif float(stats.ttest_rel(ua, ia, alternative="greater").pvalue) < P_THRESHOLD:
        verdict = "REFUTED"  # opposite direction fires
    else:
        verdict = "UNDECIDED"

    return {
        "verdict": verdict,
        "hypothesis": ("identity-initialized training retains higher readout purity "
                       "at each arm's selected epoch than uniform-initialized training"),
        "n_pairs": n,
        "mean_identity_purity_at_sel": float(np.mean(ia)),
        "mean_uniform_purity_at_sel": float(np.mean(ua)),
        "mean_diff_identity_minus_uniform": mean_diff,
        "sd_diff": float(np.std(diffs, ddof=1)),
        "t_stat": float(t_stat),
        "p_one_sided_identity_greater": p_one,
        "alpha": P_THRESHOLD,
        "test": "scipy.stats.ttest_rel(alternative='greater')",
        "per_seed_detail": detail,
    }


def h_drift(units, verbs):
    """H-DRIFT: paired one-sided t on final Frobenius distance-from-I (identity nearer).
    p < 0.05 -> SUPPORTED. Path-length-from-own-init reported alongside.
    """
    ident_frob = []
    unif_frob = []
    ident_path = []
    unif_path = []
    detail = []
    for seed in SEEDS:
        u_id = units.get(("A3-T2-identity", seed))
        u_un = units.get(("A3-T2-uniform", seed))
        if u_id is None or u_un is None:
            continue
        f_id = final_frob_from_I(u_id, verbs)
        f_un = final_frob_from_I(u_un, verbs)
        pl_id = path_length_from_init(u_id, verbs)
        pl_un = path_length_from_init(u_un, verbs)
        ident_frob.append(f_id)
        unif_frob.append(f_un)
        ident_path.append(pl_id)
        unif_path.append(pl_un)
        detail.append({
            "seed": seed,
            "identity_final_frob_from_I": f_id,
            "uniform_final_frob_from_I": f_un,
            "diff_uniform_minus_identity": f_un - f_id,  # positive -> identity nearer
            "identity_path_length": pl_id,
            "uniform_path_length": pl_un,
        })

    n = len(ident_frob)
    if n < 2:
        return {"verdict": "UNDECIDED", "reason": "too few paired units",
                "n": n, "detail": detail}

    ia = np.array(ident_frob)
    ua = np.array(unif_frob)
    # H-DRIFT: identity nearer -> uniform_frob > identity_frob
    diffs = ua - ia   # positive = identity nearer to I
    mean_diff = float(np.mean(diffs))
    t_stat, _ = stats.ttest_rel(ua, ia)
    # one-sided: H1 uniform > identity (identity nearer)
    p_one = float(stats.ttest_rel(ua, ia, alternative="greater").pvalue)

    if p_one < P_THRESHOLD and mean_diff > 0:
        verdict = "SUPPORTED"
    elif float(stats.ttest_rel(ia, ua, alternative="greater").pvalue) < P_THRESHOLD:
        verdict = "REFUTED"
    else:
        verdict = "UNDECIDED"

    return {
        "verdict": verdict,
        "hypothesis": ("identity-initialized verb blocks end training nearer to I in "
                       "Frobenius norm than uniform-initialized blocks"),
        "lazy_training_caveat": (
            "doc 78 SS0: H-DRIFT BOOLEAN reads the endpoint distance-from-I (field a). "
            "Path-length-from-own-init (field b) is reported alongside to distinguish "
            "'never moved' (path near zero) from 'moved and returned' (path > endpoint). "
            "H-COUPLING carries the causal weight."),
        "n_pairs": n,
        "mean_identity_final_frob_from_I": float(np.mean(ia)),
        "mean_uniform_final_frob_from_I": float(np.mean(ua)),
        "mean_diff_uniform_minus_identity": mean_diff,
        "sd_diff": float(np.std(diffs, ddof=1)),
        "t_stat_uniform_greater": float(t_stat),
        "p_one_sided_identity_nearer": p_one,
        "alpha": P_THRESHOLD,
        "test": "scipy.stats.ttest_rel(uniform_frob, identity_frob, alternative='greater')",
        "path_length_identity_mean": float(np.mean(ident_path)),
        "path_length_uniform_mean": float(np.mean(unif_path)),
        "per_seed_detail": detail,
    }


def h_coupling(units, ref, verbs):
    """H-COUPLING: Spearman rho between per-seed delta-accuracy (from results_exp82.json)
    and delta-purity (identity purity at selected epoch minus uniform purity at selected epoch).

    Fires iff rho >= 0.5 AND permutation p < 0.05 (10,000 permutations, seed 86999).
    The permutation test permutes the purity differences and recomputes rho.
    """
    delta_acc = []
    delta_purity = []
    detail = []
    for seed in SEEDS:
        u_id = units.get(("A3-T2-identity", seed))
        u_un = units.get(("A3-T2-uniform", seed))
        if u_id is None or u_un is None:
            continue
        acc_id = ref[(seed, "identity_acc")]
        acc_un = ref[(seed, "uniform_acc")]
        da = acc_id - acc_un

        ep_id = ref[(seed, "identity_selected_epoch")]
        ep_un = ref[(seed, "uniform_selected_epoch")]
        p_id = readout_purity_at_epoch(u_id, ep_id)
        p_un = readout_purity_at_epoch(u_un, ep_un)
        dp = p_id - p_un

        delta_acc.append(da)
        delta_purity.append(dp)
        detail.append({
            "seed": seed,
            "delta_acc": da,
            "delta_purity": dp,
        })

    n = len(delta_acc)
    if n < 4:
        return {"verdict": "UNDECIDED", "reason": "too few pairs for Spearman",
                "n": n, "detail": detail}

    da_arr = np.array(delta_acc)
    dp_arr = np.array(delta_purity)
    rho, _ = stats.spearmanr(da_arr, dp_arr)
    rho = float(rho)

    # Permutation test (seed 86999 -- the ONE allowed RNG use in the aggregate)
    rng_perm = np.random.default_rng(PERM_SEED)
    perm_rhos = []
    for _ in range(N_PERM):
        dp_perm = rng_perm.permutation(dp_arr)
        r_perm, _ = stats.spearmanr(da_arr, dp_perm)
        perm_rhos.append(float(r_perm))
    perm_rhos = np.array(perm_rhos)
    perm_p = float(np.mean(perm_rhos >= rho))

    # Fires iff rho >= 0.5 AND perm_p < 0.05
    fires = (rho >= RHO_THRESHOLD) and (perm_p < P_THRESHOLD)

    if fires:
        verdict = "SUPPORTED"
    elif rho <= -RHO_THRESHOLD and float(np.mean(perm_rhos <= rho)) < P_THRESHOLD:
        verdict = "REFUTED"   # strong negative correlation
    else:
        verdict = "UNDECIDED"

    return {
        "verdict": verdict,
        "hypothesis": ("per-seed accuracy gain (identity - uniform) tracks per-seed "
                       "purity retention (Spearman rank correlation)"),
        "fires_criterion": ("rho >= %.1f AND permutation p < %.2f"
                            % (RHO_THRESHOLD, P_THRESHOLD)),
        "n_pairs": n,
        "spearman_rho": rho,
        "fires": fires,
        "permutation_p": perm_p,
        "permutation_seed": PERM_SEED,
        "n_permutations": N_PERM,
        "rho_threshold": RHO_THRESHOLD,
        "alpha": P_THRESHOLD,
        "perm_rho_mean": float(perm_rhos.mean()),
        "perm_rho_sd": float(perm_rhos.std(ddof=1)),
        "per_seed_detail": detail,
    }


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Completeness check only; do not write results_exp86.json")
    a = ap.parse_args()

    print("[exp86_aggregate] loading units...", flush=True)
    units, missing = load_all_units()
    n_complete = len(units)
    n_expected = len(ARMS) * N_SEEDS  # 30

    print("[exp86_aggregate] complete=%d / %d  missing=%d"
          % (n_complete, n_expected, len(missing)), flush=True)
    if missing:
        print("[exp86_aggregate] missing units:", missing)

    if a.check:
        print("[exp86_aggregate] --check done. Exiting without writing.")
        return 0 if not missing else 1

    if missing:
        print("[exp86_aggregate] WARNING: %d units missing; "
              "aggregate will run on %d pairs."
              % (len(missing), min(
                  sum(1 for s in SEEDS if ("A3-T2-identity", s) in units
                      and ("A3-T2-uniform", s) in units),
                  N_SEEDS)), flush=True)

    ref = load_exp82_ref()

    # Get verbs from any unit's per_verb traces (keys are verb names from the dataset)
    first_unit = list(units.values())[0]
    verbs = sorted(first_unit["traces"]["per_verb_frob_from_I"].keys())

    # Bit-identity audit: check all loaded units
    bit_failures = []
    for (arm, seed), r in units.items():
        if not r.get("bit_identity_passed", False):
            bit_failures.append("%s_s%d" % (arm, seed))
    if bit_failures:
        print("[exp86_aggregate] BIT-IDENTITY FAILURES: %r" % bit_failures, flush=True)
        print("[exp86_aggregate] These units are invalid (doc 78 SS0). "
              "Aggregate STOPPED.", flush=True)
        return 1

    # Run the three discriminants
    print("[exp86_aggregate] computing H-PURITY...", flush=True)
    hp = h_purity(units, ref, verbs)
    print("[exp86_aggregate] H-PURITY: %s  (t=%.3f  p=%.4f)"
          % (hp["verdict"],
             hp.get("t_stat", float("nan")),
             hp.get("p_one_sided_identity_greater", float("nan"))), flush=True)

    print("[exp86_aggregate] computing H-DRIFT...", flush=True)
    hd = h_drift(units, verbs)
    print("[exp86_aggregate] H-DRIFT: %s  (t=%.3f  p=%.4f)"
          % (hd["verdict"],
             hd.get("t_stat_uniform_greater", float("nan")),
             hd.get("p_one_sided_identity_nearer", float("nan"))), flush=True)

    print("[exp86_aggregate] computing H-COUPLING (perm seed %d, %d perms)..."
          % (PERM_SEED, N_PERM), flush=True)
    hc = h_coupling(units, ref, verbs)
    print("[exp86_aggregate] H-COUPLING: %s  (rho=%.3f  perm_p=%.4f)"
          % (hc["verdict"], hc.get("spearman_rho", float("nan")),
             hc.get("permutation_p", float("nan"))), flush=True)

    # Completeness summary
    complete_pairs = [s for s in SEEDS
                      if ("A3-T2-identity", s) in units and ("A3-T2-uniform", s) in units]
    wall_times = {str(arm) + "_s" + str(seed): r["wall_sec"]
                  for (arm, seed), r in units.items()}

    out = {
        "experiment": "exp86 -- WHY IDENTITY INIT HELPS -- MECHANISM STUDY",
        "spec": SPEC,
        "written_by": "exp86_aggregate.py (sole writer of this file)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "n_units_loaded": n_complete,
        "n_units_expected": n_expected,
        "n_complete_pairs": len(complete_pairs),
        "missing_units": missing,
        "bit_identity_all_passed": len(bit_failures) == 0,
        "bit_identity_failures": bit_failures,
        "verdict_rule": {
            "SUPPORTED": "test statistic meets the pre-registered threshold in the expected direction",
            "REFUTED": "opposite direction fires at p < 0.05 (one-sided, other direction)",
            "UNDECIDED": "neither SUPPORTED nor REFUTED (includes p >= 0.05 without directional reversal)",
        },
        "H_PURITY": hp,
        "H_DRIFT": hd,
        "H_COUPLING": hc,
        "rng_discipline": {
            "unit_rng": "ZERO (all per-epoch instrumentation under torch.no_grad on detached copies)",
            "aggregate_rng": ("np.random.default_rng(%d) for the %d-permutation test ONLY; "
                              "no other RNG use in the aggregate" % (PERM_SEED, N_PERM)),
        },
        "results_exp82_source": RESULTS_EXP82,
        "per_unit_wall_sec": wall_times,
    }

    json.dump(out, open(RESULTS_EXP86, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp86_aggregate] wrote %s" % RESULTS_EXP86, flush=True)
    print("[exp86_aggregate] H-PURITY=%s  H-DRIFT=%s  H-COUPLING=%s"
          % (hp["verdict"], hd["verdict"], hc["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
