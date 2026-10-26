# -*- coding: utf-8 -*-
"""exp79r_aggregate.py -- SOLE WRITER of results_exp79r.json (doc 71 §12, registered e790627).

Reads the clean C-RESET units in exp79r_ckpt/ (read-only) and evaluates the frozen §12 criteria:
  creset_clean_run_finite        all 5 seeds completed 30 epochs, no non-finite gradient
  channel_preserves_purity_clean §3's criterion (exp79_aggregate.aggregate_purity, imported unedited:
                                 every queried-pair purity > 0.75 at T = 5-10), on the clean seeds
Descriptive: union accuracy per seed beside §10's 0.5099; per-epoch subnormal instrumentation.
Twice-evaluation: purity and union accuracy are recomputed independently from the raw rows and
must agree with the imported path.
"""
import os, json, hashlib, statistics
import exp79_aggregate as A

CK = "exp79r_ckpt"
OUT = "results_exp79r.json"
SEEDS = range(5)
ORIG_UNION_MEAN = 0.5099          # doc 71 §10.1
ORIG_PURITY = "results_exp79.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    assert not os.path.exists(OUT), "Rule 3: results_exp79r.json exists"
    units, metas, inputs = [], {}, {}
    for s in SEEDS:
        p = os.path.join(CK, "CRESET_s%d.json" % s)
        m = os.path.join(CK, "CRESET_s%d_clean_meta.json" % s)
        units.append((s, json.load(open(p, encoding="utf-8"))))
        metas[s] = json.load(open(m))
        inputs[p] = sha(p); inputs[m] = sha(m)

    # ---- creset_clean_run_finite
    fin = {s: {"status": metas[s]["status"], "epochs_run": d["epochs_run"],
               "nonfinite": len(d["nonfinite_skips_doc71_SS9_3"])} for s, d in units}
    finite = all(v["status"] == "completed" and v["epochs_run"] == 30 and v["nonfinite"] == 0
                 for v in fin.values())

    # ---- purity (path 1: registered function, imported unedited)
    pur = A.aggregate_purity(units, "C-RESET-clean")
    # ---- purity (path 2: independent recomputation)
    rows = [r for _, d in units for r in d["purity_rows"]]
    inr = [r["pair_purity_at_readout"] for r in rows if A.PURITY_T_LO <= r["T"] <= A.PURITY_T_HI]
    ind = {"n": len(inr), "mean": sum(inr) / len(inr), "all_above": all(v > A.PURITY_BAR for v in inr),
           "frac_above": sum(v > A.PURITY_BAR for v in inr) / len(inr)}
    assert ind["n"] == pur["n_items_in_T_range"] and ind["all_above"] == pur["all_above_bar"]
    assert abs(round(ind["mean"], 4) - pur["mean_purity_in_T_range"]) < 1e-9
    cpp_clean = bool(pur["channel_preserves_purity"])

    orig = json.load(open(ORIG_PURITY))["purity_by_arm"]["C-RESET"]
    orig_rows_frac = None

    # ---- union accuracy (descriptive), twice
    union = {}
    for s, d in units:
        a1 = d["test"]["test_union"]["accuracy"]
        v = d["union_per_item_correct"]
        a2 = sum(v) / len(v) if isinstance(v, list) else sum(v.values()) / len(v)
        assert abs(a1 - a2) < 1e-9, (s, a1, a2)
        union[s] = a1

    # ---- instrumentation (descriptive)
    inst = {s: {"subnormal_total": sum(e["n_subnormal"] for e in metas[s]["epoch_readout_stats"]),
                "min_abs": min(e["min_abs"] for e in metas[s]["epoch_readout_stats"]),
                "epochs_with_subnormals": [i + 1 for i, e in enumerate(metas[s]["epoch_readout_stats"])
                                           if e["n_subnormal"] > 0]} for s in SEEDS}

    res = {
        "component": "results_exp79r -- C-RESET clean re-run (doc 71 §12, e790627; erratum §12.3)",
        "criteria": {
            "creset_clean_run_finite": {"value": finite, "per_seed": fin},
            "channel_preserves_purity_clean": {
                "value": cpp_clean,
                "definition": "doc 71 §3/§1: every queried-pair purity > 0.75 at T=5-10 "
                              "(exp79_aggregate.aggregate_purity, imported unedited)",
                "registered_original": False,
                "consequence_rule": ("SUPERSEDES §10 for C-RESET" if cpp_clean is not False
                                     else "AGREES with §10 (FALSE): confirmed on clean code"),
            },
        },
        "purity_clean": pur,
        "purity_twice_evaluation": {"independent": ind, "agrees": True},
        "purity_original_for_comparison": {"mean_purity_in_T_range": orig["mean_purity_in_T_range"],
                                           "all_above_bar": orig["all_above_bar"]},
        "union_accuracy_descriptive": {"per_seed": union, "mean": statistics.mean(union.values()),
                                       "sd": statistics.stdev(union.values()),
                                       "original_section10_mean": ORIG_UNION_MEAN},
        "val_acc_per_seed": {s: d["val_acc"] for s, d in units},
        "readout_instrumentation": inst,
        "determinism_note": "s0/s2/s4 relaunch per-epoch logs identical to exp79r_logs/failed_v1 (30/30 epochs)",
        "inputs_sha256": inputs,
        "aggregator_sha256": sha(__file__),
        "exp79_aggregate_sha256": sha(A.__file__),
    }
    json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
    print("creset_clean_run_finite:", finite)
    print("channel_preserves_purity_clean:", cpp_clean, "| mean purity T5-10 %.4f (orig %.4f), frac>0.75 %.3f"
          % (pur["mean_purity_in_T_range"], orig["mean_purity_in_T_range"], ind["frac_above"]))
    print("union acc: %s mean %.4f (orig %.4f)" % ({s: round(a, 4) for s, a in union.items()},
                                                  res["union_accuracy_descriptive"]["mean"], ORIG_UNION_MEAN))


if __name__ == "__main__":
    main()
