# -*- coding: utf-8 -*-
"""exp62_a0.py -- doc 48 SS2: the SELECTION-GAP DIAGNOSTIC.

Re-runs existing exp61 configurations with held-out accuracy logged EVERY epoch,
then compares what best-val selection actually picked against what the best
epoch would have given.

  oracle_heldout  = max over epochs of held-out accuracy   <- DIAGNOSTIC ONLY
  selected_heldout= held-out of the best-val checkpoint    <- what exp61 reported

doc 48 SS2.3 FROZEN DECISION RULE:
  selection_costs_transfer = TRUE iff mean(oracle - selected) >= 0.06 at ANY cell
  TRUE  -> A (3-way split) runs BEFORE B and C
  FALSE -> A descoped to a note; doc 47 SS17.3's headline stands

doc 48 SS2.2 FAITHFULNESS: the instrumented rerun must reproduce each stored
checkpoint's train/val/heldout EXACTLY. Per-epoch eval consumes no RNG, so this
is asserted, not argued. A mismatch invalidates the diagnostic.
"""
import copy, json, math, os, random, sys, time
import torch
torch.set_num_threads(1)
import exp42_compiler as comp
import exp61_bands as B
import exp61_run as R

PROBES = [("S1", "M2"), ("S1", "M3"), ("S2", "M2w")]      # doc 48 SS2
SEEDS = 5
OUT = "results_exp62_a0.json"
MARGIN = 0.06


def instrumented(arm, cell, seed, items, meta, verbs):
    ang = R.angles_for(arm, cell, seed, meta)
    data = {s: comp.compile_all([i for i in items if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = R.CapModel("A3", verbs, seed, reup=R.REUP.get(arm, 0))
    opt = torch.optim.Adam(m.parameters(), lr=R.LR)
    rng = random.Random(R.RNG_BASE + seed)
    best = {"val": -1.0, "train": -1.0, "heldout": None}
    curve = []
    for ep in range(1, R.MAX_EPOCHS + 1):
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        for s in range(0, len(order), R.BATCH):
            opt.zero_grad()
            bl = 0.0
            for i in order[s:s + R.BATCH]:
                cs = data["train"][i]
                bl = bl + torch.nn.functional.cross_entropy(
                    torch.log(m.story_values(cs) + 1e-12).unsqueeze(0),
                    torch.tensor([cs.answer]))
            (bl / min(R.BATCH, len(order) - s)).backward()
            opt.step()
        with torch.no_grad():
            va = R.evaluate(m, data["val"])
            te = R.evaluate(m, data["test"])          # DIAGNOSTIC, never selects
        curve.append({"epoch": ep, "val": va, "heldout": te})
        if va > best["val"]:
            with torch.no_grad():
                best = {"val": va, "train": R.evaluate(m, data["train"]),
                        "heldout": te}
        if ep % 50 == 0:
            print("[a0 %s %s s%d] ep %3d val=%.3f held=%.3f best_val=%.3f"
                  % (arm, cell, seed, ep, va, te, best["val"]), flush=True)
    oracle = max(c["heldout"] for c in curve)
    return {"arm": arm, "cell": cell, "seed": seed,
            "selected_val": best["val"], "selected_train": best["train"],
            "selected_heldout": best["heldout"], "oracle_heldout": oracle,
            "oracle_minus_selected": oracle - best["heldout"],
            "oracle_epoch": max(curve, key=lambda c: c["heldout"])["epoch"],
            "curve": curve}


def main():
    blob = json.load(open(R.DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    verbs = sorted(meta["verb_inventory"].keys())
    rows, faithful = [], []
    for arm, cell in PROBES:
        for s in range(SEEDS):
            r = instrumented(arm, cell, s, items, meta, verbs)
            ref = json.load(open("exp61_ckpt/%s_%s_s%d.json" % (arm, cell, s),
                                 encoding="utf-8"))
            ok = (r["selected_train"] == ref["train_acc"]
                  and r["selected_val"] == ref["val_acc"]
                  and r["selected_heldout"] == ref["heldout_test_acc"])
            faithful.append(ok)
            r["faithful_to_exp61"] = bool(ok)
            rows.append(r)
            print("[a0] %-3s %-4s s%d  selected=%.4f  oracle=%.4f (ep %d)  "
                  "gap=%+.4f  faithful=%s"
                  % (arm, cell, s, r["selected_heldout"], r["oracle_heldout"],
                     r["oracle_epoch"], r["oracle_minus_selected"], ok), flush=True)
    assert all(faithful), ("doc 48 SS2.2 FAITHFULNESS FAILED -- diagnostic "
                           "invalid: %s" % [r for r, f in zip(rows, faithful) if not f])
    per_cell = {}
    for arm, cell in PROBES:
        g = [r for r in rows if r["arm"] == arm and r["cell"] == cell]
        per_cell["%s/%s" % (arm, cell)] = {
            "mean_selected": sum(r["selected_heldout"] for r in g) / len(g),
            "mean_oracle": sum(r["oracle_heldout"] for r in g) / len(g),
            "mean_gap": sum(r["oracle_minus_selected"] for r in g) / len(g),
            "n_seeds": len(g)}
    trig = [k for k, v in per_cell.items() if v["mean_gap"] >= MARGIN]
    out = {"experiment": "exp62-A0",
           "spec": "qnlp_private_docs/48_exp62_selection_optimisation_width.md (ed4927b, FROZEN)",
           "margin": MARGIN, "per_cell": per_cell,
           "faithfulness_all_exact": bool(all(faithful)),
           "runs": [{k: v for k, v in r.items() if k != "curve"} for r in rows],
           "curves": {"%s/%s/s%d" % (r["arm"], r["cell"], r["seed"]): r["curve"]
                      for r in rows},
           "verdicts": {"selection_costs_transfer": bool(trig),
                        "cells_triggering": trig,
                        "consequence": ("A (3-way split) RUNS BEFORE B and C"
                                        if trig else
                                        "A DESCOPED to a note; doc 47 SS17.3 stands")},
           "meta_notes": {"oracle_is_diagnostic":
                          "oracle_heldout measures what selection COST. It is "
                          "never a reported model performance and never a "
                          "selection criterion (doc 48 SS2.1)."}}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[a0] VERDICTS")
    for k, v in out["verdicts"].items():
        print("   %-28s %s" % (k, v))
    for k, v in per_cell.items():
        print("   %-12s selected %.4f  oracle %.4f  gap %+.4f"
              % (k, v["mean_selected"], v["mean_oracle"], v["mean_gap"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
