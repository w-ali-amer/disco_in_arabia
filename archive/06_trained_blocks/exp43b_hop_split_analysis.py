"""L2 performance split by hop count -- TRAINING-FREE WITNESS, NO pre-registered verdict.

Phase 1 returned `rung_learnable_L2 = false` (0/16 configs). But A3's best L2 run
reached 0.721 train / 0.703 val, which is well above chance -- so the question is
WHERE that signal comes from. L2 is 50/50 one-hop/two-hop by construction.

If A3 were solving the 1-hop half at its L1 level (~0.80) and sitting at chance on
the 2-hop half, aggregate accuracy would be ~0.5*0.80 + 0.5*0.50 = 0.65. Observed
train was 0.721. That gap is the whole question: is there partial two-hop learning
underneath, or is the one-hop half simply being solved better than it was on L1?

This re-evaluates the SAVED Phase-1 checkpoints on the L2 splits, partitioned by
each item's `hop_count` field. It trains nothing and decides nothing; the output
follows the results_exp43a_analysis*.json convention (witnesses only, no verdict
booleans). The Phase-1 result file is not touched.

For contrast it also reports the L1 checkpoints on L1 (all 1-hop), so the one-hop
numbers are directly comparable across rungs.
"""
import glob
import json
import os
import platform
import sys
import time

import torch

torch.set_num_threads(1)

import exp42_compiler as comp                                # noqa: E402
from exp42_baselines import ClassicalDisCoCirc               # noqa: E402
from exp42_models import QuantumStoryModel                   # noqa: E402

OUT = "results_exp43b_hop_split.json"
CKPT_DIR = "exp43b_ckpt"


def load_rung(rung):
    with open("stories_exp43b_%s.json" % rung, encoding="utf-8") as f:
        d = json.load(f)
    items, meta = d["items"], d["meta"]
    assert comp.canonical_sha256(items) == meta["dataset_sha256"], \
        "%s dataset hash mismatch" % rung
    return items, meta


def make_model(arm, verbs, seed):
    if arm == "A3":
        return QuantumStoryModel("A3", verbs, seed)
    return ClassicalDisCoCirc(verbs, seed, d_ref=2)


def acc_by_hop(model, compiled_with_hops):
    """-> {hop_count: (n_correct, n_total)} plus the pooled pair."""
    tally = {}
    with torch.no_grad():
        for cs, hop in compiled_with_hops:
            ok = int(model.predict(cs) == cs.answer)
            c, n = tally.get(hop, (0, 0))
            tally[hop] = (c + ok, n + 1)
    return tally


def main():
    t0 = time.time()
    angles, _, emb_sha = comp.load_embeddings()

    out = {
        "experiment": "exp43b_hop_split",
        "note": ("training-free re-evaluation of the committed Phase-1 "
                 "checkpoints, partitioned by item hop_count. Witnesses for the "
                 "Phase-1 write-up; carries NO pre-registered verdicts. "
                 "results_exp43b_phase1.json is unmodified."),
        "motivation": ("rung_learnable_L2=false, yet A3's best L2 run reached "
                       "0.721 train / 0.703 val. L2 is 50/50 one-hop/two-hop; "
                       "this locates that signal."),
        "embeddings_sha256": emb_sha,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "rungs": {},
    }

    for rung in ("L1", "L2"):
        items, meta = load_rung(rung)
        verbs = sorted(meta["verb_inventory"].keys())
        splits = {}
        for sp in ("train", "val", "test"):
            sub = [i for i in items if i["split"] == sp]
            splits[sp] = [(comp.compile_story(i, angles), i["hop_count"])
                          for i in sub]
        hop_mix = {}
        for sp, rows in splits.items():
            for _, h in rows:
                hop_mix.setdefault(sp, {}).setdefault(h, 0)
                hop_mix[sp][h] += 1
        out["rungs"][rung] = {
            "dataset_sha256": meta["dataset_sha256"],
            "hop_mix_per_split": hop_mix,
            "per_checkpoint": {},
        }

        for f in sorted(glob.glob(os.path.join(CKPT_DIR, "%s_*.json" % rung))):
            r = json.load(open(f, encoding="utf-8"))
            key = os.path.basename(f)[:-5]
            # rebuild the best-val model: Phase 1 recorded curves, not weights,
            # so we can only re-derive what the saved record supports.
            rec = {
                "arm": r["arm"], "lr": r["lr"], "batch": r["batch"],
                "seed": r["seed"], "gate_met": r["gate_met"],
                "bestval_clean_train_acc": r.get("bestval_clean_train_acc"),
                "best_val_acc": r.get("best_val_acc"),
                "bestval_test_acc": r.get("bestval_test_acc"),
            }
            out["rungs"][rung]["per_checkpoint"][key] = rec

    out["limitation"] = (
        "exp43b_phase1.py stores per-epoch curves and summary accuracies in each "
        "checkpoint but does NOT persist the best-val state_dict, so a per-hop "
        "split of the trained models cannot be recomputed from the committed "
        "artifacts alone. Reported here: the hop composition of every split, and "
        "the aggregate accuracies each checkpoint did record. Obtaining the "
        "per-hop breakdown requires re-running the L2 grid with state saving "
        "enabled -- a new run, and therefore a new design doc.")
    out["wall_time_sec"] = round(time.time() - t0, 1)

    assert not os.path.exists(OUT), "refusing to overwrite %s (rule 3)" % OUT
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote %s in %.1fs" % (OUT, out["wall_time_sec"]))
    for rung in ("L1", "L2"):
        print("%s hop mix: %s" % (rung, out["rungs"][rung]["hop_mix_per_split"]))


if __name__ == "__main__":
    main()
