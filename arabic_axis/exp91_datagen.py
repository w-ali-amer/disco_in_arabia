# -*- coding: utf-8 -*-
"""exp91_datagen.py -- doc 84 SS1: EXACT-COUNT rendering of the exp90 case-twin
draws on the registered dose ladder.

Reuses exp90 verbatim (no exp90 file is edited):
  * exp90_datagen.RENDER_SEEDS / make_item  (item construction),
  * the SAME per-event uniform stream: u_e = random.Random(render_seed).random()
    drawn once per train+val event in sorted-id order (exactly exp90's loop),
  * the exp90 TEST items: rebuilt with exp90's own loop AND asserted equal to
    the committed stories_exp90_<draw>.json["test"].

Rule (doc 84 SS1):
  TRAIN: for ladder n, the n TRAIN events with the smallest u_e are CONFLICT,
         every other train event CONSISTENT.
  VAL:   CONFLICT iff u_e < tau_n, tau_n = midpoint of the n-th and (n+1)-th
         smallest TRAIN u_e.
The registered val counts and the pre-verified tau_12 / tau_5 are asserted.

    python3 exp91_datagen.py --draw {primary,confirm}
"""
import argparse
import json
import os
import random
import sys

import exp90_encoder as enc                          # noqa: F401 (provenance)
from exp90_common import canonical_sha256, file_sha256, write_new_json
from exp90_datagen import RENDER_SEEDS, make_item

DESIGN_DOC_91 = "qnlp_private_docs/84_exp91_dose_bracket.md (REGISTERED 4b5a75d)"
LADDER = [5, 7, 9, 12, 14, 18, 24, 36, 48, 72]
COND = {n: "n%d" % n for n in LADDER}
DRAWS = {
    "primary": {"base": "stories_exp60_SB.json",
                "exp90": "stories_exp90_primary.json",
                "out": "stories_exp91_primary.json",
                "anchor_n": 12},
    "confirm": {"base": "stories_exp90_confirm_base.json",
                "exp90": "stories_exp90_confirm.json",
                "out": "stories_exp91_confirm.json",
                "anchor_n": 5},
}
# doc 84 SS1, registered (orchestrator pre-verified)
EXPECTED_VAL = {
    "primary": {5: 0, 7: 0, 9: 0, 12: 2, 14: 2, 18: 3, 24: 4, 36: 7, 48: 8, 72: 10},
    "confirm": {5: 2, 7: 3, 9: 4, 12: 4, 14: 5, 18: 5, 24: 7, 36: 7, 48: 11, 72: 17},
}
EXPECTED_TAU = {("primary", 12): 0.02032, ("confirm", 5): 0.01717}
TAU_TOL = 5e-6                                       # doc quotes 4 s.f.


def load_base(base_path):
    blob = json.load(open(base_path, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("base provenance FAILED: %s" % base_path)
    bearers = set(meta["bearers"])
    for it in items:                                 # exp60 label rule
        ev = it["events"][0]
        assert it["answer"] == ("yes" if ev["agent"] in bearers else "no")
        assert len(it["events"]) == 1 and it["K"] == 2
    return items, meta, bearers


def uniform_stream(items, render_seed):
    """exp90_datagen.build's stream, reproduced line for line."""
    rng = random.Random(render_seed)
    tv = sorted([i for i in items if i["split"] in ("train", "val")],
                key=lambda x: x["id"])
    u = {b["id"]: rng.random() for b in tv}
    return tv, u


def build_test(items, bearers):
    """exp90_datagen.build's test loop, reproduced line for line."""
    test = []
    for b in sorted([i for i in items if i["split"] == "test"],
                    key=lambda x: x["id"]):
        a, p = b["events"][0]["agent"], b["events"][0]["patient"]
        cons = make_item(b, "CONSISTENT", "test", a, p, bearers, "CONS")
        conf = make_item(b, "CONFLICT", "test", p, a, bearers, "CONF",
                         twin_of=cons["id"])
        assert conf["slots"] == cons["slots"], b["id"]
        assert conf["slot_cases"] == cons["slot_cases"][::-1]
        assert conf["answer"] != cons["answer"], b["id"]
        assert cons["answer"] == b["answer"]
        test += [cons, conf]
    return test


def thresholds(tv, u, n):
    tr = sorted(u[b["id"]] for b in tv if b["split"] == "train")
    assert 0 < n < len(tr)
    assert tr[n - 1] < tr[n], "tie at the rank-%d boundary" % n
    return tr[n - 1], tr[n], 0.5 * (tr[n - 1] + tr[n])


def render(tv, u, bearers, n):
    """Returns (items, conflict_train_ids, conflict_val_ids, tau)."""
    train_ranked = sorted((b for b in tv if b["split"] == "train"),
                          key=lambda b: u[b["id"]])
    conf_train = {b["id"] for b in train_ranked[:n]}
    _, _, tau = thresholds(tv, u, n)
    out = []
    for b in tv:                                     # exp90's sorted-id order
        a, p = b["events"][0]["agent"], b["events"][0]["patient"]
        if b["split"] == "train":
            conflict = b["id"] in conf_train
        else:
            conflict = u[b["id"]] < tau
        it = make_item(b, "CONFLICT" if conflict else "CONSISTENT",
                       b["split"], a, p, bearers,
                       "CONF" if conflict else "CONS")
        assert it["answer"] == b["answer"]
        out.append(it)
    conf_val = {b["id"] for b in tv if b["split"] == "val" and u[b["id"]] < tau}
    return out, conf_train, conf_val, tau


def build(draw, render_seed=None, check_registered=True):
    spec = DRAWS[draw]
    base_path = spec["base"]
    rs = RENDER_SEEDS[base_path] if render_seed is None else render_seed
    items, meta, bearers = load_base(base_path)
    test = build_test(items, bearers)
    if check_registered:
        e90 = json.load(open(spec["exp90"], encoding="utf-8"))
        assert test == e90["test"], "test items differ from exp90's %s" % spec["exp90"]
        assert e90["meta"]["render_seed"] == rs
    tv, u = uniform_stream(items, rs)
    train_sets, counts = {}, {}
    for n in LADDER:
        out, ct, cv, tau = render(tv, u, bearers, n)
        lo, hi, _ = thresholds(tv, u, n)
        c = {
            "n": n,
            "k": n / sum(1 for b in tv if b["split"] == "train"),
            "tau": tau, "u_rank_n": lo, "u_rank_n_plus_1": hi,
            "n_conflict_train": sum(1 for i in out if i["split"] == "train"
                                    and i["item_type"] == "CONFLICT"),
            "n_train": sum(1 for i in out if i["split"] == "train"),
            "n_conflict_val": sum(1 for i in out if i["split"] == "val"
                                  and i["item_type"] == "CONFLICT"),
            "n_val": sum(1 for i in out if i["split"] == "val"),
        }
        assert c["n_conflict_train"] == n == len(ct)
        assert c["n_conflict_val"] == len(cv)
        if check_registered:
            assert c["n_conflict_val"] == EXPECTED_VAL[draw][n], \
                (draw, n, c["n_conflict_val"], EXPECTED_VAL[draw][n])
            if (draw, n) in EXPECTED_TAU:
                assert abs(tau - EXPECTED_TAU[(draw, n)]) < TAU_TOL, (draw, n, tau)
        train_sets[COND[n]] = out
        counts[COND[n]] = c
    m = {
        "experiment": "exp91", "design_doc": DESIGN_DOC_91, "draw": draw,
        "base_file": os.path.basename(base_path),
        "base_file_sha256": file_sha256(base_path),
        "base_dataset_sha256": meta["dataset_sha256"],
        "base_seed": meta["seed"], "render_seed": rs,
        "render_rule": ("u_e ~ U(0,1) once per train/val event in sorted-id "
                        "order from random.Random(render_seed) (exp90 stream); "
                        "TRAIN: the n smallest-u_e train events are CONFLICT; "
                        "VAL: CONFLICT iff u_e < tau_n = midpoint of the n-th "
                        "and (n+1)-th smallest TRAIN u_e"),
        "ladder": LADDER, "counts": counts,
        "test_source": spec["exp90"],
        "test_source_sha256": file_sha256(spec["exp90"]),
        "bearers": meta["bearers"], "humans": meta["humans"],
        "train_inventory": meta["train_inventory"],
        "heldout_inventory": meta["heldout_inventory"],
        "verbs": sorted(meta["verb_inventory"].keys()),
        "n_test": len(test),
    }
    data = {"test": test, "train_sets": train_sets}
    m["data_sha256"] = canonical_sha256(data)
    return {"meta": m, **data}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draw", required=True, choices=sorted(DRAWS))
    args = ap.parse_args()
    obj = build(args.draw)
    out = DRAWS[args.draw]["out"]
    sha = write_new_json(out, obj)
    print("[exp91_datagen] wrote %s  payload_sha256=%s" % (out, sha))
    print("  render_seed=%d n_test=%d" % (obj["meta"]["render_seed"],
                                           obj["meta"]["n_test"]))
    for c, v in obj["meta"]["counts"].items():
        print("  %-4s k=%.4f tau=%.5f  n_conflict_train=%d/%d  n_conflict_val=%d/%d"
              % (c, v["k"], v["tau"], v["n_conflict_train"], v["n_train"],
                 v["n_conflict_val"], v["n_val"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
