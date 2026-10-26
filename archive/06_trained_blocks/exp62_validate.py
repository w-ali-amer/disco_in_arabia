# -*- coding: utf-8 -*-
"""exp62_validate.py -- doc 48 SS8.6: the V-suite as a KILL-GATE on the
three-way dataset. V2-H and V6 are evaluated on BOTH novel inventories.

Reuses exp60_validate's gate functions verbatim (doc-42 discipline: no re-copy).
"""
import argparse, copy, json, os, sys
import exp42_compiler as comp
import exp60_validate as V

DATA = "stories_exp62_SB3.json"
OUT = "results_exp62_SB3_validation.json"


def remap(items, novel_split):
    """Present the 4-split data to the 3-split suite: val_indist -> val, and
    the chosen novel inventory -> test."""
    out = []
    for it in items:
        s = it["split"]
        if s == "val_indist":
            s2 = "val"
        elif s == novel_split:
            s2 = "test"
        elif s == "train":
            s2 = "train"
        else:
            continue
        j = copy.deepcopy(it); j["split"] = s2
        out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    # Rule 6: results files are IMMUTABLE. This writer had no guard and
    # silently overwrote a committed validation file on 2026-08-22.
    if os.path.exists(a.out):
        raise RuntimeError("refusing to overwrite an existing results file: %s" % a.out)
    blob = json.load(open(a.data, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("dataset self-hash mismatch")
    r = {"experiment": "exp62-A Stage A validation", "design_doc": meta["design_doc"],
         "dataset_sha256": meta["dataset_sha256"],
         "split_counts": meta["split_counts"], "per_novel_inventory": {}}
    gates = {}
    for novel in ("test", "val_novel"):
        it = remap(items, novel)
        sub = {"V1_bag_of_embeddings": V.v1_bag(it), "V2_ngram": V.v2_ngram(it),
               "V3_balance": V.v3_balance(it, meta), "V4_dedup": V.v4_dedup(it),
               "V6_token_identity_probe": V.v6_token_probe(it, meta)}
        r["per_novel_inventory"][novel] = sub
        gates["V1@" + novel] = sub["V1_bag_of_embeddings"]["passed"]
        gates["V2H@" + novel] = sub["V2_ngram"]["V2H_passed"]
        gates["V3@" + novel] = sub["V3_balance"]["passed"]
        gates["V4@" + novel] = sub["V4_dedup"]["passed"]
        gates["V6@" + novel] = sub["V6_token_identity_probe"]["passed"]
        print("[exp62-A] novel inventory = %s" % novel)
        print("   V1  %.4f  V2(train-nouns) %.4f  V2H %.4f  V3 min_p %.4f  "
              "V4 contra %d  V6 train %.4f / novel %.4f"
              % (sub["V1_bag_of_embeddings"]["test_accuracy"],
                 sub["V2_ngram"]["train_nouns_val_accuracy"],
                 sub["V2_ngram"]["heldout_accuracy"], sub["V3_balance"]["min_p"],
                 sub["V4_dedup"]["n_contradictory"],
                 sub["V6_token_identity_probe"]["train_nouns_val_accuracy"],
                 sub["V6_token_identity_probe"]["heldout_accuracy"]))
    r["gates"] = gates
    r["kill_gate_passed"] = bool(all(gates.values()))
    json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    bad = [k for k, v in gates.items() if not v]
    print("\n  KILL-GATE: %s%s" % ("PASSED" if r["kill_gate_passed"] else "FAILED",
                                   "" if not bad else "  failing: %s" % bad))
    return 0 if r["kill_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
