# -*- coding: utf-8 -*-
"""exp43b_datagen.py -- difficulty-ladder rung datasets L1/L2 (doc 23,
commit ef15d76, criteria FROZEN) + V-suite records.

Rungs (doc 23 exp43b):
  L1 (atomic swap): 6 verbs, K=4, T=5, ONE-hop swap-twin questions only
     (generator Q0 pairs -- bag-identical twins, answer flips under
     argument exchange). N = 1200/300/300 (train/val/test).
  L2 (mixed): 6 verbs, K=4-6, T=5-8, 50/50 one-hop/two-hop. Same sizes.

Implementation decisions (FLAGGED, fixed before any Phase-1 training):
- "6 verbs" = the six swap-capable verbs of the exp41 inventory
  (human agent AND human patient), sorted: they are the only verbs usable
  in swap-twin questions, so any other choice could not fill L1.
- L2's two-hop half = Q1 bridge questions (the doc-23 exp43a bridge form).
  Q2 (pro-drop) is excluded from the rungs: it entangles the pro-drop axis
  with the hop axis, which the ladder is meant to separate.
- V-suite "for the record" (doc 23): V1-V4 must pass on both rungs.
  V5 (single-sentence oracle) is N/A-BY-DESIGN on L1 -- one-hop items ARE
  single-sentence-answerable; V5 failing there is the design working, not
  a leak. On L2 the full-test V5 number is expected elevated for the same
  reason, so V5's pass criterion is applied to L2's two-hop subset only;
  the full-test number is recorded alongside. Both interpretations are
  recorded in the results JSON, not silently applied.

Generator modules are imported and rebound (VERBS/SWAP_VERBS/SPLIT_SPEC);
the exp41 generator file and all frozen exp41/exp42 artifacts are
untouched. New fixed seeds: L1=4311, L2=4312.
"""

import hashlib
import json
from collections import Counter

import exp41_story_generator as gen
import validate_exp41_data as val

FULL_VERBS = dict(gen.VERBS)
FULL_SPLIT_SPEC = dict(gen.SPLIT_SPEC)

RUNG_VERBS = sorted(v for v, c in FULL_VERBS.items() if c["swap"])

RUNGS = {
    "L1": {
        "seed": 4311,
        "spec": {
            "train": {"Q1": 0, "Q2": 0, "Q0": 600, "K": (4, 4), "T": (5, 5)},
            "val":   {"Q1": 0, "Q2": 0, "Q0": 150, "K": (4, 4), "T": (5, 5)},
            "test":  {"Q1": 0, "Q2": 0, "Q0": 150, "K": (4, 4), "T": (5, 5)},
        },
        "v5_mode": "na_by_design",
    },
    "L2": {
        "seed": 4312,
        "spec": {
            "train": {"Q1": 300, "Q2": 0, "Q0": 300, "K": (4, 6), "T": (5, 8)},
            "val":   {"Q1": 75,  "Q2": 0, "Q0": 75,  "K": (4, 6), "T": (5, 8)},
            "test":  {"Q1": 75,  "Q2": 0, "Q0": 75,  "K": (4, 6), "T": (5, 8)},
        },
        "v5_mode": "two_hop_subset",
    },
}


def canonical_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_rung(rung, cfg):
    gen.VERBS = {v: FULL_VERBS[v] for v in RUNG_VERBS}
    gen.SWAP_VERBS = [v for v, c in gen.VERBS.items() if c["swap"]]
    gen.SPLIT_SPEC = cfg["spec"]
    try:
        items, holdout = gen.generate(cfg["seed"])
    finally:
        gen.VERBS = dict(FULL_VERBS)
        gen.SWAP_VERBS = [v for v, c in gen.VERBS.items() if c["swap"]]
        gen.SPLIT_SPEC = dict(FULL_SPLIT_SPEC)

    sha = canonical_sha256(items)
    counts = Counter((it["split"], it["qtype"]) for it in items)
    meta = {
        "experiment": "exp43b",
        "rung": rung,
        "design_doc": "qnlp_private_docs/23_exp43_ladder_representability_"
                      "design.md (commit ef15d76, FROZEN)",
        "seed": cfg["seed"],
        "dataset_sha256": sha,
        "n_items": len(items),
        "split_counts": dict(Counter(it["split"] for it in items)),
        "split_qtype_counts": {"%s/%s" % (s, q): n
                               for (s, q), n in sorted(counts.items())},
        "verb_inventory": {v: {"human_patients": c["human_patients"],
                               "swap": c["swap"], "objects": c["objects"]}
                           for v, c in FULL_VERBS.items()
                           if v in RUNG_VERBS},
        "humans": gen.HUMANS, "objects": gen.OBJECTS,
        "holdout_verb_argument_pairs": sorted([list(p) for p in holdout]),
        "notes": [
            "difficulty-ladder rung dataset (doc 23); frozen exp41/exp42 "
            "artifacts untouched",
            "6 verbs = the swap-capable subset (only verbs usable in "
            "swap-twin questions)",
            "L2 two-hop half = Q1 bridges; Q2 pro-drop excluded from the "
            "ladder (axis separation)",
        ],
    }
    out = "stories_exp43b_%s.json" % rung
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False,
                  indent=1)
    with open("stories_exp43b_%s_manifest.json" % rung, "w",
              encoding="utf-8") as f:
        json.dump({k: v for k, v in meta.items() if k != "verb_inventory"},
                  f, ensure_ascii=False, indent=1)
    print("[exp43b] wrote %s: %d items sha=%s" % (out, len(items), sha))
    return items, meta


def vsuite_rung(rung, cfg, items, meta):
    """Run the frozen V-checks with rung-appropriate wiring (the exp41
    main() hard-codes main-dataset shape: test = 3 splits, >=500 items,
    hop>=2; rungs have one test split of 300 with rung-defined hops)."""
    train = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    test = [i for i in items if i["split"] == "test"]
    verbs = sorted(meta["verb_inventory"].keys())
    referents = meta["humans"] + meta["objects"]
    name_set = set(referents)

    # run_v1/run_v2 hard-code per-split loops over test_a/b/c and crash on
    # empty subsets; the rung has ONE homogeneous test split, so it is
    # round-robined across the three labels for these two calls only (an
    # arbitrary partition -- headline test_accuracy is unaffected; the
    # per_split thirds are recorded as arbitrary partitions).
    test_v12 = [dict(i, split="test_" + "abc"[n % 3])
                for n, i in enumerate(test)]
    v1 = val.run_v1(train, va, test_v12)
    v2 = val.run_v2(train, va, test_v12)
    for r in (v1, v2):
        r["per_split_note"] = ("single rung test split round-robined over "
                               "test_a/b/c; thirds are arbitrary partitions")
    v3 = val.run_v3(items, verbs, referents)
    v4 = val.run_v4(items, name_set)
    v5_full = val.run_v5(train, va, test)

    v5_mode = cfg["v5_mode"]
    if v5_mode == "na_by_design":
        v5_verdict = {
            "mode": "na_by_design",
            "note": "L1 is one-hop by construction: every item IS "
                    "single-sentence-answerable, so the oracle SHOULD "
                    "succeed; V5 is recorded, not gated (doc 23: V-suite "
                    "'for the record').",
            "full_test": v5_full, "gated": False, "passed": None,
        }
        v5_pass_for_all = True                            # excluded
    else:
        two_hop = [i for i in test if i["hop_count"] >= 2]
        v5_sub = val.run_v5(train, va, two_hop)
        v5_verdict = {
            "mode": "two_hop_subset",
            "note": "L2 mixes one-hop items, which elevate full-test V5 "
                    "by design; the pass criterion applies to the "
                    "two-hop subset (n=%d)." % len(two_hop),
            "full_test": v5_full, "two_hop_subset": v5_sub,
            "gated": True, "passed": v5_sub["passed"],
        }
        v5_pass_for_all = bool(v5_sub["passed"])

    results = {
        "experiment": "exp43b rung validation",
        "rung": rung,
        "dataset_sha256": meta["dataset_sha256"],
        "n_train": len(train), "n_val": len(va), "n_test": len(test),
        "note_n_test": "rung size fixed by doc 23 (300 < the doc-22 "
                       "main-dataset 500 minimum; CI bounds recorded at "
                       "n=300)",
        "V1_bag_of_embeddings": v1,
        "V2_ngram_logistic": v2,
        "V3_balance_audit": v3,
        "V4_dedup": v4,
        "V5_single_sentence_oracle": v5_verdict,
        "v1_passed": v1["passed"], "v2_passed": v2["passed"],
        "v3_passed": v3["passed"], "v4_passed": v4["passed"],
        "all_passed_v1_to_v4": bool(all(x["passed"]
                                        for x in (v1, v2, v3, v4))),
        "all_passed": bool(all(x["passed"] for x in (v1, v2, v3, v4))
                           and v5_pass_for_all),
    }
    out = "results_exp43b_%s_validation.json" % rung
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[exp43b] %s V-suite: v1=%s v2=%s v3=%s v4=%s v5(%s)=%s -> %s"
          % (rung, v1["passed"], v2["passed"], v3["passed"], v4["passed"],
             v5_mode, v5_verdict["passed"], out))
    return results


def main():
    print("[exp43b] rung verbs (6, swap-capable): %s" % RUNG_VERBS)
    for rung, cfg in RUNGS.items():
        items, meta = generate_rung(rung, cfg)
        vsuite_rung(rung, cfg, items, meta)


if __name__ == "__main__":
    main()
