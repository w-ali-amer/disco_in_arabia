# -*- coding: utf-8 -*-
"""exp42_c2_datagen.py -- dedicated datasets for control C2 (doc 22 SS6).

C2 informative null: blocks trained on a DISJOINT verb inventory,
evaluated on the held-out verbs' stories, must be <= chance + 5pp.

The frozen main dataset (stories_exp41.json) mixes verbs freely, so it
contains almost no verb-disjoint stories (46 verbset-A-only train items,
zero all-B test items). This script therefore runs the exp41 generator
TWICE with restricted verb inventories (pre-training generator work is
explicitly allowed by doc 22 SS2.3; the frozen main dataset is untouched):

  stories_exp42_c2_trainA.json  (seed 4201, verbset A)  -> C2 training
  stories_exp42_c2_evalB.json   (seed 4202, verbset B)  -> C2 evaluation

The A/B split is the deterministic swap-balanced rule in
exp42_controls.c2_verb_split (see its FLAGGED CHANGE note). The exp41
V-suite (validate_exp41_data, criteria frozen) is run on BOTH datasets;
verdicts land in results_exp42_c2_validation_trainA*.json /
..._evalB*.json. Hashes of both datasets are recorded in their meta and
consumed by exp42_c2_run.py.

exp41_story_generator is imported and its module-level VERBS/SWAP_VERBS
are rebound for each run -- the generator file itself is not modified.
"""

import hashlib
import json
from collections import Counter

import exp41_story_generator as gen
import validate_exp41_data as val
from exp42_controls import c2_verb_split

FULL_VERBS = dict(gen.VERBS)                              # snapshot


def canonical_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_restricted(subset, seed, out, manifest_out, label):
    gen.VERBS = {v: FULL_VERBS[v] for v in subset}
    gen.SWAP_VERBS = [v for v, c in gen.VERBS.items() if c["swap"]]
    assert len(gen.SWAP_VERBS) >= 2, \
        "verbset %s has <2 swap verbs -- Q1 bridges degenerate" % label
    try:
        items, holdout = gen.generate(seed)
    finally:
        gen.VERBS = dict(FULL_VERBS)                      # restore
        gen.SWAP_VERBS = [v for v, c in gen.VERBS.items() if c["swap"]]

    sha = canonical_sha256(items)
    counts = Counter((it["split"], it["qtype"]) for it in items)
    split_counts = Counter(it["split"] for it in items)
    meta = {
        "experiment": "exp42_c2",
        "purpose": "control C2 informative-null dataset (%s)" % label,
        "design_doc": "qnlp_private_docs/22_phase2_trained_story_qa_"
                      "design.md SS6 C2",
        "seed": seed,
        "dataset_sha256": sha,
        "n_items": len(items),
        "split_counts": dict(split_counts),
        "split_qtype_counts": {"%s/%s" % (s, q): n
                               for (s, q), n in sorted(counts.items())},
        "verb_inventory": {v: {"human_patients": c["human_patients"],
                               "swap": c["swap"], "objects": c["objects"]}
                           for v, c in FULL_VERBS.items() if v in subset},
        "humans": gen.HUMANS, "objects": gen.OBJECTS,
        "holdout_verb_argument_pairs": sorted([list(p) for p in holdout]),
        "notes": ["restricted-inventory generator run for control C2; "
                  "main frozen dataset untouched",
                  "verb subset (%s): %s" % (label, sorted(subset))],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False,
                  indent=1)
    manifest = {k: v for k, v in meta.items() if k != "verb_inventory"}
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("[c2-datagen] wrote %s: %d items, sha256=%s" % (out, len(items),
                                                          sha))
    return sha


def run_vsuite(data_path, results_base):
    val.DATA = data_path
    val.RESULTS_BASE = results_base
    print("[c2-datagen] V-suite on %s -> %s*.json"
          % (data_path, results_base))
    val.main()


def main():
    inventory = {v: {"swap": c["swap"]} for v, c in FULL_VERBS.items()}
    va, vb = c2_verb_split(inventory)
    print("[c2-datagen] verbset A (train inventory): %s" % va)
    print("[c2-datagen] verbset B (eval inventory):  %s" % vb)

    generate_restricted(va, 4201, "stories_exp42_c2_trainA.json",
                        "stories_exp42_c2_trainA_manifest.json", "trainA")
    generate_restricted(vb, 4202, "stories_exp42_c2_evalB.json",
                        "stories_exp42_c2_evalB_manifest.json", "evalB")

    run_vsuite("stories_exp42_c2_trainA.json",
               "results_exp42_c2_validation_trainA")
    run_vsuite("stories_exp42_c2_evalB.json",
               "results_exp42_c2_validation_evalB")


if __name__ == "__main__":
    main()
