# -*- coding: utf-8 -*-
"""exp53_validate.py -- doc 37 leak gate: the V1-V5 suite on rung RP.

BLOCKING. doc 29 SS3: "If any of V1-V5 fails, exp46 is VOID and is reported
as void, not as a result." Runs before any block is trained.

The rung wiring below is inherited VERBATIM from
exp43b_datagen.vsuite_rung -- validate_exp41_data.main() hard-codes the
main exp41 dataset's shape (three test splits, >=500 items, hop>=2) and
crashes on a rung, which has ONE homogeneous test split of 300 with
rung-defined hops. Only the experiment label and output filename differ, so
that an exp46 result is never written into an exp43b results file.

V5 mode is "na_by_design", exactly as for L1: W1 is one-hop by
construction, so every item IS single-sentence-answerable and the oracle
SHOULD succeed. V5 is recorded, not gated. V1-V4 are gated.

New-verb OOV risk is nil: V1 and V5 build their vocabulary from the surface
tokens of this dataset and train their own embeddings, so the Form II / III
surface forms introduced here carry no pretrained-embedding exposure.
"""

import json
import platform

import torch

import validate_exp41_data as val
from exp42_run import fresh_results_path

RUNG = "RP"
DESIGN_DOC = ("qnlp_private_docs/37_exp53_root_pattern_composition.md "
              "(1ba1469+38a9fd4+b52d1f9, FROZEN)")


def main():
    with open("stories_exp53_%s.json" % RUNG, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]

    train = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    test = [i for i in items if i["split"] == "test"]
    verbs = sorted(meta["verb_inventory"].keys())
    referents = meta["humans"] + meta["objects"]
    name_set = set(referents)

    # inherited workaround (exp43b_datagen.vsuite_rung): run_v1/run_v2 loop
    # over test_a/b/c and crash on empty subsets; the rung has ONE
    # homogeneous test split, round-robined across the three labels for
    # these two calls only. Headline test_accuracy is unaffected.
    test_v12 = [dict(i, split="test_" + "abc"[n % 3])
                for n, i in enumerate(test)]
    print("[exp53-val] V1 (bag-of-embeddings) ...", flush=True)
    v1 = val.run_v1(train, va, test_v12)
    print("[exp53-val] V2 (1-2gram logistic) ...", flush=True)
    v2 = val.run_v2(train, va, test_v12)
    for r in (v1, v2):
        r["per_split_note"] = ("single rung test split round-robined over "
                               "test_a/b/c; thirds are arbitrary partitions")
    print("[exp53-val] V3 (balance audit) ...", flush=True)
    v3 = val.run_v3(items, verbs, referents)
    print("[exp53-val] V4 (dedup) ...", flush=True)
    v4 = val.run_v4(items, name_set)
    print("[exp53-val] V5 (single-sentence oracle, ungated) ...", flush=True)
    v5_full = val.run_v5(train, va, test)

    v5_verdict = {
        "mode": "na_by_design",
        "note": ("RP is one-hop by construction: every item IS "
                 "single-sentence-answerable, so the oracle SHOULD succeed. "
                 "V5 is recorded, not gated -- identical treatment to L1 "
                 "(doc 23 V-suite 'for the record')."),
        "full_test": v5_full, "gated": False, "passed": None,
    }

    gated = (v1, v2, v3, v4)
    all_passed = bool(all(x["passed"] for x in gated))
    results = {
        "experiment": "exp53 rung validation (doc 37 leak gate)",
        "design_doc": DESIGN_DOC,
        "rung": RUNG,
        # doc 24 SS8 P5: platform/python/torch recorded in every results JSON
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "env_note": ("scipy 1.18.0 and scikit-learn 1.9.0 were installed into "
                     "the Mac env on 2026-08-13 for this gate: "
                     "validate_exp41_data imports chi2_contingency and "
                     "CountVectorizer. Additive only -- torch 2.13.0 and "
                     "numpy 2.5.1 unchanged. The exp44/45 training line still "
                     "imports torch alone."),
        "dataset_sha256": meta["dataset_sha256"],
        "grid_provenance": meta["grid_provenance"],
        "held_out_cells": meta["held_out_cells"],
        "n_train": len(train), "n_val": len(va), "n_test": len(test),
        "note_n_test": ("rung size fixed by doc 35 SS11 (300 < the doc-22 "
                        "main-dataset 500 minimum; CI bounds recorded at "
                        "n=300)"),
        "oov_note": ("V1/V5 build vocabulary from this dataset's surface "
                     "tokens and train their own embeddings, so the Form II "
                     "and Form III surface forms introduced here carry no "
                     "pretrained-embedding OOV exposure."),
        "V1_bag_of_embeddings": v1,
        "V2_ngram_logistic": v2,
        "V3_balance_audit": v3,
        "V4_dedup": v4,
        "V5_single_sentence_oracle": v5_verdict,
        "v1_passed": v1["passed"], "v2_passed": v2["passed"],
        "v3_passed": v3["passed"], "v4_passed": v4["passed"],
        "all_passed_v1_to_v4": all_passed,
        "leak_gate_passed": all_passed,
        "consequence_if_failed": ("doc 35 SS7: exp46 is VOID and reported as "
                                  "void with the failing check named; no "
                                  "clustering number is quoted anywhere."),
    }
    # Rule 3: never overwrite a results file; new script version -> new name
    out = fresh_results_path("results_exp53_%s_validation" % RUNG)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n[exp53] V-suite: v1=%s(%.4f) v2=%s(%.4f) v3=%s v4=%s v5=recorded"
          % (v1["passed"], v1["test_accuracy"], v2["passed"],
             v2["test_accuracy"], v3["passed"], v4["passed"]))
    print("[exp53] NOTE: V1 at ~0.500 is STRUCTURAL here (twin bag-identity, "
          "doc 37). If V1 is materially above chance the GENERATOR is "
          "buggy, not the task hard.")
    print("[exp53] LEAK GATE: %s -> %s"
          % ("PASS" if all_passed else "FAIL (exp46 VOID)", out))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
