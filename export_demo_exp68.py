# -*- coding: utf-8 -*-
"""export_demo_exp68.py -- emit the browser demo's exp68 "width lever" assets.

Track 2 of the demonstration phase. Same discipline as export_demo_assets.py
(demo_engine_spec.md SS5): every number written here traces to a committed
artefact, and the checkpoint is verified to reproduce its committed per-seed
held-out accuracy BEFORE anything is written.

  demo/model_exp68.json    per-arm trained weights (F1, FA, F2) for the
                           val-selected seed, plus the FROZEN feature formulas
                           baked in as constants and the per-noun angles the
                           page looks up (the page never recomputes PCA).
  demo/testset_exp68.json  the 300 held-out test items, compiled fields only,
                           with pair structure so twins can be shown together.

SEED SELECTION -- READ THIS.
`exp68_aggregate.py` performs NO per-arm seed selection: it reports 5-seed
means, and its "model_selection: best in-distribution val accuracy" refers to
the per-EPOCH checkpoint choice inside one run. A demo must run ONE model per
arm, so a selection rule is required and is stated here explicitly rather than
implied: argmax over the committed `val_per_seed`, ties broken by the LOWEST
seed index. Selection uses in-distribution val only and never peeks at
held-out. The rule is recorded verbatim in the exported JSON.
"""
import hashlib
import json
import os
import sys

import torch

import exp42_compiler as comp
import exp68_run as E
import instrument_sb_bounds as INS

RESULTS = "results_exp68.json"
DATA = "stories_exp68_WN.json"
DIAG = "diagnostic_sb_bounds.json"
RAW = "aravec_raw_vectors_44.json"
CKPT_DIR = "exp68_ckpt"
OUT = "demo"

ARMS = ["F1", "FA", "F2"]
SELECTION_RULE = ("argmax over results_exp68.json arms[<arm>].val_per_seed "
                  "(in-distribution validation accuracy of the selected "
                  "checkpoint); ties broken by the LOWEST seed index. "
                  "Held-out accuracy is never consulted. NOTE: "
                  "exp68_aggregate.py itself selects no seed -- it reports "
                  "5-seed means -- so this rule is stated here, not inherited.")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def select_seed(block):
    """(seed, index) of the val-selected run. Deterministic; see SELECTION_RULE."""
    pairs = list(zip(block["seeds"], block["val_per_seed"]))
    best = max(pairs, key=lambda sv: (sv[1], -sv[0]))
    idx = block["seeds"].index(best[0])
    return best[0], idx


def main():
    os.makedirs(OUT, exist_ok=True)

    res = json.load(open(RESULTS, encoding="utf-8"))
    vsuite = json.load(open(res["vsuite_results"], encoding="utf-8"))
    if not vsuite["gates"]["V1"]:
        raise RuntimeError("V-suite V1 gate is not green in %s"
                           % res["vsuite_results"])
    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("exp68 dataset provenance FAILED (self-hash)")
    if meta["dataset_sha256"] != res["dataset_sha256"]:
        raise RuntimeError("dataset sha != results_exp68.json dataset_sha256")
    verbs = sorted(meta["verb_inventory"].keys())

    # The frozen feature layer, through the committed instrument (never retyped).
    fm = E.feature_maps()
    diag = json.load(open(DIAG, encoding="utf-8"))
    divisors = [float(x) for x in diag["families"]["raw"]["norm_divisors"]]
    sign_pin = diag["families"]["raw"]["plane_agreement_with_frozen2d"][
        "sign_pin_applied"]

    test_items = [i for i in items if i["split"] == "test"]
    print("[export] %d held-out test items" % len(test_items))

    arms_out, selected = {}, {}
    for arm in ARMS:
        blk = res["arms"][arm]
        seed, idx = select_seed(blk)
        pt = os.path.join(CKPT_DIR, "%s_s%d.pt" % (arm, seed))
        sha = sha256_file(pt)
        if sha != blk["ckpt_sha256"][idx]:
            raise RuntimeError("%s: checkpoint sha mismatch %s vs committed %s"
                               % (pt, sha, blk["ckpt_sha256"][idx]))
        run = json.load(open(os.path.join(CKPT_DIR, "%s_s%d.json" % (arm, seed)),
                             encoding="utf-8"))
        if run["ckpt_sha256"] != sha or run["seed"] != seed:
            raise RuntimeError("%s: per-run JSON disagrees with checkpoint" % pt)

        # ---- reproduce the committed per-seed held-out accuracy BEFORE export
        m = E.build_model(arm, verbs, seed, fm)
        m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
        ang, _t1 = E.arm_angles(arm, fm)
        cs = comp.compile_all(test_items, ang)
        with torch.no_grad():
            acc = sum(int(torch.argmax(m.story_values(c)).item()) == c.answer
                      for c in cs) / len(cs)
        committed = blk["per_seed"][idx]
        if abs(acc - committed) > 1e-12:
            raise RuntimeError("%s s%d does not reproduce its committed held-out "
                               "accuracy: %.10f vs %.10f" % (arm, seed, acc, committed))
        print("[export] %-2s seed %d  val %.6f  held-out %.10f == committed  "
              "(ckpt %s)" % (arm, seed, blk["val_per_seed"][idx], acc, sha[:12]))

        wires_per_ref = 2 if arm == "F2" else 1
        arms_out[arm] = {
            "arm": arm,
            "model_class": ("exp68_run.F2Model (exp45_model.TwoQubitReferentModel "
                            "'A4' + the doc 54 SS6.4 second-RY override)"
                            if arm == "F2" else
                            "exp42_models.QuantumStoryModel 'A3'"),
            "qubits_per_referent": wires_per_ref,
            "n_qubits_per_block": 2 * wires_per_ref,
            "n_layers": E.LAYERS if hasattr(E, "LAYERS") else 3,
            "params_per_layer": 2 * (2 * wires_per_ref) + (2 * wires_per_ref - 1),
            "verbs": verbs,
            "blocks": {v: m.blocks[v].theta.detach().tolist() for v in verbs},
            "q_pos": m.q_pos.theta.detach().tolist(),
            "q_neg": m.q_neg.theta.detach().tolist(),
            "selected_seed": seed,
            "selected_seed_index": idx,
            "selection_rule": SELECTION_RULE,
            "checkpoint": pt,
            "checkpoint_sha256": sha,
            "selected_epoch": blk["selected_epoch_per_seed"][idx],
            "val_acc_selected_seed": blk["val_per_seed"][idx],
            "train_acc_selected_seed": blk["train_per_seed"][idx],
            "heldout_acc_selected_seed": committed,
            "heldout_acc_recomputed_at_export": acc,
            "n_params": blk["n_params"],
            "wires": blk["wires"],
            # committed 5-seed aggregates -- the HEADLINE numbers, distinct
            # from the single live seed above (Rule 6: label both).
            "committed_mean_5seed": blk["mean"],
            "committed_sd_5seed": blk["sd"],
            "committed_per_seed": blk["per_seed"],
            "floor_mean_5seed": res["floors"][arm]["mean"],
            "floor_sd_5seed": res["floors"][arm]["sd"],
            "floor_clearance": res["floor_clearance"][arm],
            "comparator_mean_5seed": res["comparator"]["arms"][arm]["mean"],
            "comparator_sd_5seed": res["comparator"]["arms"][arm]["sd"],
            "comparator_feature_dim": res["comparator"]["arms"][arm]["feature_dim"],
            "comparator_floor_mean_5seed": res["comparator"]["floors"][arm]["mean"],
            "bound_per_twin_doc54_SS12": E.PER_TWIN_BOUNDS[arm],
            "feature_formula": res["feature_maps"][arm],
        }
        selected[arm] = (seed, m, ang)

    # ---- the frozen per-noun angle table: LOOKED UP by the page, never recomputed
    nouns = {}
    for n in INS.HUMANS:
        p1, p2 = fm["coords"][n]
        nouns[n] = {
            "gloss": INS.GLOSS[n],
            "p1": float(p1), "p2": float(p2),
            "theta_pc1": float(fm["theta_pc1"][n]),     # F1 angle, F2 qubit 0
            "theta_pc2": float(fm["theta_pc2"][n]),     # F2 qubit 1
            "phi_atan2": float(fm["phi_exp42"][n]),     # FA angle
            "is_bearer": n in res["target_category"]["bearers"],
        }

    model = {
        "experiment": "exp68",
        "lever": res["lever"],
        "spec": res["spec"],
        "generated_by": "export_demo_exp68.py",
        "arms": arms_out,
        "arm_order": ARMS,
        "verbs": verbs,
        "nouns": nouns,
        "target_category": res["target_category"],
        "split_note": res["split"],
        "training": res["training"],
        "verdicts": res["verdicts"],
        "frozen_feature_constants": {
            "note": ("doc 54 SS6.2, FROZEN. Baked in as CONSTANTS: the page "
                     "looks these up and never recomputes a PCA."),
            "pca": ("PCA over all 44 referents of aravec_raw_vectors_44.json, "
                    "centred, PC signs pinned to the frozen exp42 vec2 plane"),
            "sign_pin_applied": list(sign_pin),
            "normalisation": ("p_k(r) = PC_k(r) / max_{r' in the 24 humans} "
                              "|PC_k(r')|  ->  [-1, 1]"),
            "norm_divisor_pc1": divisors[0],
            "norm_divisor_pc2": divisors[1],
            "scaling_name": fm["scaling"]["name"],
            "scaling_formula": fm["scaling"]["formula"],
            "theta_formula": "theta_k(r) = (p_k(r) + 1) * pi   ->  [0, 2pi)",
            "atan2_map": ("FA: phi(r) = atan2(PC2(r), PC1(r)) -- the existing "
                          "exp42 map, verified equal to the frozen embedding "
                          "angle at export"),
            "F1": res["feature_maps"]["F1"],
            "F2": res["feature_maps"]["F2"],
            "FA": res["feature_maps"]["FA"],
            "provenance": res["feature_maps"]["provenance"],
        },
        "validation": {
            "source": res["vsuite_results"],
            "source_sha256": sha256_file(res["vsuite_results"]),
            "v1_bag_of_embeddings_test_accuracy":
                vsuite["V1_bag_of_embeddings"]["test_accuracy"],
            "v1_twin_features_identical":
                vsuite["V1_bag_of_embeddings"]["twin_features_identical"],
            "v1_n_twin_pairs_checked":
                vsuite["V1_bag_of_embeddings"]["n_twin_pairs_checked"],
            "note": ("V1: any function of the cast alone is constant within a "
                     "swap-twin pair, so it scores exactly chance. Re-measured "
                     "for this dataset, not inherited."),
        },
        "inventory": {
            "n_humans": len(nouns),
            "n_referents_per_story": sorted({i["K"] for i in test_items})[0],
            "n_verbs": len(verbs),
            "embedding_dim": len(next(iter(json.load(
                open(RAW, encoding="utf-8"))["vectors"].values()))),
        },
        "provenance": {
            "results_json": RESULTS,
            "results_sha256": sha256_file(RESULTS),
            "dataset": DATA,
            "dataset_sha256": meta["dataset_sha256"],
            "raw_vectors": RAW,
            "raw_vectors_sha256": sha256_file(RAW),
            "frozen_embeddings": "exp42_noun_embeddings.json",
            "frozen_embeddings_sha256": fm["embeddings_sha256"],
            "bounds_diagnostic": DIAG,
            "bounds_diagnostic_sha256": sha256_file(DIAG),
            "comparator_json": res["comparator"]["source"],
            "design_doc": "qnlp_private_docs/54_exp68_width_lever.md",
            "note": ("exp68 SB rung: 6 verbs, 2 referents, 24-noun human "
                     "inventory, constructed stories, simulation only. The "
                     "classical comparator ties or exceeds every arm."),
        },
    }
    json.dump(model, open(os.path.join(OUT, "model_exp68.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- the test set: compiled fields for all three arms + pair structure
    by_pair = {}
    for it in test_items:
        by_pair.setdefault(it["pair_id"], []).append(it["id"])
    out_items = []
    for it in test_items:
        wires = {r["name"]: i for i, r in enumerate(it["referents"])}
        q = it["question"]
        out_items.append({
            "id": it["id"], "pair_id": it["pair_id"], "split": it["split"],
            "twin_ids": sorted(by_pair[it["pair_id"]]),
            "K": len(wires),
            "referents": [r["name"] for r in it["referents"]],
            "wires": wires,
            "intro": [[w, n] for n, w in sorted(wires.items(), key=lambda kv: kv[1])],
            "events": [[e["verb"], wires[e["agent"]], wires[e["patient"]]]
                       for e in it["events"]],
            "question": [[q["v"]], wires[q["a"]], wires[q["p"]]],
            "answer": comp.LABELS[it["answer"]],
            "answer_label": it["answer"],
            "sentences": it["sentences"],
            "question_surface": q["surface"],
            "gold_agent": it["events"][0]["agent"],
            "gold_patient": it["events"][0]["patient"],
            "bearer": next(n for n in [r["name"] for r in it["referents"]]
                           if n in model["target_category"]["bearers"]),
        })
    ts = {"experiment": "exp68", "dataset": DATA,
          "dataset_sha256": meta["dataset_sha256"],
          "split": "test (held-out unseen PAIRINGS -- doc 54 SS6.1.1)",
          "n_items": len(out_items), "n_pairs": len(by_pair),
          "compiled_by": ("export_demo_exp68.py, mirroring "
                          "exp42_compiler.compile_story; angles are looked up "
                          "per arm from model_exp68.json nouns[]"),
          "items": out_items}
    json.dump(ts, open(os.path.join(OUT, "testset_exp68.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)

    print("[export] wrote demo/model_exp68.json (%d arms, %d nouns, %d verbs)"
          % (len(arms_out), len(nouns), len(verbs)))
    print("[export] wrote demo/testset_exp68.json (%d items, %d pairs)"
          % (len(out_items), len(by_pair)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
