# -*- coding: utf-8 -*-
"""export_demo_assets.py -- emit the browser demo's static assets.

demo_engine_spec.md SS5. Phase 1 rests entirely on exp51's committed A3
checkpoint; no exp55-59 artifact is required. Every number this writes traces
to results_exp51.json.

  demo/model.json      120 trained floats + verbs + provenance hashes
  demo/vocab_core.json the 44 frozen nouns -> angle (Tier 1)
  demo/testset.json    the 300 VT test items, compiled fields only
"""
import hashlib, json, os, sys
import torch

import exp42_compiler as comp
import exp41_story_generator as gen
from exp42_models import QuantumStoryModel

CKPT = "exp51_ckpt/VT_A3_lr0.02_b8_s0.pt"     # best val-selected A3 (results_exp51.json)
RESULTS = "results_exp51.json"
DATA = "stories_exp51_VT.json"
EMB = "exp42_noun_embeddings.json"
OUT = "demo"
COMMITTED_TEST_ACC = 0.7266666666666667        # results_exp51.json per_arm A3


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    verbs = sorted(meta["verb_inventory"].keys())
    base, emb_meta, emb_sha = comp.load_embeddings()

    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = QuantumStoryModel("A3", verbs, 0)
    m.load_state_dict(sd)

    # ---- verify against the committed number BEFORE exporting anything
    test = [i for i in items if i["split"] == "test"]
    cs = comp.compile_all(test, base)
    with torch.no_grad():
        acc = sum(int(torch.argmax(m.story_values(c)).item()) == c.answer
                  for c in cs) / len(cs)
    if abs(acc - COMMITTED_TEST_ACC) > 1e-12:
        raise RuntimeError("checkpoint does not reproduce the committed "
                           "accuracy: %.10f vs %.10f" % (acc, COMMITTED_TEST_ACC))
    print("[export] checkpoint reproduces committed test acc %.10f" % acc)

    model = {
        "arm": "A3", "n_qubits_per_block": 2, "n_layers": 3,
        "verbs": verbs,
        "blocks": {v: m.blocks[v].theta.detach().tolist() for v in verbs},
        "q_pos": m.q_pos.theta.detach().tolist(),
        "q_neg": m.q_neg.theta.detach().tolist(),
        "labels": {"yes": 0, "no": 1},
        "provenance": {
            "checkpoint": CKPT, "checkpoint_sha256": sha256_file(CKPT),
            "results_json": RESULTS, "results_sha256": sha256_file(RESULTS),
            "dataset_sha256": meta["dataset_sha256"],
            "embeddings_sha256": emb_sha,
            "committed_test_acc": COMMITTED_TEST_ACC,
            "design_doc": "qnlp_private_docs/35_exp51_verb_twin_rung.md",
            "note": ("trained on the VT verb-twin rung; 6 verbs, 4 referent "
                     "wires, synthetic corpus, simulation only"),
        },
    }
    json.dump(model, open(os.path.join(OUT, "model.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)

    vocab = {"tier": 1,
             "source": emb_meta.get("source", ""),
             "intro_convention": emb_meta.get("intro_convention", ""),
             "embeddings_sha256": emb_sha,
             "humans": sorted(gen.HUMANS),
             "angles": {n: base[n] for n in sorted(base)}}
    json.dump(vocab, open(os.path.join(OUT, "vocab_core.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)

    ts = []
    for it, c in zip(test, cs):
        ts.append({"id": c.item_id, "pair_id": c.pair_id, "K": c.K,
                   "intro": [[w, n, a] for w, n, a in c.intro],
                   "events": [[v, wa, wp] for v, wa, wp in c.events],
                   "question": [list(c.question[0]), c.question[1],
                                c.question[2]],
                   "answer": c.answer,
                   "sentences": it["sentences"],
                   "question_surface": it["question"]["surface"]})
    json.dump({"n": len(ts), "dataset_sha256": meta["dataset_sha256"],
               "items": ts},
              open(os.path.join(OUT, "testset.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    print("[export] wrote %s/model.json (%d verbs, %d floats), "
          "vocab_core.json (%d nouns), testset.json (%d items)"
          % (OUT, len(verbs), sum(p.numel() for p in m.parameters()),
             len(base), len(ts)))
    if os.path.exists("results_exp59.json"):
        export_spread_models()
    else:
        print("[export] results_exp59.json absent -- Phase 2 assets skipped")
    return 0




# ---------------------------------------------------------------- Phase 2
def export_spread_models():
    """demo_engine_spec.md SS13: the trained-at-spread series from exp59.

    Every condition ships the weights of ONE seed -- the seed whose test_acc is
    closest to that condition's 5-seed mean -- alongside the committed mean, sd
    and gate rate. The page must label the live number as that single seed's,
    never as the 5-seed mean (honesty trap T1's sibling)."""
    import statistics
    R = json.load(open("results_exp59.json", encoding="utf-8"))
    if not R["verdicts"]["anchors_reproduce"]:
        raise RuntimeError("exp59 anchors did not reproduce -- weights are NOT "
                           "licensed for the demo (doc 45 SS4)")
    blob = json.load(open(DATA, encoding="utf-8"))
    verbs = sorted(blob["meta"]["verb_inventory"].keys())
    conds = []
    for row in R["curve"]:
        c = row["condition"]
        rs = R["per_condition"][c]
        mean = row["mean_test"]
        rep = min(rs, key=lambda r: abs(r["test_acc"] - mean))
        sd = torch.load(rep["ckpt_path"], map_location="cpu", weights_only=True)
        # A-scale's CondModel carries an extra `scale` parameter. At inference
        # it only rescales the intro angles, and we ship the ALREADY-compressed
        # post-training angle map, so dropping it is exact, not an approximation.
        sd = {k: v for k, v in sd.items() if k != "scale"}
        m = QuantumStoryModel("A3", verbs, 0)
        m.load_state_dict(sd)
        angles = dict(rep["drawn_angles"])
        if rep.get("post_angles"):
            angles = dict(rep["post_angles"])        # A-scale: learned map
        conds.append({
            "id": c, "spread": row["spread"],
            "learnable_scale": row["learnable_scale"],
            "mean_test": mean, "sd_test": row["sd_test"],
            "gate_met": row["gate_met"], "n_seeds": len(rs),
            "seed_shipped": rep["seed"], "seed_test_acc": rep["test_acc"],
            "learned_scale": rep.get("learned_scale"),
            "half_width": rep["half_width"],
            "angles": angles,
            "blocks": {v: m.blocks[v].theta.detach().tolist() for v in verbs},
            "q_pos": m.q_pos.theta.detach().tolist(),
            "q_neg": m.q_neg.theta.detach().tolist(),
            "ckpt_sha256": rep["ckpt_sha256"],
        })
    out = {"source": "results_exp59.json", "design_doc": R["spec"],
           "verdicts": R["verdicts"], "conditions": conds}
    json.dump(out, open(os.path.join(OUT, "spread_models.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)
    print("[export] wrote %s/spread_models.json (%d trained-at-spread models)"
          % (OUT, len(conds)))


if __name__ == "__main__":
    sys.exit(main())
