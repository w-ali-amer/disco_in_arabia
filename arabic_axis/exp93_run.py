# -*- coding: utf-8 -*-
"""exp93_run.py -- doc 88 §4/§5: train one unit or a grid shard.

Unit: python exp93_run.py --draw primary --arm Q-AGR --seed 0
Grid: python exp93_run.py --grid --worker w --nworkers N [--draws primary,confirm]
Recipe: Adam lr 0.01, batch 8, 40 epochs, best-val checkpoint (val = training stems).
Twin sets evaluated in fixed order, both twins adjacent in the same batch (G-FLOOR, doc 88 §6).
Output: exp93_ckpt/<draw>_<arm>_s<seed>.{json,pt} (Rule 3; os.makedirs).
"""
import argparse, copy, json, os, random, time
import torch
import exp93_common as C
import exp93_models as M

EPOCHS, LR, BATCH, EVAL_BATCH = 40, 0.01, 8, 200
CKPT = "exp93_ckpt"
ARMS = ["Q-AGR", "Q-BLIND", "Q-GOLD", "SURF-MLP"]
V1_IDS, V2_IDS = sorted(C.V1), sorted(C.V2)


def load(draw):
    return json.load(open("stories_exp93_%s.json" % draw, encoding="utf-8"))


def encode(stories, angles):
    cols = {k: [] for k in ("v1", "v2", "vq", "theta0", "theta1", "theta_o", "fem0", "fem1", "suffix", "x_wire",
                            "gold", "label", "stem0", "stem1", "obj")}
    for st in stories:
        c = C.compile_story(st)                                  # SURFACE only
        cols["v1"].append(V1_IDS.index(c["v1"])); cols["v2"].append(V2_IDS.index(c["v2"]))
        cols["vq"].append(V2_IDS.index(c["vq"]))
        cols["theta0"].append(angles[c["stem0"]]); cols["theta1"].append(angles[c["stem1"]])
        cols["theta_o"].append(angles[c["obj"]])
        cols["fem0"].append(c["fem0"]); cols["fem1"].append(c["fem1"]); cols["suffix"].append(c["suffix_fem"])
        cols["x_wire"].append(c["x_wire"]); cols["gold"].append(st["gold_agent_wire"]); cols["label"].append(st["label"])
        cols["stem0"].append(C.STEMS.index(c["stem0"])); cols["stem1"].append(C.STEMS.index(c["stem1"]))
        cols["obj"].append(C.OBJECTS.index(c["obj"]))
    return {k: torch.tensor(v, dtype=torch.float32 if k.startswith("theta") else torch.long) for k, v in cols.items()}


def sub(b, j):
    return {k: v[j] for k, v in b.items()}


def make_model(arm, seed):
    if arm == "SURF-MLP":
        return M.SurfMLP(len(V1_IDS), len(V2_IDS), len(C.STEMS), len(C.OBJECTS), seed)
    return M.QStory(V1_IDS, V2_IDS, seed, arm.split("-")[1])


def evaluate(model, enc, stories=None):
    model.eval()
    preds, scores = [], []
    with torch.no_grad():
        n = enc["label"].shape[0]
        for i in range(0, n, EVAL_BATCH):                        # twins adjacent (even/odd), same batch
            p, s = model.predict(sub(enc, torch.arange(i, min(i + EVAL_BATCH, n))))
            preds += p.tolist(); scores += s.tolist()
    model.train()
    acc = sum(int(p == y) for p, y in zip(preds, enc["label"].tolist())) / len(preds)
    return acc, preds, scores


def run_unit(draw, arm, seed, outdir=CKPT, data=None):
    os.makedirs(outdir, exist_ok=True)
    tag = "%s_%s_s%d" % (draw, arm, seed)
    jpath, ppath = os.path.join(outdir, tag + ".json"), os.path.join(outdir, tag + ".pt")
    if os.path.exists(jpath):
        print("[skip]", tag, flush=True); return
    torch.set_num_threads(1)
    d = data or load(draw)
    sd = d["seeds"][str(seed)]
    E = {k: encode(sd[k], d["intro_angles"]) for k in ("train", "val", "test_heldout", "test_seen")}
    model = make_model(arm, seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = random.Random(93900 + seed)
    idx = list(range(E["train"]["label"].shape[0]))
    best = {"val": -1.0, "epoch": -1, "state": None}; hist = []; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        rng.shuffle(idx); tot = 0.0
        for i in range(0, len(idx), BATCH):
            j = torch.tensor(idx[i:i + BATCH])
            loss, _ = model.loss(sub(E["train"], j))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss %s ep %d" % (tag, ep))
            opt.zero_grad(); loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("non-finite grad %s ep %d" % (tag, ep))
            opt.step(); tot += loss.item() * len(j)
        va = evaluate(model, E["val"])[0]
        hist.append({"epoch": ep, "loss": tot / len(idx), "val": va})
        if va > best["val"]:
            best = {"val": va, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
        if ep % 5 == 0:
            print("[%s] ep %d loss %.4f val %.3f best %.3f@%d %.0fs" % (tag, ep, tot / len(idx), va, best["val"],
                  best["epoch"], time.time() - t0), flush=True)
    model.load_state_dict(best["state"])
    res = {}
    for k in ("train", "val", "test_heldout", "test_seen"):
        acc, pred, score = evaluate(model, E[k])
        res[k] = {"acc": acc, "n": len(pred)}
        if k.startswith("test"):
            res[k]["per_item"] = [{"id": st["id"], "pair": st["pair"], "label": st["label"], "pred": p, "score": s}
                                  for st, p, s in zip(sd[k], pred, score)]
            pairs = {}
            for r in res[k]["per_item"]:
                pairs.setdefault(r["pair"], []).append(r)
            res[k]["twin_pred_mismatch"] = sum(1 for v in pairs.values() if v[0]["pred"] != v[1]["pred"])
            res[k]["twin_max_score_diff"] = max(abs(v[0]["score"] - v[1]["score"]) for v in pairs.values())
    extra = {}
    if arm == "Q-AGR":
        extra["match_table"] = model.match_table()
    if arm.startswith("Q"):
        extra["sigma"] = model.sigma.detach().tolist(); extra["gamma"] = model.gamma.detach().tolist()
    out = {"doc": "88 (c1caeb4)", "draw": draw, "arm": arm, "seed": seed, "held_out": sd["held_out"],
           "best_epoch": best["epoch"], "best_val": best["val"], "history": hist, "results": res, "extra": extra,
           "wall_sec": time.time() - t0, "torch": torch.__version__, "provenance": C.provenance()}
    torch.save(best["state"], ppath)
    C.write_new_json(jpath, out)
    print("[done] %s heldout %.3f seen %.3f train %.3f mismatch %d %.0fs" % (tag, res["test_heldout"]["acc"],
          res["test_seen"]["acc"], res["train"]["acc"], res["test_heldout"]["twin_pred_mismatch"], time.time() - t0),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw"); ap.add_argument("--arm", choices=ARMS); ap.add_argument("--seed", type=int)
    ap.add_argument("--grid", action="store_true"); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1); ap.add_argument("--draws", default="primary,confirm")
    ap.add_argument("--outdir", default=CKPT)
    a = ap.parse_args()
    if a.grid:
        units = sorted([(dr, ar, s) for dr in a.draws.split(",") for ar in ARMS for s in range(8)],
                       key=lambda u: (u[1] == "SURF-MLP", u[0], u[1], u[2]))
        cache = {}
        for k, u in enumerate(units):
            if k % a.nworkers == a.worker:
                if u[0] not in cache:
                    cache[u[0]] = load(u[0])
                run_unit(*u, outdir=a.outdir, data=cache[u[0]])
    else:
        run_unit(a.draw, a.arm, a.seed, outdir=a.outdir)


if __name__ == "__main__":
    main()
