# -*- coding: utf-8 -*-
"""exp98_run.py -- doc 93: units and grid (derived from the exp97 runner; per-event register under LEARNED control + TF/LSTM grids).

Arms: Q-AGR, CL-AGR, Q-BLIND, Q-GOLD (registered circuit recipe: Adam lr 0.01, batch 8, 40 epochs, best ID-val)
      TF_d{32,64,128}_lr{1e-3,3e-3}, LSTM_h{..}_lr{..} (AdamW wd 0.01, batch 32, 60 epochs, best ID-val)
      TF-FULL: per seed, the ID-val-selected TF config retrained on train_full (all 6 classes; expressivity oracle).
Selection uses VAL only; OOD sets never reach selection (doc 89 §5).
Output: exp98_ckpt/<draw>_<arm>_s<seed>.json (+ .pt for circuit arms). Rule 3; os.makedirs.
"""
import argparse, copy, glob, json, os, random, time
import torch
import exp94_common as C
import exp95_models as M
import exp98_models as M96
import exp97_models as M97G
import hashlib
import exp95_data as E

CK = "exp98_ckpt"
CIRCUIT = M96.ARMS
DATA_SHA = json.load(open("exp98_gates/data_sha.json"))["sha256"] if os.path.exists("exp98_gates/data_sha.json") else {}
GRID = {"TF": [(d, lr) for d in (32, 64, 128) for lr in (1e-3, 3e-3)],
        "LSTM": [(d, lr) for d in (32, 64, 128) for lr in (1e-3, 3e-3)]}
TESTS = ["test_id", "test_oodl", "test_long0"]


def load(draw):
    p = "stories_exp95_%s.json" % draw
    if draw in DATA_SHA:
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == DATA_SHA[draw], "G-DATA: exp95 draw changed"
    return json.load(open(p, encoding="utf-8"))


def batches(enc, bs, rng):
    out = []
    for K, (b, idx) in enc.items():
        order = list(range(len(idx))); rng.shuffle(order)
        out += [(K, torch.tensor(order[i:i + bs])) for i in range(0, len(order), bs)]
    rng.shuffle(out)
    return out


def evaluate(model, enc, n):
    model.eval()
    pred, score = [None] * n, [None] * n
    with torch.no_grad():
        for K, (b, idx) in enc.items():
            for i in range(0, len(idx), 256):
                j = torch.arange(i, min(i + 256, len(idx)))
                p, s = model.predict(E.sub(b, j))
                for k, pp, ss in zip(j.tolist(), p.tolist(), s.tolist()):
                    pred[idx[k]], score[idx[k]] = pp, ss
    model.train()
    return pred, score


def train(model, d, train_key, seed, lr, epochs, bs, opt_name):
    enc_tr = E.encode(d[train_key], d["intro_angles"]); enc_va = E.encode(d["val"], d["intro_angles"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01) if opt_name == "adamw" else torch.optim.Adam(params, lr=lr)
    rng = random.Random(98900 + seed)
    best = {"val": -1, "val_loss": float("inf"), "epoch": -1, "state": None}; hist = []
    yv = [st["label"] for st in d["val"]]
    for ep in range(1, epochs + 1):
        tot = 0.0; nb = 0
        for K, j in batches(enc_tr, bs, rng):
            loss = model.loss(E.sub(enc_tr[K][0], j))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss ep %d" % ep)
            opt.zero_grad(); loss.backward()
            for p in params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("non-finite grad ep %d" % ep)
            opt.step(); tot += loss.item(); nb += 1
        pv, _ = evaluate(model, enc_va, len(d["val"]))
        va = sum(int(p == y) for p, y in zip(pv, yv)) / len(yv)
        vl = val_loss(model, enc_va)
        hist.append({"epoch": ep, "loss": tot / nb, "val": va, "val_loss": vl})
        if vl < best["val_loss"]:          # doc 89 §12: checkpoint by MIN val loss (val acc saturates at 1.0)
            best = {"val": va, "val_loss": vl, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    return best, hist


def val_loss(model, enc):
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for K, (b, idx) in enc.items():
            l = model.loss(b); tot += l.item() * len(idx); n += len(idx)
    model.train()
    return tot / n


def results(model, d):
    res = {}
    for k in TESTS + ["train"]:
        pr, sc = evaluate(model, E.encode(d[k], d["intro_angles"]), len(d[k]))
        rows = [{"id": st["id"], "pair": st["pair"], "label": st["label"], "pred": p, "score": s,
                 "T": st["meta"]["T"], "L": st["meta"]["L"], "B": st["meta"]["B"], "q": st["meta"]["q"]}
                for st, p, s in zip(d[k], pr, sc)]
        acc = sum(r["pred"] == r["label"] for r in rows) / len(rows)
        pairs = {}
        for r in rows:
            pairs.setdefault(r["pair"], []).append(r)
        res[k] = {"acc": acc, "n": len(rows),
                  "twin_pred_mismatch": sum(1 for v in pairs.values() if len(v) == 2 and v[0]["pred"] != v[1]["pred"]),
                  "twin_max_score_diff": max(abs(v[0]["score"] - v[1]["score"]) for v in pairs.values() if len(v) == 2)}
        if k != "train":
            res[k]["per_item"] = rows
    return res


def run_unit(draw, arm, seed, data=None, outdir=CK):
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, "%s_%s_s%d.json" % (draw, arm, seed))
    if os.path.exists(jpath):
        print("[skip]", jpath, flush=True); return
    torch.set_num_threads(1)
    d = data or load(draw)
    t0 = time.time()
    if arm in CIRCUIT:
        model = CIRCUIT[arm](seed)
        best, hist = train(model, d, "train", seed, 0.01, 40, 8, "adam")
        cfg = {"lr": 0.01, "epochs": 40, "batch": 8, "opt": "adam"}
    else:
        fam = arm.split("_")[0]
        if fam == "TF-FULL":
            sel = select(draw, "TF", seed, outdir)
            raise ValueError("no TF-FULL in exp95")
        else:
            dd = int(arm.split("_")[1][1:]); lr = float(arm.split("_lr")[1]); train_key = "train"
            mk = M.TFChain if fam == "TF" else M.LSTMChain
        model = mk(seed, dd)
        best, hist = train(model, d, train_key, seed, lr, 60, 32, "adamw")
        cfg = {"family": fam, "d": dd, "lr": lr, "epochs": 60, "batch": 32, "opt": "adamw", "train_key": train_key}
    res = results(model, d)
    extra = {}
    if getattr(model, "field", None) == "R":
        extra["G_REAL_max_imag"] = max(float(m.matrix().imag.detach().abs().max()) for m in model.modules()
                                       if isinstance(m, M97G.Gate))
    extra["train_acc"] = res["train"]["acc"]
    if getattr(model, "morph", None) is not None:
        extra["morph_table"] = model.morph.table()
    out = {"doc": "93 (23d4dfb)", "draw": draw, "arm": arm, "seed": seed, "cfg": cfg, "best_epoch": best["epoch"],
           "best_val": best["val"], "best_val_loss": best["val_loss"], "history": hist, "results": res, "extra": extra,
           "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "wall_sec": time.time() - t0, "torch": torch.__version__, "provenance": C.provenance()}
    if arm in CIRCUIT:
        torch.save(model.state_dict(), jpath[:-5] + ".pt")
    C.write_new_json(jpath, out)
    print("[done] %s id %.3f oodl %.3f long0 %.3f val %.3f %.0fs" % (os.path.basename(jpath),
          res["test_id"]["acc"], res["test_oodl"]["acc"], res["test_long0"]["acc"], best["val"], time.time() - t0), flush=True)


def select(draw, fam, seed, outdir=CK):
    """Practitioner's pick (doc 89 §12; descriptive + TF-FULL config): min ID val loss, then smallest d, then smallest lr.
    Never an OOD set."""
    cands = []
    for dd, lr in GRID[fam]:
        p = os.path.join(outdir, "%s_%s_d%d_lr%g_s%d.json" % (draw, fam, dd, lr, seed))
        u = json.load(open(p, encoding="utf-8"))
        cands.append((u["best_val_loss"], dd, lr, u["arm"], u["best_val"]))
    vl, dd, lr, arm, va = min(cands)
    return {"val_loss": vl, "val": va, "d": dd, "lr": lr, "arm": arm}


def units(draws):
    us = []
    for dr in draws:
        for s in range(8):
            us += [(dr, a, s) for a in CIRCUIT]
            us += [(dr, "%s_d%d_lr%g" % (f, dd, lr), s) for f in GRID for dd, lr in GRID[f]]
    return us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw"); ap.add_argument("--arm"); ap.add_argument("--seed", type=int)
    ap.add_argument("--grid", action="store_true"); ap.add_argument("--full-stage", action="store_true")
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--draws", default="e98p,e98c"); ap.add_argument("--outdir", default=CK)
    a = ap.parse_args()
    cache = {}
    if a.grid or a.full_stage:
        us = units(a.draws.split(",")) if a.grid else [(dr, "TF-FULL", s) for dr in a.draws.split(",") for s in range(8)]
        us = sorted(us, key=lambda u: (not (u[1] in CIRCUIT and u[1] != "CL-AGR"), u[0], u[1], u[2]))
        for k, u in enumerate(us):
            if k % a.nworkers == a.worker:
                if u[0] not in cache:
                    cache[u[0]] = load(u[0])
                run_unit(*u, data=cache[u[0]], outdir=a.outdir)
    else:
        run_unit(a.draw, a.arm, a.seed, outdir=a.outdir)


if __name__ == "__main__":
    main()
