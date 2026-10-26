# -*- coding: utf-8 -*-
"""exp92_run.py -- doc 87 §4/§5: train one unit, or a grid shard.

Unit:  python exp92_run.py --draw primary --arm Q-COMP --seed 0 [--n-i 4]
Grid:  python exp92_run.py --grid --worker w --nworkers N [--draws primary,confirm]
Recipe: Adam lr 0.02, batch 8, 80 epochs, BCE, best-val checkpoint; val from training cells only.
Held-out derived cells are TEST ONLY; a held-out root keeps its Form I cell (first n_I items if --n-i).
Output: exp92_ckpt/<draw>_<arm>[_nI<n>]_s<seed>.{json,pt}  (Rule 3; os.makedirs).
"""
import argparse, copy, json, os, random, sys, time
import torch
import exp92_common as C
import exp92_models as M

EPOCHS, LR, BATCH = 80, 0.02, 8
CKPT = "exp92_ckpt"
ARMS = list(M.ARMS)
LOWSHOT = {"arms": ["Q-COMP", "CL-MLP-FACT"], "n_i": [4, 12], "draw": "primary"}


def load(draw):
    return json.load(open("stories_exp92_%s.json" % draw, encoding="utf-8"))


def encode(items, roots, vtab):
    pidx = {r["pair"]: i for i, r in enumerate(roots)}
    nidx = {n: i for i, n in enumerate(C.NOUNS)}
    fidx = {f: i for i, f in enumerate(M.FORMS)}
    P, F, N, Q, S, Y = [], [], [], [], [], []
    for it in items:
        pair, form, nouns = C.compile_surface(it["tokens"], vtab, nidx)       # SURFACE only
        qn = nidx[C.strip_marks(it["question"][1])]
        qpos = nouns.index(qn)
        assert qpos == it["q_pos"] and pair == it["pair"] and form == it["form"]
        P.append(pidx[pair]); F.append(fidx[form]); N.append(nouns); Q.append(qpos)
        S.append(pidx[it["q_state"]]); Y.append(it["label"])
    t = torch.tensor
    return [t(P), t(F), t(N), t(Q), t(S), t(Y)]


def splits(d, seed, n_i=None):
    held = set(d["held_out"][str(seed)])
    it = d["items"]
    def keep_trainval(x):
        return x["form"] == "I" or x["pair"] not in held
    train = [x for x in it if x["split"] == "train" and keep_trainval(x)]
    if n_i is not None:
        cut = {}
        for x in sorted(train, key=lambda z: z["id"]):
            if x["pair"] in held and x["form"] == "I":
                cut.setdefault(x["pair"], []).append(x)
        drop = {x["id"] for v in cut.values() for x in v[n_i:]}
        train = [x for x in train if x["id"] not in drop]
    val = [x for x in it if x["split"] == "val" and keep_trainval(x)]
    test_h = [x for x in it if x["split"] == "test_heldout" and x["pair"] in held]
    test_s = [x for x in it if x["split"] == "test_seen" and keep_trainval(x)]
    return train, val, test_h, test_s, sorted(held)


def evaluate(model, enc):
    model.eval()
    with torch.no_grad():
        if model.kind == "quantum":
            p = model.prob_yes(*enc[:5]); pred = (p > 0.5).long(); score = p
        else:
            z = model.logit(*enc[:5]); pred = (z > 0).long(); score = z
    model.train()
    return (pred == enc[5]).float().mean().item(), pred.tolist(), score.tolist()


def run_unit(draw, arm, seed, n_i=None, outdir=CKPT):
    os.makedirs(outdir, exist_ok=True)
    tag = "%s_%s%s_s%d" % (draw, arm, "_nI%d" % n_i if n_i else "", seed)
    jpath, ppath = os.path.join(outdir, tag + ".json"), os.path.join(outdir, tag + ".pt")
    if os.path.exists(jpath):
        print("[skip] exists", tag, flush=True)
        return
    torch.set_num_threads(1)
    d = load(draw)
    roots = d["roots"]
    vtab = C.verb_table(roots)
    tr, va, th, ts, held = splits(d, seed, n_i)
    E = {k: encode(v, roots, vtab) for k, v in (("train", tr), ("val", va), ("test_heldout", th), ("test_seen", ts))}
    model = M.ARMS[arm](len(roots), len(C.NOUNS), len(roots), seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = random.Random(92900 + seed)
    idx = list(range(len(tr)))
    best = {"val": -1.0, "epoch": -1, "state": None}
    hist = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        rng.shuffle(idx)
        tot = 0.0
        for b in range(0, len(idx), BATCH):
            j = torch.tensor(idx[b:b + BATCH])
            batch = [x[j] for x in E["train"]]
            loss, _ = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss %s epoch %d" % (tag, ep))
            opt.zero_grad(); loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("non-finite grad %s epoch %d" % (tag, ep))
            opt.step(); tot += loss.item() * len(j)
        va_acc = evaluate(model, E["val"])[0]
        hist.append({"epoch": ep, "loss": tot / len(idx), "val": va_acc})
        if va_acc > best["val"]:
            best = {"val": va_acc, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
        if ep % 10 == 0:
            print("[%s] ep %d loss %.4f val %.3f best %.3f@%d %.0fs" % (tag, ep, tot / len(idx), va_acc,
                  best["val"], best["epoch"], time.time() - t0), flush=True)
    model.load_state_dict(best["state"])
    res = {}
    for k in ("train", "val", "test_heldout", "test_seen"):
        acc, pred, score = evaluate(model, E[k])
        res[k] = {"acc": acc, "n": len(pred)}
        if k.startswith("test"):
            items = {"train": tr, "val": va, "test_heldout": th, "test_seen": ts}[k]
            res[k]["per_item"] = [{"id": x["id"], "alt": x["alt"], "form": x["form"], "kind": x["kind"],
                                   "label": x["label"], "pred": p, "score": s} for x, p, s in zip(items, pred, score)]
    formI = [i for i, x in enumerate(tr) if x["form"] == "I"]
    jI = torch.tensor(formI)
    res["train_formI_acc"] = evaluate(model, [x[jI] for x in E["train"]])[0]
    for alt, form in (("ANTI", "VII"), ("CAUS", "IV")):
        rows = [r for r in res["test_heldout"]["per_item"] if r["alt"] == alt]
        res["heldout_" + alt] = sum(r["pred"] == r["label"] for r in rows) / len(rows)
    extra = {}
    if arm in ("Q-COMP", "OP-ORACLE", "Q-COMP-SG"):
        pidx = {r["pair"]: i for i, r in enumerate(roots)}
        for fi, fname, alt in ((1, "VII", "ANTI"), (2, "IV", "CAUS")):
            ids_h = [pidx[p] for p in held if dict((r["pair"], r["alt"]) for r in roots)[p] == alt]
            ids_s = [pidx[r["pair"]] for r in roots if r["alt"] == alt and r["pair"] not in held]
            extra["routing_fidelity_%s_heldout_roots" % fname] = M.routing_fidelity(model, ids_h, fi)
            extra["routing_fidelity_%s_seen_roots" % fname] = M.routing_fidelity(model, ids_s, fi)
        with torch.no_grad():
            for fi, fname in ((1, "VII"), (2, "IV")):
                Fm = model.F_mat(fi)
                extra["F_%s_swap_fidelity" % fname] = abs(torch.trace(Fm.conj().T @ M.SWAP).item()) / 4
                extra["F_%s_matrix_re" % fname] = Fm.real.tolist()
                extra["F_%s_matrix_im" % fname] = Fm.imag.tolist()
    out = {"doc": "87 (59f6621)", "draw": draw, "arm": arm, "seed": seed, "n_i": n_i, "held_out": held,
           "best_epoch": best["epoch"], "best_val": best["val"], "history": hist, "results": res,
           "extra": extra, "wall_sec": time.time() - t0, "torch": torch.__version__,
           "sizes": {k: len(v) for k, v in (("train", tr), ("val", va), ("test_heldout", th), ("test_seen", ts))},
           "provenance": C.provenance()}
    torch.save(best["state"], ppath)
    C.write_new_json(jpath, out)
    print("[done] %s heldout ANTI %.3f CAUS %.3f formI %.3f %.0fs" % (tag, res["heldout_ANTI"], res["heldout_CAUS"],
          res["train_formI_acc"], time.time() - t0), flush=True)


def grid_units(draws):
    units = [(dr, a, s, None) for dr in draws for a in ARMS for s in range(8)]
    if LOWSHOT["draw"] in draws:
        units += [(LOWSHOT["draw"], a, s, n) for a in LOWSHOT["arms"] for n in LOWSHOT["n_i"] for s in range(8)]
    # quantum units first (slow), interleaved across workers
    return sorted(units, key=lambda u: (not u[1].startswith(("Q", "OP")), u[0], u[1], u[3] or 0, u[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw"); ap.add_argument("--arm", choices=ARMS); ap.add_argument("--seed", type=int)
    ap.add_argument("--n-i", type=int, default=None)
    ap.add_argument("--grid", action="store_true"); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1); ap.add_argument("--draws", default="primary,confirm")
    ap.add_argument("--outdir", default=CKPT)
    a = ap.parse_args()
    if a.grid:
        units = grid_units(a.draws.split(","))
        for k, u in enumerate(units):
            if k % a.nworkers == a.worker:
                run_unit(*u, outdir=a.outdir)
    else:
        run_unit(a.draw, a.arm, a.seed, a.n_i, outdir=a.outdir)


if __name__ == "__main__":
    main()
