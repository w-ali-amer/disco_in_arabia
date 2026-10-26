# -*- coding: utf-8 -*-
"""exp94_gates.py -- doc 89 §6: G-V (data) and G-SIM (exact mixture == full statevector at K=2,3).

Usage: python exp94_gates.py --draw primary   -> exp94_gates/gv_<draw>.json ; --sim -> exp94_gates/gsim.json (Rule 3)
"""
import argparse, collections, json, os, torch
import exp94_common as C
import exp94_datagen as D
import exp94_models as M
import exp94_encode as E

SPLITS = ["train", "val", "test_id", "test_oodc", "test_oodk", "test_oodck", "train_full"]


def check(d):
    rep = {"twins": [], "balance": [], "no_fdu_train_val": True, "distinct": True, "oracles": {}}; ok = True
    for sp in SPLITS:
        L = d[sp]; pairs = collections.defaultdict(list)
        for st in L:
            pairs[st["pair"]].append(st)
            c = C.compile_story(st)                                   # round-trip
            if len(set(c["cands"][i][0] for i in range(len(c["cands"])))) != len(c["cands"]):
                rep["distinct"] = ok = False
            if len(set(st["meta"]["classes"])) != st["meta"]["K"]:
                rep["distinct"] = ok = False
            if sp in ("train", "val") and ("f-du" in st["meta"]["classes"] or st["meta"]["suffix"] == "f-du"):
                rep["no_fdu_train_val"] = ok = False
        for p, v in pairs.items():
            if len(v) != 2 or v[0]["label"] == v[1]["label"] or v[0]["intro"] != v[1]["intro"] or v[0]["q"] != v[1]["q"] \
                    or v[0]["event"][0] != v[1]["event"][0] or v[0]["event"][2] != v[1]["event"][2] \
                    or v[0]["event"][1] == v[1]["event"][1]:
                rep["twins"].append([sp, p]); ok = False; break
        if 2 * sum(st["label"] for st in L) != len(L):
            rep["balance"].append(sp); ok = False
    cnt = collections.defaultdict(lambda: [0, 0])
    for st in d["train"]:
        cnt[st["meta"]["suffix"]][0] += st["label"]; cnt[st["meta"]["suffix"]][1] += 1
    for sp in ("test_id", "test_oodc", "test_oodk", "test_oodck"):
        L = d[sp]
        acc = lambda f: sum(int(f(st)) == st["label"] for st in L) / len(L)
        rep["oracles"][sp] = {
            "suffix_alone": acc(lambda st: cnt[st["meta"]["suffix"]][1] and cnt[st["meta"]["suffix"]][0] / cnt[st["meta"]["suffix"]][1] > 0.5),
            "x_position": acc(lambda st: st["meta"]["x"] == 0),
            "first_listed": acc(lambda st: st["meta"]["x"] == 0),
            "last_listed": acc(lambda st: st["meta"]["x"] == st["meta"]["K"] - 1),
            "x_class_fsg": acc(lambda st: st["meta"]["classes"][st["meta"]["x"]] == "f-sg")}
        ok &= all(0.45 <= v <= 0.55 for v in rep["oracles"][sp].values())
    rep["pass"] = bool(ok)
    return rep


def gsim():
    d = D.build("smoke"); out = {}
    for K in (2, 3):
        enc = E.encode(d["test_oodc"] if K == 2 else d["test_id"], d["intro_angles"])[K][0]
        mx = 0.0
        for trial in range(50):
            m = M.QAgr(1000 + trial)
            b = E.sub(enc, torch.arange(trial % 20, trial % 20 + 3))
            with torch.no_grad():
                mx = max(mx, (m.scores(b) - M.full_statevector_scores(m, b)).abs().max().item())
        out["K%d_max_abs_diff" % K] = mx
    out["pass"] = all(v < 1e-5 for k, v in out.items() if k.endswith("diff"))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draw"); ap.add_argument("--sim", action="store_true")
    a = ap.parse_args()
    if a.sim:
        r = gsim(); C.write_new_json("exp94_gates/gsim.json", {"doc": "89 §6 G-SIM", "result": r, "provenance": C.provenance()})
        print(r); return
    d = json.load(open("stories_exp94_%s.json" % a.draw, encoding="utf-8"))
    rep = check(d)
    bad = D.build(a.draw, corrupt=True)
    fl = sum(int(st["meta"]["x"] == 0) == st["label"] for st in bad["test_id"]) / len(bad["test_id"])
    out = {"doc": "89 §6 G-V", "draw": a.draw, "gate": rep, "designed_to_fail": {"first_listed_on_corrupt": fl,
           "failed_as_designed": fl > 0.55}, "provenance": C.provenance()}
    out["G_V_PASS"] = bool(rep["pass"] and fl > 0.55)
    C.write_new_json("exp94_gates/gv_%s.json" % a.draw, out)
    print(json.dumps({"G_V_PASS": out["G_V_PASS"], "twins": rep["twins"][:2], "balance": rep["balance"],
                      "no_fdu": rep["no_fdu_train_val"], "distinct": rep["distinct"], "oracles": rep["oracles"], "dtf": fl},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
