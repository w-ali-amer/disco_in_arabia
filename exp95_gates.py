# -*- coding: utf-8 -*-
"""exp95_gates.py -- doc 90 §5: G-V and G-SIM.  --draw D -> exp95_gates/gv_D.json ; --sim -> exp95_gates/gsim.json"""
import argparse, collections, json, torch
import exp94_common as C
import exp95_data as D
import exp95_models as M

SPL = ["train", "val", "test_id", "test_oodl", "test_long0"]


def check(d):
    rep = {"twins": [], "L_exact": True, "no_L2_trainval": True, "balance": [], "oracles": {}}; ok = True
    want = {"train": {0, 1}, "val": {0, 1}, "test_id": {0, 1}, "test_oodl": {2, 3}, "test_long0": {0}}
    for sp in SPL:
        pairs = collections.defaultdict(list)
        for st in d[sp]:
            pairs[st["pair"]].append(st); D.compile_story(st)
            if st["meta"]["L"] not in want[sp]:
                rep["L_exact"] = ok = False
        for p, v in pairs.items():
            a, b = v
            q = a["meta"]["q"]
            diff = [t for t in range(a["meta"]["T"]) if a["events"][t] != b["events"][t]]
            if a["label"] == b["label"] or a["intro"] != b["intro"] or a["q"] != b["q"] or diff != [q] \
                    or a["meta"]["L"] != b["meta"]["L"]:
                rep["twins"].append([sp, p]); ok = False; break
        if 2 * sum(st["label"] for st in d[sp]) != len(d[sp]):
            rep["balance"].append(sp); ok = False
    cnt = collections.defaultdict(lambda: [0, 0])
    for st in d["train"]:
        k = st["events"][st["meta"]["q"]][1]; cnt[k][0] += st["label"]; cnt[k][1] += 1
    for sp in ("test_id", "test_oodl", "test_long0"):
        L = d[sp]; acc = lambda f: sum(int(bool(f(st))) == st["label"] for st in L) / len(L)
        prev = lambda st: st["meta"]["q"] >= 1 and st["gold_agents"][st["meta"]["q"] - 1] == st["meta"]["x"]
        rep["oracles"][sp] = {
            "queried_suffix_alone": acc(lambda st: cnt[st["events"][st["meta"]["q"]][1]][1] and
                                        cnt[st["events"][st["meta"]["q"]][1]][0] / cnt[st["events"][st["meta"]["q"]][1]][1] > 0.5),
            "x_class_fsg": acc(lambda st: st["meta"]["classes"][st["meta"]["x"]] == "f-sg"),
            "previous_agent": acc(prev), "first_listed": acc(lambda st: st["meta"]["x"] == 0)}
        ok &= all(0.45 <= v <= 0.55 for v in rep["oracles"][sp].values())
    rep["pass"] = bool(ok)
    return rep


def gsim():
    d = D.build("smoke"); out = {}
    for T, sp in ((1, "test_id"), (2, "test_id"), (3, "test_oodl")):
        enc = D.encode(d[sp], d["intro_angles"])[T][0]; mx = 0.0
        for trial in range(50 if T < 3 else 10):
            m = M.QChain(2000 + trial)
            b = D.sub(enc, torch.arange(trial % 20, trial % 20 + 2))
            with torch.no_grad():
                mx = max(mx, (m.scores(b) - M.full_statevector_scores(m, b)).abs().max().item())
        out["T%d_max_abs_diff" % T] = mx
    out["pass"] = all(v < 1e-5 for k, v in out.items() if k.endswith("diff"))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draw"); ap.add_argument("--sim", action="store_true"); a = ap.parse_args()
    if a.sim:
        r = gsim(); C.write_new_json("exp95_gates/gsim.json", {"doc": "90 §5 G-SIM", "result": r, "provenance": C.provenance()})
        print(r); return
    d = json.load(open("stories_exp95_%s.json" % a.draw, encoding="utf-8"))
    rep = check(d)
    bad = D.build(a.draw, corrupt=True)
    L = [st for st in bad["test_oodl"] if st["meta"]["q"] >= 1] + [st for st in bad["test_id"] if st["meta"]["q"] >= 1]
    prev = sum(int(st["gold_agents"][st["meta"]["q"] - 1] == st["meta"]["x"]) == st["label"] for st in L) / len(L)
    out = {"doc": "90 §5 G-V", "draw": a.draw, "gate": rep, "designed_to_fail": {"previous_agent_on_corrupt": prev,
           "failed_as_designed": prev > 0.55}, "provenance": C.provenance()}
    out["G_V_PASS"] = bool(rep["pass"] and prev > 0.55)
    C.write_new_json("exp95_gates/gv_%s.json" % a.draw, out)
    print(json.dumps({"G_V_PASS": out["G_V_PASS"], "twins": rep["twins"][:2], "L": rep["L_exact"], "bal": rep["balance"],
                      "oracles": rep["oracles"], "dtf": prev}, ensure_ascii=False))


if __name__ == "__main__":
    main()
