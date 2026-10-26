# -*- coding: utf-8 -*-
"""exp93_gates.py -- doc 88 §6 G-V, before any learner unit.

(a) twins differ ONLY in V2's token and labels flip; exact balance (label, S1-subject gender, X position, X gender)
(b) held-out stems absent from train/val
(c) on test_heldout, per seed: previous-agent rule, suffix-alone, X-gender-alone, X-position-alone,
    S1-subject-gender-alone (single-feature majority rules fit on train) all within [0.45, 0.55]
(d) every story round-trips through the compiler
Designed-to-fail: an in-memory corrupted draw (V2 always agrees with the S1 subject) must FAIL (c).
Usage: python exp93_gates.py --draw primary -> exp93_gates/gv_<draw>.json (Rule 3)
"""
import argparse, collections, json, os
import exp93_common as C
import exp93_datagen as D


def feats(st):
    c = C.compile_story(st)
    xfem = c["fem0"] if c["x_wire"] == 0 else c["fem1"]
    return {"suffix": c["suffix_fem"], "x_gender": xfem, "x_position": c["x_wire"], "s1_subject_gender": c["fem0"]}


def check(d):
    rep = {"a": [], "b": [], "c": {}, "d": True}; ok = True
    for s, sd in d["seeds"].items():
        held = set(sd["held_out"])
        for split in ("train", "val", "test_heldout", "test_seen"):
            L = sd[split]
            pairs = collections.defaultdict(list)
            for st in L:
                pairs[st["pair"]].append(st)
                C.compile_story(st)                                     # (d)
            for p, v in pairs.items():
                if len(v) != 2 or v[0]["label"] == v[1]["label"] or v[0]["s1"] != v[1]["s1"] or v[0]["q"] != v[1]["q"] \
                        or v[0]["s2"][0] != v[1]["s2"][0] or v[0]["s2"][2] != v[1]["s2"][2] or v[0]["s2"][1] == v[1]["s2"][1]:
                    rep["a"].append([s, split, p]); ok = False; break
            n = len(L)
            for key in ("subj_fem", "x_is_a", "x_fem"):
                if sum(st["meta"][key] for st in L) * 2 != n:
                    rep["a"].append([s, split, "balance", key]); ok = False
            if sum(st["label"] for st in L) * 2 != n:
                rep["a"].append([s, split, "label balance"]); ok = False
            if split in ("train", "val"):
                used = {C.read_noun(st["s1"][1])[0] for st in L} | {C.read_noun(st["s1"][2])[0] for st in L}
                if used & held:
                    rep["b"].append([s, split]); ok = False
        # (c)
        tr, th = sd["train"], sd["test_heldout"]
        acc = {"previous_agent_rule": sum(int(st["meta"]["x_is_a"]) == st["label"] for st in th) / len(th)}
        for name in ("suffix", "x_gender", "x_position", "s1_subject_gender"):
            cnt = collections.defaultdict(lambda: [0, 0])
            for st in tr:
                f = feats(st)[name]; cnt[f][0] += st["label"]; cnt[f][1] += 1
            acc[name] = sum(int(cnt[feats(st)[name]][0] / cnt[feats(st)[name]][1] > 0.5) == st["label"] for st in th) / len(th)
        rep["c"][s] = acc
        ok &= all(0.45 <= v <= 0.55 for v in acc.values())
    rep["pass"] = bool(ok)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draw", required=True); a = ap.parse_args()
    d = json.load(open("stories_exp93_%s.json" % a.draw, encoding="utf-8"))
    rep = check(d)
    bad = D.build(a.draw, corrupt=True)
    # corrupt draw has no twins -> judge it on (c) only
    b_prev = max(sum(int(st["meta"]["x_is_a"]) == st["label"] for st in sd["test_heldout"]) / len(sd["test_heldout"])
                 for sd in bad["seeds"].values())
    out = {"doc": "88 §6 G-V", "draw": a.draw, "gate": rep,
           "designed_to_fail": {"previous_agent_rule_on_corrupt": b_prev, "failed_as_designed": b_prev > 0.55},
           "provenance": C.provenance()}
    out["G_V_PASS"] = bool(rep["pass"] and b_prev > 0.55)
    C.write_new_json(os.path.join("exp93_gates", "gv_%s.json" % a.draw), out)
    print(json.dumps({"G_V_PASS": out["G_V_PASS"], "a": rep["a"][:3], "b": rep["b"][:3], "c_seed0": rep["c"]["0"],
                      "dtf_prev_agent": b_prev}, ensure_ascii=False))


if __name__ == "__main__":
    main()
