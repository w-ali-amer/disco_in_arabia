# -*- coding: utf-8 -*-
"""exp92_gates.py -- doc 87 §6 G-V (data validity), run before any learner unit.

Usage: python exp92_gates.py --draw {primary,confirm}   -> exp92_gates/gv_<draw>.json (Rule 3)
(a) per cell/split: YES = n/2, NO split exactly n/4 (non-undergoer, S_root) + n/4 (undergoer, S_other);
    per-state train YES-rate REPORTED (not gated; reviewer rev. 1).
(b) per seed: no held-out derived item in train/val.
(c) role-only oracle and state-only oracle == 0.75 in EVERY cell (all splits); per seed, every single-token
    majority predictor (root, form, noun1, noun2, position, state; fit on that seed's train) <= 0.60 on held-out test.
(d) every item round-trips through the S-T compiler (exp92_run.encode asserts).
Designed-to-fail: the same checks on an in-memory corrupted draw (NO = 100% non-undergoer/S_root) must FAIL (c).
"""
import argparse, collections, json, os
import exp92_common as C
import exp92_datagen as D
import exp92_run as R


def check(d):
    rep = {"a": {}, "b": {}, "c": {}, "d": {}}
    ok = True
    cells = collections.defaultdict(list)
    for x in d["items"]:
        cells[(x["pair"], x["form"], x["split"])].append(x)
    bad_a = []
    for k, v in cells.items():
        n = len(v); c = collections.Counter(x["kind"] for x in v)
        if not (c["yes"] == n // 2 and c["nonU_Sr"] == c["U_So"] == n // 4 and n % 4 == 0):
            bad_a.append([list(k), dict(c)])
    rep["a"]["cells"] = len(cells); rep["a"]["bad"] = bad_a[:10]; rep["a"]["pass"] = not bad_a
    tr = [x for x in d["items"] if x["split"] == "train"]
    st = collections.defaultdict(list)
    for x in tr:
        st[x["q_state"]].append(x["label"])
    rates = {k: sum(v) / len(v) for k, v in st.items()}
    rep["a"]["state_yes_rate_train_reported"] = {"min": min(rates.values()), "max": max(rates.values())}
    ok &= rep["a"]["pass"]
    # (c) oracles per cell
    bad_c = []
    for k, v in cells.items():
        role = sum((x["q_pos"] == x["undergoer_pos"]) == bool(x["label"]) for x in v) / len(v)
        state = sum((x["q_state"] == x["pair"]) == bool(x["label"]) for x in v) / len(v)
        if abs(role - 0.75) > 1e-9 or abs(state - 0.75) > 1e-9:
            bad_c.append([list(k), role, state])
    rep["c"]["oracle_bad_cells"] = bad_c[:10]; rep["c"]["oracle_pass"] = not bad_c
    ok &= rep["c"]["oracle_pass"]
    # (b)+(c single-token)+(d)
    roots = d["roots"]; vtab = C.verb_table(roots)
    feats = {"root": lambda x: x["pair"], "form": lambda x: x["form"],
             "noun1": lambda x: C.strip_marks(x["tokens"][1]), "noun2": lambda x: C.strip_marks(x["tokens"][-1]),
             "position": lambda x: x["q_pos"], "state": lambda x: x["q_state"]}
    worst = {}
    for s in range(8):
        tr_s, va_s, th_s, ts_s, held = R.splits(d, s)
        leak = [x["id"] for x in tr_s + va_s if x["pair"] in held and x["form"] != "I"]
        rep["b"][str(s)] = len(leak); ok &= not leak
        for name, f in feats.items():
            cnt = collections.defaultdict(lambda: [0, 0])
            for x in tr_s:
                cnt[f(x)][0] += x["label"]; cnt[f(x)][1] += 1
            pred = lambda x: 1 if (f(x) in cnt and cnt[f(x)][0] / cnt[f(x)][1] > 0.5) else 0
            acc = sum(pred(x) == x["label"] for x in th_s) / len(th_s)
            worst[name] = max(worst.get(name, 0.0), acc)
        if s == 0:
            for part in (tr_s, va_s, th_s, ts_s):
                R.encode(part, roots, vtab)          # (d): asserts inside
    rep["c"]["single_token_max_heldout"] = worst
    rep["c"]["single_token_pass"] = all(v <= 0.60 for v in worst.values())
    ok &= rep["c"]["single_token_pass"]
    rep["d"]["pass"] = True
    rep["pass"] = bool(ok)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True, choices=list(D.DRAWS))
    a = ap.parse_args()
    d = json.load(open("stories_exp92_%s.json" % a.draw, encoding="utf-8"))
    rep = check(d)
    bad = check(D.build(a.draw, corrupt=True))
    out = {"doc": "87 §6 G-V", "draw": a.draw, "gate": rep,
           "designed_to_fail": {"pass": bad["pass"], "oracle_pass": bad["c"]["oracle_pass"],
                                "must_fail": True, "failed_as_designed": not bad["pass"]},
           "provenance": C.provenance()}
    out["G_V_PASS"] = bool(rep["pass"] and not bad["pass"])
    C.write_new_json(os.path.join("exp92_gates", "gv_%s.json" % a.draw), out)
    print(json.dumps({"G_V_PASS": out["G_V_PASS"], "a": rep["a"]["pass"], "state_rate": rep["a"]["state_yes_rate_train_reported"],
                      "oracle": rep["c"]["oracle_pass"], "single_token": rep["c"]["single_token_max_heldout"],
                      "leaks": rep["b"], "dtf_failed": not bad["pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
