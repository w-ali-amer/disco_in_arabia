# -*- coding: utf-8 -*-
"""exp60_validate.py -- doc 46 Stage A kill-gate (SS8, SS9).

The V-suite is RE-DERIVED for SB, not inherited from VT (doc 46 SS8):

  V1   bag-of-embeddings      must be EXACTLY 0.500 (shared cast)
  V2   n-gram, train nouns    recorded, NO BAR -- Arabic word order legitimately
                              marks role, so above-chance here is expected
  V2-H n-gram, HELD-OUT nouns <= 0.55   <- the real gate
  V3   balance audit          answer independent of verb / bearer position /
                              order flag / noun
  V4   dedup                  on COMPILED CIRCUIT content, not surfaces
  V6   token-identity probe   must SUCCEED on train nouns and FAIL on held-out

Any gate red => Stage B never runs and a new doc is required (doc 46 SS9).
"""
import json, math, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency

import exp42_compiler as comp

DATA = "stories_exp60_SB.json"
OUT = "results_exp60_SB_validation.json"
V2H_BAR, V6_HELD_BAR = 0.55, 0.55


def split_items(items, s):
    return [i for i in items if i["split"] == s]


def acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def v1_bag(items):  # noqa
    """Bag of referent embeddings. Both twins share a cast => identical
    features => identical prediction => exactly 0.500 on opposite labels."""
    base, _m, _s = comp.load_embeddings()
    def feat(it):
        v = sorted(base[r["name"]] for r in it["referents"])
        return [math.cos(v[0]), math.sin(v[0]), math.cos(v[1]), math.sin(v[1]),
                math.cos(v[0]) + math.cos(v[1]), math.sin(v[0]) + math.sin(v[1])]
    tr, te = split_items(items, "train"), split_items(items, "test")
    Xtr = np.array([feat(i) for i in tr]); ytr = [i["answer"] for i in tr]
    Xte = np.array([feat(i) for i in te]); yte = [i["answer"] for i in te]
    m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    # structural check: twins must receive identical feature rows
    byp = defaultdict(list)
    for i, it in enumerate(te):
        byp[it["pair_id"]].append(i)
    ident = all(np.allclose(Xte[a], Xte[b]) for a, b in byp.values())
    return {"test_accuracy": acc(yte, m.predict(Xte)),
            "twin_features_identical": bool(ident),
            "passed": bool(abs(acc(yte, m.predict(Xte)) - 0.5) < 1e-12 and ident)}


def tokens(it):
    return " ".join(it["sentences"] + [it["question"]["surface"]])


def v2_ngram(items):
    tr = split_items(items, "train")
    va, te = split_items(items, "val"), split_items(items, "test")
    vec = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\S+")
    Xtr = vec.fit_transform([tokens(i) for i in tr])
    m = LogisticRegression(max_iter=3000, C=10.0).fit(
        Xtr, [i["answer"] for i in tr])
    a_val = acc([i["answer"] for i in va],
                m.predict(vec.transform([tokens(i) for i in va])))
    a_hel = acc([i["answer"] for i in te],
                m.predict(vec.transform([tokens(i) for i in te])))
    return {"train_nouns_val_accuracy": a_val, "heldout_accuracy": a_hel,
            "n_features": int(Xtr.shape[1]),
            "V2H_bar": V2H_BAR, "V2H_passed": bool(a_hel <= V2H_BAR)}


def v3_balance(items, meta):
    out, worst = {}, 1.0
    bearers = set(meta["bearers"])
    def cells(keyfn):
        t = defaultdict(Counter)
        for i in items:
            t[keyfn(i)][i["answer"]] += 1
        rows = [[c["yes"], c["no"]] for c in t.values()]
        rows = [r for r in rows if sum(r) > 0]
        if len(rows) < 2:
            return 1.0
        return float(chi2_contingency(np.array(rows))[1])
    # doc 46 SS8 + SS16.1: GATED on the registered nuisance variables only.
    checks = {
        "verb": lambda i: i["events"][0]["verb"],
        "bearer_wire": lambda i: 0 if i["referents"][0]["name"] in bearers else 1,
        "order_flag": lambda i: i["order_flags"][0],
        "wire0_noun": lambda i: i["referents"][0]["name"],
    }
    for k, f in checks.items():
        p = cells(f); out[k + "_p"] = p; worst = min(worst, p)
    # DESCRIPTIVE, never gated (doc 46 SS16.1): the answer IS "was the agent a
    # bearer", so agent-token predictivity is the task definition, not a leak.
    # It is the intended shortcut, defeated by the held-out split and measured
    # by V6. Gating on it would make the rung unbuildable by definition.
    out["agent_noun_p_DESCRIPTIVE"] = cells(lambda i: i["events"][0]["agent"])
    n = Counter(i["answer"] for i in items)
    out.update({"n_yes": n["yes"], "n_no": n["no"],
                "classes_balanced": n["yes"] == n["no"],
                "min_p": worst, "passed": bool(worst > 0.01 and n["yes"] == n["no"])})
    return out


def v4_dedup(items):
    """doc 46 SS5: dedup on what the CIRCUIT sees, not the surface."""
    seen = defaultdict(set)
    for i in items:
        w = {r["name"]: k for k, r in enumerate(i["referents"])}
        e = i["events"][0]
        key = (tuple(r["name"] for r in i["referents"]), e["verb"],
               w[e["agent"]], w[e["patient"]], i["question"]["surface"])
        seen[key].add(i["answer"])
    dup = sum(1 for k, v in seen.items() if len(v) > 1)
    return {"n_distinct_circuits": len(seen), "n_items": len(items),
            "n_contradictory": dup, "passed": bool(dup == 0)}


def v6_token_probe(items, meta):
    """The shortcut found during design, converted into an instrument: a model
    keyed on NOUN IDENTITY x POSITION must succeed on train nouns and fail on
    held-out ones."""
    nouns = sorted(meta["humans"])
    idx = {n: k for k, n in enumerate(nouns)}
    def feat(it):
        # doc 46 SS16.2: keyed on the AGENT noun. The target is
        # is_bearer(noun at the agent wire) -- an INTERACTION of
        # (wire0, wire1) x agent_wire, which a linear probe cannot represent;
        # the earlier form sat at chance and would have falsely certified that
        # no token shortcut exists.
        v = np.zeros(2 * len(nouns))
        v[idx[it["events"][0]["agent"]]] = 1
        v[len(nouns) + idx[it["events"][0]["patient"]]] = 1
        return v
    tr, va, te = (split_items(items, s) for s in ("train", "val", "test"))
    m = LogisticRegression(max_iter=4000, C=10.0).fit(
        np.array([feat(i) for i in tr]), [i["answer"] for i in tr])
    a_val = acc([i["answer"] for i in va],
                m.predict(np.array([feat(i) for i in va])))
    a_hel = acc([i["answer"] for i in te],
                m.predict(np.array([feat(i) for i in te])))
    return {"train_nouns_val_accuracy": a_val, "heldout_accuracy": a_hel,
            "must_succeed_on_train": bool(a_val >= 0.80),
            "must_fail_on_heldout": bool(a_hel <= V6_HELD_BAR),
            "passed": bool(a_val >= 0.80 and a_hel <= V6_HELD_BAR)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    blob = json.load(open(a.data, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("dataset self-hash mismatch")
    r = {"experiment": "exp60 Stage A validation", "design_doc": meta["design_doc"],
         "rung": "SB", "dataset_sha256": meta["dataset_sha256"],
         "split_counts": meta["split_counts"],
         "V1_bag_of_embeddings": v1_bag(items),
         "V2_ngram": v2_ngram(items),
         "V3_balance": v3_balance(items, meta),
         "V4_dedup_on_circuits": v4_dedup(items),
         "V6_token_identity_probe": v6_token_probe(items, meta)}
    gates = {"V1": r["V1_bag_of_embeddings"]["passed"],
             "V2H": r["V2_ngram"]["V2H_passed"],
             "V3": r["V3_balance"]["passed"],
             "V4": r["V4_dedup_on_circuits"]["passed"],
             "V6": r["V6_token_identity_probe"]["passed"]}
    r["gates"] = gates
    r["stage_a_kill_gate_passed"] = bool(all(gates.values()))
    json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp60 Stage A]")
    print("  V1  bag-of-embeddings   %.4f  twins-identical=%s  %s"
          % (r["V1_bag_of_embeddings"]["test_accuracy"],
             r["V1_bag_of_embeddings"]["twin_features_identical"],
             "PASS" if gates["V1"] else "FAIL"))
    print("  V2  n-gram val (train nouns) %.4f   [recorded, no bar]"
          % r["V2_ngram"]["train_nouns_val_accuracy"])
    print("  V2H n-gram held-out     %.4f  bar<=%.2f  %s"
          % (r["V2_ngram"]["heldout_accuracy"], V2H_BAR,
             "PASS" if gates["V2H"] else "FAIL"))
    print("  V3  balance min_p       %.4f  %s"
          % (r["V3_balance"]["min_p"], "PASS" if gates["V3"] else "FAIL"))
    print("  V4  contradictory circuits %d of %d distinct  %s"
          % (r["V4_dedup_on_circuits"]["n_contradictory"],
             r["V4_dedup_on_circuits"]["n_distinct_circuits"],
             "PASS" if gates["V4"] else "FAIL"))
    print("  V6  token probe: train %.4f (must>=0.80), held-out %.4f (must<=%.2f)  %s"
          % (r["V6_token_identity_probe"]["train_nouns_val_accuracy"],
             r["V6_token_identity_probe"]["heldout_accuracy"], V6_HELD_BAR,
             "PASS" if gates["V6"] else "FAIL"))
    print("\n  STAGE A KILL-GATE: %s"
          % ("PASSED" if r["stage_a_kill_gate_passed"] else "FAILED"))
    return 0 if r["stage_a_kill_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
