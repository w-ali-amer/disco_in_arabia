# -*- coding: utf-8 -*-
"""exp68_validate.py -- doc 54 SS6.5's kill-gate, RE-DERIVED not inherited.

doc 46 SS8's rule, and doc 46 SS8's lesson, are the same rule read twice: a
V-suite is derived for the rung it guards. doc 54 SS6.5 states it explicitly --
"full V-suite kill-gate first, RE-DERIVED not inherited, with the
held-out-pairing split substituted for the held-out-referent split and V1
RE-MEASURED rather than assumed to still be 0.500."

WHAT CHANGED, AND WHY TWO OF exp60'S BARS CANNOT BE INHERITED
-------------------------------------------------------------
exp60 split by REFERENT INVENTORY, so a model keyed on noun identity could
memorise noun -> bearer on train and had to fail on a disjoint held-out
inventory. V2-H (n-gram on held-out nouns <= 0.55) and V6's held-out clause
(token probe must FAIL on held-out) were the instruments that enforced that, and
they were the right instruments for the claim exp60 made.

exp68 splits by PAIRING (doc 54 SS6.1.1). The 24 nouns are the same in train and
test; only the (bearer, non-bearer) combinations are unseen. A token-identity
model therefore TRANSFERS BY DESIGN, and importing exp60's bars would fail this
dataset for doing exactly what doc 54 registered it to do. Inheriting them
blindly is the failure doc 46 SS8 warns about, pointing the other way.

So V2's held-out clause and V6's held-out clause are RECORDED WITH NO BAR, each
carrying the reason and doc 54 SS10's scope limit in the JSON. This is not a
weakened gate; it is the gate the design licenses, and the price is paid in the
CLAIM, not in the check: exp68 may never be reported as novel-filler
generalisation. The circuit itself never sees a token -- `exp42_compiler`
compiles items to per-referent angles and nothing else -- so the surface
shortcut is available to surface baselines and to nothing exp68 measures.

TWO GATES ARE ADDED, because the new split needs enforcing where the old bars
used to sit:

  V5  SPLIT INTEGRITY   test casts disjoint from train/val casts, and every
                        bearer AND every non-bearer present in >= 1 train cast
                        (doc 54 SS6.1.1). Without the coverage half, "unseen
                        pairing" silently degenerates into "unseen noun" for
                        whichever referent got no train cast.
  V7  LABEL DEFINITION  answer == yes iff the AGENT is a real bearer, on every
                        item; classes balanced in every split.

THE SUITE
---------
  V1  bag-of-embeddings    EXACTLY 0.500, re-measured        GATE
  V2  n-gram               recorded, NO BAR (see above)      no gate
  V3  balance audit        answer independent of verb / bearer wire /
                           order flag / wire-0 noun          GATE
  V4  dedup                on COMPILED CIRCUIT content       GATE
  V5  split integrity      disjointness + train coverage     GATE
  V6  token-identity probe must SUCCEED on in-distribution val (a functional
                           check on the probe); held-out recorded, NO BAR   GATE
  V7  label definition     agent-bearer <-> yes, balanced    GATE

Any gate red => no training runs and a new doc is required (doc 46 SS9).

The probes are the committed exp60_validate.py probes, transposed only where the
split forced it: same CountVectorizer(1,2) on whitespace tokens, same
LogisticRegression settings, same chi2 balance test, same circuit-level dedup
key, same agent/patient one-hot token probe.

    python exp68_validate.py
    python exp68_validate.py --data stories_exp68_WN_confirm.json --out ...
"""
import argparse, json, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency

import exp42_compiler as comp
import instrument_sb_bounds as INS

DESIGN_DOC = "qnlp_private_docs/54_exp68_width_lever.md (1025d21, FROZEN)"
DATA = "stories_exp68_WN.json"
OUT = "results_exp68_WN_validation.json"
V6_TRAIN_BAR = 0.80          # the probe must WORK, else V6 certifies nothing
N_TEST_ITEMS = 300           # doc 54 SS6.1.1, frozen


def split_items(items, s):
    return [i for i in items if i["split"] == s]


def acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def v1_bag(items):
    """Bag of referent embeddings. Both twins share a cast => identical feature
    rows => identical prediction => exactly 0.500 on opposite labels.

    RE-MEASURED, not assumed (doc 54 SS6.5). The structural argument is
    unchanged by the split -- it is a statement about TWINS, not about which
    casts are held out -- but the whole point of doc 46 SS8's rule is that the
    argument is checked on the dataset that was actually built."""
    base, _m, _s = comp.load_embeddings()
    def feat(it):
        v = sorted(base[r["name"]] for r in it["referents"])
        return [np.cos(v[0]), np.sin(v[0]), np.cos(v[1]), np.sin(v[1]),
                np.cos(v[0]) + np.cos(v[1]), np.sin(v[0]) + np.sin(v[1])]
    tr, te = split_items(items, "train"), split_items(items, "test")
    Xtr = np.array([feat(i) for i in tr]); ytr = [i["answer"] for i in tr]
    Xte = np.array([feat(i) for i in te]); yte = [i["answer"] for i in te]
    m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    a = acc(yte, m.predict(Xte))
    byp = defaultdict(list)
    for i, it in enumerate(te):
        byp[it["pair_id"]].append(i)
    ident = all(np.allclose(Xte[x], Xte[y]) for x, y in byp.values())
    return {"test_accuracy": a, "twin_features_identical": bool(ident),
            "n_twin_pairs_checked": len(byp),
            "passed": bool(abs(a - 0.5) < 1e-12 and ident)}


def tokens(it):
    return " ".join(it["sentences"] + [it["question"]["surface"]])


def v2_ngram(items):
    """RECORDED, NO BAR. Under a pairing split the surface names both referents
    and the noun inventory is shared, so an n-gram model transfers to unseen
    pairings BY DESIGN. exp60's V2H_bar = 0.55 was an instrument for a
    held-out-REFERENT claim exp68 does not make (doc 54 SS10)."""
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
    return {"seen_pairings_val_accuracy": a_val,
            "unseen_pairings_test_accuracy": a_hel,
            "n_features": int(Xtr.shape[1]),
            "gated": False,
            "exp60_V2H_bar_not_inherited": 0.55,
            "why_no_bar": (
                "exp60 split by referent inventory, so a surface model HAD to fail "
                "on held-out nouns and 0.55 was the instrument that enforced it. "
                "exp68 splits by PAIRING (doc 54 SS6.1.1): the same 24 nouns appear "
                "in train and test, the surface names both referents, so token "
                "identity transfers BY DESIGN. Importing the bar would fail the "
                "dataset for satisfying its own registration. The price is paid in "
                "the CLAIM instead -- doc 54 SS10: exp68 is not novel-filler "
                "generalisation and may never be reported as such. The compiled "
                "circuit consumes per-referent angles only and never sees a token, "
                "so this shortcut is available to surface baselines and to nothing "
                "exp68 measures.")}


def v3_balance(items, meta):
    """Answer independent of the registered nuisance variables (doc 46 SS8/SS16.1)."""
    out, worst = {}, 1.0
    bearers = set(meta["bearers"])

    def cells(keyfn):
        t = defaultdict(Counter)
        for i in items:
            t[keyfn(i)][i["answer"]] += 1
        rows = [[c["yes"], c["no"]] for c in t.values() if sum(c.values()) > 0]
        if len(rows) < 2:
            return 1.0
        return float(chi2_contingency(np.array(rows))[1])

    checks = {
        "verb": lambda i: i["events"][0]["verb"],
        "bearer_wire": lambda i: 0 if i["referents"][0]["name"] in bearers else 1,
        "order_flag": lambda i: i["order_flags"][0],
        "wire0_noun": lambda i: i["referents"][0]["name"],
    }
    for k, f in checks.items():
        p = cells(f)
        out[k + "_p"] = p
        worst = min(worst, p)
    # DESCRIPTIVE, never gated (doc 46 SS16.1): the answer IS "was the agent a
    # bearer", so agent-token predictivity is the task definition, not a leak.
    out["agent_noun_p_DESCRIPTIVE"] = cells(lambda i: i["events"][0]["agent"])
    n = Counter(i["answer"] for i in items)
    out.update({"n_yes": n["yes"], "n_no": n["no"],
                "classes_balanced": n["yes"] == n["no"],
                "min_p": worst,
                "passed": bool(worst > 0.01 and n["yes"] == n["no"])})
    return out


def v4_dedup(items):
    """doc 46 SS5: dedup on what the CIRCUIT sees, not on the surface."""
    seen = defaultdict(set)
    for i in items:
        w = {r["name"]: k for k, r in enumerate(i["referents"])}
        e = i["events"][0]
        key = (tuple(r["name"] for r in i["referents"]), e["verb"],
               w[e["agent"]], w[e["patient"]], i["question"]["surface"])
        seen[key].add(i["answer"])
    dup = sum(1 for v in seen.values() if len(v) > 1)
    return {"n_distinct_circuits": len(seen), "n_items": len(items),
            "n_contradictory": dup,
            "all_circuits_distinct": bool(len(seen) == len(items)),
            "passed": bool(dup == 0)}


def v5_split_integrity(items, meta):
    """NEW for exp68: the pairing split's own load-bearing invariants."""
    cast = lambda i: tuple(sorted(r["name"] for r in i["referents"]))  # noqa: E731
    by = {s: {cast(i) for i in split_items(items, s)}
          for s in ("train", "val", "test")}
    disjoint = not (by["test"] & (by["train"] | by["val"])) \
        and not (by["train"] & by["val"])
    in_train = {n for c in by["train"] for n in c}
    bearers, nons = set(meta["bearers"]), set(meta["nonbearers"])
    missing = sorted((bearers | nons) - in_train)
    n_test = len(split_items(items, "test"))
    return {"n_train_casts": len(by["train"]), "n_val_casts": len(by["val"]),
            "n_test_casts": len(by["test"]),
            "casts_disjoint_across_splits": bool(disjoint),
            "bearers_covered_in_train": bool(bearers <= in_train),
            "nonbearers_covered_in_train": bool(nons <= in_train),
            "referents_missing_from_train_casts": missing,
            "n_test_items": n_test,
            "n_test_items_is_registered_300": bool(n_test == N_TEST_ITEMS),
            "why_coverage_matters": (
                "without it, 'unseen pairing' degenerates into 'unseen noun' for "
                "any referent absent from every train cast, and the split stops "
                "measuring what doc 54 SS6.1.1 says it measures"),
            "passed": bool(disjoint and not missing and n_test == N_TEST_ITEMS)}


def v6_token_probe(items, meta):
    """Noun-identity x role probe. GATED on the in-distribution half only.

    exp60 gated BOTH halves (succeed on train nouns, FAIL on held-out nouns).
    Under a pairing split the held-out half is expected to SUCCEED, so only the
    functional half survives as a gate: a probe that cannot fit the seen
    pairings certifies nothing about the unseen ones."""
    nouns = sorted(meta["humans"])
    idx = {n: k for k, n in enumerate(nouns)}

    def feat(it):
        v = np.zeros(2 * len(nouns))
        v[idx[it["events"][0]["agent"]]] = 1
        v[len(nouns) + idx[it["events"][0]["patient"]]] = 1
        return v

    tr, va, te = (split_items(items, s) for s in ("train", "val", "test"))
    m = LogisticRegression(max_iter=4000, C=10.0).fit(
        np.array([feat(i) for i in tr]), [i["answer"] for i in tr])
    a_val = acc([i["answer"] for i in va], m.predict(np.array([feat(i) for i in va])))
    a_hel = acc([i["answer"] for i in te], m.predict(np.array([feat(i) for i in te])))
    return {"seen_pairings_val_accuracy": a_val,
            "unseen_pairings_test_accuracy": a_hel,
            "must_succeed_on_seen_pairings": bool(a_val >= V6_TRAIN_BAR),
            "train_bar": V6_TRAIN_BAR,
            "heldout_gated": False,
            "exp60_heldout_bar_not_inherited": 0.55,
            "why_heldout_not_gated": (
                "the noun inventory is shared across splits by construction "
                "(doc 54 SS6.1.1), so a noun-identity probe transfers to unseen "
                "PAIRINGS. That is a property of the registered split, not a "
                "defect. It is exactly why doc 54 SS10 forbids the novel-filler "
                "reading, and it is recorded here so the limitation travels with "
                "the dataset."),
            "passed": bool(a_val >= V6_TRAIN_BAR)}


def v7_label_definition(items, meta):
    """The label IS the task: yes iff the agent is a real bearer."""
    bset = set(meta["bearers"])
    bad = [i["id"] for i in items
           if (i["answer"] == "yes") != (i["events"][0]["agent"] in bset)]
    one_bearer = [i["id"] for i in items
                  if sum(1 for r in i["referents"] if r["name"] in bset) != 1]
    bal = {s: Counter(i["answer"] for i in split_items(items, s))
           for s in ("train", "val", "test")}
    balanced = all(c["yes"] == c["no"] for c in bal.values())
    return {"items_violating_label_rule": bad[:8], "n_violations": len(bad),
            "items_without_exactly_one_bearer": one_bearer[:8],
            "n_cast_violations": len(one_bearer),
            "per_split_balance": {s: dict(c) for s, c in bal.items()},
            "balanced_in_every_split": bool(balanced),
            "passed": bool(not bad and not one_bearer and balanced)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    blob = json.load(open(a.data, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("dataset self-hash mismatch -- refusing to validate")

    r = {"experiment": "exp68 V-suite kill-gate",
         "spec": DESIGN_DOC,
         "design_doc_of_dataset": meta["design_doc"],
         "rung": "SB (width lever)",
         "dataset": a.data,
         "dataset_sha256": meta["dataset_sha256"],
         "dataset_seed": meta["seed"],
         "category": meta["category"],
         "bearers": meta["bearers"],
         "split_counts": meta["split_counts"],
         "split_unit": meta["split_unit"],
         "rederivation_note": (
             "RE-DERIVED for the held-out-PAIRING split (doc 54 SS6.5, doc 46 SS8). "
             "exp60's V2-H bar and V6 held-out bar are NOT inherited: under a "
             "pairing split a token model transfers by design, so those bars would "
             "fail the dataset for satisfying its registration. They are replaced "
             "by V5 (split integrity + train coverage) and V7 (label definition), "
             "and the un-gated numbers are recorded in full with doc 54 SS10's "
             "scope limit attached."),
         "V1_bag_of_embeddings": v1_bag(items),
         "V2_ngram": v2_ngram(items),
         "V3_balance": v3_balance(items, meta),
         "V4_dedup_on_circuits": v4_dedup(items),
         "V5_split_integrity": v5_split_integrity(items, meta),
         "V6_token_identity_probe": v6_token_probe(items, meta),
         "V7_label_definition": v7_label_definition(items, meta)}
    gates = {"V1": r["V1_bag_of_embeddings"]["passed"],
             "V3": r["V3_balance"]["passed"],
             "V4": r["V4_dedup_on_circuits"]["passed"],
             "V5": r["V5_split_integrity"]["passed"],
             "V6": r["V6_token_identity_probe"]["passed"],
             "V7": r["V7_label_definition"]["passed"]}
    r["gates"] = gates
    r["ungated_recorded"] = ["V2 (both halves)", "V6 unseen-pairings half"]
    r["kill_gate_passed"] = bool(all(gates.values()))
    r["consequence_if_failed"] = ("doc 46 SS9: any gate red and Stage B never runs "
                                  "-- a new doc is required, not a patch.")
    json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    v1, v2, v3, v4, v5, v6, v7 = (r["V1_bag_of_embeddings"], r["V2_ngram"],
                                  r["V3_balance"], r["V4_dedup_on_circuits"],
                                  r["V5_split_integrity"],
                                  r["V6_token_identity_probe"],
                                  r["V7_label_definition"])
    print("[exp68 V-suite]  %s  sha %s" % (a.data, meta["dataset_sha256"][:12]))
    print("  V1  bag-of-embeddings   %.6f  twins-identical=%s (%d pairs)  %s"
          % (v1["test_accuracy"], v1["twin_features_identical"],
             v1["n_twin_pairs_checked"], "PASS" if gates["V1"] else "FAIL"))
    print("  V2  n-gram  seen-pairings val %.4f | unseen-pairings test %.4f"
          "   [RECORDED, NO BAR -- pairing split]"
          % (v2["seen_pairings_val_accuracy"], v2["unseen_pairings_test_accuracy"]))
    print("  V3  balance min_p       %.4f  (verb %.3f, bearer-wire %.3f, "
          "order %.3f, wire0-noun %.3f)  %s"
          % (v3["min_p"], v3["verb_p"], v3["bearer_wire_p"], v3["order_flag_p"],
             v3["wire0_noun_p"], "PASS" if gates["V3"] else "FAIL"))
    print("  V4  contradictory circuits %d of %d distinct (%d items)  %s"
          % (v4["n_contradictory"], v4["n_distinct_circuits"], v4["n_items"],
             "PASS" if gates["V4"] else "FAIL"))
    print("  V5  casts %d/%d/%d disjoint=%s  coverage B=%s N=%s  n_test=%d  %s"
          % (v5["n_train_casts"], v5["n_val_casts"], v5["n_test_casts"],
             v5["casts_disjoint_across_splits"], v5["bearers_covered_in_train"],
             v5["nonbearers_covered_in_train"], v5["n_test_items"],
             "PASS" if gates["V5"] else "FAIL"))
    print("  V6  token probe: seen-pairings %.4f (must>=%.2f), unseen-pairings "
          "%.4f [NO BAR]  %s"
          % (v6["seen_pairings_val_accuracy"], V6_TRAIN_BAR,
             v6["unseen_pairings_test_accuracy"], "PASS" if gates["V6"] else "FAIL"))
    print("  V7  label rule violations %d, cast violations %d, balanced=%s  %s"
          % (v7["n_violations"], v7["n_cast_violations"],
             v7["balanced_in_every_split"], "PASS" if gates["V7"] else "FAIL"))
    print("\n  KILL-GATE: %s" % ("PASSED" if r["kill_gate_passed"] else "FAILED"))
    print("[exp68 V-suite] wrote %s" % a.out)
    return 0 if r["kill_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
