#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate_exp41r_data.py -- the exp41 REVIVAL V-suite (doc 22 SS2.3 verbatim,
doc 65 SS2.3's designed-to-fail smoke battery).

WHY THIS IS A NEW FILE AND NOT AN EDIT TO validate_exp41_data.py
    `validate_exp41_data.py` is not a private exp41 script: it is a COMMITTED
    SHARED LIBRARY. `exp43a_family_datagen`, `exp43b_datagen`, `exp50_validate`,
    `exp51_validate` and `exp53_validate` all `import validate_exp41_data as val`
    and call `val.run_v1 .. val.run_v5` (exp43a also sets `val.DATA` /
    `val.RESULTS_BASE` and calls `val.main()`). Changing `run_v5`'s semantics in
    place would silently change what five committed results files MEAN, in
    breach of research rule 3 and of doc 65 SS2.3's twice-evaluation rule. The
    house pattern for a new dataset is a new validator that IMPORTS the shared
    runners -- exp46/50/51/53 are the precedent; this is the next instance.

    V1-V4 are therefore the committed runners, called unchanged.

V5 IS STRENGTHENED, AND THAT IS THE POINT (doc 22 SS2.3 licenses generator
fixes before training; here the fix went to the INSTRUMENT first, then to the
generator).
    Registered text: "a single-sentence oracle (sees any ONE sentence + the
    question) must be at chance on the test set." The August instantiation was a
    mean-pooled bag-of-embeddings model over (sentence + question) tokens. Mean
    pooling is ORDER-BLIND, and every twin pair in this task family is
    bag-identical by construction -- so the August V5 returns exactly 0.500 on
    ANY dataset in the family, including one that is trivially single-hop. It
    cannot fail, so it was not a gate. Proof, run below: the `singlehop` and
    `nodistract` smoke builds pass the August V5 and fail this one.

    The revival oracle reads ONE sentence at a time, and reads it properly:
    it parses the 3-word SVO / VSO / pro-drop frame over the closed vocabulary
    (verbs and referents are known token sets, so this is deterministic surface
    parsing, not a NLP model), and scores question-relative features -- does the
    sentence's verb match the question's verb, is a question-named referent the
    agent or the patient, does the sentence exactly match the queried triple --
    alongside the sentence's own lexical identity and its 1-2 grams. It is
    trained at the sentence level and aggregated to the item by
    max P(yes) over sentences (threshold tuned on the in-distribution val split,
    never on test) and by the most-confident sentence. Both must be at chance.
    A mean-over-sentences aggregation is reported as a DESCRIPTIVE row: it can
    see per-pattern COUNTS, which a strict "sees ONE sentence" reading does not
    license, so it does not gate -- but it is recorded because the generator's
    balancing event was designed to close that channel too.

SMOKE BATTERY (doc 65 SS2.3): designed-to-fail builds run through the SAME
V-runners as the real data, aggregated into a SEPARATE block, never mixed into
the five verdict booleans, and their data deleted afterwards.
    qleak       question particle differs by answer class  -> V2 MUST fail
    singlehop   Q0 single-hop items placed in test         -> V5 MUST fail
    nodistract  the August construction (no balancing event) -> V5 MUST fail
A smoke build that PASSES indicts the validator, not the generator.

OUTPUT: results_exp41r_validation.json (rule 3: never overwrites; suffixes).
"""

import datetime
import hashlib
import json
import math
import os
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse

import exp41_datagen as G
import validate_exp41_data as val          # the committed V1-V4 runners
from exp41_story_generator import VERBS, HUMANS, OBJECTS

SPEC = ("qnlp_private_docs/22_phase2_trained_story_qa_design.md SS2.3 "
        "+ 65_exp40_42_revival.md SS2.3")
DATA = "stories_exp41r.json"
RESULTS_BASE = "results_exp41r_validation"
ACC_BOUND = val.ACC_BOUND               # 0.55
ALPHA_P = val.ALPHA_P                   # 0.05
MIN_TEST_PER_SPLIT = 500
SEED = 41000

VERB_SET = set(VERBS)
NAME_SET = set(HUMANS) | set(OBJECTS)
SAB_FILES = {m: f"stories_exp41r_sab_{m}.json" for m in G.SABOTAGE_MODES}


# ---------------------------------------------------------------- V5 (strong)
def parse_sentence(tokens):
    """Deterministic surface parse of a 3-word SVO/VSO frame (2 words when the
    subject is pro-dropped). Returns (verb, agent, patient, order)."""
    if len(tokens) == 2 and tokens[0] in VERB_SET:
        return tokens[0], None, tokens[1], "PRO"
    if len(tokens) == 3:
        if tokens[0] in VERB_SET:
            return tokens[0], tokens[1], tokens[2], "VSO"
        if tokens[1] in VERB_SET:
            return tokens[1], tokens[0], tokens[2], "SVO"
    return None, None, None, "?"


def q_slots(q):
    """(verbs named by the question, referents named by the question)."""
    if q["type"] == "Q1":
        return {"qv_main": q["v2"], "qv_rel": q["v1"]}, \
               {"qr_main": q["y"], "qr_rel": q["x"]}
    return {"qv_main": q["v"], "qv_rel": None}, \
           {"qr_main": q["p"], "qr_rel": q["a"]}


def sentence_features(sent, item):
    """Symbolic features of ONE sentence read against the question. No feature
    may reference any other sentence -- that is the whole point of V5."""
    q = item["question"]
    toks = sent.split()
    v, ag, pat, order = parse_sentence(toks)
    qv, qr = q_slots(q)
    f = [f"qtype={q['type']}", f"order={order}", f"verb={v}"]
    for vname, vtok in qv.items():
        f.append(f"vmatch:{vname}={v is not None and v == vtok}")
    for rname, rtok in qr.items():
        role = "agent" if ag == rtok else "patient" if pat == rtok else "absent"
        f.append(f"role:{rname}={role}")
        for vname, vtok in qv.items():
            if vtok is not None and v == vtok:
                f.append(f"vx:{vname}x{rname}={role}")
    # the exact single-hop reads, in both directions
    if q["type"] in ("Q0", "Q2"):
        f.append(f"exact={v == q['v'] and ag == q['a'] and pat == q['p']}")
        f.append(f"reversed={v == q['v'] and ag == q['p'] and pat == q['a']}")
    else:
        f.append(f"exact_main={v == q['v2'] and pat == q['y']}")
        f.append(f"exact_rel={v == q['v1'] and pat == q['x']}")
    # lexical identity of the slots (so names cannot hide a signal)
    f += [f"lex_ag={ag}", f"lex_pat={pat}", f"lex_v={v}"]
    return f


def expand_sentences(items):
    """One row per (item, sentence). Text = sentence + question surface (the
    August n-gram channel); symbolic = the question-relative features."""
    texts, feats, ys, owners = [], [], [], []
    for n, it in enumerate(items):
        qs = it["question"]["surface"]
        for s in it["sentences"]:
            texts.append(s + " " + qs)
            feats.append(" ".join(sentence_features(s, it)))
            ys.append(val.label(it))
            owners.append(n)
    return texts, feats, np.array(ys), np.array(owners)


def _tune_threshold(scores, y):
    best = (-1.0, 0.5)
    for t in np.unique(np.round(scores, 4)):
        acc = float(((scores > t).astype(int) == y).mean())
        if acc > best[0]:
            best = (acc, float(t))
    return best[1], best[0]


def run_v5_strong(train, valid, test):
    tr_tx, tr_ft, tr_y, _ = expand_sentences(train)
    va_tx, va_ft, va_y, va_o = expand_sentences(valid)
    te_tx, te_ft, te_y, te_o = expand_sentences(test)

    tv = CountVectorizer(token_pattern=r"[^\s]+", ngram_range=(1, 2))
    fv = CountVectorizer(token_pattern=r"[^\s]+", ngram_range=(1, 1),
                         binary=True)
    Xtr = sparse.hstack([tv.fit_transform(tr_tx), fv.fit_transform(tr_ft)]).tocsr()
    Xva = sparse.hstack([tv.transform(va_tx), fv.transform(va_ft)]).tocsr()
    Xte = sparse.hstack([tv.transform(te_tx), fv.transform(te_ft)]).tocsr()

    def item_scores(P, owners, n_items):
        mx = np.full(n_items, -np.inf)
        mn = np.zeros(n_items)
        cnt = np.zeros(n_items)
        conf = np.zeros(n_items)
        best = np.zeros(n_items)
        for p, o in zip(P, owners):
            mx[o] = max(mx[o], p)
            mn[o] += p
            cnt[o] += 1
            if abs(p - 0.5) > conf[o]:
                conf[o] = abs(p - 0.5)
                best[o] = p
        return mx, mn / np.maximum(cnt, 1), best

    y_item_va = np.array([val.label(i) for i in valid])
    y_item_te = np.array([val.label(i) for i in test])

    best_pick = None
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=4000)
        clf.fit(Xtr, tr_y)
        Pva = clf.predict_proba(Xva)[:, 1]
        mx, mean, bestp = item_scores(Pva, va_o, len(valid))
        t_max, a_max = _tune_threshold(mx, y_item_va)
        t_mean, a_mean = _tune_threshold(mean, y_item_va)
        t_best, a_best = _tune_threshold(bestp, y_item_va)
        if best_pick is None or a_max > best_pick[0]:
            best_pick = (a_max, C, clf, t_max, t_mean, t_best,
                         a_max, a_mean, a_best)
    _, C, clf, t_max, t_mean, t_best, va_max, va_mean, va_best = best_pick

    Pte = clf.predict_proba(Xte)[:, 1]
    mx, mean, bestp = item_scores(Pte, te_o, len(test))
    acc_max = float(((mx > t_max).astype(int) == y_item_te).mean())
    acc_mean = float(((mean > t_mean).astype(int) == y_item_te).mean())
    acc_best = float(((bestp > t_best).astype(int) == y_item_te).mean())
    n = len(test)

    per_split = {}
    for s in sorted({i["split"] for i in test}):
        m = np.array([i["split"] == s for i in test])
        per_split[s] = float(((mx[m] > t_max).astype(int) == y_item_te[m]).mean())

    return {
        "oracle": "sentence-level logistic over question-relative symbolic "
                  "features + sentence/question 1-2 grams; item score "
                  "aggregated over sentences; thresholds tuned on val only",
        "selected_C": C, "n_features": int(Xtr.shape[1]),
        "n_test": n,
        "test_accuracy_max_aggregation": acc_max,
        "ci95_max_aggregation": val.ci95(acc_max, n),
        "test_accuracy_best_sentence": acc_best,
        "ci95_best_sentence": val.ci95(acc_best, n),
        "val_accuracy_max_aggregation": va_max,
        "val_accuracy_best_sentence": va_best,
        "per_split_accuracy_max_aggregation": per_split,
        "descriptive_mean_aggregation": {
            "note": "mean over sentences can see per-pattern COUNTS, which a "
                    "strict 'sees ONE sentence' reading does not license; "
                    "recorded, NOT gating",
            "test_accuracy": acc_mean,
            "ci95": val.ci95(acc_mean, n),
            "val_accuracy": va_mean,
        },
        "passed": bool(val.chance_bound(acc_max, n)
                       and val.chance_bound(acc_best, n)),
    }


# ------------------------------------------------------------------- helpers
def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    items, meta = d["items"], d["meta"]
    sha = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True)
                         .encode("utf-8")).hexdigest()
    return items, meta, sha


def splits(items):
    return ([i for i in items if i["split"] == "train"],
            [i for i in items if i["split"] == "val"],
            [i for i in items if i["split"].startswith("test_")])


def vsuite(items, meta, want=("V1", "V2", "V3", "V4", "V5", "V5aug")):
    tr, va, te = splits(items)
    verbs = sorted(meta["verb_inventory"])
    referents = meta["humans"] + meta["objects"]
    out = {}
    if "V1" in want:
        out["V1_bag_of_embeddings"] = val.run_v1(tr, va, te)
    if "V2" in want:
        out["V2_ngram_logistic"] = val.run_v2(tr, va, te)
    if "V3" in want:
        out["V3_balance_audit"] = val.run_v3(items, verbs, referents)
    if "V4" in want:
        out["V4_dedup"] = val.run_v4(items, set(referents))
    if "V5" in want:
        out["V5_single_sentence_oracle"] = run_v5_strong(tr, va, te)
    if "V5aug" in want:
        r = val.run_v5(tr, va, te)
        r["note"] = ("the August order-blind mean-pooled oracle; DESCRIPTIVE "
                     "ONLY, non-gating -- it is bag-identical-blind and so "
                     "returns 0.5 on any dataset in this family")
        out["V5_august_orderblind_descriptive"] = r
    return out


# -------------------------------------------------------------- smoke battery
SMOKE = {
    "qleak":      {"must_fail": "V2", "want": ("V1", "V2", "V3")},
    "singlehop":  {"must_fail": "V5", "want": ("V2", "V5", "V5aug")},
    "nodistract": {"must_fail": "V5", "want": ("V2", "V5", "V5aug")},
}
KEY = {"V1": "V1_bag_of_embeddings", "V2": "V2_ngram_logistic",
       "V3": "V3_balance_audit", "V4": "V4_dedup",
       "V5": "V5_single_sentence_oracle"}


def run_smoke(seed):
    block = {
        "purpose": "doc 65 SS2.3 designed-to-fail builds; separate aggregation, "
                   "never mixed into the five verdict booleans; a build that "
                   "PASSES indicts the validator, not the generator",
        "builds": {},
    }
    all_fired = True
    for mode, cfg in SMOKE.items():
        path = SAB_FILES[mode]
        items, holdout = G.generate(seed, sabotage=mode)
        sha, _, _ = G.write(items, holdout, seed, path, None, sabotage=mode)
        its, meta, sha2 = load(path)
        assert sha == sha2
        res = vsuite(its, meta, want=cfg["want"])
        target = res[KEY[cfg["must_fail"]]]
        fired = not target["passed"]
        all_fired &= fired
        block["builds"][mode] = {
            "dataset_file": path, "dataset_sha256": sha,
            "n_items": len(its),
            "must_fail": cfg["must_fail"],
            "fired": bool(fired),
            "results": res,
        }
        print(f"[smoke/{mode}] {cfg['must_fail']} fired={fired}")
    block["all_fired"] = bool(all_fired)
    return block


# ------------------------------------------------------------------------ main
def main():
    items, meta, sha = load(DATA)
    hash_ok = sha == meta["dataset_sha256"]
    if not hash_ok:
        print("FATAL: dataset hash mismatch -- refusing to validate.")
        return 2
    tr, va, te = splits(items)
    per_split_n = Counter(i["split"] for i in te)
    assert all(i["hop_count"] >= 2 for i in te), "single-hop item leaked into test"
    for s, n in per_split_n.items():
        assert n >= MIN_TEST_PER_SPLIT, f"{s} has {n} items (< {MIN_TEST_PER_SPLIT})"

    print(f"[exp41r-val] n_train={len(tr)} n_val={len(va)} n_test={len(te)} "
          f"{dict(per_split_n)}")
    res = vsuite(items, meta)
    v1, v2 = res["V1_bag_of_embeddings"], res["V2_ngram_logistic"]
    v3, v4 = res["V3_balance_audit"], res["V4_dedup"]
    v5 = res["V5_single_sentence_oracle"]
    vsuite_passed = all(r["passed"] for r in (v1, v2, v3, v4, v5))

    print("[exp41r-val] running the designed-to-fail smoke battery ...")
    smoke = run_smoke(SEED)

    out = {
        "experiment": "exp41r_validation",
        "design_doc": SPEC,
        "timestamp": datetime.datetime.now().isoformat(),
        "python": sys.version.split()[0],
        "generator": "exp41_datagen.py",
        "generator_seed": meta["seed"],
        "vocabulary_source": meta["vocabulary_source"],
        "dataset_file": DATA,
        "dataset_sha256": sha,
        "dataset_hash_matches_meta": hash_ok,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_test_per_split": dict(per_split_n),
        "criteria": {
            "accuracy_bound": ACC_BOUND,
            "ci": "95% normal approx must contain 0.5",
            "chi2_alpha": ALPHA_P,
            "n_test_per_split_min": MIN_TEST_PER_SPLIT,
        },
        **res,
        "v1_passed": v1["passed"], "v2_passed": v2["passed"],
        "v3_passed": v3["passed"], "v4_passed": v4["passed"],
        "v5_passed": v5["passed"],
        "vsuite_all_passed": bool(vsuite_passed),
        "smoke_battery": smoke,
        "smoke_battery_all_fired": smoke["all_fired"],
        "ready_for_training": bool(vsuite_passed and smoke["all_fired"]),
    }

    path = RESULTS_BASE + ".json"
    suffix = ord("b")
    while os.path.exists(path):                      # research rule 3
        path = f"{RESULTS_BASE}_{chr(suffix)}.json"
        suffix += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    for p in SAB_FILES.values():                     # quarantined data, deleted
        if os.path.exists(p):
            os.remove(p)

    print("\n" + "=" * 70)
    print(f"{'exp41 revival -- generator validation':^70}")
    print("=" * 70)
    rows = [
        ("V1 bag-of-embeddings", f"acc={v1['test_accuracy']:.4f} "
         f"CI=[{v1['ci95'][0]:.3f},{v1['ci95'][1]:.3f}]", v1["passed"]),
        ("V2 1-2gram logistic", f"acc={v2['test_accuracy']:.4f} "
         f"CI=[{v2['ci95'][0]:.3f},{v2['ci95'][1]:.3f}]", v2["passed"]),
        ("V3 balance audit", f"50/50={v3['classes_balanced_50_50']} "
         f"min_p={v3['min_p']:.3f} (n={v3['n_tests']})", v3["passed"]),
        ("V4 dedup", f"verbatim={v4['n_verbatim_duplicates']} "
         f"near_groups={v4['n_near_dup_groups']}", v4["passed"]),
        ("V5 1-sentence oracle", f"max={v5['test_accuracy_max_aggregation']:.4f} "
         f"best={v5['test_accuracy_best_sentence']:.4f} "
         f"(mean={v5['descriptive_mean_aggregation']['test_accuracy']:.4f})",
         v5["passed"]),
    ]
    for name, detail, ok in rows:
        print(f"{name:22s} {detail:40s} {'PASS' if ok else 'FAIL'}")
    print("-" * 70)
    for mode, b in smoke["builds"].items():
        print(f"smoke {mode:12s} must fail {b['must_fail']:3s} "
              f"-> fired={b['fired']}")
    print("-" * 70)
    print(f"{'V-SUITE':22s} {'':40s} {'PASS' if vsuite_passed else 'FAIL'}")
    print(f"{'SMOKE BATTERY':22s} {'':40s} "
          f"{'PASS' if smoke['all_fired'] else 'FAIL'}")
    print(f"results -> {path}")
    print(f"dataset sha256 = {sha}")
    return 0 if out["ready_for_training"] else 1


if __name__ == "__main__":
    sys.exit(main())
