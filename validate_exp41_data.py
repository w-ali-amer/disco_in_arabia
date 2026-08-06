#!/usr/bin/env python3
"""
exp41 — Pre-registered generator validation (doc 22 §2.3, criteria FROZEN).

V1  trainable bag-of-embeddings classifier (torch: embedding, mean-pool, MLP)
    on the QA task: pass = test accuracy <= 0.55 AND 95% CI contains 0.5,
    N >= 500 items.
V2  surface n-gram (1-2gram) logistic regression (sklearn): same bound.
V3  balance audit: answer classes 50/50; chi-square of verb/referent/template
    marginals vs answer class all non-significant (p > 0.05).
V4  dedup: zero verbatim duplicate stories; near-dup canonical-form report.
V5  single-sentence oracle: bag-of-embeddings model trained on (one sentence +
    question) pairs; item prediction = best (most confident) sentence, also
    reported with the max-P(yes) aggregation; both must be at chance on the
    multi-hop test set (same numeric bound as V1).

Verdict booleans are written to results_exp41_validation.json. Existing results
files are NEVER overwritten (research rule 3): if the file exists, the next
free suffix (_b, _c, ...) is used.
"""

import datetime
import hashlib
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 41
DATA = "stories_exp41.json"
RESULTS_BASE = "results_exp41_validation"
ACC_BOUND = 0.55
ALPHA_P = 0.05


def story_text(item):
    return " . ".join(item["sentences"])


def full_text(item):
    return story_text(item) + " " + item["question"]["surface"]


def label(item):
    return 1 if item["answer"] == "yes" else 0


def ci95(acc, n):
    half = 1.96 * math.sqrt(max(acc * (1 - acc), 1e-12) / n)
    return [acc - half, acc + half]


def chance_bound(acc, n):
    lo, hi = ci95(acc, n)
    return bool(acc <= ACC_BOUND and lo <= 0.5 <= hi)


# ---------------------------------------------------------------- V1 / V5 model
class BagModel(nn.Module):
    def __init__(self, vocab, d=64, h=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, idx):
        mask = (idx != 0).float().unsqueeze(-1)
        pooled = (self.emb(idx) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.mlp(pooled)


def build_vocab(token_lists):
    vocab = {"<pad>": 0, "<unk>": 1}
    for toks in token_lists:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def encode(token_lists, vocab):
    L = max(len(t) for t in token_lists)
    arr = np.zeros((len(token_lists), L), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        for j, t in enumerate(toks):
            arr[i, j] = vocab.get(t, 1)
    return torch.from_numpy(arr)


def train_bag_model(train_toks, train_y, val_toks, val_y, epochs=40, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    vocab = build_vocab(train_toks)
    Xtr, Xva = encode(train_toks, vocab), encode(val_toks, vocab)
    ytr = torch.tensor(train_y)
    yva = torch.tensor(val_y)
    model = BagModel(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    best_va, best_state = -1.0, None
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = (model(Xva).argmax(1) == yva).float().mean().item()
        if va > best_va:
            best_va = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, vocab, best_va


def eval_bag_model(model, vocab, toks, y):
    with torch.no_grad():
        pred = model(encode(toks, vocab)).argmax(1).numpy()
    return float((pred == np.array(y)).mean())


# ------------------------------------------------------------------------- V1
def run_v1(train, val, test):
    tr = [full_text(i).split() for i in train]
    va = [full_text(i).split() for i in val]
    te = [full_text(i).split() for i in test]
    model, vocab, best_va = train_bag_model(
        tr, [label(i) for i in train], va, [label(i) for i in val])
    acc = eval_bag_model(model, vocab, te, [label(i) for i in test])
    per_split = {}
    for s in ("test_a", "test_b", "test_c"):
        sub = [i for i in test if i["split"] == s]
        per_split[s] = eval_bag_model(model, vocab,
                                      [full_text(i).split() for i in sub],
                                      [label(i) for i in sub])
    train_acc = eval_bag_model(model, vocab, tr, [label(i) for i in train])
    return {
        "test_accuracy": acc, "ci95": ci95(acc, len(test)), "n_test": len(test),
        "train_accuracy": train_acc, "best_val_accuracy": best_va,
        "per_split_accuracy": per_split,
        "passed": chance_bound(acc, len(test)),
    }


# ------------------------------------------------------------------------- V2
def run_v2(train, val, test):
    vec = CountVectorizer(token_pattern=r"[^\s]+", ngram_range=(1, 2))
    Xtr = vec.fit_transform([full_text(i) for i in train])
    Xva = vec.transform([full_text(i) for i in val])
    Xte = vec.transform([full_text(i) for i in test])
    ytr = [label(i) for i in train]
    yva = np.array([label(i) for i in val])
    yte = np.array([label(i) for i in test])
    best = (-1, None, None)
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xtr, ytr)
        va = float((clf.predict(Xva) == yva).mean())
        if va > best[0]:
            best = (va, C, clf)
    va, C, clf = best
    acc = float((clf.predict(Xte) == yte).mean())
    per_split = {}
    for s in ("test_a", "test_b", "test_c"):
        m = np.array([i["split"] == s for i in test])
        per_split[s] = float((clf.predict(Xte[m]) == yte[m]).mean())
    return {
        "test_accuracy": acc, "ci95": ci95(acc, len(test)), "n_test": len(test),
        "selected_C": C, "best_val_accuracy": va,
        "n_features": Xtr.shape[1], "per_split_accuracy": per_split,
        "passed": chance_bound(acc, len(test)),
    }


# ------------------------------------------------------------------------- V3
def marginal_chi2(items, feature_fn, values):
    """2xk chi-square of feature presence vs answer class; returns min p."""
    out = {}
    for v in values:
        table = np.zeros((2, 2))
        for it in items:
            present = 1 if v in feature_fn(it) else 0
            table[label(it), present] += 1
        if (table.sum(0) == 0).any() or (table.sum(1) == 0).any():
            out[str(v)] = 1.0  # degenerate margin: no association measurable
            continue
        try:
            _, p, _, _ = chi2_contingency(table)
        except ValueError:
            p = 1.0
        out[str(v)] = float(p)
    return out


def run_v3(items, verbs, referents):
    yes = sum(label(i) for i in items)
    no = len(items) - yes
    balance_ok = yes == no

    verb_p = marginal_chi2(items, lambda i: {e["verb"] for e in i["events"]},
                           verbs)
    ref_p = marginal_chi2(
        items, lambda i: ({e["agent"] for e in i["events"]}
                          | {e["patient"] for e in i["events"]}), referents)
    templ_p = marginal_chi2(items, lambda i: {i["qtype"]}, ["Q0", "Q1", "Q2"])
    pro_p = marginal_chi2(items, lambda i: {i["pro_drop_item"]}, [True, False])

    all_p = list(verb_p.values()) + list(ref_p.values()) + \
        list(templ_p.values()) + list(pro_p.values())
    min_p = min(all_p)
    return {
        "n_yes": yes, "n_no": no, "classes_balanced_50_50": balance_ok,
        "min_p": float(min_p),
        "n_tests": len(all_p),
        "verb_min_p": float(min(verb_p.values())),
        "referent_min_p": float(min(ref_p.values())),
        "template_min_p": float(min(templ_p.values())),
        "pro_drop_min_p": float(min(pro_p.values())),
        "passed": bool(balance_ok and min_p > ALPHA_P),
    }


# ------------------------------------------------------------------------- V4
def canonical_key(item, name_set):
    order = []

    def idx(name):
        if name not in order:
            order.append(name)
        return f"R{order.index(name)}"

    ev = [(e["verb"], idx(e["agent"]), idx(e["patient"]), e["pro_drop"])
          for e in item["events"]]
    q = item["question"]
    qc = [q["type"]] + [idx(t) if t in name_set else t
                        for k, t in sorted(q.items())
                        if k not in ("type", "surface")]
    return json.dumps({"ev": ev, "q": qc, "a": item["answer"]},
                      ensure_ascii=False, sort_keys=True)


def run_v4(items, name_set):
    verbatim = Counter(full_text(i) for i in items)
    n_verbatim_dups = sum(c - 1 for c in verbatim.values() if c > 1)

    canon = {}
    for it in items:
        canon.setdefault(canonical_key(it, name_set), []).append(it)
    near_groups = []
    for k, group in canon.items():
        if len(group) > 1:
            pids = sorted({g["pair_id"] for g in group})
            near_groups.append({"n_items": len(group), "pair_ids": pids,
                                "ids": sorted(g["id"] for g in group)})
    return {
        "n_verbatim_duplicates": n_verbatim_dups,
        "n_near_dup_groups": len(near_groups),
        "near_dup_groups_sample": near_groups[:20],
        "passed": n_verbatim_dups == 0,
    }


# ------------------------------------------------------------------------- V5
def run_v5(train, val, test):
    def expand(items):
        toks, ys, owner = [], [], []
        for n, it in enumerate(items):
            q = it["question"]["surface"]
            for s in it["sentences"]:
                toks.append((s + " " + q).split())
                ys.append(label(it))
                owner.append(n)
        return toks, ys, owner

    tr_t, tr_y, _ = expand(train)
    va_t, va_y, va_o = expand(val)

    # epoch selection on item-level (best-sentence) val accuracy would need a
    # custom loop; sentence-level val accuracy is a fine (and generous) proxy.
    model, vocab, best_va = train_bag_model(tr_t, tr_y, va_t, va_y)

    def item_scores(items):
        preds_conf, preds_maxyes = [], []
        with torch.no_grad():
            for it in items:
                q = it["question"]["surface"]
                toks = [(s + " " + q).split() for s in it["sentences"]]
                p = torch.softmax(model(encode(toks, vocab)), dim=1)[:, 1].numpy()
                conf_best = p[np.argmax(np.abs(p - 0.5))]
                preds_conf.append(1 if conf_best > 0.5 else 0)
                preds_maxyes.append(1 if p.max() > 0.5 else 0)
        return np.array(preds_conf), np.array(preds_maxyes)

    yte = np.array([label(i) for i in test])
    pc, pm = item_scores(test)
    acc_conf = float((pc == yte).mean())
    acc_max = float((pm == yte).mean())
    return {
        "test_accuracy_best_sentence": acc_conf,
        "ci95_best_sentence": ci95(acc_conf, len(test)),
        "test_accuracy_max_yes": acc_max,
        "ci95_max_yes": ci95(acc_max, len(test)),
        "n_test": len(test),
        "sentence_level_val_accuracy": best_va,
        "passed": bool(chance_bound(acc_conf, len(test))
                       and chance_bound(acc_max, len(test))),
    }


# ------------------------------------------------------------------------ main
def main():
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]

    payload = json.dumps(items, ensure_ascii=False, sort_keys=True)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    hash_ok = sha == meta["dataset_sha256"]
    if not hash_ok:
        print("FATAL: dataset hash mismatch — refusing to validate.")

    train = [i for i in items if i["split"] == "train"]
    val = [i for i in items if i["split"] == "val"]
    test = [i for i in items if i["split"].startswith("test_")]
    assert all(i["hop_count"] >= 2 for i in test), \
        "single-hop item leaked into test"
    assert len(test) >= 500, "test set below pre-registered minimum"

    verbs = sorted(meta["verb_inventory"].keys())
    referents = meta["humans"] + meta["objects"]
    name_set = set(referents)

    print(f"[exp41-val] n_train={len(train)} n_val={len(val)} n_test={len(test)}")
    print("[exp41-val] running V1 (bag-of-embeddings, torch) ...")
    v1 = run_v1(train, val, test) if hash_ok else {"passed": False}
    print("[exp41-val] running V2 (1-2gram logistic regression) ...")
    v2 = run_v2(train, val, test) if hash_ok else {"passed": False}
    print("[exp41-val] running V3 (balance audit) ...")
    v3 = run_v3(items, verbs, referents) if hash_ok else {"passed": False}
    print("[exp41-val] running V4 (dedup) ...")
    v4 = run_v4(items, name_set) if hash_ok else {"passed": False}
    print("[exp41-val] running V5 (single-sentence oracle) ...")
    v5 = run_v5(train, val, test) if hash_ok else {"passed": False}

    all_passed = bool(hash_ok and all(v["passed"] for v in (v1, v2, v3, v4, v5)))
    results = {
        "experiment": "exp41_validation",
        "design_doc": "qnlp_private_docs/22_phase2_trained_story_qa_design.md §2.3 (commit cacac31)",
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset_file": DATA,
        "dataset_sha256": sha,
        "dataset_hash_matches_meta": hash_ok,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "criteria": {
            "accuracy_bound": ACC_BOUND,
            "ci": "95% normal approx must contain 0.5",
            "chi2_alpha": ALPHA_P,
        },
        "V1_bag_of_embeddings": v1,
        "V2_ngram_logistic": v2,
        "V3_balance_audit": v3,
        "V4_dedup": v4,
        "V5_single_sentence_oracle": v5,
        "v1_passed": v1["passed"], "v2_passed": v2["passed"],
        "v3_passed": v3["passed"], "v4_passed": v4["passed"],
        "v5_passed": v5["passed"],
        "all_passed": all_passed,
    }

    out = RESULTS_BASE + ".json"
    suffix = ord("b")
    while os.path.exists(out):  # research rule 3: never overwrite
        out = f"{RESULTS_BASE}_{chr(suffix)}.json"
        suffix += 1
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 66)
    print(f"{'exp41 generator validation':^66}")
    print("=" * 66)
    rows = [
        ("V1 bag-of-embeddings", f"acc={v1['test_accuracy']:.4f} "
         f"CI=[{v1['ci95'][0]:.3f},{v1['ci95'][1]:.3f}]", v1["passed"]),
        ("V2 1-2gram logistic", f"acc={v2['test_accuracy']:.4f} "
         f"CI=[{v2['ci95'][0]:.3f},{v2['ci95'][1]:.3f}]", v2["passed"]),
        ("V3 balance audit", f"50/50={v3['classes_balanced_50_50']} "
         f"min_p={v3['min_p']:.3f}", v3["passed"]),
        ("V4 dedup", f"verbatim={v4['n_verbatim_duplicates']} "
         f"near_groups={v4['n_near_dup_groups']}", v4["passed"]),
        ("V5 1-sentence oracle", f"acc={v5['test_accuracy_best_sentence']:.4f}"
         f"/{v5['test_accuracy_max_yes']:.4f}", v5["passed"]),
    ]
    for name, detail, ok in rows:
        print(f"{name:24s} {detail:32s} {'PASS' if ok else 'FAIL'}")
    print("-" * 66)
    print(f"{'ALL CRITERIA':24s} {'':32s} {'PASS' if all_passed else 'FAIL'}")
    print(f"results -> {out}")
    print(f"dataset sha256 = {sha}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
