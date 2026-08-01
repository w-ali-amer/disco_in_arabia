#!/usr/bin/env python3
"""Repair the WordOrderMatched pair_id column in sentences.json.

Discovered 2026-08-01 (see ERRATUM.md): the pair_id column did not link the
matched SVO/VSO twins. The construction claim itself holds — every one of the
120 sentences has a word-multiset twin carrying the opposite label (60 true
pairs) — only the bookkeeping column was wrong.

This script rebuilds pair_id by grouping sentences on their sorted word
multiset, preserves the old values as pair_id_legacy, verifies the result,
and flags the two multisets that contain verbatim duplicate pairs (4
duplicated sentences; duplicates are paired by order of appearance).

No published experiment used pair_id (exp13 stratifies on labels only), so
no reported number is affected by this repair.
"""
import json
from collections import defaultdict

PATH = "sentences.json"

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)
entries = data["WordOrderMatched"]

groups = defaultdict(list)
for i, e in enumerate(entries):
    groups[tuple(sorted(e["sentence"].split()))].append(i)

pid = 0
for idxs in sorted(groups.values(), key=lambda g: g[0]):
    svo = [i for i in idxs if entries[i]["label"].endswith("SVO")]
    vso = [i for i in idxs if entries[i]["label"].endswith("VSO")]
    assert len(svo) == len(vso), f"unbalanced multiset group: {idxs}"
    for a, b in zip(svo, vso):
        for i in (a, b):
            entries[i]["pair_id_legacy"] = entries[i]["pair_id"]
            entries[i]["pair_id"] = pid
        pid += 1

# --- verification ---
by_pid = defaultdict(list)
for e in entries:
    by_pid[e["pair_id"]].append(e)
assert len(by_pid) == 60, f"expected 60 pairs, got {len(by_pid)}"
multiset_count = defaultdict(int)
for p in sorted(by_pid):
    x, y = by_pid[p]
    ms = tuple(sorted(x["sentence"].split()))
    assert ms == tuple(sorted(y["sentence"].split()))
    assert {x["label"], y["label"]} == {"WordOrder_SVO", "WordOrder_VSO"}
    multiset_count[ms] += 1
n_dup = 0
for ms, c in multiset_count.items():
    if c > 1:
        n_dup += c - 1
        print(f"duplicate multiset (x{c}, paired by appearance):", " ".join(ms))
print(f"OK: 60 pairs, each one SVO + one VSO with identical word multiset; "
      f"{n_dup} duplicated pair(s) flagged")

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
print("sentences.json rewritten (pair_id repaired, pair_id_legacy preserved)")
