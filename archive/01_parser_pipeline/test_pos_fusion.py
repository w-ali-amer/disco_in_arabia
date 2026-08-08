# -*- coding: utf-8 -*-
"""POS-fusion gate test: masdar rescue fires flag-on, nothing changes flag-off.

1. Rescue: reordered WSD sentences for فتح/رفع/قطع must produce a VSO diagram
   (verb box with type s·n.l·n.l) with ARABIC_POS_FUSION=1.
2. Regression: for a sweep across all published splits, flag-on diagrams must
   be byte-identical (repr) to flag-off diagrams whenever the flag-off parse
   already found a verb or the sentence is ال-initial nominal.
"""
import os, json
import arabic_dep_reader as adr

data = json.load(open("sentences.json", encoding="utf-8"))

def parse(sent):
    return adr.sentence_to_diagram(sent)

def is_vso(diag):
    r = repr(diag)
    return diag is not None and "Ty(s) @ Ty(n).l @ Ty(n).l" in r and "[SWAP;" in r

# ── 1. rescue on the diagnosed masdar failures ───────────────────────────
RESCUE = ["فتح الرجل الباب", "فتح الولد الصندوق", "رفع الطالب الملف",
          "رفع المدير الورقة", "قطع النجار الخشب", "قطع العامل الحبل"]
os.environ["ARABIC_POS_FUSION"] = "0"
off = {s: repr(parse(s)) for s in RESCUE}
os.environ["ARABIC_POS_FUSION"] = "1"
on = {s: repr(parse(s)) for s in RESCUE}
n_changed = sum(off[s] != on[s] for s in RESCUE)
n_vso = 0
os.environ["ARABIC_POS_FUSION"] = "1"
for s in RESCUE:
    d = parse(s)
    vso = is_vso(d)
    n_vso += vso
    print(f"[fusion-test] {s!r}: changed={off[s] != on[s]} vso_swap={vso}",
          flush=True)
print(f"[fusion-test] RESCUE: {n_changed}/{len(RESCUE)} changed, "
      f"{n_vso}/{len(RESCUE)} now VSO", flush=True)

# ── 2. regression sweep: published splits unchanged ──────────────────────
sweep = []
for split in ("WordOrderMatched", "WordOrder", "TenseBinary",
              "WordSenseDisambiguation_v2"):
    ds = data[split]
    sweep += [d["sentence"] for d in ds[:: max(1, len(ds) // 15)][:15]]
os.environ["ARABIC_POS_FUSION"] = "0"
off_s = [repr(parse(s)) for s in sweep]
os.environ["ARABIC_POS_FUSION"] = "1"
on_s = [repr(parse(s)) for s in sweep]
diffs = [s for s, a, b in zip(sweep, off_s, on_s) if a != b]
print(f"[fusion-test] REGRESSION: {len(diffs)}/{len(sweep)} diagrams changed "
      f"with flag on", flush=True)
for s in diffs:
    print(f"[fusion-test]   changed: {s!r}", flush=True)
print("[fusion-test] DONE", flush=True)
