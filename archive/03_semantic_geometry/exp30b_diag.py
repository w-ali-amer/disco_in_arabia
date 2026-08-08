# -*- coding: utf-8 -*-
"""Diagnose exp30b build failures: which stage kills each frame?"""
import json
import numpy as np
src = open("exp28a_real_gates.py", encoding="utf-8").read()
head = src[:src.index("# ── parse frames")]
ns = {}
exec(compile(head, "exp28a_head", "exec"), ns)
exp13, NumpyModel = ns["exp13"], ns["NumpyModel"]

data = json.load(open("sentences.json", encoding="utf-8"))
def hn(w): return w.replace("أ","ا").replace("إ","ا").replace("آ","ا")
tests = []
for split in ["WordSenseDisambiguation","WordSenseDisambiguation_v2"]:
    for d in data[split]:
        t = d["sentence"].split()
        lab = d["label"]
        wv = lab.split("_")[1]
        if len(t) >= 3 and hn(t[1]) == hn(wv) and hn(wv) in ("فتح","رفع","قطع"):
            tests.append((wv, " ".join([t[1], t[0]] + t[2:]), t[0]))
from collections import Counter
tests = tests[:36]
sents = [s for _, s, _ in tests]
diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)
dmap = dict(zip(sents, diagrams))
ansatz = exp13.make_ansatz(1, 1)
stage = Counter()
for wv, sent, subj in tests:
    d = dmap.get(sent)
    if d is None:
        stage[(wv, "parse_None")] += 1; continue
    try:
        c = ansatz(exp13._remove_cups(d))
    except Exception as e:
        stage[(wv, "ansatz_fail")] += 1; continue
    names = [str(s) for s in NumpyModel.from_diagrams([c], use_jit=False).symbols]
    vn = [nm for nm in names if sent.split()[0] in nm]
    if not any("s@n.l@n.l" in nm for nm in names):
        typ = vn[0].split("__")[-1] if vn else "?"
        stage[(wv, f"no3leg:{typ}")] += 1; continue
    sym_index = {nm: i for i, nm in enumerate(names)}
    prog = ns["compile_circuit"](c, sym_index)
    info, _ = ns["track_wires"](prog, names)
    try:
        ns["find_subject_wire"](prog, names, info, subj)
        stage[(wv, "OK")] += 1
    except AssertionError:
        stage[(wv, "subjwire_fail")] += 1
for k, v in sorted(stage.items()):
    print(k, v, flush=True)
