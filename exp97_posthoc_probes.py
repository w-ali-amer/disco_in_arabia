# -*- coding: utf-8 -*-
"""exp97_posthoc_probes.py -- POST-HOC DESCRIPTIVE (reviewer 2026-09-27 ~22:05): is R-8-LOCAL using x's wire?
swap-question: readout pointed at wire 1-x, labels unchanged (legitimate use of x -> accuracy collapses/flips).
scrambled-x: x's intro angle replaced by a fresh random angle (legitimate use -> accuracy drops).
Writes results_exp97_posthoc_probes.json (Rule 3)."""
import json, os, torch, statistics
import exp94_common as C, exp95_data as D, exp97_models as M

OUT = "results_exp97_posthoc_probes.json"
assert not os.path.exists(OUT)
d = json.load(open("stories_exp95_fresh.json", encoding="utf-8"))
enc = D.encode(d["test_oodl"], d["intro_angles"])
res = {}
for arm in ("R-8-LOCAL", "Q-INV", "R-INV", "R-8"):
    per = {"normal": [], "swap_x": [], "scramble_x": []}
    for s in range(8):
        m = M.ARMS[arm](s); m.load_state_dict(torch.load("exp97_ckpt/fresh_%s_s%d.pt" % (arm, s), weights_only=True)); m.eval()
        tot = {k: [0, 0] for k in per}
        g = torch.Generator().manual_seed(97000 + s)
        with torch.no_grad():
            for T, (b, idx) in enc.items():
                for mode in per:
                    bb = dict(b)
                    if mode == "swap_x":
                        bb["x"] = 1 - b["x"]
                    if mode == "scramble_x":
                        th = b["theta"].clone(); th[torch.arange(th.shape[0]), b["x"]] = torch.rand(th.shape[0], generator=g) * 3.14159
                        bb["theta"] = th
                    p, _ = m.predict(bb)
                    tot[mode][0] += int((p == b["label"]).sum()); tot[mode][1] += len(p)
        for k in per:
            per[k].append(tot[k][0] / tot[k][1])
    res[arm] = {k: {"mean": statistics.mean(v), "seeds": v} for k, v in per.items()}
    print(arm, {k: round(v["mean"], 3) for k, v in res[arm].items()})
C.write_new_json(OUT, {"label": "POST-HOC DESCRIPTIVE (not registered)", "results": res, "provenance": C.provenance()})
