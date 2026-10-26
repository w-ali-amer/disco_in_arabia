# -*- coding: utf-8 -*-
"""exp92_posthoc_identity.py -- POST-HOC, DESCRIPTIVE (not registered; reviewer 2026-09-26 ~21:45).

Mechanism check for doc 87 §13: Q-COMP scores BELOW chance on held-out derived cells (primary ANTI 0.468, CAUS 0.399).
Hypothesis: F learned no routing, so F^dag R F ~ R and the root's state lands on the BASE-form undergoer wire,
which is the wrong wire in every derived form. Metric: the registered routing_fidelity function (exp92_models), but
against IDENTITY routing (plain R) instead of SWAP R SWAP, with the same trained R. If identity-fidelity >> swap-fidelity
on held-out roots, "no routing learned; state delivered to the base-form position".
Writes results_exp92_posthoc_identity.json (Rule 3).
"""
import json, math, os, statistics, torch
import exp92_models as M
import exp92_common as C

OUT = "results_exp92_posthoc_identity.json"
assert not os.path.exists(OUT), "Rule 3"
g = torch.Generator().manual_seed(92990)
th = torch.rand(64, 2, generator=g) * math.pi
psi = (M.ry_state(th[:, 0])[:, :, None] * M.ry_state(th[:, 1])[:, None, :]).reshape(64, 4)


def fid_vs(model, root_ids, fi, target):
    out = []
    with torch.no_grad():
        Fm = model.F_mat(fi)
        for r in root_ids:
            Rm = model.blocks[r].matrix()
            A = (Fm.conj().T @ Rm @ Fm @ psi.T).T.reshape(-1, 2, 2)
            B = ((Rm if target == "identity" else M.SWAP @ Rm @ M.SWAP) @ psi.T).T.reshape(-1, 2, 2)
            out.append(statistics.mean(M._fid(M._rho1(A, w), M._rho1(B, w)).mean().item() for w in (0, 1)))
    return statistics.mean(out)


res = {}
for draw in ("primary", "confirm"):
    d = json.load(open("stories_exp92_%s.json" % draw, encoding="utf-8"))
    roots = d["roots"]; pidx = {r["pair"]: i for i, r in enumerate(roots)}; alt_of = {r["pair"]: r["alt"] for r in roots}
    for arm in ("Q-COMP", "Q-COMP-SG"):
        rows = {"VII": {"identity": [], "swap": []}, "IV": {"identity": [], "swap": []}}
        for s in range(8):
            u = json.load(open("exp92_ckpt/%s_%s_s%d.json" % (draw, arm, s), encoding="utf-8"))
            m = M.ARMS[arm](len(roots), len(C.NOUNS), len(roots), s)
            m.load_state_dict(torch.load("exp92_ckpt/%s_%s_s%d.pt" % (draw, arm, s), weights_only=True))
            for fi, fname, alt in ((1, "VII", "ANTI"), (2, "IV", "CAUS")):
                ids = [pidx[p] for p in u["held_out"] if alt_of[p] == alt]
                for tgt in ("identity", "swap"):
                    rows[fname][tgt].append(fid_vs(m, ids, fi, tgt))
        res["%s|%s" % (draw, arm)] = {f: {t: {"mean": statistics.mean(v), "seeds": v} for t, v in x.items()}
                                      for f, x in rows.items()}
json.dump({"label": "POST-HOC DESCRIPTIVE (not registered)", "results": res, "provenance": C.provenance()},
          open(OUT, "w"), indent=1, sort_keys=True)
for k, v in res.items():
    print(k, {f: {t: round(x[t]["mean"], 3) for t in x} for f, x in v.items()})
