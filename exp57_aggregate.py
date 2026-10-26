# -*- coding: utf-8 -*-
"""exp57_aggregate.py -- the missing aggregation for exp57 (doc 43), repaired.

Third instance of the results-without-aggregator gap-class (doc 49 SS11.1,
doc 53 SS7.3): exp57's 35 checkpoints (7 conditions x 5 seeds) were written
2026-08-19 and no results JSON ever was, so its registered verdicts lived only
in a working session. This script computes doc 43's frozen criteria from the
checkpoints and writes the results file of record.

doc 43 SS4 (frozen): rescue fraction rho = (acc - M-wide) / (M-tight - M-wide)
on seed means; readout_capacity_explains iff rho(M-wide-R8) >= 0.50;
optimisation_explains iff rho(M-wide-ROT) >= 0.50; both false =>
spread_mechanism_open = true. Instrument gate (doc 43, frozen): the Q1 verdicts
are read only if M-tight - M-wide >= 0.15 on seed means.
doc 43 SS5 (frozen): semantics_helps_when_conditioned iff C-AraVec > M-tight
AND C-AraVec > C-AraVec-Pw, both one-sided paired permutation p < 0.05.
"""
import glob, hashlib, json, os, platform, random, statistics as st

from exp42_run import fresh_results_path, git_note

DESIGN_DOC = "qnlp_private_docs/43_exp57_mechanism_and_compressed_aravec.md (FROZEN)"
CKPT = "exp57_ckpt"
CONDS = ["M-tight", "M-wide", "M-wide-R8", "M-wide-ROT", "M-tight-R8",
         "C-AraVec", "C-AraVec-Pw"]
GATE_BAR, RHO_BAR, P_BAR = 0.15, 0.50, 0.05
PERM_DRAWS, PERM_SEED = 20000, 57900


def load():
    rows = {}
    for f in sorted(glob.glob(os.path.join(CKPT, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        rows.setdefault(d["condition"], []).append(d)
    for c in CONDS:
        assert c in rows and len(rows[c]) == 5, "missing runs for %s" % c
        rows[c].sort(key=lambda r: r["seed"])
    return rows


def perm_p(diffs):
    """One-sided paired sign-flip permutation on per-item differences."""
    obs = sum(diffs) / len(diffs)
    rng = random.Random(PERM_SEED)
    ge = 0
    for _ in range(PERM_DRAWS):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if s / len(diffs) >= obs:
            ge += 1
    return (ge + 1) / (PERM_DRAWS + 1), obs


def item_rates(runs):
    n = len(runs[0]["test_correct"])
    assert all(len(r["test_correct"]) == n for r in runs)
    return [sum(r["test_correct"][i] for r in runs) / len(runs) for i in range(n)]


def main():
    rows = load()
    per = {}
    for c in CONDS:
        accs = [r["test_acc"] for r in rows[c]]
        per[c] = {"mean": st.mean(accs), "sd": st.stdev(accs), "per_seed": accs,
                  "gate_met": ["%s" % r["gate_met"] for r in rows[c]]}
    mt, mw = per["M-tight"]["mean"], per["M-wide"]["mean"]
    gate = mt - mw
    gate_ok = bool(gate >= GATE_BAR)
    rho = {c: (per[c]["mean"] - mw) / (mt - mw) for c in ("M-wide-R8", "M-wide-ROT")}
    readout = bool(gate_ok and rho["M-wide-R8"] >= RHO_BAR)
    optim = bool(gate_ok and rho["M-wide-ROT"] >= RHO_BAR)
    open_ = bool(gate_ok and not readout and not optim)

    ca, mt_r, pw = (item_rates(rows[c]) for c in ("C-AraVec", "M-tight", "C-AraVec-Pw"))
    p1, d1 = perm_p([a - b for a, b in zip(ca, mt_r)])
    p2, d2 = perm_p([a - b for a, b in zip(ca, pw)])
    q2 = bool(p1 < P_BAR and p2 < P_BAR)

    out = {
        "experiment": "exp57", "spec": DESIGN_DOC,
        "aggregated_from": "%s (35 checkpoints, 7 conditions x 5 seeds, written 2026-08-19)" % CKPT,
        "platform": platform.platform(), "python": platform.python_version(),
        "per_condition": per,
        "instrument_gate": {"M_tight_minus_M_wide": gate, "bar": GATE_BAR, "passed": gate_ok},
        "rescue_fractions": rho,
        "q2_contrasts": {
            "C_AraVec_vs_M_tight": {"mean_paired_diff": d1, "one_sided_p": p1},
            "C_AraVec_vs_C_AraVec_Pw": {"mean_paired_diff": d2, "one_sided_p": p2},
            "method": "per-item correctness averaged over 5 seeds per condition; "
                      "one-sided paired sign-flip permutation, %d draws, seed %d"
                      % (PERM_DRAWS, PERM_SEED)},
        "verdicts": {"readout_capacity_explains": readout,
                     "optimisation_explains": optim,
                     "spread_mechanism_open": open_,
                     "semantics_helps_when_conditioned": q2},
        "meta_notes": {
            "repair": "results file written 2026-08-25 from checkpoints untouched since "
                      "2026-08-19; the third instance of the results-without-aggregator "
                      "gap-class (doc 49 SS11.1, doc 53 SS7.3). Criteria computed from "
                      "doc 43's frozen bars, not asserted.",
            "reading": "doc 43 SS6's consequence table applies to whichever verdicts hold."},
        "git_note": git_note(),
    }
    p = fresh_results_path("results_exp57")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for c in CONDS:
        print("%-12s %.4f +- %.4f" % (c, per[c]["mean"], per[c]["sd"]))
    print("gate %.4f (>=%.2f: %s) | rho R8 %.3f ROT %.3f" % (gate, GATE_BAR, gate_ok, rho["M-wide-R8"], rho["M-wide-ROT"]))
    print("Q2: dCA-Mt %.4f p=%.4f | dCA-Pw %.4f p=%.4f" % (d1, p1, d2, p2))
    print(json.dumps(out["verdicts"]))
    print("wrote", p)


if __name__ == "__main__":
    main()
