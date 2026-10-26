# -*- coding: utf-8 -*-
"""verify_demo_parity_lab.py -- the BLOCKING gates for the exp69 language-lab
page, following verify_demo_parity_exp68.py's pattern (that file is committed
and is not modified; neither is verify_demo_parity.py).

Four gate families, ALL blocking:

  F  featuriser  Python (the committed exp65 closure) vs JS (the exported basis)
                 max |angle diff| < 1e-9 on all 24 inventory nouns AND on the
                 20 REGISTERED novel words, which include faail-pattern and
                 non-faail forms, one with tatweel and one with diacritics.
  P  parity      per arm, over its own 300 committed held-out items:
                 300/300 identical argmax vs PyTorch,
                 max |raw value diff| and max |P(yes) diff| < 1e-4,
                 max |block-matrix element diff| < 1e-4.
  A  accuracy    JS accuracy == the arm's committed per-seed held-out accuracy
                 (results_exp69.json arms[<arm>].per_seed[<selected>]), exactly.
  T  traces      the T = 2 extension of exp68's trace gate:
                 trace() reproduces storyValues() bit-for-bit;
                 every wire is PURE after the intro RYs;
                 every item is ENTANGLED after EACH event;
                 JS per-wire Tr rho^2 == torch's `exp69_run.forward_traced`
                 at every stage of every item (bar 1e-4, the same complex64
                 dtype residual as the values; measured near 1e-6);
                 doc 55 SS19.4's asymmetry: the readout-vs-post-event-1 purity
                 gap is SMALLER for q = 2 than for q = 1 at T = 2, and the
                 exact-undo subset (queried event is last AND wired (0,1), so
                 block-dagger cancels it exactly) restores purity on 100% of
                 its members.
  C  arc bound   the browser's client-side d = 3 arc calculus == the Python one,
                 exactly, for both angle families.

The torch side runs through the COMMITTED pipeline -- exp69_run.feature_maps /
build_model / arm_angles / compile_item / forward_traced -- so the reference is
the experiment's own code, never a re-implementation.

Writes demo/parity_report_lab.json. Any gate red => the page does not build.

NOTE ON THE q = 2 GATE. The inverse question block acts on wires (0, 1). A
queried event wired (1, 0) is NOT undone by it, so "exact undo" is asserted only
on the subset where the wirings coincide -- doc 55 SS20.5 reports a q = 2 mean
gap of 0.0268, not 0.0, and a gate that demanded zero everywhere would be
demanding something the committed record says is false.
"""
import json
import math
import subprocess
import sys

import torch

import exp65_featurizer as WZ
import exp41_story_generator as gen
import exp69_run as R
import instrument_sb_bounds as INS

TOL = 1e-4
TOL_PURITY = 1e-4
TOL_FEAT = 1e-9
RESULTS = "results_exp69.json"


def main():
    model = json.load(open("demo/model_exp69.json", encoding="utf-8"))
    ts = json.load(open("demo/testset_lab.json", encoding="utf-8"))
    fspec = json.load(open("demo/featurizer_lab.json", encoding="utf-8"))
    res = json.load(open(RESULTS, encoding="utf-8"))

    js = json.loads(subprocess.run(["node", "demo/_parity_node_lab.js"],
                                   capture_output=True, text=True,
                                   check=True).stdout)

    report = {
        "spec": ("demo_engine_spec.md SS11 gates P/A/Z, transposed to exp69 and "
                 "extended with the featuriser gate F, the T = 2 trace gate T "
                 "and the arc-calculus gate C "
                 "(qnlp_private_docs/55_exp69_reunion.md)"),
        "reference": ("PyTorch through the committed exp69 pipeline: "
                      "exp69_run.feature_maps / build_model / arm_angles / "
                      "compile_item / forward_traced, and "
                      "exp65_featurizer.fit_featurizer for the wazn angles"),
        "tolerances": {"values_and_probabilities": TOL, "purity": TOL_PURITY,
                       "featurizer_angles": TOL_FEAT},
        "tolerance_note": ("the reference model runs in complex64 (torch_sv_sim is "
                           "the simulator of record for exp42-exp69 and is not "
                           "re-typed for this gate); the page runs in float64, "
                           "so the residual is a dtype difference, not a port "
                           "difference. Purity is quadratic in the amplitudes, "
                           "so it carries the same residual and the same 1e-4 "
                           "bar; the MEASURED values are reported per arm and "
                           "sit near 1e-6. The featuriser is float64 on both "
                           "sides, which is why its bar is 1e-9."),
    }

    # ==================================================== gate F -- featuriser
    H = sorted(gen.HUMANS)
    closure = WZ.fit_featurizer(H)                # the COMMITTED code path
    inv_rows, mx_inv = [], 0.0
    for n in sorted(INS.HUMANS):
        py, j = float(closure(n)), float(js["featurizer"]["inventory"][n])
        mx_inv = max(mx_inv, abs(py - j))
        inv_rows.append({"word": n, "python": py, "js": j, "abs_diff": abs(py - j)})
    novel_rows, mx_nov = [], 0.0
    for w in fspec["novel_word_gate_list"]:
        jrow = js["featurizer"]["novel"][w["raw"]]
        if jrow["normalized"] != w["normalized"]:
            raise RuntimeError("input hygiene differs for %r: py %r js %r"
                               % (w["raw"], w["normalized"], jrow["normalized"]))
        py = float(closure(w["normalized"]))
        d = abs(py - float(jrow["angle"]))
        mx_nov = max(mx_nov, d)
        novel_rows.append({"word": w["raw"], "normalized": w["normalized"],
                           "stripped": w["stripped"], "gloss": w["gloss"],
                           "faail_pattern_by_eye": w["faail_pattern_by_eye"],
                           "python": py, "js": float(jrow["angle"]),
                           "abs_diff": d, "unseen_tokens": jrow["unseen_tokens"],
                           "n_tokens": jrow["n_tokens"],
                           "degenerate": jrow["degenerate"]})
    n_faail = sum(1 for r in novel_rows if r["faail_pattern_by_eye"])
    report["F_featurizer"] = {
        "vocab_size": js["featurizer"]["vocab_size"],
        "fit_protocol": fspec["fit_protocol"],
        "n_inventory_words": len(inv_rows),
        "max_abs_diff_inventory": mx_inv,
        "n_novel_words": len(novel_rows),
        "n_novel_faail_pattern": n_faail,
        "n_novel_not_faail": len(novel_rows) - n_faail,
        "max_abs_diff_novel": mx_nov,
        "inventory_rows": inv_rows,
        "novel_rows": novel_rows,
        "F_passed": bool(mx_inv < TOL_FEAT and mx_nov < TOL_FEAT
                         and len(novel_rows) == 20 and n_faail >= 5
                         and (len(novel_rows) - n_faail) >= 5),
    }
    print("  F  featuriser: inventory max|d| %.3e (%d words)   novel max|d| %.3e "
          "(%d words, %d faail / %d not)  %s"
          % (mx_inv, len(inv_rows), mx_nov, len(novel_rows), n_faail,
             len(novel_rows) - n_faail,
             "PASS" if report["F_featurizer"]["F_passed"] else "FAIL"))

    # ==================================================== gate C -- arc calculus
    arc_rows, arc_ok = {}, True
    for key, py in model["arc_calculus"].items():
        j = js["arcs"][key]
        ok = (j["best"] == py["best_correct"] and j["n"] == py["n_nouns"]
              and j["window_start"] == py["window_start_index_in_angle_order"]
              and j["window_length"] == py["window_length"])
        arc_ok = arc_ok and ok
        arc_rows[key] = {"python_best": py["best_correct"], "js_best": j["best"],
                         "n": py["n_nouns"],
                         "python_accuracy": py["best_accuracy"],
                         "js_accuracy": j["accuracy"],
                         "window_matches": bool(
                             j["window_start"] == py["window_start_index_in_angle_order"]
                             and j["window_length"] == py["window_length"]),
                         "identical": bool(ok)}
        print("  C  arc %-20s python %d/%d  js %d/%d  %s"
              % (key, py["best_correct"], py["n_nouns"], j["best"], j["n"],
                 "PASS" if ok else "FAIL"))
    report["C_arc_calculus"] = {"rows": arc_rows, "C_passed": bool(arc_ok),
                                "statement": model["arc_calculus"][
                                    list(model["arc_calculus"])[0]]["statement"]}

    # ============================================ gates P / A / T -- the circuits
    fm = R.feature_maps()
    store = {}
    for part, path in (("A", R.DATA_A), ("B", R.DATA_B)):
        store[part] = json.load(open(path, encoding="utf-8"))["items"]

    arms, all_ok = {}, True
    for arm in model["arm_order"]:
        spec = model["arms"][arm]
        key = spec["testset_key"]
        bundled = ts["sets"][key]["items"]
        test = [i for i in store[spec["part"]]
                if i["split"] == "test" and i["T"] == spec["T"]]
        if [i["id"] for i in test] != [i["id"] for i in bundled]:
            raise RuntimeError("%s: bundled testset is not the committed held-out "
                               "split, in order -- parity would not cover what "
                               "ships" % arm)

        seed = spec["selected_seed"]
        m = R.build_model(arm, sorted(model["verbs"]), seed, fm)
        m.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu",
                                     weights_only=True))
        bw = spec["n_qubits_per_block"]
        d = 1 << bw
        J = js["arms"][arm]
        if J["n_items"] != len(test):
            raise RuntimeError("%s: JS scored %d items, torch has %d"
                               % (arm, J["n_items"], len(test)))

        # ---- block matrices (the real port risk)
        mx_block = 0.0
        with torch.no_grad():
            for v in list(model["verbs"]) + ["__q_pos__", "__q_neg__"]:
                M = (m.blocks[v].matrix() if v in model["verbs"]
                     else (m.q_pos if v == "__q_pos__" else m.q_neg).matrix())
                flat = []
                for r in range(d):
                    for c in range(d):
                        flat += [float(M[r, c].real), float(M[r, c].imag)]
                mx_block = max(mx_block, max(abs(a - b)
                                             for a, b in zip(flat, J["blocks"][v])))

        # ---- per-item values, probabilities, argmax and PURITY TRACES
        ang, _t1 = R.arm_angles(arm, fm)
        cs = [R.compile_item(i, ang) for i in test]
        mx_val = mx_p = mx_pur = 0.0
        agree = correct = 0
        worst = None
        py_gaps = {1: [], 2: []}
        with torch.no_grad():
            for i, c in enumerate(cs):
                vals, tr = R.forward_traced(m, c)
                py = [float(vals[0]), float(vals[1])]
                dv = max(abs(py[k] - J["vals"][i][k]) for k in (0, 1))
                p_py = py[0] / (py[0] + py[1])
                dp = abs(p_py - J["pyes"][i])
                if dp > mx_p:
                    worst = {"item": c.item_id, "p_torch": p_py,
                             "p_js": J["pyes"][i], "abs_diff": dp}
                mx_val, mx_p = max(mx_val, dv), max(mx_p, dp)
                for stage, vv in tr.items():
                    jv = J["purity"][i][stage]
                    mx_pur = max(mx_pur, max(abs(a - b) for a, b in zip(vv, jv)))
                g = abs(sum(tr["at_readout"]) / len(tr["at_readout"])
                        - sum(tr["after_event1"]) / len(tr["after_event1"]))
                py_gaps[c.qidx].append(g)
                pa = int(torch.argmax(vals).item())
                agree += int(pa == J["argmax"][i])
                correct += int(pa == c.answer)

        blk = res["arms"][arm]
        committed = blk["per_seed"][spec["selected_seed_index"]]

        # ---- the T gates
        ent = J["trace_entangled_after_event"]
        q1m = J["q1_mean_abs_gap_vs_after_event1"]
        q2m = J["q2_mean_abs_gap_vs_after_event1"]
        # doc 55 SS19.4's asymmetry only exists at T = 2 (at T = 1 there is no
        # q = 2 cell at all), so the direction is only asserted where it applies.
        asym_applicable = bool(spec["T"] == 2 and J["q1_n"] and J["q2_n"])
        asym_ok = (q2m < q1m) if asym_applicable else True
        # torch must reproduce the same gap statistic
        mx_gap = 0.0
        for q in (1, 2):
            if not py_gaps[q]:
                continue
            pym = sum(py_gaps[q]) / len(py_gaps[q])
            jm = q1m if q == 1 else q2m
            if jm is not None:
                mx_gap = max(mx_gap, abs(pym - jm))

        t_passed = bool(
            J["trace_max_delta_vs_storyValues"] == 0.0
            and J["trace_intro_all_wires_pure"]
            and len(ent) == spec["T"]
            and all(e == len(test) for e in ent)
            and mx_pur < TOL_PURITY
            and mx_gap < TOL_PURITY
            and asym_ok
            and J["exact_undo_n"] > 0
            and J["exact_undo_restored"] == J["exact_undo_n"])

        g = {
            "T": spec["T"], "part": spec["part"], "encoding": spec["encoding"],
            "testset_key": key,
            "selected_seed": seed,
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "qubits_per_referent": spec["qubits_per_referent"],
            "circuit_wires": spec["wires"],
            "P_parity_argmax": "%d/%d" % (agree, len(cs)),
            "P_max_value_diff": mx_val,
            "P_max_prob_diff": mx_p,
            "P_worst_prob_item": worst,
            "P_max_block_matrix_diff": mx_block,
            "P_passed": bool(agree == len(cs) and mx_p < TOL and mx_val < TOL
                             and mx_block < TOL),
            "T_trace_max_delta_vs_storyValues":
                J["trace_max_delta_vs_storyValues"],
            "T_trace_intro_all_wires_pure": J["trace_intro_all_wires_pure"],
            "T_trace_entangled_after_event": ent,
            "T_n_items": len(test),
            "T_max_purity_diff_vs_torch": mx_pur,
            "T_max_gap_statistic_diff_vs_torch": mx_gap,
            "T_q1_n": J["q1_n"],
            "T_q1_mean_abs_gap": q1m,
            "T_q1_max_abs_gap": J["q1_max_abs_gap_vs_after_event1"],
            "T_q2_n": J["q2_n"],
            "T_q2_mean_abs_gap": q2m,
            "T_q2_max_abs_gap": J["q2_max_abs_gap_vs_after_event1"],
            "T_asymmetry_applicable": asym_applicable,
            "T_asymmetry_q2_gap_below_q1": bool(asym_ok),
            "T_exact_undo_subset_n": J["exact_undo_n"],
            "T_exact_undo_restored": J["exact_undo_restored"],
            "T_exact_undo_worst_purity_gap": J["exact_undo_worst_purity_gap"],
            "T_passed": t_passed,
            "A_js_accuracy": J["acc"],
            "A_py_accuracy": correct / len(cs),
            "A_committed_heldout_selected_seed": committed,
            "A_committed_mean_5seed": blk["mean"],
            "A_passed": bool(abs(J["acc"] - committed) < 1e-12
                             and abs(correct / len(cs) - committed) < 1e-12),
        }
        g["ALL_PASSED"] = bool(g["P_passed"] and g["A_passed"] and g["T_passed"])
        all_ok = all_ok and g["ALL_PASSED"]
        arms[arm] = g
        print("  %-5s T=%d %-2s  argmax %-8s  max|dp| %.2e  max|dblock| %.2e  "
              "max|dpurity| %.2e  acc %.10f (committed %.10f)  ent %s  "
              "undo %d/%d  %s"
              % (arm, spec["T"], spec["encoding"], g["P_parity_argmax"], mx_p,
                 mx_block, mx_pur, J["acc"], committed, ent,
                 J["exact_undo_restored"], J["exact_undo_n"],
                 "PASS" if g["ALL_PASSED"] else "FAIL"))

    report["n_items_per_arm"] = 300
    report["items_are"] = ("each arm's own full committed held-out split "
                           "(unseen PAIRINGS, doc 55 SS4.5)")
    report["datasets"] = {k: {"dataset": v["dataset"],
                              "dataset_sha256": v["dataset_sha256"],
                              "n_items": v["n_items"], "n_pairs": v["n_pairs"]}
                          for k, v in ts["sets"].items()}
    report["arms"] = arms
    report["ALL_PASSED"] = bool(all_ok and report["F_featurizer"]["F_passed"]
                                and report["C_arc_calculus"]["C_passed"])
    json.dump(report, open("demo/parity_report_lab.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("  ALL_PASSED %s" % report["ALL_PASSED"])
    return 0 if report["ALL_PASSED"] else 1


if __name__ == "__main__":
    sys.exit(main())
