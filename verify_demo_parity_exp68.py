# -*- coding: utf-8 -*-
"""verify_demo_parity_exp68.py -- the BLOCKING parity gate for the width-lever
page, following verify_demo_parity.py's pattern (that file is committed and is
not modified).

For each arm (F1, FA, F2), over ALL 300 bundled held-out test items:

  P  parity     300/300 identical argmax vs PyTorch
                max |raw value diff| and max |P(yes) diff| < 1e-4
                max |block-matrix element diff| < 1e-4
  A  accuracy   JS accuracy == the arm's committed per-seed held-out accuracy
                (results_exp68.json arms[<arm>].per_seed[<selected>]), exactly

The torch side runs through the COMMITTED pipeline -- exp68_run.feature_maps /
build_model / arm_angles and exp42_compiler.compile_story -- so the reference
is the experiment's own code, not a re-implementation.

Writes demo/parity_report_exp68.json. Any gate red => the page does not ship.
"""
import json
import subprocess
import sys

import torch

import exp42_compiler as comp
import exp68_run as E

TOL = 1e-4
DATA = "stories_exp68_WN.json"
RESULTS = "results_exp68.json"


def main():
    model = json.load(open("demo/model_exp68.json", encoding="utf-8"))
    ts = json.load(open("demo/testset_exp68.json", encoding="utf-8"))
    res = json.load(open(RESULTS, encoding="utf-8"))

    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("exp68 dataset provenance FAILED (self-hash)")
    if meta["dataset_sha256"] != ts["dataset_sha256"]:
        raise RuntimeError("bundled testset sha != dataset sha")
    verbs = sorted(meta["verb_inventory"].keys())
    test = [i for i in items if i["split"] == "test"]
    if [i["id"] for i in test] != [i["id"] for i in ts["items"]]:
        raise RuntimeError("bundled testset is not the committed held-out split, "
                           "in order -- parity would not cover what ships")

    fm = E.feature_maps()
    js = json.loads(subprocess.run(["node", "demo/_parity_node_exp68.js"],
                                   capture_output=True, text=True,
                                   check=True).stdout)
    if js["n_items"] != len(test):
        raise RuntimeError("JS scored %d items, torch has %d"
                           % (js["n_items"], len(test)))

    arms, all_ok = {}, True
    for arm in model["arm_order"]:
        spec = model["arms"][arm]
        seed = spec["selected_seed"]
        m = E.build_model(arm, verbs, seed, fm)
        m.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu",
                                     weights_only=True))
        bw = spec["n_qubits_per_block"]
        d = 1 << bw
        J = js["arms"][arm]

        # ---- block matrices (the real port risk)
        mx_block = 0.0
        with torch.no_grad():
            for v in verbs + ["__q_pos__", "__q_neg__"]:
                M = (m.blocks[v].matrix() if v in verbs
                     else (m.q_pos if v == "__q_pos__" else m.q_neg).matrix())
                flat = []
                for r in range(d):
                    for c in range(d):
                        flat += [float(M[r, c].real), float(M[r, c].imag)]
                mx_block = max(mx_block, max(abs(a - b)
                                             for a, b in zip(flat, J["blocks"][v])))

        # ---- per-item values, probabilities and argmax
        ang, _t1 = E.arm_angles(arm, fm)
        cs = comp.compile_all(test, ang)
        mx_val = mx_p = 0.0
        agree = correct = 0
        worst = None
        with torch.no_grad():
            for i, c in enumerate(cs):
                vals = m.story_values(c)
                py = [float(vals[0]), float(vals[1])]
                dv = max(abs(py[k] - J["vals"][i][k]) for k in (0, 1))
                p_py = py[0] / (py[0] + py[1])
                dp = abs(p_py - J["pyes"][i])
                if dp > mx_p:
                    worst = {"item": c.item_id, "p_torch": p_py,
                             "p_js": J["pyes"][i], "abs_diff": dp}
                mx_val, mx_p = max(mx_val, dv), max(mx_p, dp)
                pa = int(torch.argmax(vals).item())
                agree += int(pa == J["argmax"][i])
                correct += int(pa == c.answer)

        blk = res["arms"][arm]
        committed = blk["per_seed"][spec["selected_seed_index"]]
        g = {
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
            # the circuit diagram is driven by ArmModel.trace(); it must be the
            # SAME forward pass, not a second implementation of it
            "T_trace_max_delta_vs_storyValues":
                J["trace_max_delta_vs_storyValues"],
            "T_trace_intro_all_wires_pure": J["trace_intro_all_wires_pure"],
            "T_trace_items_entangled_after_verb":
                J["trace_items_entangled_after_verb"],
            "T_readout_product_when_order_matches":
                J["trace_readout_product_when_order_matches"],
            "T_readout_product_when_order_differs":
                J["trace_readout_product_when_order_differs"],
            "T_passed": bool(
                J["trace_max_delta_vs_storyValues"] == 0.0
                and J["trace_intro_all_wires_pure"]
                # the inverse-preparation mechanism, counted not claimed
                and (J["trace_readout_product_when_order_matches"][0]
                     == J["trace_readout_product_when_order_matches"][1])
                and J["trace_readout_product_when_order_differs"][0] == 0),
            "A_js_accuracy": J["acc"],
            "A_py_accuracy": correct / len(cs),
            "A_committed_heldout_selected_seed": committed,
            "A_passed": bool(abs(J["acc"] - committed) < 1e-12
                             and abs(correct / len(cs) - committed) < 1e-12),
        }
        g["ALL_PASSED"] = bool(g["P_passed"] and g["A_passed"]
                                and g["T_passed"])
        all_ok = all_ok and g["ALL_PASSED"]
        arms[arm] = g
        print("  %-2s  argmax %-8s  max|dp| %.3e  max|dblock| %.3e  "
              "acc %.10f (committed %.10f)  trace d=%.1e ent %d/%d  %s"
              % (arm, g["P_parity_argmax"], mx_p, mx_block, J["acc"], committed,
                 g["T_trace_max_delta_vs_storyValues"],
                 g["T_trace_items_entangled_after_verb"], len(cs),
                 "PASS" if g["ALL_PASSED"] else "FAIL"))

    report = {
        "spec": ("demo_engine_spec.md SS11 gate P/A, transposed to exp68 "
                 "(qnlp_private_docs/54_exp68_width_lever.md)"),
        "reference": ("PyTorch through the committed exp68 pipeline: "
                      "exp68_run.feature_maps / build_model / arm_angles + "
                      "exp42_compiler.compile_story"),
        "tolerance": TOL,
        "tolerance_note": ("the reference model runs in complex64; the page "
                           "runs in float64, so the residual is a dtype "
                           "difference, not a port difference"),
        "n_items": len(test),
        "items_are": "the full committed held-out split (unseen PAIRINGS)",
        "dataset_sha256": meta["dataset_sha256"],
        "arms": arms,
        "ALL_PASSED": bool(all_ok),
    }
    json.dump(report, open("demo/parity_report_exp68.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("  ALL_PASSED %s" % all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
