# -*- coding: utf-8 -*-
"""verify_demo_parity.py -- demo_engine_spec.md SS11 gates P / A / S.

  P  parity    300/300 identical argmax vs PyTorch; max |value diff| < 1e-6
  A  accuracy  JS test accuracy == the committed 0.7266666666666667
  S  sentinel  JS tie-the-verbs reads exactly 0.5000

Writes demo/parity_report.json. Any gate red => the demo does not ship.
"""
import json, subprocess, sys
import torch
import exp42_compiler as comp
from exp42_models import QuantumStoryModel

COMMITTED = 0.7266666666666667
TOL = 1e-6


def main():
    spec = json.load(open("demo/model.json", encoding="utf-8"))
    ts = json.load(open("demo/testset.json", encoding="utf-8"))
    verbs = spec["verbs"]
    sd = torch.load(spec["provenance"]["checkpoint"], map_location="cpu",
                    weights_only=True)
    m = QuantumStoryModel("A3", verbs, 0)
    m.load_state_dict(sd)

    js = json.loads(subprocess.run(["node", "demo/_parity_node.js"],
                                   capture_output=True, text=True,
                                   check=True).stdout)

    # ---- block matrices (the real port risk)
    mx_block = 0.0
    with torch.no_grad():
        for v in verbs + ["__q_pos__", "__q_neg__"]:
            M = (m.blocks[v].matrix() if v in m.blocks
                 else (m.q_pos if v == "__q_pos__" else m.q_neg).matrix())
            flat = []
            for r in range(4):
                for c in range(4):
                    flat += [float(M[r, c].real), float(M[r, c].imag)]
            mx_block = max(mx_block, max(abs(a - b)
                                         for a, b in zip(flat, js["blocks"][v])))

    # ---- per-item values and argmax
    items = [i for i in json.load(open("stories_exp51_VT.json",
                                       encoding="utf-8"))["items"]
             if i["split"] == "test"]
    base, _mm, _s = comp.load_embeddings()
    cs = comp.compile_all(items, base)
    mx_val, agree, correct = 0.0, 0, 0
    with torch.no_grad():
        for i, c in enumerate(cs):
            vals = m.story_values(c)
            py = [float(vals[0]), float(vals[1])]
            mx_val = max(mx_val, max(abs(py[k] - js["vals"][i][k])
                                     for k in (0, 1)))
            pa = int(torch.argmax(vals).item())
            agree += int(pa == js["argmax"][i])
            correct += int(pa == c.answer)

    gates = {
        "P_parity_argmax": "%d/%d" % (agree, len(cs)),
        "P_parity_passed": bool(agree == len(cs) and mx_val < TOL
                                and mx_block < TOL),
        "P_max_value_diff": mx_val,
        "P_max_block_matrix_diff": mx_block,
        "A_js_accuracy": js["acc"],
        "A_py_accuracy": correct / len(cs),
        "A_passed": bool(abs(js["acc"] - COMMITTED) < 1e-12),
        "S_tied_accuracy": js["tiedAcc"],
        "S_tied_verb": js["canon"],
        "S_passed": bool(abs(js["tiedAcc"] - 0.5) < 1e-12),
    }
    gates["ALL_PASSED"] = bool(gates["P_parity_passed"] and gates["A_passed"]
                               and gates["S_passed"])
    report = {"spec": "demo_engine_spec.md SS11",
              "committed_reference": COMMITTED, "tolerance": TOL,
              "n_items": len(cs), "gates": gates,
              "model_provenance": spec["provenance"]}
    json.dump(report, open("demo/parity_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for k, v in gates.items():
        print("  %-26s %s" % (k, v))
    return 0 if gates["ALL_PASSED"] else 1


if __name__ == "__main__":
    sys.exit(main())
