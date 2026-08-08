# -*- coding: utf-8 -*-
"""Exp 35: hardware export + exact verification for the argument-swap
plausibility demo (exp34a) on a real QPU.

Reuses exp34a's harness verbatim (exec of its source up to the scoring
loop): same frames, same LOO theta fits, same encoding.  For every
(orig, swap) sentence pair it:
  1. computes the HARDWARE-NATIVE score classically:
         p = P(s-wire = S_OUT | all post-selections pass)
     from the run_sf amplitudes (this differs from exp34a's align score
     only in normalization; the decision agreement is reported);
  2. builds a qiskit QuantumCircuit from the compiled op-list —
     subject state prep as Ry(2*atan2(v1,v0)), SWAPs as wire relabeling,
     every parameterized/fixed gate as an explicit UnitaryGate (immune to
     rotation-convention mismatches), cups as deferred end-of-circuit
     measurements with post-selection;
  3. verifies the circuit: qiskit Statevector conditional probability must
     match the run_sf value to ~1e-9 (any qubit-ordering error would show
     here);
  4. transpiles to {rz,sx,x,cx} for depth/2q-gate stats;
  5. saves measured circuits (QPY) + metadata JSON for submission.

No IBM account needed for this step.
"""
import json, math
import numpy as np

src34 = open("exp34a_swap_plausibility.py", encoding="utf-8").read()
head34 = src34[:src34.index('OUT = {"verbs"')]
g = {}
exec(compile(head34, "exp34a_head", "exec"), g)
USABLE, surface, subjects = g["USABLE"], g["surface"], g["subjects"]
pref_from, vec, enc = g["pref_from"], g["vec"], g["enc"]
fit, align, run_sf = g["fit"], g["align"], g["run_sf"]
ns, KETS, FORMS = g["ns"], g["KETS"], g["FORMS"]
S_OUT = g["S_OUT"]

from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


def hw_native_score(G, theta, subj_vec):
    """P(s = S_OUT | post-selections pass) from run_sf amplitudes."""
    w = G["w"].copy()
    for k, i in enumerate(G["solve_idx"]):
        w[i] = theta[k]
    st = run_sf(G, w, np.asarray(subj_vec, complex)).flatten()
    den = float(np.sum(np.abs(st) ** 2))
    return float(np.abs(st[S_OUT]) ** 2 / den) if den > 1e-30 else 0.5


def build_qiskit(G, theta, subj_vec):
    w = G["w"].copy()
    for k, i in enumerate(G["solve_idx"]):
        w[i] = theta[k]
    prog, info, swid = G["prog"], G["info"], G["swid"]
    skip = set(info[swid]["symops"])
    ket_oi = info[swid]["ket"]
    n_kets = sum(1 for oi, (kind, _, _) in enumerate(prog)
                 if kind == "ket" and oi not in skip)
    qc = QuantumCircuit(n_kets)
    axes, nextq, post = [], 0, []
    for oi, (kind, off, arg) in enumerate(prog):
        if oi in skip:
            continue
        if kind == "ket":
            q = nextq; nextq += 1
            axes.insert(off, q)
            if oi == ket_oi:
                a = math.atan2(float(np.real(subj_vec[1])),
                               float(np.real(subj_vec[0])))
                qc.ry(2 * a, q)
            elif int(arg) == 1:
                qc.x(q)
        elif kind == "bra":
            post.append((axes[off], int(arg)))
            axes.pop(off)
        elif kind == "scalar":
            pass
        elif kind == "H":
            qc.h(axes[off])
        elif kind == "SWAP":
            axes[off], axes[off + 1] = axes[off + 1], axes[off]
        elif kind == "fixed1":
            qc.append(UnitaryGate(np.asarray(ns["FIXED1"][arg], complex)),
                      [axes[off]])
        elif kind in ("Rx", "Rz"):
            qc.append(UnitaryGate(np.asarray(FORMS[kind](w[arg]), complex)),
                      [axes[off]])
        elif kind in ("CRz", "CRx"):
            qc.append(UnitaryGate(np.asarray(FORMS[kind](w[arg]), complex)),
                      [axes[off + 1], axes[off]])
        else:
            qc.append(UnitaryGate(np.asarray(ns["FIXED2"][arg], complex)),
                      [axes[off + 1], axes[off]])
    assert len(axes) == 1, f"expected one open wire, got {len(axes)}"
    return qc, axes[0], post


def sv_conditional(qc, s_q, post):
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    num = den = 0.0
    for idx, p in enumerate(probs):
        if any(((idx >> q) & 1) != out for q, out in post):
            continue
        den += p
        if ((idx >> s_q) & 1) == S_OUT:
            num += p
    return num / den if den > 1e-30 else 0.5


circuits, meta = [], []
n_agree = n_pairs = 0
tot_hw_correct = 0
max_dev = 0.0
depths, twoqs, accepts = [], [], []
for k, pairs in USABLE.items():
    v = surface[k]
    for i, p in enumerate(pairs):
        words = [pp["subj"] for j, pp in enumerate(pairs) if j != i]
        words += [w_ for w_ in subjects[k]
                  if w_ not in [pp["subj"] for pp in pairs]]
        tgt = pref_from(words)
        if tgt is None:
            continue
        fb = fit([pp["orig"] for j, pp in enumerate(pairs) if j != i], tgt)
        vs_, vo_ = vec(p["subj"]), vec(p["obj"])
        if vs_ is None or vo_ is None:
            continue
        entry = {"verb": v, "pair": i, "sent": p["sent"],
                 "theta": [float(x) for x in fb.x], "roles": {}}
        scores = {}
        for role, G, wvec in (("orig", p["orig"], enc(vs_)),
                              ("swap", p["swap"], enc(vo_))):
            p_native = hw_native_score(G, fb.x, wvec)
            qc, s_q, post = build_qiskit(G, fb.x, wvec)
            p_sv = sv_conditional(qc, s_q, post)
            dev = abs(p_native - p_sv)
            max_dev = max(max_dev, dev)
            # measured version for submission
            qm = qc.copy()
            mq = [q for q, _ in post] + [s_q]
            qm.add_register(__import__("qiskit").ClassicalRegister(len(mq)))
            for ci, q in enumerate(mq):
                qm.measure(q, ci)
            tq = transpile(qm, basis_gates=["rz", "sx", "x", "cx"],
                           optimization_level=3, seed_transpiler=7)
            depths.append(tq.depth())
            twoqs.append(sum(1 for inst in tq.data
                             if inst.operation.name == "cx"))
            # post-selection acceptance (for shot budgeting)
            sv = Statevector.from_instruction(qc)
            acc = sum(pr for idx, pr in enumerate(sv.probabilities())
                      if all(((idx >> q) & 1) == out for q, out in post))
            accepts.append(float(acc))
            circuits.append(qm)
            role_rec = {
                "circuit_index": len(circuits) - 1,
                "n_qubits": qc.num_qubits, "s_qubit": int(s_q),
                "post": [[int(q), int(o)] for q, o in post],
                "clbit_order": [int(q) for q in mq],
                "p_predicted": p_native, "sv_check_dev": float(dev),
                "acceptance": float(acc),
            }
            # basis-input circuits: align = P_v / (P_0 + P_1) on hardware,
            # global scalars cancel in the ratio
            basis_idx, basis_praw = [], []
            for bvec in (np.array([1, 0], complex), np.array([0, 1], complex)):
                qb, s_qb, post_b = build_qiskit(G, fb.x, bvec)
                svb = Statevector.from_instruction(qb)
                praw = sum(pr for idx2, pr in enumerate(svb.probabilities())
                           if all(((idx2 >> q) & 1) == out for q, out in post_b)
                           and ((idx2 >> s_qb) & 1) == S_OUT)
                qmb = qb.copy()
                mqb = [q for q, _ in post_b] + [s_qb]
                qmb.add_register(__import__("qiskit").ClassicalRegister(len(mqb)))
                for ci2, q2 in enumerate(mqb):
                    qmb.measure(q2, ci2)
                circuits.append(qmb)
                basis_idx.append(len(circuits) - 1)
                basis_praw.append(float(praw))
            svv = Statevector.from_instruction(qc)
            p_raw_v = sum(pr for idx2, pr in enumerate(svv.probabilities())
                          if all(((idx2 >> q) & 1) == out for q, out in post)
                          and ((idx2 >> s_qb) & 1 if False else (idx2 >> s_q) & 1) == S_OUT)
            role_rec["basis_circuit_indices"] = basis_idx
            role_rec["basis_p_raw"] = basis_praw
            role_rec["p_raw_v"] = float(p_raw_v)
            role_rec["align_ratio_pred"] = float(
                p_raw_v / (basis_praw[0] + basis_praw[1])
                if basis_praw[0] + basis_praw[1] > 1e-30 else 0.5)
            entry["roles"][role] = role_rec
            scores[role] = p_native
        hw_decision = scores["orig"] > scores["swap"]
        # exp34a decision used align-normalization; recompute for agreement
        a_o = align(p["orig"], fb.x, enc(vs_))
        a_s = align(p["swap"], fb.x, enc(vo_))
        entry["hw_native_correct"] = bool(hw_decision)
        entry["align_correct"] = bool(a_o > a_s)
        entry["align_margin"] = float(abs(a_o - a_s))
        ro, rs = entry["roles"]["orig"], entry["roles"]["swap"]
        entry["ratio_correct"] = bool(
            ro["align_ratio_pred"] > rs["align_ratio_pred"])
        entry["min_p_raw"] = float(min(
            ro["p_raw_v"], rs["p_raw_v"],
            *ro["basis_p_raw"], *rs["basis_p_raw"]))
        n_pairs += 1
        n_agree += int(hw_decision == (a_o > a_s))
        tot_hw_correct += int(hw_decision)
        meta.append(entry)
    print(f"[35] {v}: pairs exported", flush=True)

n_ratio = sum(1 for e in meta if e["ratio_correct"])
n_meas = sum(1 for e in meta if e["align_margin"] > 0.05
             and e["min_p_raw"] > 1e-3)
print(f"[35] ratio-score correct {n_ratio}/{n_pairs} (must match exp34a "
      f"align exactly) | measurable pairs (margin>0.05, p_raw>1e-3): "
      f"{n_meas}/{n_pairs}", flush=True)
print(f"[35] pairs={n_pairs} circuits={len(circuits)} | hw-native correct "
      f"{tot_hw_correct}/{n_pairs} | decision agreement with exp34a "
      f"{n_agree}/{n_pairs} | max statevector deviation {max_dev:.2e}",
      flush=True)
print(f"[35] transpiled: depth med={int(np.median(depths))} "
      f"max={max(depths)} | cx med={int(np.median(twoqs))} max={max(twoqs)} "
      f"| post-selection acceptance med={np.median(accepts):.3f} "
      f"min={min(accepts):.3f}", flush=True)
with open("exp35_circuits.qpy", "wb") as f:
    qpy.dump(circuits, f)
json.dump({"pairs": meta, "S_OUT": S_OUT,
           "n_pairs": n_pairs, "hw_native_correct": tot_hw_correct,
           "agreement_with_exp34a": n_agree},
          open("exp35_meta.json", "w"), indent=2, ensure_ascii=False)
print("[35] saved exp35_circuits.qpy + exp35_meta.json", flush=True)
print("[35] DONE", flush=True)
