# -*- coding: utf-8 -*-
"""Exp 31: does sentence ORDER change the verdict?

Part A (real pipeline, consumed-token/register mode): each sentence gate
contributes per-branch scalar multipliers to the register -> scalars
commute -> pre-registered claim: final posterior and total p_post are
ORDER-INVARIANT over all 24 permutations of the 4 real gates (trajectories
may differ). Numerical proof or refutation.

Part B (stylized persistent open-wire gates, exp27 family — NON-conformal
directional filters): matrix products do not commute -> order should
change endpoints. Quantify the spread over all orders. If confirmed:
sentence-order sensitivity (taqdim/ta'khir as physics) requires
non-conformal open-wire gates — the CRx-class ansatz redesign.
"""
import os, json
from itertools import permutations
import numpy as np

S_OUT = int(os.environ.get("S_OUT", "0"))
src = open("exp28a_real_gates.py", encoding="utf-8").read()
head = src[:src.index("# ── parse frames")]
ns = {}
exec(compile(head, "exp28a_head", "exec"), ns)
exp13, NumpyModel = ns["exp13"], ns["NumpyModel"]
KETS, FORMS, h01 = ns["KETS"], ns["FORMS"], ns["h01"]
apply1, apply2 = ns["apply1"], ns["apply2"]
rx, rz = ns["rx"], ns["rz"]
euler = ns["euler"]

FRAMES_V = [("قرأ الطالب كتاب النحو", "الطالب"),
            ("حمل الولد حقيبة المدرسة", "الولد"),
            ("فتح المدير باب المكتب", "المدير"),
            ("كتب الطالب الدرس الجديد", "الطالب")]
diagrams = exp13.sentences_to_diagrams([s for s, _ in FRAMES_V],
                                       log_interval=999)
ansatz = exp13.make_ansatz(1, 1)

def run_sf(prog, names, w, swid, info, input_vec):
    H_M, FIXED1, FIXED2 = ns["H_M"], ns["FIXED1"], ns["FIXED2"]
    skip = set(info[swid]["symops"])
    st = np.array(1.0 + 0j)
    for oi, (kind, off, arg) in enumerate(prog):
        if oi in skip:
            continue
        if kind == "ket":
            vec = input_vec if oi == info[swid]["ket"] else KETS[arg]
            st = np.moveaxis(np.tensordot(st, vec, 0), -1, off)
        elif kind == "bra":
            st = np.take(st, arg, axis=off)
        elif kind == "scalar":
            st = st * arg
        elif kind == "H":
            st = apply1(st, H_M, off)
        elif kind == "fixed1":
            st = apply1(st, FIXED1[arg], off)
        elif kind == "SWAP":
            st = np.swapaxes(st, off, off + 1)
        elif kind in ("Rx", "Rz"):
            st = apply1(st, FORMS[kind](w[arg]), off)
        elif kind in ("CRz", "CRx"):
            st = apply2(st, FORMS[kind](w[arg]), off, off + 1)
        else:
            st = apply2(st, FIXED2[arg], off, off + 1)
    return np.asarray(st)

gates = []
for (sent, SUBJ), d in zip(FRAMES_V, diagrams):
    c = ansatz(exp13._remove_cups(d))
    model = NumpyModel.from_diagrams([c], use_jit=False)
    names = [str(s) for s in model.symbols]
    sym_index = {nm: i for i, nm in enumerate(names)}
    prog = ns["compile_circuit"](c, sym_index)
    info, _ = ns["track_wires"](prog, names)
    swid = ns["find_subject_wire"](prog, names, info, SUBJ)
    w = ns["weights_for"](names)
    W = np.eye(2, dtype=complex)
    for oi in sorted(info[swid]["symops"]):
        kind, off, arg = prog[oi]
        W = FORMS[kind](w[arg]) @ W
    mine = run_sf(prog, names, w, swid, info, W @ KETS[0]).flatten()
    syms = sorted(c.free_symbols, key=str)
    vals = [w[sym_index[str(s)]] for s in syms]
    ref = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
    f = abs(np.vdot(mine / np.linalg.norm(mine),
                    ref / np.linalg.norm(ref))) ** 2
    assert f > 1 - 1e-9
    cols = [run_sf(prog, names, w, swid, info, KETS[b]).flatten()
            for b in (0, 1)]
    gates.append(np.stack(cols, axis=1))
print(f"[31] {len(gates)} real gates rebuilt (closures passed)", flush=True)

def anchor(root, lex):
    return [2 * h01(f"{root}::{lex}|{i}") for i in range(3)]

wA = euler(anchor("ر.ج.ل", "man")) @ KETS[0]
wB = euler(anchor("ر.ج.ل", "leg")) @ KETS[0]

# Part A: register mode, all 24 orders
finals, pposts, trajs = [], [], []
for perm in permutations(range(4)):
    amp = np.array([np.sqrt(0.75), np.sqrt(0.25)], dtype=complex)  # [B, A]
    traj = [0.25]
    pp = 1.0
    for i in perm:
        fA = (gates[i] @ wA)[S_OUT]
        fB = (gates[i] @ wB)[S_OUT]
        amp = np.array([amp[0] * fB, amp[1] * fA])
        n2 = float(np.sum(np.abs(amp) ** 2))
        pp *= n2
        amp = amp / np.sqrt(n2)
        traj.append(float(abs(amp[1]) ** 2))
    finals.append(traj[-1])
    pposts.append(pp)
    trajs.append(traj)
finals, pposts = np.array(finals), np.array(pposts)
inv_final = float(finals.max() - finals.min())
inv_pp = float(pposts.max() - pposts.min())
mid_spread = float(np.std([t[2] for t in trajs]))
print(f"[31] A: final P(A) spread over 24 orders = {inv_final:.2e} "
      f"(ORDER-INVARIANT: {inv_final < 1e-12})", flush=True)
print(f"[31] A: total p_post spread = {inv_pp:.2e}", flush=True)
print(f"[31] A: mid-trajectory (after 2 sentences) std across orders = "
      f"{mid_spread:.4f} — the JOURNEY depends on order, the VERDICT does not",
      flush=True)

# Part B: stylized persistent open-wire non-conformal filters (exp27 family)
def styl_gate(axis, g):
    F = np.diag([1.0, np.cos(np.pi * g)]).astype(complex)
    return rx(-axis) @ F @ rx(axis)

filters = [styl_gate(2 * h01(f"v{i}|ax"), 0.30 + 0.15 * h01(f"v{i}|g"))
           for i in range(4)]
finB, ppB = [], []
for perm in permutations(range(4)):
    Mtot = np.eye(2, dtype=complex)
    for i in perm:
        Mtot = filters[i] @ Mtot
    a = Mtot @ (np.sqrt(0.75) * wB + 0j)
    b = Mtot @ (np.sqrt(0.25) * wA + 0j)
    # register-entangled token passing through the chain per branch
    sA = np.sqrt(0.25) * (Mtot @ wA)
    sB = np.sqrt(0.75) * (Mtot @ wB)
    nA, nB = np.linalg.norm(sA) ** 2, np.linalg.norm(sB) ** 2
    finB.append(nA / (nA + nB))
    ppB.append(nA + nB)
finB, ppB = np.array(finB), np.array(ppB)
print(f"[31] B (persistent non-conformal): final P(A) over 24 orders: "
      f"min={finB.min():.4f} max={finB.max():.4f} "
      f"spread={finB.max()-finB.min():.4f}", flush=True)
print(f"[31] B: p_post min={ppB.min():.4f} max={ppB.max():.4f}", flush=True)
best = int(np.argmax(finB))
worst = int(np.argmin(finB))
perms = list(permutations(range(4)))
print(f"[31] B: best order {perms[best]} worst {perms[worst]} — "
      f"sentence order is PHYSICAL once gates are non-conformal open-wire",
      flush=True)
json.dump({"A_final_spread": inv_final, "A_ppost_spread": inv_pp,
           "A_mid_traj_std": mid_spread,
           "B_final_min": float(finB.min()), "B_final_max": float(finB.max()),
           "B_best_order": perms[best], "B_worst_order": perms[worst]},
          open("results_exp31.json", "w"), indent=2)
print("[31] DONE", flush=True)
