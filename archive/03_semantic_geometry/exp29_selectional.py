# -*- coding: utf-8 -*-
"""Exp 29: calibrated selectional semantics on real compiled gates.

Question: do real-pipeline sentence gates have SYSTEMATIC, TRANSFERABLE
selectional behavior? Method: measure each gate's passing direction (top
input singular vector); CALIBRATE the animacy pole on ONE verb only
(qara'a); test — untouched — on the remaining three gates.

Pre-registered:
  A1: gates share a passing direction (report pairwise |<v_i,v_j>|^2;
      transfer requires mean alignment >> 0.5)
  A2: 3/3 uncalibrated verbs select the animate pole (sel > 0)
  A3: null prediction — 'ayn (eye/spring, no animacy contrast, placed on
      the equator) shows mean |sel| < half the animate-contrast mean
  A4: 3-sentence real-text accumulation with calibrated placement drives
      PB from a hostile 0.75 prior across 0.5
If A1 fails (idiosyncratic directions), calibration cannot transfer and
A2 fails too — that outcome is reported as the finding.
"""
import os, json
import numpy as np

S_OUT = int(os.environ.get("S_OUT", "0"))
src = open("exp28a_real_gates.py", encoding="utf-8").read()
head = src[:src.index("# ── parse frames")]
ns = {}
exec(compile(head, "exp28a_head", "exec"), ns)
exp13, NumpyModel = ns["exp13"], ns["NumpyModel"]
KETS, FORMS = ns["KETS"], ns["FORMS"]
apply1, apply2 = ns["apply1"], ns["apply2"]

FRAMES_V = [("قرأ الطالب كتاب النحو", "الطالب"),
            ("حمل الولد حقيبة المدرسة", "الولد"),
            ("فتح المدير باب المكتب", "المدير"),
            ("كتب الطالب الدرس الجديد", "الطالب")]
CAL_FRAME = 0  # calibrate on qara'a ONLY

diagrams = exp13.sentences_to_diagrams([s for s, _ in FRAMES_V],
                                       log_interval=999)
ansatz = exp13.make_ansatz(1, 1)

def run_state_form(prog, names, w, swid, info, input_vec):
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

gates, tags = [], []
for (sent, SUBJ), d in zip(FRAMES_V, diagrams):
    tag = sent.split()[0]
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
    wstate = W @ KETS[0]
    mine = run_state_form(prog, names, w, swid, info, wstate).flatten()
    syms = sorted(c.free_symbols, key=str)
    vals = [w[sym_index[str(s)]] for s in syms]
    ref = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
    f = abs(np.vdot(mine / np.linalg.norm(mine),
                    ref / np.linalg.norm(ref))) ** 2
    assert f > 1 - 1e-9, f"CLOSURE FAILED {tag}: {f}"
    cols = [run_state_form(prog, names, w, swid, info, KETS[b]).flatten()
            for b in (0, 1)]
    gates.append(np.stack(cols, axis=1))
    tags.append(tag)
    print(f"[29] gate {tag}: closure={f:.12f}", flush=True)

# A1: passing directions and alignment
# passing direction INTO the selected outcome = conj of the S_OUT row
# (the top singular vector maximizes total output, not the evidence
# component — v1 used it by mistake; the CAL verb own-selectivity of ~0
# exposed the error)
vs = []
for M in gates:
    r = M[S_OUT, :].conj()
    vs.append(r / np.linalg.norm(r))
align = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        align[i, j] = abs(np.vdot(vs[i], vs[j])) ** 2
off_diag = [align[i, j] for i in range(4) for j in range(4) if i != j]
print(f"[29] A1 alignment |<vi,vj>|^2:\n{align.round(3)}", flush=True)
print(f"[29] A1 mean off-diagonal alignment: {np.mean(off_diag):.3f}",
      flush=True)

# calibration on frame 0 only
pole = vs[CAL_FRAME]
orth = np.array([-np.conj(pole[1]), np.conj(pole[0])])
eq1 = (pole + orth) / np.sqrt(2)
eq2 = (pole + 1j * orth) / np.sqrt(2)

def sel(M, a, b):
    pa = abs((M @ a)[S_OUT]) ** 2
    pb = abs((M @ b)[S_OUT]) ** 2
    return (pa - pb) / (pa + pb)

res = {"alignment": align.tolist(),
       "mean_offdiag_alignment": float(np.mean(off_diag))}
test_sels, null_sels = [], []
for i, (tag, M) in enumerate(zip(tags, gates)):
    s_anim = sel(M, pole, orth)     # >0 means animate pole passes better
    s_null = sel(M, eq1, eq2)
    role = "CAL" if i == CAL_FRAME else "TEST"
    if i != CAL_FRAME:
        test_sels.append(s_anim)
        null_sels.append(abs(s_null))
    res[tag] = {"role": role, "sel_animate": float(s_anim),
                "sel_null_ayn": float(s_null)}
    print(f"[29] {role} {tag}: sel(animate)={s_anim:+.4f} "
          f"sel(ayn-equator)={s_null:+.4f}", flush=True)
a2 = sum(1 for s in test_sels if s > 0)
a3 = float(np.mean(null_sels)) < 0.5 * float(np.mean(np.abs(test_sels)))
print(f"[29] A2: animate selected on {a2}/3 uncalibrated verbs", flush=True)
print(f"[29] A3: ayn null prediction "
      f"(mean|null|={np.mean(null_sels):.4f} vs "
      f"mean|test|={np.mean(np.abs(test_sels)):.4f}): {a3}", flush=True)

# A4: 3-sentence accumulation, calibrated man/leg, hostile prior 0.75 on leg
amp = np.array([np.sqrt(0.75), np.sqrt(0.25)], dtype=complex)  # [leg, man]
traj, cross = [0.25], None
ppost = 1.0
for n, i in enumerate([1, 2, 3], 1):   # the three UNCALIBRATED verbs
    fa = (gates[i] @ pole)[S_OUT]      # man branch
    fb = (gates[i] @ orth)[S_OUT]      # leg branch
    amp = np.array([amp[0] * fb, amp[1] * fa])
    n2 = float(np.sum(np.abs(amp) ** 2))
    ppost *= n2
    amp = amp / np.sqrt(n2)
    pman = float(abs(amp[1]) ** 2)
    traj.append(pman)
    if cross is None and pman > 0.5:
        cross = n
print(f"[29] A4: P(man) " + " -> ".join(f"{x:.3f}" for x in traj) +
      f" | crossed n={cross} p_post={ppost:.4f}", flush=True)
res["A2_animate_selected"] = a2
res["A3_null_ok"] = bool(a3)
res["A4"] = {"traj": traj, "crossed_at": cross, "p_post": ppost}
json.dump(res, open(f"results_exp29_s{S_OUT}.json", "w"), indent=2,
          ensure_ascii=False)
print("[29] DONE", flush=True)
