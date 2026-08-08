# -*- coding: utf-8 -*-
"""Exp 28b: consumed-token sentence gates from true-VSO circuits.

Diagnostic finding: VSO subjects compile as STATES consumed by explicit
Bell cups (SWAP/CX/H/postselects) — word content is ONLY the Euler prep;
the cup is grammar. Surgery: replace subject ket input, skip its Euler,
keep everything else. Gate = 2x2 map referent -> sentence wire.
Closure: input = the actual word state must reproduce lambdify EXACTLY.
Battery: token consumed; register conditioning via per-branch scalars
phi_b = (M w_sense)[S_OUT]; selectivity + accumulation race.
"""
import os, json
import numpy as np

S_OUT = int(os.environ.get("S_OUT", "0"))
src = open("exp28a_real_gates.py", encoding="utf-8").read()
head = src[:src.index("# ── parse frames")]
ns = {}
exec(compile(head, "exp28a_head", "exec"), ns)
exp13, NumpyModel = ns["exp13"], ns["NumpyModel"]
KETS, FORMS, h01 = ns["KETS"], ns["FORMS"], ns["h01"]
rx_, rz_, ry_, euler = ns["rx"], ns["rz"], ns["ry"], ns["euler"]
apply1, apply2 = ns["apply1"], ns["apply2"]

FRAMES_V = [("قرأ الطالب كتاب النحو", "الطالب"),
            ("حمل الولد حقيبة المدرسة", "الولد"),
            ("فتح المدير باب المكتب", "المدير"),
            ("كتب الطالب الدرس الجديد", "الطالب")]
diagrams = exp13.sentences_to_diagrams([s for s, _ in FRAMES_V],
                                       log_interval=999)
ansatz = exp13.make_ansatz(1, 1)

def run_state_form(prog, names, w, swid, info, input_vec, skip_word=True):
    """Execute; subject ket -> input_vec; subject Euler skipped if skip_word.
    All bras kept (cups are grammar). Returns final tensor (open wires)."""
    skip = set(info[swid]["symops"]) if skip_word else set()
    H_M, FIXED1, FIXED2 = ns["H_M"], ns["FIXED1"], ns["FIXED2"]
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

OUT = {"s_outcome": S_OUT, "form": "state(consumed-token)"}
gates = {}
for (sent, SUBJ), d in zip(FRAMES_V, diagrams):
    tag = sent.split()[0] + "+" + sent.split()[2]
    c = ansatz(exp13._remove_cups(d))
    model = NumpyModel.from_diagrams([c], use_jit=False)
    names = [str(s) for s in model.symbols]
    sym_index = {nm: i for i, nm in enumerate(names)}
    prog = ns["compile_circuit"](c, sym_index)
    info, _ = ns["track_wires"](prog, names)
    swid = ns["find_subject_wire"](prog, names, info, SUBJ)
    w = ns["weights_for"](names)
    # subject word state |w> from its own Euler ops in application order
    W = np.eye(2, dtype=complex)
    for oi in sorted(info[swid]["symops"]):
        kind, off, arg = prog[oi]
        W = FORMS[kind](w[arg]) @ W
    wstate = W @ KETS[0]
    # closure: surgical(input=|w>) must equal lambdify exactly
    mine = run_state_form(prog, names, w, swid, info, wstate).flatten()
    syms = sorted(c.free_symbols, key=str)
    vals = [w[sym_index[str(s)]] for s in syms]
    ref = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
    f = abs(np.vdot(mine / np.linalg.norm(mine),
                    ref / np.linalg.norm(ref))) ** 2
    assert f > 1 - 1e-9, f"CLOSURE FAILED {tag}: {f}"
    # gate M: referent basis -> s wire
    cols = [run_state_form(prog, names, w, swid, info, KETS[b]).flatten()
            for b in (0, 1)]
    M = np.stack(cols, axis=1)
    sv = np.linalg.svd(M, compute_uv=False)
    gates[tag] = M
    OUT.setdefault("gates", {})[tag] = {
        "singulars": sv.tolist(),
        "anisotropy": float(sv[0] / max(sv[1], 1e-12))}
    print(f"[28b] {tag}: closure={f:.12f} singulars={sv.round(4)} "
          f"anisotropy={sv[0]/max(sv[1],1e-12):.3f}", flush=True)

def anchor(root, lex):
    return [2 * h01(f"{root}::{lex}|{i}") for i in range(3)]

REFS = {"رجل(man/leg)": ("ر.ج.ل", "man", "leg"),
        "جمل(camel/beauty)": ("ج.م.ل", "camel", "beauty"),
        "عين(eye/spring)": ("ع.#.ن", "eye", "spring")}
for ref, (root, lA, lB) in REFS.items():
    wA = euler(anchor(root, lA)) @ KETS[0]
    wB = euler(anchor(root, lB)) @ KETS[0]
    spec = {}
    for tag, M in gates.items():
        pA = abs((M @ wA)[S_OUT]) ** 2
        pB = abs((M @ wB)[S_OUT]) ** 2
        spec[tag] = {"pA": pA, "pB": pB, "sel": (pB - pA) / (pB + pA)}
    dis = max(spec, key=lambda t: abs(spec[t]["sel"]))
    sel = spec[dis]["sel"]
    # accumulation race: prior 0.75 against the gate direction
    pb = 0.25 if sel > 0 else 0.75
    amp = np.array([np.sqrt(1 - pb), np.sqrt(pb)], dtype=complex)
    traj, cross, ppost = [pb], None, 1.0
    for n in range(1, 6):
        fA = (gates[dis] @ wA)[S_OUT]
        fB = (gates[dis] @ wB)[S_OUT]
        amp = np.array([amp[0] * fA, amp[1] * fB])
        n2 = float(np.sum(np.abs(amp) ** 2))
        ppost *= n2
        amp = amp / np.sqrt(n2)
        pb_n = float(abs(amp[1]) ** 2)
        traj.append(pb_n)
        if cross is None and ((pb_n > 0.5) == (sel > 0)):
            cross = n
    OUT.setdefault("battery", {})[ref] = {
        "spectrum": spec, "disambig": dis, "sel": sel,
        "PB_traj": traj, "crossed_at": cross, "p_post": ppost}
    print(f"[28b] {ref}: dis={dis} sel={sel:+.4f} | PB " +
          " -> ".join(f"{x:.3f}" for x in traj) +
          f" | crossed n={cross} p_post={ppost:.4f}", flush=True)

n_cross = sum(1 for r in OUT["battery"].values() if r["crossed_at"])
print(f"[28b] REAL-GATE evidence beats hostile prior: {n_cross}/3", flush=True)
json.dump(OUT, open(f"results_exp28b_s{S_OUT}.json", "w"), indent=2,
          ensure_ascii=False)
print("[28b] DONE", flush=True)
