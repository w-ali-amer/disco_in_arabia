# -*- coding: utf-8 -*-
"""Exp 28a: DisCoCirc gates from REAL compiled pipeline circuits.

Merge of exp27 (architecture) with the actual pregroup pipeline: parse real
VS frames, compile with the exp21 interpreter, then CIRCUIT SURGERY on the
subject wire — its ket becomes the referent INPUT, its word-effect
(Euler+Bra) is removed so the wire exits as OUTPUT; all compositional ops
(H layers, verb CRz, other words, their post-selections) are kept; the
sentence wire is post-selected <0| as evidence. Each frame therefore yields
a measured 2x2 non-unitary filter M — a sentence-as-map derived from the
published pipeline, replacing exp27's stylized gate.

CLOSURE GATE (hard abort): contracting the opened gate's outputs with the
removed word-effect must reproduce the original lambdify pipeline state to
fidelity 1 for every frame. If surgery bookkeeping is wrong, this fails.

Battery (exp27 protocol, real gates): selectivity spectrum -> D1
purification, D3 MAP direction, D7c fresh-wire accumulation vs hostile
prior. Verb/word parameters: root anchors via exp23_roots.json.
"""
import json, hashlib, math, os
import numpy as np
S_OUT = int(os.environ.get("S_OUT", "0"))

t_imports = True
OUT = {}
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel

ROOTS = json.load(open("exp23_roots.json"))["words"]

def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def rx(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -1j * s], [-1j * s, c]])

def rz(t):
    return np.diag([np.exp(-1j * np.pi * t), np.exp(1j * np.pi * t)])

def ry(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -s], [s, c]], dtype=complex)

H_M = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
KETS = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
FIXED1 = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Z": np.diag([1.0 + 0j, -1.0])}
FIXED2 = {"CX": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
          "CZ": np.diag([1, 1, 1, -1]).astype(complex)}

def euler(p):
    return rx(p[0]) @ rz(p[1]) @ rx(p[2])

def apply1(st, U, q):
    st = np.tensordot(U, st, ([1], [q]))
    return np.moveaxis(st, 0, q)

def apply2(st, U4, qa, qb):
    T = U4.reshape(2, 2, 2, 2)
    st = np.tensordot(T, st, ([2, 3], [qa, qb]))
    return np.moveaxis(st, [0, 1], [qa, qb])

def crz_m(t):
    return np.diag([1, 1, np.exp(-1j * np.pi * t),
                    np.exp(1j * np.pi * t)]).astype(complex)

def gate_forms():
    return {"Rx": rx, "Rz": rz, "CRz": crz_m,
            "CRx": lambda t: np.block([[np.eye(2), np.zeros((2, 2))],
                                       [np.zeros((2, 2)), rx(t)]])}
FORMS = gate_forms()

def base_of(name):
    return name.split("__")[0].split("_")[0]

def compile_circuit(c, sym_index):
    try:
        pairs = list(zip(c.boxes, c.offsets))
    except AttributeError:
        pairs = [(lay.box, len(lay.left)) for lay in c.layers]
    prog = []
    for box, off in pairs:
        nm = str(getattr(box, "name", box))
        fs = list(getattr(box, "free_symbols", []))
        si = sym_index[str(fs[0])] if fs else None
        nd, nc = len(box.dom), len(box.cod)
        if nd == 0 and nc == 1:
            prog.append(("ket", off, int(nm) if nm in "01" else 0))
        elif nd == 1 and nc == 0:
            prog.append(("bra", off, int(nm) if nm in "01" else 0))
        elif nd == 0 and nc == 0:
            prog.append(("scalar", 0,
                         complex(np.asarray(box.array).flatten()[0])))
        elif nm == "H":
            prog.append(("H", off, None))
        elif nm.startswith("Rx("):
            prog.append(("Rx", off, si))
        elif nm.startswith("Rz("):
            prog.append(("Rz", off, si))
        elif nm.startswith("CRz("):
            prog.append(("CRz", off, si))
        elif nm.startswith("CRx("):
            prog.append(("CRx", off, si))
        elif nm.upper().startswith("SWAP"):
            prog.append(("SWAP", off, None))
        elif nm in ("CX", "CNOT", "CZ") and nd == 2:
            prog.append(("fixed2", off, "CX" if nm != "CZ" else "CZ"))
        elif nm in ("X", "Z") and nd == 1:
            prog.append(("fixed1", off, nm))
        else:
            raise ValueError(f"unknown box {nm!r}")
    return prog

def track_wires(prog, names):
    """Wire identity tracking. Returns per-wire-id op indices + final order."""
    wires, nid = [], 0
    info = {}
    for oi, (kind, off, arg) in enumerate(prog):
        if kind == "ket":
            wires.insert(off, nid)
            info[nid] = {"ket": oi, "ops": [], "bra": None, "symops": []}
            nid += 1
        elif kind == "bra":
            wid = wires.pop(off)
            info[wid]["bra"] = oi
        elif kind == "scalar":
            continue
        elif kind in ("H", "Rx", "Rz", "fixed1"):
            wid = wires[off]
            info[wid]["ops"].append(oi)
            if kind in ("Rx", "Rz"):
                info[wid]["symops"].append(oi)
        else:  # 2-qubit
            for wid in (wires[off], wires[off + 1]):
                info[wid]["ops"].append(oi)
    return info, wires  # wires = ids still open at end (the s wire(s))

def find_subject_wire(prog, names, info, subj_base):
    cands = [wid for wid, d in info.items()
             if any(base_of(names[prog[oi][2]]) == subj_base
                    for oi in d["symops"])]
    assert len(cands) == 1, f"subject wire ambiguous: {cands}"
    return cands[0]

def run_surgical(prog, names, w, subj_wid, info, input_vec,
                 keep_subject_word=False, postselect_s=True):
    """Execute with subject ket replaced by input_vec; subject word ops
    (its Rx/Rz symbols + bra) removed unless keep_subject_word."""
    skip = set()
    if not keep_subject_word:
        skip = set(info[subj_wid]["symops"])
        if info[subj_wid]["bra"] is not None:
            skip.add(info[subj_wid]["bra"])
    st = np.array(1.0 + 0j)
    wires = []
    for oi, (kind, off, arg) in enumerate(prog):
        if oi in skip:
            continue
        if kind == "ket":
            vec = input_vec if (oi == info[subj_wid]["ket"]) else KETS[arg]
            st = np.moveaxis(np.tensordot(st, vec, 0), -1, off)
            wires.insert(off, oi)
        elif kind == "bra":
            st = np.take(st, arg, axis=off)
            wires.pop(off)
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
            k2 = "CRx" if (kind == "CRz" and
                           os.environ.get("GATE_SWAP") == "crx") else kind
            st = apply2(st, FORMS[k2](w[arg]), off, off + 1)
        else:
            st = apply2(st, FIXED2[arg], off, off + 1)
    # open wires now: subject-out (ket-op id of subj) and s wire(s)
    subj_axis = wires.index(info[subj_wid]["ket"]) \
        if not keep_subject_word and info[subj_wid]["bra"] is not None else None
    return st, wires, subj_axis

def weights_for(names):
    w = np.empty(len(names))
    for i, nm in enumerate(names):
        b = base_of(nm)
        try:
            idx = int(nm.rsplit("_", 1)[-1])
        except ValueError:
            idx = 0
        root = ROOTS.get(b, {}).get("top")
        if root:
            w[i] = 2 * h01(f"{root}|{idx}") + 0.35 * (2 * h01(f"{b}|{idx}") - 1)
        else:
            w[i] = 2 * h01(nm)
    return w

# ── parse frames, compile, surgery, closure ─────────────────────────────────
SUBJ = "الرجل"
# TRANSITIVE frames: the OBJECT is the potential disambiguator — 2-word
# intransitives yielded unitary (evidence-free) gates in the first run
FRAMES_T = [("فتح", "الباب"), ("قرا", "الكتاب"), ("اكل", "الطعام"),
            ("شرب", "الحليب"), ("حمل", "الحقيبة"), ("كتب", "الدرس")]
FRAME_VERBS = [f"{v}+{o}" for v, o in FRAMES_T]
sent_list = [f"{v} {SUBJ} {o}" for v, o in FRAMES_T]
diagrams = exp13.sentences_to_diagrams(sent_list, log_interval=999)
ansatz = exp13.make_ansatz(1, 1)

gates = {}
for v, d in zip(FRAME_VERBS, diagrams):
    if d is None:
        print(f"[28a] frame {v}: parse failed — skipped", flush=True)
        continue
    try:
        c = ansatz(exp13._remove_cups(d))
    except Exception as e:
        print(f"[28a] frame {v}: ansatz failed {e}", flush=True)
        continue
    model = NumpyModel.from_diagrams([c], use_jit=False)
    names = [str(s) for s in model.symbols]
    sym_index = {nm: i for i, nm in enumerate(names)}
    prog = compile_circuit(c, sym_index)
    info, open_end = track_wires(prog, names)
    try:
        swid = find_subject_wire(prog, names, info, SUBJ)
    except AssertionError as e:
        print(f"[28a] frame {v}: {e} — skipped", flush=True)
        continue
    w = weights_for(names)

    # CLOSURE: opened gate (+ s open) contracted with removed effect must
    # equal the lambdify pipeline state.
    st, wires, saxis = run_surgical(prog, names, w, swid, info, KETS[0],
                                    postselect_s=False)
    subj_axis = wires.index(info[swid]["ket"])
    eff_ops = sorted(info[swid]["symops"])
    E = np.eye(2, dtype=complex)
    for oi in eff_ops:
        kind, off, arg = prog[oi]
        E = FORMS[kind](w[arg]) @ E
    eff_row = E[prog[info[swid]["bra"]][2], :] \
        if info[swid]["bra"] is not None else E[0, :]
    closed = np.tensordot(eff_row, st, ([0], [subj_axis]))
    closed = closed.flatten()
    syms = sorted(c.free_symbols, key=str)
    vals = [w[sym_index[str(s)]] for s in syms]
    ref = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
    f_close = abs(np.vdot(closed / np.linalg.norm(closed),
                          ref / np.linalg.norm(ref))) ** 2
    assert f_close > 1 - 1e-9, f"CLOSURE FAILED frame {v}: {f_close}"

    # gate matrix: input basis -> subject-out, s post-selected <0|
    cols = []
    for b in (0, 1):
        st, wires, _ = run_surgical(prog, names, w, swid, info, KETS[b])
        subj_ax = wires.index(info[swid]["ket"])
        s_axes = [a for a in range(st.ndim) if a != subj_ax]
        v_out = st
        for a in sorted(s_axes, reverse=True):
            v_out = np.take(v_out, S_OUT, axis=a)
        cols.append(v_out)
    M = np.stack(cols, axis=1)
    gates[v] = M
    OUT_M = OUT.setdefault("matrices", {})
    OUT_M[v] = {"re": np.real(M).tolist(), "im": np.imag(M).tolist(),
                "singulars": np.linalg.svd(M, compute_uv=False).tolist()}
    print(f"[28a] frame {v}: closure fid={f_close:.12f}, "
          f"|M| singulars={np.linalg.svd(M, compute_uv=False).round(3)}",
          flush=True)

assert gates, "no gates built"

print(f"[28a] {len(gates)} real-circuit gates built, ALL closures passed",
      flush=True)

# ── battery with real gates ─────────────────────────────────────────────────
def anchor(root, lex):
    return [2 * h01(f"{root}::{lex}|{i}") for i in range(3)]

REFS = {"رجل(man/leg)": ("ر.ج.ل", "man", "leg"),
        "جمل(camel/beauty)": ("ج.م.ل", "camel", "beauty"),
        "عين(eye/spring)": ("ع.#.ن", "eye", "spring")}

def blockprep(root, lexA, lexB):
    return euler(anchor(root, lexA)), euler(anchor(root, lexB))

OUT.update({"closure": "all passed", "gates": list(gates),
            "s_outcome": S_OUT})
results = {}
for ref, (root, lA, lB) in REFS.items():
    EA, EB = blockprep(root, lA, lB)
    # selectivity: apply each gate to each sense token state
    spec = {}
    for v, M in gates.items():
        wA, wB = M @ (EA @ KETS[0]), M @ (EB @ KETS[0])
        pA, pB = np.linalg.norm(wA) ** 2, np.linalg.norm(wB) ** 2
        spec[v] = {"passA": pA, "passB": pB,
                   "sel": (pB - pA) / (pB + pA)}
    dis_v = max(spec, key=lambda v: abs(spec[v]["sel"]))
    neu_v = min(spec, key=lambda v: abs(spec[v]["sel"]))
    # register + fresh-wire accumulation vs hostile prior
    sel_dir = spec[dis_v]["sel"] > 0     # True => selects B
    p_B_prior = 0.25 if sel_dir else 0.75
    psi = ry(np.arcsin(np.sqrt(p_B_prior)) / np.pi) @ KETS[0]  # register
    psi = psi.reshape(2)
    traj = [float(abs(psi[1]) ** 2)]
    cross = None
    p_post = 1.0
    for n in range(1, 6):
        psi = np.tensordot(psi, KETS[0], 0)         # fresh token wire
        wax = psi.ndim - 1
        U = np.zeros((4, 4), dtype=complex)
        U[:2, :2], U[2:, 2:] = EA, EB
        psi = apply2(psi, U, 0, wax)                 # re-ground from register
        psi = apply1(psi, gates[dis_v], wax)         # REAL sentence gate
        n2 = float(np.sum(np.abs(psi) ** 2))
        psi = psi / np.sqrt(n2)
        p_post *= n2
        rho = np.moveaxis(psi, 0, psi.ndim - 1).reshape(-1, 2)
        rho = rho.T @ rho.conj()
        pb = float(np.real(rho[1, 1] / np.trace(rho)))
        traj.append(pb)
        if cross is None and ((pb > 0.5) == sel_dir):
            cross = n
    results[ref] = {"spectrum": spec, "disambig": dis_v, "neutral": neu_v,
                    "sel": spec[dis_v]["sel"], "PB_traj": traj,
                    "crossed_at": cross, "p_post": p_post}
    print(f"[28a] {ref}: dis={dis_v} sel={spec[dis_v]['sel']:+.3f} "
          f"neu={neu_v} sel={spec[neu_v]['sel']:+.3f} | PB " +
          " -> ".join(f"{x:.3f}" for x in traj) +
          f" | crossed n={cross} p_post={p_post:.3f}", flush=True)

n_cross = sum(1 for r in results.values() if r["crossed_at"])
print(f"[28a] REAL-GATE accumulation beats hostile prior: "
      f"{n_cross}/{len(results)}", flush=True)
OUT["battery"] = results
json.dump(OUT, open(f"results_exp28a_s{S_OUT}.json", "w"), indent=2, ensure_ascii=False)
print("[28a] DONE — results_exp28a.json", flush=True)
