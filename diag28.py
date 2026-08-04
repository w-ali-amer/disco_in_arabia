import importlib.util, sys, types, json
import numpy as np

# load exp28a WITHOUT executing module tail: read source, exec only up to the frame loop
src = open("exp28a_real_gates.py", encoding="utf-8").read()
cut = src.index("# ── parse frames")
head = src[:cut]
ns = {}
exec(compile(head, "exp28a_head", "exec"), ns)

exp13 = ns["exp13"]
NumpyModel = ns["NumpyModel"]

SENT, SUBJ = "قرأ الطالب كتاب النحو", "الطالب"
d = exp13.sentences_to_diagrams([SENT], log_interval=999)[0]
c = ns["exp13"].make_ansatz(1, 1)(exp13._remove_cups(d))
model = NumpyModel.from_diagrams([c], use_jit=False)
names = [str(s) for s in model.symbols]
sym_index = {nm: i for i, nm in enumerate(names)}
prog = ns["compile_circuit"](c, sym_index)
info, open_end = ns["track_wires"](prog, names)
swid = ns["find_subject_wire"](prog, names, info, SUBJ)
w = ns["weights_for"](names)

print("[diag] prog:")
for oi, (k, off, a) in enumerate(prog):
    tag = names[a][:30] if k in ("Rx", "Rz", "CRz", "CRx") else a
    onsub = "S" if oi in info[swid]["ops"] or oi in (info[swid]["ket"], info[swid]["bra"]) else " "
    print(f"  {oi:2d} {onsub} {k:7s} off={off} {tag}")
print(f"[diag] subject wid={swid} ket={info[swid]['ket']} "
      f"bra={info[swid]['bra']} symops={info[swid]['symops']}")

# (a) keep_subject_word=True must equal lambdify exactly
st, wires, _ = ns["run_surgical"](prog, names, w, swid, info,
                                  ns["KETS"][0], keep_subject_word=True)
mine = np.asarray(st).flatten()
syms = sorted(c.free_symbols, key=str)
vals = [w[sym_index[str(s)]] for s in syms]
ref = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
f = abs(np.vdot(mine / np.linalg.norm(mine), ref / np.linalg.norm(ref))) ** 2
print(f"[diag] keep=True vs lambdify fidelity: {f:.12f}")
