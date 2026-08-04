"""Rewrite run_surgical with stash semantics: on the subject's bra, move its
axis to the tensor end (preserving downstream offsets) instead of skipping
the removal. Fixes VSO closure failures (offset drift after mid-circuit
subject effects)."""
import io, re

PATH = "exp28a_real_gates.py"
src = io.open(PATH, encoding="utf-8").read()

start = src.index("def run_surgical(")
end = src.index("def weights_for(")
new_fn = '''def run_surgical(prog, names, w, subj_wid, info, input_vec,
                 keep_subject_word=False):
    """Execute with subject ket replaced by input_vec; subject Euler ops
    skipped; subject bra STASHED (axis moved to tensor end so downstream
    offsets stay valid). Returns (state, open_wire_ids, subj_axis)."""
    skip_syms = set() if keep_subject_word else set(info[subj_wid]["symops"])
    subj_bra = None if keep_subject_word else info[subj_wid]["bra"]
    st = np.array(1.0 + 0j)
    wires = []
    stashed = 0
    for oi, (kind, off, arg) in enumerate(prog):
        if oi in skip_syms:
            continue
        if kind == "ket":
            vec = input_vec if (oi == info[subj_wid]["ket"]) else KETS[arg]
            st = np.moveaxis(np.tensordot(st, vec, 0), -1, off)
            wires.insert(off, oi)
        elif kind == "bra":
            if oi == subj_bra:
                st = np.moveaxis(st, off, st.ndim - 1)
                wires.pop(off)
                stashed += 1
                continue
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
    if stashed:
        subj_axis = st.ndim - 1
    elif not keep_subject_word:
        subj_axis = wires.index(info[subj_wid]["ket"])
    else:
        subj_axis = None
    return st, wires, subj_axis


'''
src = src[:start] + new_fn + src[end:]

# closure caller: use returned subj_axis
old = """    st, wires, saxis = run_surgical(prog, names, w, swid, info, KETS[0],
                                    postselect_s=False)
    subj_axis = wires.index(info[swid]["ket"])"""
new = """    st, wires, subj_axis = run_surgical(prog, names, w, swid, info, KETS[0])"""
assert src.count(old) == 1, "closure caller"
src = src.replace(old, new)

# gate-matrix caller: use returned subj_axis
old2 = """    for b in (0, 1):
        st, wires, _ = run_surgical(prog, names, w, swid, info, KETS[b])
        subj_ax = wires.index(info[swid]["ket"])
        s_axes = [a for a in range(st.ndim) if a != subj_ax]"""
new2 = """    for b in (0, 1):
        st, wires, subj_ax = run_surgical(prog, names, w, swid, info, KETS[b])
        s_axes = [a for a in range(st.ndim) if a != subj_ax]"""
assert src.count(old2) == 1, "gate caller"
src = src.replace(old2, new2)

io.open(PATH, "w", encoding="utf-8").write(src)
print("SURGERY-REWRITTEN")
