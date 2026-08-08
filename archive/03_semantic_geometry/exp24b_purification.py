# -*- coding: utf-8 -*-
"""Exp 24b (probe): sense-superposition mixedness via the 2025 H3 machinery
(AmbiguousLexicalBox + ControlledSenseFunctor from common_qnlp_types).

Tonight's scope is MECHANICAL VALIDATION, defensively coded (the H3 test
files needed monkey-patches in 2025): build [ambiguous noun >> verb] with a
2-sense controlled ansatz, evaluate, trace the sense-control wire, and test
whether the sentence state is mixed (entropy > 0) versus a fixed-sense
construction (entropy ~ 0). If the old API breaks, dump signatures and exit
gracefully — the log is the deliverable either way.
"""
import inspect, json, time
import numpy as np

t0 = time.time()
print("[24b] importing common_qnlp_types...", flush=True)
import common_qnlp_types as cqt

for name in ("AmbiguousLexicalBox", "ControlledSenseFunctor"):
    obj = getattr(cqt, name, None)
    print(f"[24b] {name}: "
          f"{inspect.signature(obj.__init__) if obj else 'MISSING'}",
          flush=True)

try:
    from common_qnlp_types import (AmbiguousLexicalBox, ControlledSenseFunctor,
                                   N_ARABIC, S_ARABIC)
    from lambeq.backend.grammar import Box

    noun_amb = AmbiguousLexicalBox("رجل_ambiguous", N_ARABIC,
                                   senses=["man", "leg"])
    verb = Box("جاء", N_ARABIC, S_ARABIC)
    diagram = noun_amb >> verb
    print(f"[24b] diagram built: {type(diagram).__name__}, "
          f"cod={diagram.cod}", flush=True)

    functor = ControlledSenseFunctor(ob_map={N_ARABIC: 1, S_ARABIC: 1}, n_layers=1, n_single_qubit_params=3)
    circ = functor(diagram)
    print(f"[24b] circuit: {type(circ).__name__}, is_mixed="
          f"{getattr(circ, 'is_mixed', '?')}, "
          f"n_free_symbols={len(list(circ.free_symbols))}", flush=True)

    import hashlib, math
    syms = sorted(circ.free_symbols, key=str)
    vals = [(int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)
             / 0xFFFFFFFF) * 2 for s in syms]
    T = (circ.lambdify(*syms)(*vals).eval() if syms else circ.eval())
    T = np.asarray(T)
    print(f"[24b] eval OK: shape={T.shape}, dtype={T.dtype}", flush=True)

    def entropy(rho):
        ev = np.clip(np.real(np.linalg.eigvalsh(
            (rho + rho.conj().T) / 2)), 1e-12, None)
        ev = ev / ev.sum()
        return float(-np.sum(ev * np.log2(ev)))

    v = T.flatten()
    n_q = int(round(np.log2(v.size))) if v.size > 1 else 0
    if getattr(circ, "is_mixed", False):
        d = int(round(v.size ** 0.5))
        rho = T.reshape(d, d)
        rho = rho / np.trace(rho)
        print(f"[24b] mixed output: S(rho) = {entropy(rho):.4f} bits", flush=True)
    elif n_q >= 2:
        # pure state with extra open wires: trace all but the last qubit
        psi = v / np.linalg.norm(v)
        A = psi.reshape(2 ** (n_q - 1), 2)
        rho_s = A.conj().T @ A
        print(f"[24b] pure {n_q}-qubit output; sentence-wire "
              f"S(rho_s) = {entropy(rho_s):.4f} bits "
              f"(>0 means sense-wire entanglement = ambiguity as mixedness)",
              flush=True)
    else:
        print(f"[24b] single-wire pure output — no sense wire to trace; "
              f"construction collapsed the superposition", flush=True)
    print("[24b] MECHANICAL VALIDATION COMPLETE", flush=True)
except Exception as e:
    import traceback
    print(f"[24b] FAILED at: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    print("[24b] (diagnostic run — failure log is the deliverable)", flush=True)
print(f"[24b] done in {time.time()-t0:.1f}s", flush=True)
