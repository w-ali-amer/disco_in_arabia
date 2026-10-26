# -*- coding: utf-8 -*-
"""kernel_factorisation_check.py -- committed producer for doc 52 SS1.1.

doc 52 SS1.1 states the headline closure of the advantage question for
per-word angle encodings: the product-state kernel factorises,

    <psi(theta) | psi(theta')> = prod_i cos( (theta_i - theta'_i) / 2 ),

so the quantum kernel is classically computable in O(n) and no accuracy
advantage is available from this encoding family. The number quoted there
(max deviation 3.3e-16) came from session-only computation -- the doc 53
SS7.3 / paper-draft B1 gap-class. This script is the committed producer:
it verifies the identity against the repo's own statevector simulator
(torch_sv_sim, the simulator every experiment ran on) and writes
diagnostic_kernel_factorisation.json.

Identity: RY(t)|0> = (cos(t/2), sin(t/2)), all-real single-qubit states, so
<u(t)|u(t')> = cos(t/2)cos(t'/2) + sin(t/2)sin(t'/2) = cos((t-t')/2), and a
product state factorises the inner product across wires. The script checks
the SIMULATOR realises this, not merely the algebra.
"""
import hashlib, json, math, platform, random

import torch

import torch_sv_sim as S

SEED = 5201             # registered here; deterministic
NS = [1, 2, 3, 5, 8, 12]
PAIRS_PER_N = 200


def ry(theta):
    t = torch.as_tensor(theta, dtype=torch.float32)
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex64)


def product_state(thetas):
    st = S.init_state(len(thetas))
    for w, t in enumerate(thetas):
        st = S.apply_gate(st, ry(t), [w])
    return st


def main():
    rng = random.Random(SEED)
    import numpy as np

    def state64(ths):
        v = np.array([1.0])
        for t in ths:
            v = np.kron(v, np.array([math.cos(t / 2), math.sin(t / 2)]))
        return v

    rows, worst, worst64 = [], 0.0, 0.0
    for n in NS:
        dev_max, dev64_max = 0.0, 0.0
        for _ in range(PAIRS_PER_N):
            a = [rng.uniform(0.0, 2 * math.pi) for _ in range(n)]
            b = [rng.uniform(0.0, 2 * math.pi) for _ in range(n)]
            overlap = torch.vdot(product_state(a).flatten(), product_state(b).flatten())
            predicted = 1.0
            for x, y in zip(a, b):
                predicted *= math.cos((x - y) / 2.0)
            dev = abs(overlap.item() - predicted)
            dev_max = max(dev_max, dev)
            dev64 = abs(float(state64(a) @ state64(b)) - predicted)
            dev64_max = max(dev64_max, dev64)
        rows.append({"n_qubits": n, "pairs": PAIRS_PER_N,
                     "max_abs_deviation_sim_c64": dev_max,
                     "max_abs_deviation_float64": dev64_max})
        worst, worst64 = max(worst, dev_max), max(worst64, dev64_max)
        print("n=%2d  sim(c64) %.3e   float64 %.3e" % (n, dev_max, dev64_max))
    # float32 statevectors: the identity must hold to simulator precision.
    assert worst < 1e-5, "kernel factorisation FAILED at %.3e" % worst
    assert worst64 < 1e-12, "float64 factorisation FAILED at %.3e" % worst64
    out = {
        "check": "kernel_factorisation",
        "claim": "doc 52 SS1.1: <psi(theta)|psi(theta')> = prod_i cos((theta_i-theta'_i)/2) "
                 "for per-wire RY product encodings; classical O(n)",
        "simulator": "torch_sv_sim (complex64), the simulator of record for exp42-exp68",
        "seed": SEED,
        "rows": rows,
        "max_abs_deviation_sim_c64_overall": worst,
        "max_abs_deviation_float64_overall": worst64,
        "precision_note": "complex64 statevectors bound the check near 1e-6; doc 52's 3.3e-16 "
                          "was float64 algebra on the closed form. Both verify the same identity; "
                          "this file is the committed producer of record.",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    }
    with open("diagnostic_kernel_factorisation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote diagnostic_kernel_factorisation.json  (worst %.3e)" % worst)


if __name__ == "__main__":
    main()
