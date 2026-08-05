# -*- coding: utf-8 -*-
"""Exp 34b: measured classical-simulation scaling of the DisCoCirc referent
architecture.

CLAIM SCOPE (printed with results): this measures the cost of EXACTLY
simulating this circuit architecture classically.  It is not a claim about
task-level complexity (a bespoke classical algorithm for a given task may
be cheaper), and no quantum task advantage is claimed at current scale.

Two regimes, both using the project's real gate forms (CRz·CRx sentence
entanglers, exp33's zx4 family):
  pure  — k referent wires, statevector, sentence gates only.  Cost 2^k.
  mixed — the discourse features the architecture actually needs
          (exp24 entropy evidence, exp26/27 purification + discard)
          force density matrices.  Cost 4^k.
Per k: T sentence-steps (gate on a referent pair + for the mixed regime a
discard channel on one wire), wall-time per step measured, memory = state
size.  Fit slope of log2(time) vs k, extrapolate the RAM walls.
"""
import json, time
import numpy as np

rng = np.random.default_rng(7)
T_STEPS = 6
TIME_CAP = 30.0          # stop growing k when a step exceeds this (seconds)
K_MAX_PURE = 24
K_MAX_MIXED = 13

def crz(t):
    d = np.ones(4, complex)
    d[3] = np.exp(1j * np.pi * t)
    return np.diag(d)

def crx(t):
    m = np.eye(4, dtype=complex)
    c, s = np.cos(np.pi * t / 2), -1j * np.sin(np.pi * t / 2)
    m[2, 2] = m[3, 3] = c
    m[2, 3] = m[3, 2] = s
    return m

def sentence_gate():
    return crz(rng.uniform(0, 2)) @ crx(rng.uniform(0, 2))

def apply2_state(psi, U, i, k):
    psi = psi.reshape([2] * k)
    psi = np.moveaxis(psi, (i, i + 1), (0, 1)).reshape(4, -1)
    psi = U @ psi
    psi = np.moveaxis(psi.reshape([2, 2] + [2] * (k - 2)), (0, 1), (i, i + 1))
    return psi.reshape(-1)

def apply2_rho(rho, U, i, k):
    d = 2 ** k
    rho = rho.reshape([2] * k + [2] * k)
    rho = np.moveaxis(rho, (i, i + 1), (0, 1)).reshape(4, -1)
    rho = (U @ rho).reshape([2, 2] + [2] * (k - 2) + [2] * k)
    rho = np.moveaxis(rho, (0, 1), (i, i + 1)).reshape(d, d)
    rho = rho.reshape([2] * k + [2] * k)
    rho = np.moveaxis(rho, (k + i, k + i + 1), (0, 1)).reshape(4, -1)
    rho = (U.conj() @ rho).reshape([2, 2] + [2] * (2 * k - 2))
    rho = np.moveaxis(rho, (0, 1), (k + i, k + i + 1))
    return rho.reshape(d, d)

def discard_channel(rho, i, k):
    """Measure-and-forget wire i (the exp26/27 discard): Kraus {|0><0|,|1><1|}."""
    d = 2 ** k
    rho = rho.reshape([2] * k + [2] * k)
    keep = np.moveaxis(rho, (i, k + i), (0, 1))
    # wire i dephased: populations kept, coherences dropped
    new = np.zeros_like(keep)
    new[0, 0] = keep[0, 0]
    new[1, 1] = keep[1, 1]
    rho = np.moveaxis(new, (0, 1), (i, k + i))
    return rho.reshape(d, d)

results = {"pure": [], "mixed": []}

k = 2
while k <= K_MAX_PURE:
    psi = np.zeros(2 ** k, complex)
    psi[0] = 1.0
    times = []
    for t in range(T_STEPS):
        i = int(rng.integers(0, k - 1))
        U = sentence_gate()
        t0 = time.perf_counter()
        psi = apply2_state(psi, U, i, k)
        times.append(time.perf_counter() - t0)
    mt = float(np.median(times))
    mem = psi.nbytes
    results["pure"].append({"k": k, "median_step_s": mt, "bytes": mem})
    print(f"[34b] pure  k={k:2d}: step={mt:.2e}s state={mem/1e6:.1f}MB",
          flush=True)
    if mt > TIME_CAP:
        break
    k += 1

k = 2
while k <= K_MAX_MIXED:
    d = 2 ** k
    rho = np.zeros((d, d), complex)
    rho[0, 0] = 1.0
    times = []
    for t in range(T_STEPS):
        i = int(rng.integers(0, k - 1))
        U = sentence_gate()
        t0 = time.perf_counter()
        rho = apply2_rho(rho, U, i, k)
        rho = discard_channel(rho, i, k)
        times.append(time.perf_counter() - t0)
    mt = float(np.median(times))
    mem = rho.nbytes
    results["mixed"].append({"k": k, "median_step_s": mt, "bytes": mem})
    print(f"[34b] mixed k={k:2d}: step={mt:.2e}s state={mem/1e6:.1f}MB",
          flush=True)
    if mt > TIME_CAP:
        break
    k += 1

def slope(rows, kmin):
    xs = [r["k"] for r in rows if r["k"] >= kmin and r["median_step_s"] > 1e-5]
    ys = [np.log2(r["median_step_s"]) for r in rows
          if r["k"] >= kmin and r["median_step_s"] > 1e-5]
    if len(xs) < 3:
        return None
    return float(np.polyfit(xs, ys, 1)[0])

s_pure = slope(results["pure"], 12)
s_mixed = slope(results["mixed"], 7)
print(f"[34b] fitted slopes (log2 time per wire): pure={s_pure} "
      f"(theory 1.0), mixed={s_mixed} (theory 2.0)", flush=True)

# RAM walls for the mixed (discourse) regime: 16 bytes * 4^k
for name, ram in [("16GB laptop", 16e9), ("1TB node", 1e12),
                  ("exascale-ish 100PB", 1e17)]:
    kw = int(np.floor(np.log(ram / 16) / np.log(4)))
    print(f"[34b] mixed-regime RAM wall on {name}: k={kw} referents",
          flush=True)
print("[34b] a 100-referent discourse needs 16*4^100 ≈ 2.6e121 bytes "
      "classically (exact simulation); neutral-atom register holds it in "
      "100 atoms", flush=True)
print("[34b] CAVEAT (part of the result): this bounds exact classical "
      "simulation of THIS architecture; task-level classical algorithms "
      "may be cheaper; no quantum task advantage is claimed at current "
      "scale.", flush=True)
json.dump({"results": results, "slope_pure": s_pure, "slope_mixed": s_mixed},
          open("results_exp34b.json", "w"), indent=2)
print("[34b] DONE", flush=True)
