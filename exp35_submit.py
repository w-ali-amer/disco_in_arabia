# -*- coding: utf-8 -*-
"""Exp 35 submit: send the measurable subset of exported circuits to an IBM
QPU.  Requires a saved IBM account (QiskitRuntimeService.save_account).

Selection: pairs with align margin > 0.05 AND every raw probability
> 1e-3 (shot-noise measurable).  Six circuits per pair (orig/swap x
{v, |0>, |1>} inputs).  Shots chosen so the smallest raw probability gets
>= ~100 expected counts, capped at 30k.
"""
import json
import numpy as np
from qiskit import qpy, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

SHOT_CAP = 30000
meta = json.load(open("exp35_meta.json"))
with open("exp35_circuits.qpy", "rb") as f:
    circuits = qpy.load(f)

pairs = [e for e in meta["pairs"]
         if e["align_margin"] > 0.05 and e["min_p_raw"] > 1e-3]
print(f"[35s] measurable pairs: {len(pairs)}/{len(meta['pairs'])}")
sel = []
for e in pairs:
    for role in ("orig", "swap"):
        r = e["roles"][role]
        sel.append(r["circuit_index"])
        sel.extend(r["basis_circuit_indices"])
shots = min(SHOT_CAP, int(100 / min(e["min_p_raw"] for e in pairs)))
print(f"[35s] circuits={len(sel)} shots={shots} "
      f"(~{len(sel)*shots/1e6:.1f}M total)")

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False,
                             min_num_qubits=7)
print(f"[35s] backend: {backend.name}")
isa = [transpile(circuits[i], backend=backend, optimization_level=3,
                 seed_transpiler=7) for i in sel]
sampler = SamplerV2(mode=backend)
job = sampler.run(isa, shots=shots)
print(f"[35s] job id: {job.job_id()}")
json.dump({"job_id": job.job_id(), "backend": backend.name,
           "shots": shots, "selected_indices": sel,
           "pairs": [{"verb": e["verb"], "pair": e["pair"],
                      "sent": e["sent"]} for e in pairs]},
          open("exp35_job.json", "w"), indent=2, ensure_ascii=False)
print("[35s] saved exp35_job.json — run exp35_retrieve.py once complete")
