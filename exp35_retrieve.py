# -*- coding: utf-8 -*-
"""Exp 35 retrieve: fetch QPU counts, compute hardware align scores
(P_v / (P_0 + P_1) — global scalars and much of the noise cancel in the
ratio), compare decisions against noiseless predictions."""
import json
from qiskit_ibm_runtime import QiskitRuntimeService

jobinfo = json.load(open("exp35_job.json"))
meta = json.load(open("exp35_meta.json"))
sel = jobinfo["selected_indices"]
pos_of = {ci: k for k, ci in enumerate(sel)}

service = QiskitRuntimeService()
job = service.job(jobinfo["job_id"])
print(f"[35r] status: {job.status()}")
res = job.result()

def p_raw(circ_pos, rec):
    counts = res[circ_pos].data[
        list(res[circ_pos].data.keys())[0]].get_counts()
    tot = sum(counts.values())
    good = 0
    n_post = len(rec["post"])
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]      # bits[i] = clbit i
        if all(bits[ci] == str(out)
               for ci, (_, out) in enumerate(rec["post"])) \
                and bits[n_post] == str(meta["S_OUT"]):
            good += c
    return good / tot

pairs = [e for e in meta["pairs"]
         if e["align_margin"] > 0.05 and e["min_p_raw"] > 1e-3]
n_ok = n_match = 0
rows = []
for e in pairs:
    hw = {}
    for role in ("orig", "swap"):
        r = e["roles"][role]
        pv = p_raw(pos_of[r["circuit_index"]], r)
        p0 = p_raw(pos_of[r["basis_circuit_indices"][0]], r)
        p1 = p_raw(pos_of[r["basis_circuit_indices"][1]], r)
        hw[role] = pv / (p0 + p1) if p0 + p1 > 0 else 0.5
    hw_dec = hw["orig"] > hw["swap"]
    pred_dec = e["ratio_correct"]
    n_ok += int(hw_dec)
    n_match += int(hw_dec == pred_dec)
    rows.append({"verb": e["verb"], "sent": e["sent"],
                 "hw_orig": hw["orig"], "hw_swap": hw["swap"],
                 "hw_correct": bool(hw_dec),
                 "matches_prediction": bool(hw_dec == pred_dec)})
    print(f"[35r] {e['verb']} p{e['pair']}: hw orig={hw['orig']:.3f} "
          f"swap={hw['swap']:.3f} correct={hw_dec} "
          f"pred_match={hw_dec == pred_dec}", flush=True)
print(f"[35r] HARDWARE: correct {n_ok}/{len(pairs)} | agreement with "
      f"noiseless prediction {n_match}/{len(pairs)} | backend "
      f"{jobinfo['backend']}, {jobinfo['shots']} shots")
json.dump({"rows": rows, "hw_correct": n_ok,
           "prediction_agreement": n_match, "n": len(pairs)},
          open("results_exp35_hardware.json", "w"), indent=2,
          ensure_ascii=False)
print("[35r] DONE")
