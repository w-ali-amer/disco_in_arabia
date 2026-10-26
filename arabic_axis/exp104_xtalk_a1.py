# -*- coding: utf-8 -*-
"""exp104_xtalk_a1.py -- doc 101 §8 amendment A1 (19844ad): G-XTALK on fewer atoms spaced further apart.

R3 = traps [0, 46, 14] (34.64 um) first, then R2 = [0, 60] (40 um); register of record = largest passing (else 1 atom).
R6 = six ring-4 corners without the centre: descriptive.  G-SER re-run on the register of record (exp104_abstract_a1/).
Power/device time recomputed with atoms per shot = |register of record| from results_exp104.json (unchanged).
Run with ~/pulser_mac_env.  Writes results_exp104_a1.json (Rule 3).
"""
import json, math, os
import numpy as np
from pulser import Sequence
import exp94_common as C94
import exp104_pasqal as E

CANDS = {"R3": [0, 50, 46], "R2": [0, 60]}   # erratum doc 101 §10: registered IDs [0,46,14] were not alternating
DESCR = {"R6": [0, 10, 14, 46, 50, 60], "R3_as_first_registered": [0, 46, 14]}
AB = "exp104_abstract_a1"


def arms():
    r101 = json.load(open("results_exp101.json"))
    a = {"A-2to1": E.arm_2to1()}
    if r101["decisions"]["which_referent_uniform"]:
        a["A-WR6"] = E.arm_wr(6); a["A-WR3"] = E.arm_wr(3)
    return a


def xtalk(traps, A):
    per_arm = {}
    for name, cells in A.items():
        phys, _ = E.physical(cells); dev = []
        for t, m in phys:
            single = E.pg_noiseless(E.sequence(E.pulses_for(t, m), [traps[0]]))[0]
            sites = E.pg_noiseless(E.sequence(E.pulses_for(t, m), traps))
            dev.append(max(abs(p - single) for p in sites))
        per_arm[name] = max(dev)
    return per_arm


def main():
    os.makedirs(AB, exist_ok=True)
    A = arms(); out = {"doc": "101 §8 A1 (19844ad)", "xtalk": {}, "descriptive": {}}
    pos = E.LAYOUT.traps_dict
    for k, tr in {**CANDS, **DESCR}.items():
        d = min(np.linalg.norm(np.asarray(pos[a]) - np.asarray(pos[b])) for i, a in enumerate(tr) for b in tr[i + 1:])
        x = {k2: float(v) for k2, v in xtalk(tr, A).items()}                       # erratum doc 101 §10: JSON casts
        rec = {"traps": tr, "min_dist_um": float(d), "max_pg_dev": x, "pass": bool(max(x.values()) <= 0.01)}
        (out["descriptive"] if k in DESCR else out["xtalk"])[k] = rec
        print(k, rec, flush=True)
    rec_name = next((k for k in ("R3", "R2") if out["xtalk"][k]["pass"]), "single")
    traps = CANDS.get(rec_name, [30])
    out["register_of_record"] = {"name": rec_name, "traps": traps, "n_atoms": len(traps)}
    ser = True
    for name, cells in A.items():
        phys, _ = E.physical(cells)
        for k, (t, m) in enumerate(phys):
            js = E.sequence(E.pulses_for(t, m), traps).to_abstract_repr()
            ser &= Sequence.from_abstract_repr(js).to_abstract_repr() == js
            open(os.path.join(AB, "%s_%02d.json" % (name, k)), "w").write(js)
    for nm, pl in (("g", []), ("r", [(math.pi, 0.0)])):
        js = E.sequence(pl, traps).to_abstract_repr(); ser &= Sequence.from_abstract_repr(js).to_abstract_repr() == js
        open(os.path.join(AB, "A-CTRL_%s.json" % nm), "w").write(js)
    out["G_SER"] = bool(ser)
    r = json.load(open("results_exp104.json"))
    out["power"] = {}
    for name, p in r["power"].items():
        if p["total_samples"] is None:
            out["power"][name] = {"note": "no positive margin"}; continue
        cycles = math.ceil(p["total_samples"] / len(traps))
        mins = {hz: cycles / hz / 60 for hz in (0.5, 1.0, 3.0)}
        out["power"][name] = {"delta_pess": p["delta_pess"], "N_samples_per_sequence": p["N_samples_per_sequence"],
                              "total_samples": p["total_samples"], "atoms_per_shot": len(traps), "device_cycles": cycles,
                              "minutes_at_Hz": mins, "enters_ask": bool(p["delta_pess"] >= 0.04 and mins[0.5] <= 60)}
        print("power", name, out["power"][name], flush=True)
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp104_a1.json", out)
    print(json.dumps({"register_of_record": out["register_of_record"], "G_SER": ser}), flush=True)


if __name__ == "__main__":
    main()
