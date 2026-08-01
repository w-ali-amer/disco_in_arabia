"""Exp16 Phase 1: submit the locked sequences to Pasqal Cloud emulators.

UNTESTED STUB until credentials exist. Requires environment variables:
  PASQAL_PROJECT_ID, PASQAL_USERNAME, PASQAL_PASSWORD
from an account at portal.pasqal.cloud.

Strategy: submit the three peak-window spacings (6.8, 7.3, 7.9 um) plus the
far-spacing control (25 um) for all 60 pairs, C1 and C2 encodings, 1000 runs
each — the Phase-1 question is whether the V/Omega ~ 1 distinguishability
window survives realistic device noise. Start with EMU_FREE to validate the
round-trip, then EMU_TN for the noise-realistic result.
"""
import json
import os
import sys

import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.devices import AnalogDevice
from pulser.waveforms import InterpolatedWaveform

TARGET_SPACINGS = [6.8, 7.3, 7.9, 25.0]
BACKEND = os.environ.get("PASQAL_BACKEND", "EMU_FREE")  # EMU_FREE | EMU_TN
RUNS = 1000

T = 4000
OMEGA_MAX = 2 * np.pi * 1.0
DET_RANGE = 2 * np.pi * 2.0
ETA = 0.35


def build_sequence(ys, a):
    reg = Register({f"a{i}": ((i - 1) * a, ys[i]) for i in range(3)})
    seq = Sequence(reg, AnalogDevice)
    seq.declare_channel("ryd", "rydberg_global")
    amp = InterpolatedWaveform(T, [0.0, OMEGA_MAX, OMEGA_MAX, 0.0])
    det = InterpolatedWaveform(T, [-DET_RANGE, 0.0, DET_RANGE])
    seq.add(Pulse(amp, det, 0.0), "ryd")
    return seq


def main():
    for var in ("PASQAL_PROJECT_ID", "PASQAL_USERNAME", "PASQAL_PASSWORD"):
        if not os.environ.get(var):
            sys.exit(f"Missing {var}. Create an account at portal.pasqal.cloud, "
                     "then export PASQAL_PROJECT_ID / PASQAL_USERNAME / "
                     "PASQAL_PASSWORD and re-run.")

    from pulser_pasqal import PasqalCloud

    conn = PasqalCloud(
        username=os.environ["PASQAL_USERNAME"],
        password=os.environ["PASQAL_PASSWORD"],
        project_id=os.environ["PASQAL_PROJECT_ID"],
    )
    print("Connected. Available devices/emulators:")
    print(conn.fetch_available_devices())

    # Registers identical to Phase 0 (geometry encoding; see EXP16_RUNBOOK.md).
    # Build the job list from the same code paths as exp16_analog_geometry.py:
    from exp16_analog_geometry import yoffs, sents, twins  # noqa: E402

    jobs = []
    for a in TARGET_SPACINGS:
        jobs.append(("C1_SVO", a, [0.0, ETA * a, 0.0]))
        jobs.append(("C1_VSO", a, [ETA * a, 0.0, 0.0]))
        for pi, (i, j) in enumerate(twins):
            jobs.append((f"C2_p{pi}_SVO", a, yoffs(sents[i].split(), "C2_embedding", a)))
            jobs.append((f"C2_p{pi}_VSO", a, yoffs(sents[j].split(), "C2_embedding", a)))
    print(f"{len(jobs)} sequences to submit to {BACKEND} ({RUNS} runs each).")
    print("Submitting is gated behind --submit to avoid accidental spends.")
    if "--submit" not in sys.argv:
        return

    results = {}
    for tag, a, ys in jobs:
        seq = build_sequence(ys, a)
        # pulser_pasqal backend API (v0.23): EmuFreeBackend / EmuTNBackend
        from pulser_pasqal.backends import EmuFreeBackend, EmuTNBackend
        Backend = EmuFreeBackend if BACKEND == "EMU_FREE" else EmuTNBackend
        backend = Backend(seq, connection=conn)
        job = backend.run(job_params=[{"runs": RUNS}], wait=True)
        results[f"{tag}_a{a}"] = {str(k): int(v)
                                  for k, v in job.results[0].bitstring_counts.items()}
        print(f"done {tag} a={a}")
        json.dump(results, open(f"exp16_phase1_{BACKEND}.json", "w"), indent=2)
    print("ALL SUBMITTED AND SAVED")


if __name__ == "__main__":
    main()
