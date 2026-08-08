"""Exp16 Phase-0 smoke test: 3-atom line register, one global ramp-plateau-ramp
pulse, exact simulation, site-resolved samples. Proves the toolchain end to end."""
import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.devices import AnalogDevice
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipEmulator

spacing = 6.0  # um, inside blockade-relevant range
reg = Register({f"atom{i}": (i * spacing, 0.0) for i in range(3)})

seq = Sequence(reg, AnalogDevice)
seq.declare_channel("ryd", "rydberg_global")
T = 4000  # ns
omega_max = 2 * np.pi * 1.0  # rad/us
amp = InterpolatedWaveform(T, [0.0, omega_max, omega_max, 0.0])
det = InterpolatedWaveform(T, [-2 * np.pi * 2.0, 0.0, 2 * np.pi * 2.0])
seq.add(Pulse(amp, det, 0.0), "ryd")

sim = QutipEmulator.from_sequence(seq)
res = sim.run()
counts = res.sample_final_state(1000)
top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
print("top bitstrings:", top)
nn = AnalogDevice.interaction_coeff / spacing**6
print(f"V_nn = {nn:.3f} rad/us,  V/Omega = {nn/omega_max:.2f}")
print("SMOKE TEST OK")
