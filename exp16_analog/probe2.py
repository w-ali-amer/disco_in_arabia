import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.devices import AnalogDevice
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipEmulator

d = AnalogDevice
print("min_atom_distance:", d.min_atom_distance)
print("max_radial_distance:", d.max_radial_distance)
print("max_sequence_duration:", d.max_sequence_duration)
print("interaction_coeff:", d.interaction_coeff)
ch = d.channels["rydberg_global"]
print("max_amp:", ch.max_amp, "max_abs_detuning:", ch.max_abs_detuning)

spacing = 6.0
reg = Register({f"a{i}": ((i - 1) * spacing, 0.0) for i in range(3)})
seq = Sequence(reg, AnalogDevice)
seq.declare_channel("ryd", "rydberg_global")
T = 4000
amp = InterpolatedWaveform(T, [0, 2 * np.pi, 2 * np.pi, 0])
det = InterpolatedWaveform(T, [-4 * np.pi, 0, 4 * np.pi])
seq.add(Pulse(amp, det, 0), "ryd")
res = QutipEmulator.from_sequence(seq).run()
probs = np.abs(res.get_final_state().full().flatten()) ** 2
counts = res.sample_final_state(2000)
print("exact:", {format(i, "03b"): round(float(p), 4) for i, p in enumerate(probs) if p > 0.01})
print("sampled:", dict(sorted(counts.items(), key=lambda kv: -kv[1])[:4]))
print("PROBE2 OK")
