import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.devices import AnalogDevice
from pulser.waveforms import InterpolatedWaveform, ConstantWaveform
from pulser_simulation import QutipEmulator

print("dmm_objects:", AnalogDevice.dmm_objects)

spacing = 6.0
reg = Register({f"a{i}": ((i - 1) * spacing, 0.0) for i in range(3)})
dmap = reg.define_detuning_map({"a0": 1.0, "a1": 0.3, "a2": 0.0})

seq = Sequence(reg, AnalogDevice)
seq.declare_channel("ryd", "rydberg_global")
seq.config_detuning_map(dmap, "dmm_0")
T = 4000
amp = InterpolatedWaveform(T, [0, 2 * np.pi, 2 * np.pi, 0])
det = InterpolatedWaveform(T, [-4 * np.pi, 0, 4 * np.pi])
seq.add(Pulse(amp, det, 0), "ryd")
seq.add_dmm_detuning(ConstantWaveform(T, -2 * np.pi * 3), "dmm_0")

sim = QutipEmulator.from_sequence(seq)
res = sim.run()
probs = np.abs(res.get_final_state().full().flatten()) ** 2
counts = res.sample_final_state(2000)
print("exact:", {format(i, "03b"): round(float(p), 4) for i, p in enumerate(probs) if p > 0.01})
print("sampled:", dict(sorted(counts.items(), key=lambda kv: -kv[1])[:4]))
print("PROBE OK")
