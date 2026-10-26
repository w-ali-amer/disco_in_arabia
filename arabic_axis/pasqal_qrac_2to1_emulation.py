# 2->1 QRAC on a Pasqal AnalogDevice (global Rydberg drive) -- EMULATOR feasibility check (not a registered experiment).
import math, numpy as np, warnings
warnings.filterwarnings("ignore")
import pulser
from pulser import Pulse, Sequence, Register
from pulser.devices import AnalogDevice
from pulser.waveforms import BlackmanWaveform
from pulser_simulation import QutipEmulator
TH = {(0, 0): 7 * math.pi / 4, (0, 1): math.pi / 4, (1, 0): 5 * math.pi / 4, (1, 1): 3 * math.pi / 4}
def seq_for(word, query):
    reg = Register.from_coordinates([(0, 0)], prefix="q")
    s = Sequence(reg, AnalogDevice); s.declare_channel("ch", "rydberg_global")
    th = TH[word]; ph = 0.0
    if th > math.pi:
        th, ph = 2 * math.pi - th, math.pi                                                     # shortest rotation
    s.add(Pulse.ConstantDetuning(BlackmanWaveform(2000, th), 0.0, ph), "ch")                 # encode codeword
    if query == 1:
        s.add(Pulse.ConstantDetuning(BlackmanWaveform(1500, math.pi / 2), 0.0, 0.0), "ch")  # rotate Y-axis onto Z
    s.measure("ground-rydberg")
    return s
def success(noise=None, runs=1):
    tot = []
    for word in TH:
        for q in (0, 1):
            sim = QutipEmulator.from_sequence(seq_for(word, q), noise_model=noise) if noise else QutipEmulator.from_sequence(seq_for(word, q))
            res = sim.run()
            if noise is None:
                amp = res.states[-1].full().ravel()                      # pulser ground-rydberg basis order: [r, g]
                p_g = float(abs(amp[1]) ** 2)
            else:
                c = res.sample_final_state(N_samples=4000); n = sum(c.values()); p_g = c.get("0", 0) / n   # bitstring 0 = ground
            bit = word[q]
            tot.append(p_g if bit == 0 else 1 - p_g)
    return float(np.mean(tot)), tot
ideal, per = success()
print("noiseless mean single-shot success:", round(ideal, 4), [round(x, 3) for x in per])
for fp, fn in ((0.01, 0.05), (0.03, 0.08)):
    nm = pulser.NoiseModel(p_false_pos=fp, p_false_neg=fn, state_prep_error=0.005, runs=50, samples_per_run=40)
    m, _ = success(nm)
    print("SPAM fp=%.2f fn=%.2f: mean success %.4f   (noisy classical-bit ceiling with same readout ~ %.4f)" % (fp, fn, m, 0.75*(1-(fp+fn)/2)+0.25*(fp+fn)/2))
