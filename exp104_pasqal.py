# -*- coding: utf-8 -*-
"""exp104_pasqal.py -- doc 101 (REGISTERED 8a02346): Pasqal emulation upgrades (Pulser 1.9, AnalogDevice), EMULATION ONLY.

Arms: A-2to1 (square code, 8 physical sequences), A-WR6 / A-WR3 (exp101 which-referent; iff which_referent_uniform),
A-CTRL (|g>, |r>).  Bloch convention (probed): |g> = +z; pulse (area th, phase ph) maps +z -> (sin th sin ph,
sin th cos ph, cos th); prepare t: th = acos t_z, ph = pi/2 - azimuth(t); measure along m: inverse rotation
(phase + pi), then ground = '+'.  Register: calibrated 61-trap layout; phi_max = 0.05 rad -> d_min = 18.05 um; greedy.
Noise ranges (doc 101 §3), runs 50 x samples 40.  Gates G-NOISELESS, G-XTALK, G-SER.  Power (doc 101 §5).
Run with ~/pulser_mac_env.  Writes results_exp104.json + exp104_abstract/*.json (Rule 3).
"""
import itertools, json, math, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
import pulser
from pulser import Pulse, Sequence, NoiseModel
from pulser.devices import AnalogDevice as DEV
from pulser.waveforms import BlackmanWaveform
from pulser_simulation import QutipEmulator
import exp94_common as C94

T_PULSE = 1000
C6 = DEV.interaction_coeff
PHI_MAX, T_TOTAL = 0.05, 2.0
D_MIN = (C6 * T_TOTAL / PHI_MAX) ** (1 / 6)
LAYOUT = list(DEV.calibrated_register_layouts.values())[0]
AB = "exp104_abstract"


# ---------------------------------------------------------------- geometry of the arms
def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def arm_2to1():
    """square code in the x-z plane; queries: feature 0 -> axis z, feature 1 -> axis x. cells: (state, axis, sign, w)."""
    cells = []
    for b0, b1 in itertools.product((0, 1), repeat=2):
        t = unit([1 - 2 * b1, 0, 1 - 2 * b0])
        cells.append((t, np.array([0, 0, 1.0]), +1 if b0 == 0 else -1, 1 / 8))   # answer '+' iff bit 0
        cells.append((t, np.array([1.0, 0, 0]), +1 if b1 == 0 else -1, 1 / 8))
    return cells


def arm_wr(C):
    """which-referent: class axes n_v (C=6 octahedron, C=3 triangle in x-z plane); state (n_a - n_b)/|.|;
    query v=a -> measure n_a, '+' = first; v=b -> measure n_b, '-' = second (answer sign -1)."""
    if C == 6:
        n = [np.array(v, float) for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
    else:
        n = [np.array([math.sin(2 * math.pi * k / 3), 0, math.cos(2 * math.pi * k / 3)]) for k in range(3)]
    cells, w = [], 1 / (C * (C - 1) * 2)
    for a, b in itertools.permutations(range(C), 2):
        t = unit(n[a] - n[b])
        cells.append((t, n[a], +1, w)); cells.append((t, n[b], -1, w))
    return cells


def physical(cells):
    """merge cells sharing (state, measurement axis up to sign) into physical sequences."""
    phys, mapping = [], []
    for t, m, sg, w in cells:
        key_m = m if (m[np.argmax(np.abs(m))] > 0) else -m          # canonical axis orientation
        flip = 1 if np.allclose(key_m, m) else -1
        for i, (tt, mm) in enumerate(phys):
            if np.allclose(tt, t, atol=1e-9) and np.allclose(mm, key_m, atol=1e-9):
                break
        else:
            phys.append((t, key_m)); i = len(phys) - 1
        mapping.append((i, sg * flip, w))
    return phys, mapping


def success_from_pg(pg, mapping):
    """P(correct) = sum_c w_c * P(outcome sign = answer sign); ground = '+' along the canonical axis."""
    return float(sum(w * (pg[i] if s > 0 else 1 - pg[i]) for i, s, w in mapping))


# ---------------------------------------------------------------- pulses
def pulses_for(t, m, scale=1.0):
    out = []
    th = math.acos(max(-1, min(1, t[2]))); psi = math.atan2(t[1], t[0])
    if th > 1e-9:
        out.append((th * scale, math.pi / 2 - psi))
    thm = math.acos(max(-1, min(1, m[2]))); psim = math.atan2(m[1], m[0])
    if thm > 1e-9:
        out.append((thm * scale, math.pi / 2 - psim + math.pi))
    return out


def register(trap_ids):
    return LAYOUT.define_register(*trap_ids, qubit_ids=["q%d" % i for i in range(len(trap_ids))])


def sequence(pulses, trap_ids):
    s = Sequence(register(trap_ids), DEV); s.declare_channel("ch", "rydberg_global")
    if not pulses:
        s.delay(T_PULSE, "ch")
    for area, ph in pulses:
        s.add(Pulse.ConstantDetuning(BlackmanWaveform(T_PULSE, area), 0.0, ph % (2 * math.pi)), "ch")
    s.measure("ground-rydberg")
    return s


def pg_noiseless(seq):
    """per-atom P(ground) from the final statevector (basis [r, g] per atom)."""
    st = QutipEmulator.from_sequence(seq).run().states[-1].full().ravel()
    n = int(round(math.log2(len(st)))); probs = np.abs(st) ** 2
    out = []
    for k in range(n):
        pg = 0.0
        for idx, p in enumerate(probs):
            if (idx >> (n - 1 - k)) & 1:          # bit 1 in basis [r, g] ordering = g
                pg += p
        out.append(pg)
    return out


def pg_noisy(seq, nm):
    c = QutipEmulator.from_sequence(seq, noise_model=nm).run().sample_final_state(N_samples=2000)
    n = sum(c.values()); return c.get("0", 0) / n


# ---------------------------------------------------------------- classical ceiling through an asymmetric channel
def classical_ceiling(cells_logical, e0, e1):
    """cells_logical: list of (x, q, answer, w).  1 bit: prep g (b=0) or r (b=1); read flips g->r w.p. e0, r->g w.p. e1."""
    X = sorted({c[0] for c in cells_logical}, key=str); Q = sorted({c[1] for c in cells_logical}, key=str)
    A = {q: sorted({c[2] for c in cells_logical if c[1] == q}, key=str) for q in Q}
    chan = {0: {0: 1 - e0, 1: e0}, 1: {0: e1, 1: 1 - e1}}
    best = 0.0
    for dec in itertools.product(*[list(itertools.product(A[q], repeat=2)) for q in Q]):
        D = dict(zip(Q, dec)); tot = 0.0
        for x in X:
            tot += max(sum(w * sum(chan[b][bp] * (D[q][bp] == a) for bp in (0, 1))
                           for xx, q, a, w in cells_logical if xx == x) for b in (0, 1))
        best = max(best, tot)
    return best


def logical_2to1():
    return [((b0, b1), f, (b0, b1)[f], 1 / 8) for b0, b1 in itertools.product((0, 1), repeat=2) for f in (0, 1)]


def logical_wr(C):
    w = 1 / (C * (C - 1) * 2)
    return [((a, b), v, "first" if v == a else "second", w) for a, b in itertools.permutations(range(C), 2) for v in (a, b)]


# ---------------------------------------------------------------- main
def main():
    os.makedirs(AB, exist_ok=True)
    r101 = json.load(open("results_exp101.json"))
    arms = {"A-2to1": (arm_2to1(), logical_2to1(), 0.5 + 1 / (2 * math.sqrt(2)))}
    if r101["decisions"]["which_referent_uniform"]:
        arms["A-WR6"] = (arm_wr(6), logical_wr(6), 0.5 + (2 * math.sqrt(2) + 1) / 10)
        arms["A-WR3"] = (arm_wr(3), logical_wr(3), 0.5 + math.sqrt(3) / 4)
    out = {"doc": "101 (8a02346)", "pulser": pulser.__version__, "D_MIN_um": D_MIN, "arms": {}}
    # register: greedy from centre outwards
    traps = sorted(LAYOUT.traps_dict.items(), key=lambda kv: (round(float(np.linalg.norm(kv[1])), 6), kv[0]))
    chosen = []
    for tid, pos in traps:
        if all(np.linalg.norm(np.asarray(pos) - np.asarray(LAYOUT.traps_dict[c])) >= D_MIN for c in chosen):
            chosen.append(tid)
    centre, outer = chosen[0], chosen[-1]
    out["register"] = {"trap_ids": chosen, "n_atoms": len(chosen),
                       "radii_um": [float(np.linalg.norm(LAYOUT.traps_dict[c])) for c in chosen]}
    print("register", out["register"], flush=True)
    ctrl_seqs = {"g": [], "r": [(math.pi, 0.0)]}
    g_noiseless = g_xtalk = g_ser = True
    for name, (cells, logical, theory) in arms.items():
        phys, mapping = physical(cells)
        pg1 = [pg_noiseless(sequence(pulses_for(t, m), [centre]))[0] for t, m in phys]
        p_nl = success_from_pg(pg1, mapping)
        g_noiseless &= abs(p_nl - theory) < 1e-3
        xt = []
        for k, (t, m) in enumerate(phys):
            per_site = pg_noiseless(sequence(pulses_for(t, m), chosen))
            xt.append(max(abs(p - pg1[k]) for p in per_site))
        g_xtalk &= max(xt) <= 0.01
        for k, (t, m) in enumerate(phys):
            s = sequence(pulses_for(t, m), chosen); js = s.to_abstract_repr()
            s2 = Sequence.from_abstract_repr(js); g_ser &= s2.to_abstract_repr() == js
            open(os.path.join(AB, "%s_%02d.json" % (name, k)), "w").write(js)
        out["arms"][name] = {"n_physical": len(phys), "n_logical": len(cells), "noiseless": p_nl, "theory": theory,
                             "xtalk_max_pg_dev": float(max(xt)), "noisy": {}}
        print(name, out["arms"][name], flush=True)
    for k, (nm_, pl) in enumerate(ctrl_seqs.items()):
        s = sequence(pl, chosen); js = s.to_abstract_repr(); g_ser &= Sequence.from_abstract_repr(js).to_abstract_repr() == js
        open(os.path.join(AB, "A-CTRL_%s.json" % nm_), "w").write(js)
    base = dict(p_false_pos=0.01, p_false_neg=0.05, state_prep_error=0.005, runs=50, samples_per_run=40)
    pess_spam = dict(p_false_pos=0.03, p_false_neg=0.08, state_prep_error=0.01)
    settings = {"baseline": (base, centre, 1.0), "spam_pess": ({**base, **pess_spam}, centre, 1.0),
                "amp_sigma_0.03": ({**base, "amp_sigma": 0.03}, centre, 1.0),
                "detuning_sigma_0.2": ({**base, "detuning_sigma": 0.2}, centre, 1.0),
                "temperature_50uK": ({**base, "temperature": 50.0}, centre, 1.0),
                "dephasing_0.05": ({**base, "dephasing_rate": 0.05}, centre, 1.0),
                "dephasing_0.2": ({**base, "dephasing_rate": 0.2}, centre, 1.0),
                "laser_waist_175_outer": ({**base, "laser_waist": 175.0}, outer, 1.0),
                "calib_x0.95": (base, centre, 0.95), "calib_x1.05": (base, centre, 1.05),
                "pessimistic_combined": ({**base, **pess_spam, "amp_sigma": 0.03, "detuning_sigma": 0.2, "temperature": 50.0,
                                          "dephasing_rate": 0.2, "laser_waist": 175.0}, outer, 0.95)}
    out["control"] = {}
    for sname, (kw, site, scale) in settings.items():
        nm = NoiseModel(**kw)
        pg_g = pg_noisy(sequence([], [site]), nm); pg_r = pg_noisy(sequence([(math.pi * scale, 0.0)], [site]), nm)
        e0, e1 = 1 - pg_g, pg_r
        out["control"][sname] = {"e0_g_read_as_r": e0, "e1_r_read_as_g": e1}
        for name, (cells, logical, theory) in arms.items():
            phys, mapping = physical(cells)
            pg = [pg_noisy(sequence(pulses_for(t, m, scale), [site]), nm) for t, m in phys]
            pq = success_from_pg(pg, mapping); pc = classical_ceiling(logical, e0, e1)
            out["arms"][name]["noisy"][sname] = {"P_Q": pq, "P_C_channel": pc, "margin": pq - pc, "pg": pg}
            print(sname, name, round(pq, 4), round(pc, 4), round(pq - pc, 4), flush=True)
    # power (doc 101 §5) under pessimistic_combined
    z = 3.0902 + 1.2816
    ctrl = out["control"]["pessimistic_combined"]
    out["power"] = {}
    for name, (cells, logical, theory) in arms.items():
        phys, mapping = physical(cells)
        rec = out["arms"][name]["noisy"]["pessimistic_combined"]; pg = rec["pg"]; delta = rec["margin"]
        wphys = collections_sum(mapping, len(phys))
        var_q1 = sum((wphys[i] ** 2) * pg[i] * (1 - pg[i]) for i in range(len(phys)))       # x 1/N
        e0, e1 = ctrl["e0_g_read_as_r"], ctrl["e1_r_read_as_g"]; h = 1e-4
        d0 = (classical_ceiling(logical, e0 + h, e1) - classical_ceiling(logical, e0 - h, e1)) / (2 * h)
        d1 = (classical_ceiling(logical, e0, e1 + h) - classical_ceiling(logical, e0, e1 - h)) / (2 * h)
        var_c1 = d0 ** 2 * e0 * (1 - e0) + d1 ** 2 * e1 * (1 - e1)
        N = math.ceil((z ** 2) * (var_q1 + var_c1) / delta ** 2) if delta > 0 else None
        seqs = len(phys) + 2
        samples = N * seqs if N else None
        cycles = math.ceil(samples / len(chosen)) if N else None
        enters = bool(delta >= 0.04 and cycles is not None and cycles / 0.5 / 60 <= 60)
        out["power"][name] = {"delta_pess": delta, "N_samples_per_sequence": N, "physical_sequences_incl_ctrl": seqs,
                              "total_samples": samples, "device_cycles": cycles,
                              "minutes_at_Hz": {hz: (cycles / hz / 60 if cycles else None) for hz in (0.5, 1.0, 3.0)},
                              "enters_ask": enters}
        print("power", name, out["power"][name], flush=True)
    out["gates"] = {"G_NOISELESS": bool(g_noiseless), "G_XTALK": bool(g_xtalk), "G_SER": bool(g_ser)}   # erratum doc 101 §9: numpy bool
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp104.json", out)
    print(json.dumps(out["gates"]), flush=True)


def collections_sum(mapping, n):
    w = [0.0] * n
    for i, s, ww in mapping:
        w[i] += ww * s                      # signed: d P_Q / d p_i
    return w


if __name__ == "__main__":
    main()
