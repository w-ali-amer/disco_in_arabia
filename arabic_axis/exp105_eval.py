# -*- coding: utf-8 -*-
"""exp105_eval.py -- doc 102 (REGISTERED d5b989a): exact noise evaluator (E-X), EOM square pulses, threshold map.

E-X: Pulser's sampled waveforms (pulser.sampler.sample, 1 ns) -> H = X + X^dag, X = 0.5 a Omega e^{-i phi} |g><r|
- 0.5 (delta + Delta * 1_pulse) |r><r| (Pulser's construction, hamiltonian.py), tlist = arange(duration)/1000,
collapse sqrt(2 gamma)|r><r|, qutip mesolve.  Quasi-static noise as Pulser applies it: a = max(0, 1+eps) * waist(d_perp
to the y optical axis) * calib, Delta = doppler + detuning offset only inside pulse slots; averaged by Gauss-Hermite.
SPAM analytic: P(g) = (1-eta)[p_g(1-fp) + (1-p_g) fn] + eta(1-fp).  Gates in doc 102 §2 order; a failure stops.
Run with ~/pulser_mac_env.  Writes results_exp105.json (+ exp105_abstract/) (Rule 3).
"""
import itertools, json, math, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
import qutip
import pulser
from pulser import NoiseModel, Sequence
from pulser.sampler import sample
from pulser_simulation import QutipEmulator
from pulser._hamiltonian_data.hamiltonian_data import doppler_sigma
import exp94_common as C94
import exp104_pasqal as E

AMP_EOM = 12.0
SITE = 0                                   # trap (-20, 0): the outer single site used for every single-atom cell
R6 = [0, 10, 14, 46, 50, 60]; R3 = [0, 50, 46]
SPAM_BASE = dict(fp=0.01, fn=0.05, eta=0.005); SPAM_PESS = dict(fp=0.03, fn=0.08, eta=0.01)
AB = "exp105_abstract"
OUTFILE = "results_exp105_run3.json"                 # erratum doc 102 §7: run 1 stopped at G-NOISELESS-EX (kept)
S_RR = qutip.Qobj(np.array([[1, 0], [0, 0]], dtype=complex))          # basis (r, g)
S_GR = qutip.Qobj(np.array([[0, 0], [1, 0]], dtype=complex))          # |g><r|
RHO0 = qutip.Qobj(np.array([[0, 0], [0, 1]], dtype=complex))


# ---------------------------------------------------------------- sequences
def eom_sequence(pulses, traps):
    s = Sequence(E.register(traps), E.DEV); s.declare_channel("ch", "rydberg_global")
    if not pulses:
        s.delay(E.T_PULSE, "ch"); s.measure("ground-rydberg"); return s
    s.enable_eom_mode("ch", amp_on=AMP_EOM, detuning_on=0.0)
    for area, ph in pulses:
        T = max(16, int(round(area / AMP_EOM * 1000 / 4)) * 4)
        s.add_eom_pulse("ch", T, phase=ph % (2 * math.pi), protocol="no-delay")   # erratum doc 102 §7: min-delay gap drifts phase
    s.measure("ground-rydberg")
    return s


def build(family, spec, traps=(SITE,)):
    pulses = spec
    return (E.sequence(pulses, list(traps)) if family == "blackman" else eom_sequence(pulses, list(traps)))


def arm_specs(scale=1.0):
    r101 = json.load(open("results_exp101.json"))
    arms = {"A-2to1": (E.arm_2to1(), E.logical_2to1(), 0.5 + 1 / (2 * math.sqrt(2)))}
    if r101["decisions"]["which_referent_uniform"]:
        arms["A-WR6"] = (E.arm_wr(6), E.logical_wr(6), 0.5 + (2 * math.sqrt(2) + 1) / 10)
        arms["A-WR3"] = (E.arm_wr(3), E.logical_wr(3), 0.5 + math.sqrt(3) / 4)
    out = {}
    for name, (cells, logical, theory) in arms.items():
        phys, mapping = E.physical(cells)
        out[name] = {"specs": [E.pulses_for(t, m) for t, m in phys], "mapping": mapping, "logical": logical, "theory": theory}
    return out


# ---------------------------------------------------------------- E-X
def ex_pg(family, spec, a=1.0, Delta=0.0, gamma=0.0, modulation=False):
    seq = build(family, spec)
    smp = sample(seq, modulation=modulation)
    smp = smp.extend_duration(smp.max_duration + 1)          # as QutipEmulator (simulation.py:173); EOM tail = detuning_off
    cs = smp.channel_samples["ch"]
    n = len(cs.amp)                                                       # Pulser: sampling_times = arange(max_duration)
    amp, det, ph = np.array(cs.amp[:n]), np.array(cs.det[:n]), np.array(cs.phase[:n])
    mask = np.zeros(n)
    for sl in cs.slots:
        mask[sl.ti:min(sl.tf, n)] = 1.0
    if n < 2:
        return 1.0
    tlist = np.arange(n, dtype=float) / 1000
    X = qutip.QobjEvo([[S_GR, 0.5 * a * amp * np.exp(-1j * ph)], [S_RR, -0.5 * (det + Delta * mask)]], tlist=tlist)
    H = X + X.dag()
    c_ops = [math.sqrt(2 * gamma) * S_RR] if gamma > 0 else []
    res = qutip.mesolve(H, RHO0, tlist, c_ops=c_ops, options={"atol": 1e-10, "rtol": 1e-8, "max_step": 1e-3, "nsteps": 100000})   # erratum 2 doc 102 §8
    return float(np.real(res.states[-1].full()[1, 1]))


def gh(sigma, k):
    if sigma <= 0:
        return [(0.0, 1.0)]
    x, w = np.polynomial.hermite.hermgauss(k)
    return [(math.sqrt(2) * sigma * xi, wi / math.sqrt(math.pi)) for xi, wi in zip(x, w)]


def ex_pg_avg(args):
    family, spec, cell, k = args
    sa, sd = cell.get("amp_sigma", 0.0), math.hypot(doppler_sigma(cell.get("temperature", 0.0) * 1e-6), cell.get("detuning_sigma", 0.0))
    wst = cell.get("waist_factor", 1.0) * cell.get("calib", 1.0)
    tot = 0.0
    for e, we in gh(sa, k):
        for d, wd in gh(sd, k):
            tot += we * wd * ex_pg(family, spec, a=max(0.0, 1 + e) * wst, Delta=d, gamma=cell.get("gamma", 0.0))
    return tot


def spam(pg, s):
    return (1 - s["eta"]) * (pg * (1 - s["fp"]) + (1 - pg) * s["fn"]) + s["eta"] * (1 - s["fp"])


def waist_factor(trap, w=175.0):
    x = E.LAYOUT.traps_dict[trap][0]                                     # distance to the optical axis along y
    return math.exp(-((x / w) ** 2))


# ---------------------------------------------------------------- cells
def cells_def(outer):
    wf = waist_factor(outer)
    base = {"spam": SPAM_BASE}
    pess = {"spam": SPAM_PESS, "amp_sigma": 0.03, "detuning_sigma": 0.2, "temperature": 50.0, "gamma": 0.2,
            "waist_factor": wf, "calib": 0.95}
    return {"baseline": base, "temperature_50uK": {**base, "temperature": 50.0}, "dephasing_0.05": {**base, "gamma": 0.05},
            "dephasing_0.2": {**base, "gamma": 0.2}, "pessimistic_combined": pess}


def evaluate(pool, arms, family, cell, k=9):
    jobs, index = [], []
    for name, a in arms.items():
        for i, spec in enumerate(a["specs"]):
            jobs.append((family, spec, cell, k)); index.append((name, i))
    jobs.append((family, [], cell, k)); jobs.append((family, [(math.pi, 0.0)], cell, k))
    vals = pool.map(ex_pg_avg, jobs)
    s = cell["spam"]
    pg_g, pg_r = spam(vals[-2], s), spam(vals[-1], s)
    e0, e1 = 1 - pg_g, pg_r
    out = {"control": {"e0": e0, "e1": e1}}
    for name, a in arms.items():
        pg = [spam(vals[j], s) for j, (nm, i) in enumerate(index) if nm == name]
        pq = E.success_from_pg(pg, a["mapping"]); pc = E.classical_ceiling(a["logical"], e0, e1)
        out[name] = {"P_Q": pq, "P_C": pc, "margin": pq - pc, "pg": pg}
    return out


def power(arms, res, n_atoms):
    z = 3.0902 + 1.2816; out = {}
    e0, e1 = res["control"]["e0"], res["control"]["e1"]
    for name, a in arms.items():
        r = res[name]; delta = r["margin"]; pg = r["pg"]
        w = [0.0] * len(pg)
        for i, s_, ww in a["mapping"]:
            w[i] += ww * s_
        vq = sum(w[i] ** 2 * pg[i] * (1 - pg[i]) for i in range(len(pg)))
        h = 1e-4
        d0 = (E.classical_ceiling(a["logical"], e0 + h, e1) - E.classical_ceiling(a["logical"], e0 - h, e1)) / (2 * h)
        d1 = (E.classical_ceiling(a["logical"], e0, e1 + h) - E.classical_ceiling(a["logical"], e0, e1 - h)) / (2 * h)
        vc = d0 ** 2 * e0 * (1 - e0) + d1 ** 2 * e1 * (1 - e1)
        N = math.ceil(z ** 2 * (vq + vc) / delta ** 2) if delta > 0 else None
        seqs = len(pg) + 2
        cyc = math.ceil(N * seqs / n_atoms) if N else None
        mins = {hz: (cyc / hz / 60 if cyc else None) for hz in (0.5, 1.0, 3.0)}
        out[name] = {"delta": delta, "N_per_sequence": N, "sequences": seqs, "atoms_per_shot": n_atoms, "cycles": cyc,
                     "minutes_at_Hz": mins, "enters_ask": bool(delta >= 0.04 and cyc is not None and mins[0.5] <= 60)}
    return out


def stop(out, why):
    out["stopped_at"] = why; out["provenance"] = C94.provenance()
    C94.write_new_json(OUTFILE, out); print("STOPPED:", why, flush=True); sys.exit(0)


def main():
    os.makedirs(AB, exist_ok=True)
    arms = arm_specs(); out = {"doc": "102 (d5b989a)", "pulser": pulser.__version__, "amp_eom": AMP_EOM}
    with Pool(8) as pool:
        # 1. G-NOISELESS-EX
        worst = {}; ok = True; eom_theory = {}
        for fam in ("blackman", "eom"):
            for name, a in arms.items():
                ex = pool.map(ex_pg_avg, [(fam, sp, {}, 1) for sp in a["specs"]])
                pu = [pulser_tight(build(fam, sp)) for sp in a["specs"]]      # Pulser at atol 1e-12 (default tol ~1e-5 on EOM edges)
                dev = max(abs(x - y) for x, y in zip(ex, pu)); worst["%s/%s" % (fam, name)] = dev
                ok &= dev <= 1e-6
                succ = E.success_from_pg(ex, a["mapping"])
                tol = 1e-3 if fam == "blackman" else 5e-3
                eom_theory["%s/%s" % (fam, name)] = {"noiseless": succ, "theory": a["theory"], "within_tol": abs(succ - a["theory"]) <= tol}
                ok &= abs(succ - a["theory"]) <= tol
        out["G_NOISELESS_EX"] = {"max_dev_vs_pulser": worst, "theory": eom_theory, "pass": bool(ok)}
        print("G-NOISELESS-EX", out["G_NOISELESS_EX"], flush=True)
        if not ok:
            stop(out, "G-NOISELESS-EX")
        # modulation descriptive (noiseless, E-X on modulated samples)
        mod = {}
        for fam in ("blackman", "eom"):
            for name, a in arms.items():
                pg = [ex_pg(fam, sp, modulation=True) for sp in a["specs"]]
                mod["%s/%s" % (fam, name)] = E.success_from_pg(pg, a["mapping"]) - eom_theory["%s/%s" % (fam, name)]["noiseless"]
        out["modulation_descriptive"] = {"deviation": mod, "needs_calibration_note": bool(max(abs(v) for v in mod.values()) > 1e-2)}
        print("modulation", out["modulation_descriptive"], flush=True)
        # 2. G-DEPH
        deph = {}
        ts = [500, 1000, 2000, 4000]
        for path in ("pulser_mesolve", "EX"):
            P = []
            for t in ts:
                s = Sequence(E.register([SITE]), E.DEV); s.declare_channel("ch", "rydberg_global")
                s.add(pulser.Pulse.ConstantDetuning(pulser.waveforms.BlackmanWaveform(1000, math.pi / 2), 0.0, 0.0), "ch")
                s.delay(t, "ch")
                s.add(pulser.Pulse.ConstantDetuning(pulser.waveforms.BlackmanWaveform(1000, math.pi / 2), 0.0, math.pi), "ch")
                s.measure("ground-rydberg")
                if path == "pulser_mesolve":
                    st = QutipEmulator.from_sequence(s, noise_model=NoiseModel(dephasing_rate=0.2)).run().states[-1].full()
                    P.append(float(np.real(st[1, 1])))
                else:
                    smp = sample(s).channel_samples["ch"]; n = len(smp.amp); tl = np.arange(n) / 1000
                    X = qutip.QobjEvo([[S_GR, 0.5 * np.array(smp.amp) * np.exp(-1j * np.array(smp.phase))], [S_RR, -0.5 * np.array(smp.det)]], tlist=tl)
                    r = qutip.mesolve(X + X.dag(), RHO0, tl, c_ops=[math.sqrt(0.4) * S_RR], options={"atol": 1e-10, "rtol": 1e-8, "max_step": 1e-3, "nsteps": 100000})
                    P.append(float(np.real(r.states[-1].full()[1, 1])))
            y = np.log(np.clip(2 * np.array(P) - 1, 1e-12, None)); slope = np.polyfit(np.array(ts) / 1000, y, 1)[0]
            deph[path] = {"P": P, "Gamma": -slope, "ratio_to_gamma": -slope / 0.2}
        ok = all(abs(v["ratio_to_gamma"] - 1) <= 0.02 for v in deph.values())
        out["G_DEPH"] = {**deph, "pass": bool(ok), "T2_equals_1_over_gamma": bool(ok)}
        print("G-DEPH", out["G_DEPH"], flush=True)
        if not ok:
            stop(out, "G-DEPH")
        # 3. G-XVAL-QS against exp104's Pulser-sampler values of record (single atom; exp104 used the centre site)
        r104 = json.load(open("results_exp104.json"))
        xq = {}; ok = True
        qs_cells = {"baseline": {"spam": SPAM_BASE}, "amp_sigma_0.03": {"spam": SPAM_BASE, "amp_sigma": 0.03},
                    "detuning_sigma_0.2": {"spam": SPAM_BASE, "detuning_sigma": 0.2},
                    "temperature_50uK": {"spam": SPAM_BASE, "temperature": 50.0}}
        for cname, cell in qs_cells.items():
            res = evaluate(pool, arms, "blackman", cell)
            for name in arms:
                d = res[name]["P_Q"] - r104["arms"][name]["noisy"][cname]["P_Q"]
                xq["%s/%s" % (cname, name)] = d; ok &= abs(d) <= 0.015
        out["G_XVAL_QS"] = {"diff_EX_minus_pulser": xq, "pass": bool(ok)}
        print("G-XVAL-QS", out["G_XVAL_QS"], flush=True)
        if not ok:
            stop(out, "G-XVAL-QS")
        # 4. G-XVAL-DEPH: 20 Pulser repeats (runs=50) vs E-X on dephasing 0.2 + baseline SPAM, A-2to1
        a = arms["A-2to1"]
        nm = NoiseModel(p_false_pos=0.01, p_false_neg=0.05, state_prep_error=0.005, dephasing_rate=0.2, runs=50, samples_per_run=40)
        reps = pool.map(_pulser_rep, [(a["specs"], a["mapping"], rep) for rep in range(20)])
        ex = evaluate(pool, {"A-2to1": a}, "blackman", {"spam": SPAM_BASE, "gamma": 0.2})["A-2to1"]["P_Q"]
        mu, se = float(np.mean(reps)), float(np.std(reps, ddof=1) / math.sqrt(len(reps)))
        ok = abs(mu - ex) <= 2 * se
        out["G_XVAL_DEPH"] = {"pulser_reps": reps, "mean": mu, "se": se, "EX": ex, "pass": bool(ok)}
        print("G-XVAL-DEPH", out["G_XVAL_DEPH"], flush=True)
        if not ok:
            stop(out, "G-XVAL-DEPH")
        # 5. G-QUAD on pessimistic (EOM and Blackman)
        cd = cells_def(SITE); qd = {}
        for fam in ("blackman", "eom"):
            v7 = evaluate(pool, arms, fam, cd["pessimistic_combined"], k=7); v11 = evaluate(pool, arms, fam, cd["pessimistic_combined"], k=11)
            for name in arms:
                qd["%s/%s" % (fam, name)] = abs(v7[name]["P_Q"] - v11[name]["P_Q"])
        ok = max(qd.values()) <= 1e-3
        out["G_QUAD"] = {"abs_diff_7_vs_11": qd, "pass": bool(ok)}
        print("G-QUAD", out["G_QUAD"], flush=True)
        if not ok:
            stop(out, "G-QUAD")
        # 6. G-XTALK-EOM (Pulser noiseless), register of record
        xt = {}
        for regname, traps in (("R6", R6), ("R3", R3)):
            dev = 0.0
            for name, a in arms.items():
                for sp in a["specs"]:
                    single = E.pg_noiseless(build("eom", sp))[0]
                    sites = E.pg_noiseless(build("eom", sp, traps))
                    dev = max(dev, max(abs(p - single) for p in sites))
            xt[regname] = {"max_dev": float(dev), "pass": bool(dev <= 0.01)}
            if xt[regname]["pass"]:
                break
        rec = next((k for k in ("R6", "R3") if k in xt and xt[k]["pass"]), "single")
        traps = {"R6": R6, "R3": R3}.get(rec, [SITE])
        out["G_XTALK_EOM"] = {**xt, "register_of_record": rec, "traps": traps}
        print("G-XTALK-EOM", out["G_XTALK_EOM"], flush=True)
        # 7. G-SER (EOM, register of record)
        ser = True
        for name, a in arms.items():
            for k, sp in enumerate(a["specs"]):
                js = build("eom", sp, traps).to_abstract_repr(); ser &= Sequence.from_abstract_repr(js).to_abstract_repr() == js
                open(os.path.join(AB, "%s_%02d.json" % (name, k)), "w").write(js)
        for nm_, sp in (("g", []), ("r", [(math.pi, 0.0)])):
            js = build("eom", sp, traps).to_abstract_repr(); ser &= Sequence.from_abstract_repr(js).to_abstract_repr() == js
            open(os.path.join(AB, "A-CTRL_%s.json" % nm_), "w").write(js)
        out["G_SER"] = bool(ser)
        # cells: Blackman (exact, forward pointer for exp104) and EOM
        outer = max(traps, key=lambda t: abs(E.LAYOUT.traps_dict[t][0]))
        cd = cells_def(outer); out["cells"] = {}
        for fam in ("blackman", "eom"):
            for cname, cell in cd.items():
                r = evaluate(pool, arms, fam, cell)
                out["cells"]["%s/%s" % (fam, cname)] = r
                print(fam, cname, {n: round(r[n]["margin"], 4) for n in arms}, flush=True)
        # threshold map (EOM)
        out["map"] = {}
        for T in (5, 10, 20, 30, 50):
            for g in (0.02, 0.05, 0.1, 0.2, 0.5):
                cell = {"spam": SPAM_BASE, "amp_sigma": 0.03, "detuning_sigma": 0.2, "temperature": float(T), "gamma": g,
                        "waist_factor": waist_factor(outer), "calib": 0.95}
                r = evaluate(pool, arms, "eom", cell)
                out["map"]["T%d_g%s" % (T, g)] = {n: r[n]["margin"] for n in arms}
                print("map", T, g, {n: round(r[n]["margin"], 4) for n in arms}, flush=True)
        # power
        out["power"] = {fam: power(arms, out["cells"]["%s/pessimistic_combined" % fam], len(traps)) for fam in ("blackman", "eom")}
        print("power", json.dumps(out["power"]), flush=True)
    for v in out["cells"].values():
        for n in arms:
            v[n]["pg"] = [float(x) for x in v[n]["pg"]]
    out["provenance"] = C94.provenance()
    C94.write_new_json(OUTFILE, out)
    print("DONE", flush=True)


def pulser_tight(seq):
    st = QutipEmulator.from_sequence(seq).run(atol=1e-12, rtol=1e-10, nsteps=200000).states[-1].full().ravel()
    return float(abs(st[1]) ** 2)


def _pulser_rep(args):
    specs, mapping, rep = args
    np.random.seed(105000 + rep)
    nm = NoiseModel(p_false_pos=0.01, p_false_neg=0.05, state_prep_error=0.005, dephasing_rate=0.2, runs=50, samples_per_run=40)
    pg = [E.pg_noisy(build("blackman", sp), nm) for sp in specs]
    return E.success_from_pg(pg, mapping)


if __name__ == "__main__":
    main()
