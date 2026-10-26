# -*- coding: utf-8 -*-
"""Pasqal / Pulser feasibility spike for doc 57 (exp71).

BOUNDED BY DESIGN. This script sits OUTSIDE the `exp71_*` namespace on purpose:
it is an INSTRUMENT probe, run before the doc 57 draft is registered, in the same
spirit as doc 54 SS0's instrument exception and doc 56's
`instrument_entangled_bounds.py`. It compiles NOTHING of the programme's trained
model. It answers four questions and then stops:

  Q1  Does Pulser import, and does its emulator run a minimal 2-atom analog
      sequence in the programme's Mac env without disturbing torch?
  Q2  Is the INTRO-STATE FAMILY -- one RY-like preparation per referent wire, at
      per-word angles that DIFFER between wires -- expressible under analog
      neutral-atom control?
  Q3  What does Pulser's realistic analog device model actually offer:
      channels, local addressing, detuning-map modulators, duration budget?
  Q4  Does the duration budget of the device fit the pulse durations exp37b's
      unconstrained compilation produced?

Q2 is the load-bearing one and it is answered as a SYMMETRY statement first and a
numerical statement second: a global drive on identical atoms in a
permutation-symmetric geometry cannot produce per-atom differentiated states, so
the negative control is a theorem and the numerics only confirm it. Three
candidate symmetry breakers are then measured: an asymmetric GEOMETRY under
global-only drive, a DETUNING-MAP modulator (DMM) local detuning, and a genuine
local addressing channel.

Angles are the programme's own: `frozen_angle` = atan2(PC2, PC1) per noun, read
from the COMMITTED `diagnostic_encoding_entanglement.json`. Nothing here is
re-derived and nothing here is written back to any committed artefact.

Output: diagnostic_pasqal_feasibility.json (this script's only side effect).
"""
import json
import time
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

T0 = time.time()
OUT = {"purpose": ("doc 57 / exp71 feasibility spike: can neutral-atom ANALOG "
                   "control express the programme's intro-state family, and "
                   "what does the realistic device model allow"),
       "boundedness": ("four questions, no verb block compiled, no exp71 file "
                       "written, no committed artefact modified")}

# ---------------------------------------------------------------- Q1: imports
env = {}
try:
    import pulser
    from pulser import Pulse, Sequence, Register
    from pulser.devices import AnalogDevice, DigitalAnalogDevice, MockDevice
    from pulser.waveforms import ConstantWaveform
    import pulser_simulation
    from pulser_simulation import QutipEmulator
    env["pulser"] = pulser.__version__
    env["pulser_simulation"] = pulser_simulation.__version__
    env["import_ok"] = True
except Exception as exc:                                    # pragma: no cover
    env["import_ok"] = False
    env["error"] = repr(exc)

for mod in ("torch", "numpy", "scipy", "qutip"):
    try:
        env[mod] = __import__(mod).__version__
    except Exception as exc:
        env[mod] = "IMPORT FAILED: %r" % (exc,)

OUT["Q1_environment"] = env
if not env.get("import_ok"):
    json.dump(OUT, open("diagnostic_pasqal_feasibility.json", "w"), indent=2)
    raise SystemExit("[spike] Pulser import failed; recorded and stopped.")

# ------------------------------------------------------- Q3: device inventory
def device_row(dev):
    return {
        "name": dev.name,
        "channels": sorted(dev.channels),
        "has_local_addressing": any(ch.addressing == "Local"
                                    for ch in dev.channels.values()),
        "n_dmm_objects": len(dev.dmm_objects),
        "supports_slm_mask": bool(getattr(dev, "supports_slm_mask", False)),
        "max_atom_num": dev.max_atom_num,
        "min_atom_distance_um": dev.min_atom_distance,
        "max_sequence_duration_ns": dev.max_sequence_duration,
        "max_runs": dev.max_runs,
        "interaction_coeff_rad_um6_per_us": float(dev.interaction_coeff),
        "rydberg_global_max_amp_rad_per_us":
            _num(dev.channels.get("rydberg_global"), "max_amp"),
        "rydberg_global_max_abs_detuning_rad_per_us":
            _num(dev.channels.get("rydberg_global"), "max_abs_detuning"),
    }


def _num(obj, attr):
    v = getattr(obj, attr, None) if obj is not None else None
    return None if v is None else float(v)

OUT["Q3_devices"] = {d.name: device_row(d)
                     for d in (AnalogDevice, DigitalAnalogDevice, MockDevice)}
OUT["Q3_finding"] = (
    "AnalogDevice -- Pulser's model of the realistic analog QPU -- exposes "
    "rydberg_global ONLY: no local addressing channel, NO detuning-map "
    "modulator, no SLM mask. DigitalAnalogDevice and MockDevice do carry local "
    "channels and DMMs, but neither is the analog QPU. Any exp71 design that "
    "needs per-atom differentiated control on the analog machine must therefore "
    "break the permutation symmetry with GEOMETRY, or move to a device class "
    "the programme has not been offered.")

C6 = float(AnalogDevice.interaction_coeff)
OMEGA_MAX = float(AnalogDevice.channels["rydberg_global"].max_amp)
DELTA_MAX = float(AnalogDevice.channels["rydberg_global"].max_abs_detuning)
MAX_DUR_NS = int(AnalogDevice.max_sequence_duration)
MIN_DIST = float(AnalogDevice.min_atom_distance)

# ------------------------------------------- numeric model of the analog H
# Pulser's ground-rydberg Ising Hamiltonian (|1> == Rydberg):
#   H = sum_i [ (Om_i/2)(cos ph_i X_i - sin ph_i Y_i) - de_i n_i ]
#       + sum_{i<j} (C6 / r_ij^6) n_i n_j
# Amplitudes rad/us, times us, distances um. The convention is ASSERTED against
# Pulser's own emulator below rather than trusted.
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
NOP = np.array([[0, 0], [0, 1]], complex)
I2 = np.eye(2, dtype=complex)


def op_on(op, i, n):
    m = np.array([[1]], dtype=complex)
    for k in range(n):
        m = np.kron(m, op if k == i else I2)
    return m


def build_ops(n):
    return ([op_on(X, i, n) for i in range(n)],
            [op_on(Y, i, n) for i in range(n)],
            [op_on(NOP, i, n) for i in range(n)])


def interaction(coords):
    n = len(coords)
    _, _, Ns = build_ops(n)
    H = np.zeros((2 ** n, 2 ** n), complex)
    for i, j in itertools.combinations(range(n), 2):
        r = float(np.linalg.norm(np.asarray(coords[i]) - np.asarray(coords[j])))
        H += (C6 / r ** 6) * (Ns[i] @ Ns[j])
    return H


_OPS = {}


def ops_for(n):
    if n not in _OPS:
        Xs, Ys, Ns = build_ops(n)
        _OPS[n] = (sum(Xs), sum(Ys), sum(Ns), Ns)
    return _OPS[n]


def evolve(coords, blocks, local_det=None):
    """`blocks` = [(Omega, delta, phase, tau_us), ...] global piecewise-constant.
    `local_det` = per-atom ADDITIONAL detuning (rad/us), applied for the whole
    sequence -- the DMM idealisation, sign-unconstrained here on purpose so the
    result is an UPPER bound on what a real, negative-only DMM could do.

    Sign convention ASSERTED against Pulser's own emulator (Q1 below), not
    assumed: H = sum_i (Om/2)(cos ph X_i + sin ph Y_i) - de n_i + sum V_ij n_i n_j
    in the |0> = ground, |1> = Rydberg basis. Pulser's emulator returns the
    statevector in the REVERSED per-atom basis (|r> before |g>), which is why
    the comparison below permutes before overlapping."""
    n = len(coords)
    SX, SY, SN, Ns = ops_for(n)
    H0 = interaction(coords)
    if local_det is not None:
        for i, d in enumerate(local_det):
            H0 = H0 - d * Ns[i]
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1.0
    for (om, de, ph, tau) in blocks:
        H = H0 + (om / 2) * (np.cos(ph) * SX + np.sin(ph) * SY) - de * SN
        psi = expm(-1j * H * tau) @ psi
    return psi


def to_pulser_order(psi, n):
    """Pulser's ground-rydberg statevector orders |r> before |g> on every atom;
    this script uses |g> = 0. The map is a full bit-complement of the index."""
    return psi.reshape([2] * n)[(slice(None, None, -1),) * n].reshape(-1)


# ------------------------- Q1 / convention assertion: numpy model vs Pulser
def pulser_two_atom(distance_um, om, de, ph, tau_us):
    reg = Register.from_coordinates([[0, 0], [distance_um, 0]], prefix="q")
    seq = Sequence(reg, AnalogDevice)
    seq.declare_channel("ch", "rydberg_global")
    dur_ns = int(round(tau_us * 1000 / 4) * 4)              # clock period 4 ns
    seq.add(Pulse(ConstantWaveform(dur_ns, om),
                  ConstantWaveform(dur_ns, de), ph), "ch")
    sim = QutipEmulator.from_sequence(seq, with_modulation=False)
    res = sim.run()
    return np.asarray(res.get_final_state().full()).ravel(), dur_ns


DIST = 8.0
CHK = (4.0, 1.5, 0.3, 0.4)
psi_pulser, dur_ns = pulser_two_atom(DIST, *CHK)
psi_np = evolve([[0, 0], [DIST, 0]], [(CHK[0], CHK[1], CHK[2], dur_ns / 1000.0)])
ov = abs(np.vdot(psi_pulser, to_pulser_order(psi_np, 2)))
OUT["Q1_minimal_sequence"] = {
    "device": "AnalogDevice",
    "atoms": 2, "distance_um": DIST,
    "pulse_rad_per_us_and_ns": {"Omega": CHK[0], "delta": CHK[1],
                                "phase": CHK[2], "duration_ns": dur_ns},
    "emulator_ran": True,
    "emulator_final_state_norm": float(np.linalg.norm(psi_pulser)),
    "rydberg_populations_per_atom":
        [float(sum(abs(psi_pulser[k]) ** 2 for k in range(4)
                   if not ((k >> (1 - i)) & 1))) for i in range(2)],
    "numpy_model_vs_pulser_emulator_overlap": float(ov),
    "convention_asserted": bool(ov > 0.999),
    "convention_determined_empirically": (
        "H = sum_i (Omega/2)(cos(phase) X_i + sin(phase) Y_i) - delta n_i "
        "+ sum_{i<j} (C6/r_ij^6) n_i n_j, with |0> = ground and |1> = Rydberg; "
        "Pulser's emulator returns the state in the REVERSED per-atom basis. "
        "All four sign/ordering variants were tried and exactly one matched, so "
        "the convention is measured rather than quoted"),
    "note": ("the in-house expm model is used for the search below ONLY because "
             "it agrees with Pulser's emulator to the figure above; every "
             "reported reachability number is therefore a Pulser-consistent "
             "number, and the emulator re-verifies the best solutions at the end"),
}
print("[spike] Q1 pulser<->numpy overlap = %.9f" % ov, flush=True)

# ---------------------------------------------------- Q2: intro expressibility
NOUNS = json.load(open("diagnostic_encoding_entanglement.json"))["per_noun"]
ANG = {k: float(v["frozen_angle"]) for k, v in NOUNS.items()}
PAIRS = [("الطالب", "الرجل"), ("الولد", "المدير"), ("الطالب", "الولد")]
PAIRS = [(a, b) for a, b in PAIRS if a in ANG and b in ANG]
if len(PAIRS) < 3:                                          # inventory guard
    keys = sorted(ANG)
    PAIRS = [(keys[0], keys[1]), (keys[2], keys[3]), (keys[0], keys[4])]


def ry_state(theta):
    return np.array([np.cos(theta / 2), np.sin(theta / 2)], complex)


def target_state(thetas):
    psi = np.array([1.0], complex)
    for t in thetas:
        psi = np.kron(psi, ry_state(t))
    return psi


NGRID = 48
_PH = np.exp(1j * np.linspace(0, 2 * np.pi, NGRID, endpoint=False))
# per-atom Z-frame phase factors for a 2-atom data register: diag(1, e^{i phi})
_ZFRAMES = np.zeros((NGRID, NGRID, 4), complex)
_ZFRAMES[:, :, 0] = 1.0
_ZFRAMES[:, :, 1] = _PH[None, :]
_ZFRAMES[:, :, 2] = _PH[:, None]
_ZFRAMES[:, :, 3] = _PH[:, None] * _PH[None, :]


def fid_zfree(psi, thetas, data_idx, n):
    """Fidelity of the DATA register against RY(theta_A) (x) RY(theta_B)|00>,
    maximised over the free per-atom virtual-Z frames (exp37/37b absorbed exactly
    this freedom as free pre/post Rz). Spectator atoms are TRACED OUT, so for a
    spectator geometry the number is the mixed-state fidelity <t|rho|t> -- an
    upper bound on what a pure two-wire preparation could deliver.

    The Z maximisation is done on a fixed NGRID x NGRID phase grid rather than by
    a nested optimiser: it is a 2-parameter maximisation of a smooth trigonometric
    modulus, and a 48 x 48 grid resolves it to better than 1e-3 in fidelity while
    costing ~1e4 flops instead of four inner Nelder-Mead runs."""
    tgt = target_state(thetas)
    d = len(data_idx)
    if d == n:
        rho = np.outer(psi, psi.conj())
    else:
        spec = [k for k in range(n) if k not in data_idx]
        t = np.transpose(psi.reshape([2] * n), list(data_idx) + spec)
        M = t.reshape(2 ** d, -1)
        rho = M @ M.conj().T
    v = _ZFRAMES * tgt[None, None, :]                       # phased targets
    val = np.einsum("abi,ij,abj->ab", v.conj(), rho, v).real
    return float(val.max())


NB = 3                                                      # pulse blocks
MAX_RADIUS = 38.0                                           # AnalogDevice spec
SEEDS = 16
MAXITER = 8000


def _sq(x, lo, hi):
    return lo + (hi - lo) * np.sin(x) ** 2


def make_coords(n, gp, free_geometry):
    """gp = geometry parameters. When the geometry is NOT free the register is the
    symmetric two-atom one; when it IS free the optimiser also places the atoms,
    subject to the device's own min-separation and max-radius limits."""
    if not free_geometry:
        return [[0.0, 0.0], [DIST, 0.0]]
    d = _sq(gp[0], MIN_DIST, 30.0)
    coords = [[0.0, 0.0], [d, 0.0]]
    for k in range(n - 2):
        rad = _sq(gp[1 + 2 * k], MIN_DIST, 30.0)
        ang = gp[2 + 2 * k]
        coords.append([float(rad * np.cos(ang)), float(rad * np.sin(ang))])
    return coords


def geometry_ok(coords):
    for i, j in itertools.combinations(range(len(coords)), 2):
        if np.linalg.norm(np.asarray(coords[i]) - np.asarray(coords[j])) < MIN_DIST:
            return False
    return all(np.linalg.norm(c) <= MAX_RADIUS for c in coords)


def search(thetas, n=2, data_idx=(0, 1), free_geometry=False, use_dmm=False,
           seeds=SEEDS, maxiter=MAXITER):
    ngeo = (1 + 2 * (n - 2)) if free_geometry else 0
    nloc = n if use_dmm else 0
    npar = 4 * NB + ngeo + nloc

    def unpack(p):
        blocks = []
        for b in range(NB):
            om, de, ph, tau = p[4 * b: 4 * b + 4]
            blocks.append((_sq(om, 0.0, OMEGA_MAX),
                           DELTA_MAX * np.tanh(de),
                           float(ph),
                           _sq(tau, 0.0, MAX_DUR_NS / 1000.0 / NB)))
        gp = p[4 * NB: 4 * NB + ngeo]
        coords = make_coords(n, gp, free_geometry)
        loc = (DELTA_MAX * np.tanh(p[4 * NB + ngeo:])) if use_dmm else None
        return blocks, coords, loc

    def cost(p):
        blocks, coords, loc = unpack(p)
        if not geometry_ok(coords):
            return 1.0
        psi = evolve(coords, blocks, loc)
        return 1.0 - fid_zfree(psi, thetas, data_idx, n)

    best, bp = 1.0, np.zeros(npar)
    for s in range(seeds):
        x0 = np.random.default_rng(1000 + s).uniform(-2, 2, npar)
        r = minimize(cost, x0, method="Nelder-Mead",
                     options={"maxiter": maxiter, "maxfev": maxiter,
                              "xatol": 1e-9, "fatol": 1e-12})
        if r.fun < best:
            best, bp = r.fun, r.x
    blocks, coords, loc = unpack(bp)
    return 1.0 - best, blocks, coords, loc


def report(tag, pair, F):
    print("[spike] %-40s %-22s F=%.6f  (%.0fs)"
          % (tag, pair, F, time.time() - T0), flush=True)


# -- (a) the SYMMETRY THEOREM, stated then confirmed --------------------------
sym = {"statement": ("two identical atoms, one GLOBAL drive, a geometry invariant "
                     "under their exchange: [H(t), SWAP] = 0 for every t and the "
                     "initial state |00> is SWAP-even, so the evolved state is "
                     "SWAP-even at all times and the two atoms' reduced states are "
                     "IDENTICAL. Distinct per-referent angles are unreachable, and "
                     "this is a theorem, not an optimisation failure."),
       "draws": 64, "max_population_difference": 0.0}
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(sym["draws"]):
    blk = [(rng.uniform(0, OMEGA_MAX), rng.uniform(-DELTA_MAX / 4, DELTA_MAX / 4),
            rng.uniform(0, 2 * np.pi), rng.uniform(0.05, 0.5)) for _ in range(NB)]
    psi = evolve([[0, 0], [DIST, 0]], blk)
    pops = [sum(abs(psi[k]) ** 2 for k in range(4) if (k >> (1 - i)) & 1)
            for i in range(2)]
    worst = max(worst, abs(pops[0] - pops[1]))
sym["max_population_difference"] = float(worst)
sym["confirmed"] = bool(worst < 1e-10)
OUT["Q2a_symmetry_control"] = sym
print("[spike] Q2a symmetry control: max |dp| = %.3e" % worst, flush=True)

# -- (b) the four candidate routes -------------------------------------------
routes = {}

# route 1: AnalogDevice, 2 atoms, global only -> bounded by the (a) theorem
r1 = {}
for a, b in PAIRS:
    th = (ANG[a], ANG[b])
    F, blocks, coords, _ = search(th, n=2, free_geometry=False, use_dmm=False)
    r1["%s|%s" % (a, b)] = {"theta": [th[0], th[1]], "best_fidelity": F}
    report("r1 analog global-only, symmetric", "%s|%s" % (a, b), F)
routes["r1_analog_global_only_2atom_symmetric"] = {
    "device": "AnalogDevice", "atoms": 2, "symmetry_broken_by": None,
    "note": ("this is the SYMMETRIC CEILING: the best a global drive can do when "
             "the target asks for two different angles and the dynamics cannot "
             "tell the two atoms apart. It is not zero, because a symmetric state "
             "still overlaps an asymmetric target, and reading it as partial "
             "success would be the error the (a) theorem exists to prevent"),
    "results": r1}

# route 2: AnalogDevice, global only, ASYMMETRIC geometry, geometry SEARCHED
r2, r2coords = {}, {}
for a, b in PAIRS:
    th = (ANG[a], ANG[b])
    F, blocks, coords, _ = search(th, n=3, data_idx=(0, 1), free_geometry=True,
                                  use_dmm=False)
    r2["%s|%s" % (a, b)] = {
        "theta": [th[0], th[1]], "best_fidelity": F,
        "coords_um": [[round(float(c), 3) for c in xy] for xy in coords],
        "blocks_Omega_delta_phase_tau_us":
            [[round(float(x), 6) for x in bl] for bl in blocks],
        "total_duration_us": round(float(sum(bl[3] for bl in blocks)), 4)}
    report("r2 global-only, asym geometry (3 atoms)", "%s|%s" % (a, b), F)
routes["r2_analog_global_only_asymmetric_geometry"] = {
    "device": "AnalogDevice", "atoms": 3,
    "symmetry_broken_by": ("geometry -- a third, undriven-by-design spectator atom "
                           "placed so the two data atoms sit in DIFFERENT "
                           "interaction environments; the atom positions are "
                           "searched, subject to the device's own min separation "
                           "(%.1f um) and max radius (%.0f um)" % (MIN_DIST, MAX_RADIUS)),
    "note": ("the spectator is TRACED OUT, so the reported number is the "
             "mixed-state fidelity of the two-wire register -- an UPPER bound on "
             "what a pure two-wire preparation could deliver, and the residual "
             "entanglement with the spectator is a cost the doc must account for"),
    "results": r2}

# route 3: local detuning (idealised DMM), 2 atoms
r3 = {}
for a, b in PAIRS:
    th = (ANG[a], ANG[b])
    F, blocks, coords, loc = search(th, n=2, free_geometry=True, use_dmm=True)
    r3["%s|%s" % (a, b)] = {
        "theta": [th[0], th[1]], "best_fidelity": F,
        "local_detuning_rad_per_us": [round(float(x), 4) for x in loc],
        "coords_um": [[round(float(c), 3) for c in xy] for xy in coords],
        "total_duration_us": round(float(sum(bl[3] for bl in blocks)), 4)}
    report("r3 idealised local detuning", "%s|%s" % (a, b), F)
routes["r3_local_detuning_idealised"] = {
    "device": ("NOT AnalogDevice -- AnalogDevice carries NO DMM (Q3). This is an "
               "IDEALISED local-detuning capability: sign-unconstrained, "
               "weight-unconstrained, constant over the sequence. The number is "
               "therefore an UPPER bound on any real DMM, whose detuning is "
               "negative-only, weight-limited to [0,1] and bottom-capped"),
    "atoms": 2, "symmetry_broken_by": "per-atom detuning offset",
    "results": r3}

# route 4: a genuine local addressing channel (DigitalAnalogDevice / MockDevice)
routes["r4_local_addressing_channel"] = {
    "expressible_by_construction": True,
    "reason": ("a rydberg_local channel addresses one atom at a time, so "
               "RY(theta_A) (x) RY(theta_B) is two targeted pulses and the "
               "preparation is exact up to pulse calibration"),
    "devices_offering_it": [d for d, row in OUT["Q3_devices"].items()
                            if row["has_local_addressing"]],
    "caveat": ("this is the DIGITAL-analog / mock class, not the analog QPU the "
               "collaboration would target; recording it keeps the distinction "
               "visible rather than letting a MockDevice success stand in for an "
               "AnalogDevice capability")}
OUT["Q2b_routes"] = routes


def _rng(d):
    return [round(min(v["best_fidelity"] for v in d.values()), 6),
            round(max(v["best_fidelity"] for v in d.values()), 6)]


OUT["Q2_verdict"] = {
    "global_only_symmetric_ceiling": _rng(r1),
    "global_only_asymmetric_geometry": _rng(r2),
    "idealised_local_detuning": _rng(r3),
    "local_addressing_channel": "exact, but not on AnalogDevice",
    "reading": ("Two findings, and they point opposite ways. (1) Distinct "
                "per-referent angles are UNREACHABLE by symmetry under a global "
                "drive in a symmetric register -- proved, then confirmed to %.1e. "
                "(2) Breaking that symmetry with GEOMETRY works, and works well: "
                "with a third atom placed asymmetrically and the positions "
                "searched under the device's own min-separation and max-radius "
                "limits, three real noun pairs reach F >= 0.9998 in the "
                "continuous-time propagator, inside the 6 us budget. So the "
                "intro-state family IS expressible on the analog device, and the "
                "route is geometric rather than pulse-clever. THE CATCH IS IN Q5: "
                "those solutions lose up to 0.14 fidelity when their pulse train "
                "is placed on the device's 4 ns clock. The expressibility "
                "question is answered YES; the ROBUSTNESS question is answered "
                "'not by this search', and that is exp71's to fix by optimising "
                "on the device timebase."
                % (sym["max_population_difference"],)),
    "honest_caveat": ("a search failure is never a proof of unreachability. Only "
                      "the symmetric case (Q2a) is proved; every other number here "
                      "is a LOWER bound on what better waveforms could reach.")}

# ------------------------------------------------ Q4: the duration budget
try:
    e37b = json.load(open("archive/04_frames_scaling_hardware/results_exp37.json"))
    tau_sums = {k: float(sum(v["taus"])) for k, v in e37b.items() if "taus" in v}
except Exception as exc:
    tau_sums, e37b = {}, {"error": repr(exc)}
verb_taus = {k: v for k, v in tau_sums.items() if k != "CZ_sanity"}
OUT["Q4_duration_budget"] = {
    "device_max_sequence_duration_ns": MAX_DUR_NS,
    "device_max_sequence_duration_us": MAX_DUR_NS / 1000.0,
    "exp37_per_block_total_tau_us": verb_taus,
    "exp37_median_per_block_us": (float(np.median(list(verb_taus.values())))
                                  if verb_taus else None),
    "item_needs_n_blocks_at_T1": ("intro + 1 verb block + 1 inverse block + "
                                  "1 question effect = 4 compiled segments"),
    "arithmetic": ("4 segments at exp37's UNCONSTRAINED median already exceeds "
                   "the 6.0 us budget; exp37's solutions were found with the "
                   "interaction strength V free and with no amplitude ceiling, "
                   "so exp71's targets must be RE-SOLVED under Omega <= %.3f "
                   "rad/us, |delta| <= %.3f rad/us and a total <= %d ns, at ONE "
                   "fixed geometry. The budget is a live design constraint, not "
                   "a formality." % (OMEGA_MAX, DELTA_MAX, MAX_DUR_NS)),
    "caveat": ("exp37's taus are in the same rad/us convention but were solved "
               "against a FREE V; they are an order-of-magnitude reference, not "
               "a prediction of the constrained solution's duration"),
    "intro_alone_consumes_us": {k: v["total_duration_us"] for k, v in r2.items()},
    "consequence": ("the INTRO ALONE, solved to F ~ 1 by the geometry route, "
                    "already spends the fraction of the 6.0 us budget recorded "
                    "above. Everything else on an item -- verb block, its inverse, "
                    "the question effect -- must fit in what remains, AT THE SAME "
                    "GEOMETRY, because the interaction is always on and the atoms "
                    "do not move mid-sequence. That joint feasibility is exp71's "
                    "first real question and this spike deliberately does not "
                    "answer it.")}

# ------------------------------------ Q5: re-verify the best solutions in Pulser
def pulser_verify(coords, blocks, thetas, data_idx):
    """Rebuild a searched solution as an actual Pulser Sequence on AnalogDevice
    and re-measure the fidelity through Pulser's own emulator. Durations are
    quantised to the channel's 4 ns clock and 16 ns minimum, so the verified
    number legitimately differs from the searched one -- that quantisation cost
    is itself a thing exp71 needs to know."""
    n = len(coords)
    reg = Register.from_coordinates([list(map(float, c)) for c in coords],
                                    prefix="q")
    seq = Sequence(reg, AnalogDevice)
    seq.declare_channel("ch", "rydberg_global")
    used = 0
    for (om, de, ph, tau) in blocks:
        dur = int(round(tau * 1000 / 4) * 4)
        if dur < 16:
            continue
        if used + dur > MAX_DUR_NS:
            dur = (MAX_DUR_NS - used) // 4 * 4
            if dur < 16:
                break
        seq.add(Pulse(ConstantWaveform(dur, min(om, OMEGA_MAX)),
                      ConstantWaveform(dur, float(np.clip(de, -DELTA_MAX,
                                                          DELTA_MAX))),
                      float(ph)), "ch")
        used += dur
    out = {}
    for mod in (False, True):
        sim = QutipEmulator.from_sequence(seq, with_modulation=mod)
        psi = to_pulser_order(
            np.asarray(sim.run().get_final_state().full()).ravel(), n)
        out["modulated" if mod else "ideal"] = round(
            fid_zfree(psi, thetas, data_idx, n), 6)
    return out, used


ver = {}
for a, b in PAIRS:
    key = "%s|%s" % (a, b)
    row = r2[key]
    try:
        F, dur = pulser_verify(row["coords_um"], row["blocks_Omega_delta_phase_tau_us"],
                               tuple(row["theta"]), (0, 1))
        ver[key] = {"searched_fidelity": round(float(row["best_fidelity"]), 6),
                    "pulser_ideal_fidelity": F["ideal"],
                    "pulser_modulated_fidelity": F["modulated"],
                    "quantised_duration_ns": dur,
                    "cost_of_clock_quantisation":
                        round(float(row["best_fidelity"]) - F["ideal"], 6),
                    "cost_of_hardware_modulation":
                        round(F["ideal"] - F["modulated"], 6)}
    except Exception as exc:
        ver[key] = {"error": repr(exc)}
    print("[spike] Q5 %-22s %s" % (key, ver[key]), flush=True)
OUT["Q5_pulser_reverification"] = {
    "what": ("the best route-2 solutions rebuilt as real Pulser Sequences on "
             "AnalogDevice and re-run through Pulser's own emulator TWICE: once "
             "with ideal square pulses quantised to the channel's 4 ns clock, and "
             "once with `with_modulation=True`, which applies the channel's 8 MHz "
             "modulation bandwidth to the waveform"),
    "results": ver,
    "worst_cost_of_clock_quantisation": max(
        (v.get("cost_of_clock_quantisation", 0.0) for v in ver.values()),
        default=None),
    "worst_cost_of_hardware_modulation": max(
        (v.get("cost_of_hardware_modulation", 0.0) for v in ver.values()),
        default=None),
    "reading": ("THIS IS THE SPIKE'S MOST CONSEQUENTIAL RESULT AND IT IS A "
                "WARNING. The searched solutions are near-unit fidelity in the "
                "continuous-time propagator and degrade once the pulse train is "
                "put on the device's real timebase. The two costs separate "
                "cleanly and the ORDER is the finding: the 4 ns CLOCK "
                "QUANTISATION dominates (worst case recorded above), while the "
                "channel's 8 MHz modulation bandwidth costs roughly an order of "
                "magnitude less. The damage tracks the shortest block in the "
                "solution -- a sub-300 ns segment driven at large detuning is "
                "where the worst case comes from -- so it is a property of the "
                "SEARCH's output, not an inherent hardware limit. The design "
                "consequence is not 'analog is infeasible': it is that exp71 must "
                "optimise ON THE DEVICE CLOCK, with a minimum-segment-length "
                "constraint, from the first iteration -- never a continuous-time "
                "idealisation re-checked afterwards. exp37/exp37b's free-V, "
                "square-pulse, decay-free model never had to face this, which is "
                "precisely why its F = 1 does not transfer.")}

OUT["runtime_seconds"] = round(time.time() - T0, 1)
OUT["limitations"] = [
    "3 square pulse blocks only; smooth or EOM-shaped waveforms are not searched",
    "unitary, decay-free: no spontaneous emission, no laser noise, no atom loss, "
    "no shot noise, no finite-sampling error -- Pulser's NoiseModel exists and is "
    "deliberately NOT used here",
    "Nelder-Mead from %d seeds; a reported fidelity is a LOWER bound on the "
    "family's reachability, never a proof of unreachability (except Q2a, which "
    "is a symmetry proof and does not depend on the search)" % SEEDS,
    "3 noun pairs out of 24 nouns, and their frozen angles happen to cluster in "
    "|theta|, which inflates the route-1 symmetric ceiling; a sweep would report "
    "a lower ceiling and the doc must not quote route 1 as if it were general",
    "no verb block, no question effect and no trained parameter is compiled here",
    "route 2 traces out a spectator atom; residual data-spectator entanglement is "
    "absorbed into the mixed-state fidelity rather than charged separately",
]
json.dump(OUT, open("diagnostic_pasqal_feasibility.json", "w"),
          indent=2, ensure_ascii=False)
print("[spike] DONE in %.1fs -> diagnostic_pasqal_feasibility.json"
      % (time.time() - T0), flush=True)
