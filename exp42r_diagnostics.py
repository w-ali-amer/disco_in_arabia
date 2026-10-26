# -*- coding: utf-8 -*-
"""exp42r_diagnostics.py -- doc 65 SS7.3's TWO REGISTERED DIAGNOSTICS.

  (a) a T = 1 single-event Arabic block on the real vocabulary --
      discriminates READOUT-SOUNDNESS from DEPTH;
  (b) purity / entropy of the question-time reduced 2-wire state as a
      function of story length T, REUSING exp76's certificate machinery.

*** THESE ARE BARRED FROM THE VERDICT. ***
doc 65 SS7.3 verbatim: "Neither gates nor enters `tier_q`/`tier_s`; they
sharpen the read." This file writes `results_exp42r_diagnostics.json` and
NOTHING ELSE. `exp42r_aggregate.py` -- the sole writer of
`results_exp42r.json` -- MUST NEVER READ THIS FILE, and asserts as much in
its own source. No doc 22 SS7 boolean is computed here; every number below
is DESCRIPTIVE.

------------------------------------------------------------------------
DIAGNOSTIC (a): T = 1, ONE EVENT, ONE QUESTION ABOUT IT

What it discriminates. doc 65 SS7.2 established that on `stories_exp41r`
the A2 family fits 200 items to 0.978 train while sitting at 0.506 val,
and fits SCRAMBLED labels equally well (Delta +0.003) -- "the
composition/readout interface preserves no task signal at this design
point". Two very different mechanisms produce that:

  (i) DEPTH: the two-effect readout can express thematic role, but the
      signal is destroyed by composing T = 5-10 two-qubit blocks before
      it is read (decoherence-like mixing of the queried pair);
  (ii) READOUT: the two-effect softmax cannot express thematic role at
      ALL in this architecture, at any depth.

At T = 1 the depth channel is switched off by construction. The story is
one event, the question asks about that one event, and nothing else has
happened. If A2 generalises here, the readout is SOUND and (i) is the
live mechanism. If A2 is at chance HERE, the readout is the problem and
no amount of depth engineering rescues this design point.

Construction (minimal, and deliberately the easiest role-sensitive task
this architecture can be posed):
  cast K = 4 referents, wires fixed by referent-list order;
  one event, a SWAP verb with a human patient, between two of them;
  the yes-twin has the event (v, A, B), the no-twin has (v, B, A);
  BOTH twins carry the identical question "did A v B" (Q0 form).
  Bag of words identical, answer flipped -- doc 22 SS2.2's instrument at
  T = 1. The other two referents are SPECTATORS: they are introduced and
  never acted on, so at T = 1 the global state is a product across them
  and they are PROVABLY INERT (recorded, not assumed -- the readout's
  partial trace over a product factor contributes exactly 1). They are
  kept so the cast size, wire count and partial-trace structure mirror the
  real experiment.

The read is VAL, never train. doc 65 SS7.2 already measured that this
family memorises 200 items to ~0.98 with the story->answer link DESTROYED,
so a train number here would mean nothing. Train and val draw DISJOINT
(verb, agent, patient) triples.

Arms run: A2 under the committed uniform[0, 2pi) init (the exp40r
diagnostic family, for continuity) AND A2 under doc 65 SS2.4's registered
N(0, 0.1^2) init (so a T = 1 read cannot be dismissed as an init
artefact of the wrong family), plus A1 as the negative control -- A1's
blocks commute with SWAP, so its two twins compile to the IDENTICAL
statevector and it must sit at exactly 0.5. A1 above chance here would
mean this diagnostic's own generator leaks.

------------------------------------------------------------------------
DIAGNOSTIC (b): PURITY / ENTROPY OF THE QUESTION-TIME 2-WIRE STATE vs T

doc 65 SS7.3(b) as registered, with SS7.2's leading mechanism CANDIDATE as
the thing being instrumented: "doc 22 SS9's mini uses ancilla-and-discard
channels while SS3.1's Arabic verbs are pure 2-qubit unitaries". Under a
pure global unitary evolution the QUERIED PAIR is not a closed system --
it is entangled with the K - 2 wires the readout traces out -- so its
reduced state rho_2 is mixed, and the two-effect softmax reads
P(queried == |00>) THROUGH that mixture. Tr rho_2^2 is exactly how much
of a pure signal survives to the readout.

  Tr rho_2^2 = 1     the pair is pure: the readout sees an undegraded
                     two-qubit state and every bit of role information
                     that reached the pair is available to the effects;
  Tr rho_2^2 -> 1/4  the pair is maximally mixed: the readout sees noise
                     and NO trained effect pair can recover the answer.

If Tr rho_2^2 falls toward 1/4 as T grows while the T = 1 block of
diagnostic (a) learns, mechanism (i) DEPTH is instrumented rather than
merely inferred.

ONE PHYSICS FACT, recorded so a reader does not mistake the choice of
measurement point for a free parameter: the question's inverse
preparation is a UNITARY ON THE QUERIED PAIR ALONE, and a unitary acting
only inside a subsystem leaves that subsystem's purity invariant
(rho -> U rho U^dagger, Tr (U rho U^d)^2 = Tr rho^2). So the purity
"at question time" and "after the last event" are the SAME NUMBER. This
file computes both and asserts the identity to 1e-6 rather than assuming
it.

Machinery: `exp76_run.wire_purities` (per-wire Tr rho^2) and
`exp76_run.schmidt_cuts` (per-cut S in bits, R_1e-6) are IMPORTED AND
USED UNMODIFIED -- doc 65 SS7.3's "REUSING exp76's certificate machinery".
exp76's own definitions are 1-wire and per-cut; the 2-WIRE reduced purity
this diagnostic needs does not exist there, so `pair_purity` below is
written in exp76's exact idiom (movedim -> reshape -> rho -> sum |rho|^2)
and cross-checked against `wire_purities` on product states.

Checkpoints: doc 65 SS7.3(b) says "on the exp40r Arabic diagnostic
checkpoints (or retrain briefly if not saved)". They were NOT saved --
`exp40r_trainer.strip_state` drops `final_state` before serialisation and
`exp40r_arabic_diag` writes only the stripped record, so no state_dict
from that block exists on disk. The H-honest condition is therefore
REPRODUCED here (same 200 balanced train items, same 60 epochs, same lr,
same August-parity uniform init, seeds 0-2) and its states ARE saved.

The T-sweep items are generated FRESH at a diagnostic-only seed with the
cast size held FIXED at K = 6, for two reasons: (1) purity of a 2-wire
marginal depends on how many wires it can entangle with, so K must be
controlled if T is to be the variable; (2) test_a / test_b / test_c are
doc 22 SS7's gate sets and a diagnostic that touched them would
contaminate exp42r's pre-registered comparison. The sweep runs
T in {1, 2, 3, 5, 7, 10, 14, 20}, which extends past the train range on
both sides.

Writes results_exp42r_diagnostics.json.
"""

import argparse
import hashlib
import json
import random
import sys
import time

import torch

import exp41_story_generator as gen
import exp42_compiler as comp
import exp40r_trainer as trainer
import exp42_sim_adapter as sim
from exp42_models import QuantumStoryModel

# exp76's certificate machinery, imported and used unmodified (doc 65 SS7.3).
from exp76_run import wire_purities, schmidt_cuts, SCHMIDT_EPS

OUT = "results_exp42r_diagnostics.json"
EXP41R_PATH = "stories_exp41r.json"
EXP41R_SHA = ("2326942ffb6ccf6bca95a7ec0502c82563957fabb159b143608a9fb31f"
              "8f3113")

# diagnostic-only seeds, disjoint from the generator's 41000/42000 family
SEED_T1_TRAIN = 99001
SEED_T1_VAL = 99002
SEED_SWEEP = 99003

T_SWEEP = (1, 2, 3, 5, 7, 10, 14, 20)
SWEEP_K = 6                       # cast size HELD FIXED across the sweep
SWEEP_ITEMS = 40

NOT_A_VERDICT = (
    "doc 65 SS7.3: these are DIAGNOSTICS and are BARRED FROM THE VERDICT. "
    "They neither gate nor enter tier_q_passed / tier_s_passed / "
    "content_earned / trained_intro_helps. exp42r_aggregate.py, the sole "
    "writer of results_exp42r.json, never reads this file.")


# ======================================================================
# provenance
# ======================================================================
def load_exp41r(path=EXP41R_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXP41R_SHA:
        raise RuntimeError("exp41r provenance FAILED: computed=%s manifest=%s "
                           "pin=%s" % (sha, meta["dataset_sha256"], EXP41R_SHA))
    return items, meta, sha


# ======================================================================
# item construction (the real exp41 vocabulary, imported never retyped)
# ======================================================================
HUMAN_PATIENT_SWAP_VERBS = sorted(
    v for v in gen.SWAP_VERBS if gen.VERBS[v]["human_patients"])


def _item(iid, pair_id, refs, events, question, answer, K, T):
    """A minimal exp41-shaped item; `exp42_compiler.compile_story` consumes
    exactly these fields (surface strings included for completeness even
    though the structured arms never see them)."""
    sents, flags = [], []
    for ev in events:
        sents.append("%s %s %s" % (ev["verb"], ev["agent"], ev["patient"]))
        flags.append("VSO")
    q = dict(question)
    q["surface"] = "هل %s %s %s" % (q["v"], q["a"], q["p"])
    return {"id": iid, "pair_id": pair_id, "split": "diag", "qtype": "Q0",
            "hop_count": 1, "pro_drop_item": False, "order_dependent": False,
            "answer": answer, "K": K, "T": T,
            "referents": refs, "events": events,
            "sentences": sents, "order_flags": flags, "question": q}


def build_t1_items(n_pairs, seed, used_triples, cast_k=4):
    """T = 1 swap twins. Yes-twin event (v, A, B); no-twin event (v, B, A);
    BOTH ask the identical question "did A v B". Bag-identical, answer
    flipped. Returns (items, triples_used)."""
    rng = random.Random(seed)
    items, mine = [], set()
    guard = 0
    while len(items) < 2 * n_pairs:
        guard += 1
        if guard > 200 * n_pairs:
            raise RuntimeError("T=1 generator could not fill %d pairs"
                               % n_pairs)
        v = rng.choice(HUMAN_PATIENT_SWAP_VERBS)
        A, B = rng.sample(gen.HUMANS, 2)
        key = (v, A, B)
        if key in used_triples or key in mine:
            continue
        mine.add(key)
        # two spectators: introduced, never acted on (provably inert at T=1)
        spare = [h for h in gen.HUMANS if h not in (A, B)]
        spec = rng.sample(spare, 1) + rng.sample(gen.OBJECTS, 1)
        names = [A, B] + spec
        rng.shuffle(names)                       # wire order carries no signal
        refs = [{"name": n,
                 "type": "human" if n in gen.HUMANS else "object"}
                for n in names]
        pid = "t1_%s_%s_%s" % (v, A, B)
        q = {"type": "Q0", "v": v, "a": A, "p": B}
        items.append(_item(pid + "_yes", pid, refs,
                           [{"verb": v, "agent": A, "patient": B,
                             "pro_drop": False}], q, "yes", cast_k, 1))
        items.append(_item(pid + "_no", pid, refs,
                           [{"verb": v, "agent": B, "patient": A,
                             "pro_drop": False}], q, "no", cast_k, 1))
    rng.shuffle(items)
    return items, mine


def _patients_for(v, cast, agent):
    cfg = gen.VERBS[v]
    out = []
    if cfg["human_patients"]:
        out += [c for c in cast if c in gen.HUMANS and c != agent]
    out += [c for c in cast if c in cfg["objects"]]
    return out


def build_sweep_items(T, n_items, seed, K=SWEEP_K):
    """Diagnostic-only stories at FIXED cast size K and story length T,
    over the real vocabulary. The question is a Q0 asked about one of the
    story's own events. Labels are correct but IRRELEVANT here: this block
    measures STATES, not accuracy."""
    rng = random.Random(seed)
    items = []
    guard = 0
    while len(items) < n_items:
        guard += 1
        if guard > 500 * n_items:
            raise RuntimeError("sweep generator stalled at T=%d" % T)
        n_h = max(2, K // 2)
        cast = rng.sample(gen.HUMANS, n_h) + \
            rng.sample(gen.OBJECTS, K - n_h)
        rng.shuffle(cast)
        humans = [c for c in cast if c in gen.HUMANS]
        events = []
        for _ in range(T):
            for _try in range(50):
                v = rng.choice(sorted(gen.VERBS))
                a = rng.choice(humans)
                pool = _patients_for(v, cast, a)
                if pool:
                    events.append({"verb": v, "agent": a,
                                   "patient": rng.choice(pool),
                                   "pro_drop": False})
                    break
            else:
                break
        if len(events) != T:
            continue
        ev = rng.choice(events)
        q = {"type": "Q0", "v": ev["verb"], "a": ev["agent"],
             "p": ev["patient"]}
        refs = [{"name": c,
                 "type": "human" if c in gen.HUMANS else "object"}
                for c in cast]
        items.append(_item("sweep_T%d_%d" % (T, len(items)),
                           "sweep_T%d_%d" % (T, len(items)), refs, events,
                           q, "yes", K, T))
    return items


# ======================================================================
# the 2-wire reduced purity (exp76's idiom; exp76 itself is 1-wire)
# ======================================================================
def pair_purity(psi, w1, w2):
    """Tr rho_2^2 for the reduced state of wires (w1, w2), every other wire
    traced out. Written in `exp76_run.wire_purities`' exact idiom -- movedim
    the kept wires to the front, reshape, rho = M M^dagger, sum |rho_ij|^2 --
    generalised from 1 kept wire to 2. Range [0.25, 1.0].

    Taken in complex128, following `exp76_run.schmidt_cuts`' own convention:
    the simulator runs complex64 and its purity floor is ~1e-7, so the
    reduction is promoted before the trace. This removes float32 noise from
    the purity NUMBER; it does not remove the simulator's own complex64
    unitary error, which is what the invariance check below measures."""
    p = torch.movedim(psi.detach().to(torch.complex128),
                      (w1, w2), (0, 1)).reshape(4, -1)
    rho = p @ p.conj().transpose(0, 1)
    return float((rho.abs() ** 2).sum())


def traced_states(model, cs):
    """Walk the committed `story_values` gate order, snapshotting the state
    after the last event and again after the question's inverse preparation.
    Gate order is copied from exp42_models.QuantumStoryModel.story_values and
    checked against it by `check_forward_matches` below."""
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = model.verb_matrix(v)
        return cache[v]

    psi = sim.init_state(cs.K)
    for w, _n, angle in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(angle), (w,))
    psi_intro = psi
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, block(v), (wa, wp))
    psi_last = psi
    vseq, qa, qp = cs.question
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, (qa, qp))
    return psi_intro, psi_last, psi


def check_forward_matches(model, cs):
    """The traced walk must reproduce the committed readout bit-for-bit."""
    with torch.no_grad():
        _i, _l, psi = traced_states(model, cs)
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        mine = torch.stack([sim.prob_all_zero_on(
            sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
            for eff in (model.q_pos, model.q_neg)])
        theirs = model.story_values(cs)
    return float((mine - theirs).abs().max().item())


def sweep_metrics(model, compiled):
    """Per-item purity/entropy at the queried pair. Returns the raw rows."""
    rows = []
    with torch.no_grad():
        for cs in compiled:
            _vseq, qa, qp = cs.question
            psi_i, psi_l, psi_q = traced_states(model, cs)
            S_l, R_l, _P = schmidt_cuts(psi_l, cs.K, eps=SCHMIDT_EPS)
            wp_l = wire_purities(psi_l, cs.K)
            rows.append({
                "item_id": cs.item_id, "K": cs.K,
                "queried_wires": [int(qa), int(qp)],
                "pair_purity_after_intro": pair_purity(psi_i, qa, qp),
                "pair_purity_after_last_event": pair_purity(psi_l, qa, qp),
                "pair_purity_at_readout": pair_purity(psi_q, qa, qp),
                "wire_purity_q1": wp_l[qa], "wire_purity_q2": wp_l[qp],
                "S_max_bits_after_last_event": float(max(S_l)) if S_l else 0.0,
                "R_max_after_last_event": int(max(R_l)) if R_l else 1,
            })
    return rows


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"n": n, "mean": sum(xs) / n, "min": xs[0], "max": xs[-1],
            "median": xs[n // 2],
            "p05": xs[max(0, int(0.05 * n) - 1)],
            "p95": xs[min(n - 1, int(0.95 * n))]}


# ======================================================================
# init policies
# ======================================================================
def reinit_normal(model, seed, std=0.1):
    """doc 65 SS2.4's registered init, applied to every trained parameter of
    a quantum arm, from a dedicated stream."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return {"policy": "N(0, %.3f^2)" % std, "stream_seed": seed}


# ======================================================================
# diagnostic (a)
# ======================================================================
def run_t1(angles, args):
    train_items, used = build_t1_items(args.t1_pairs, SEED_T1_TRAIN, set())
    val_items, _ = build_t1_items(args.t1_pairs, SEED_T1_VAL, used)
    tri_train = {(i["question"]["v"], i["question"]["a"], i["question"]["p"])
                 for i in train_items}
    tri_val = {(i["question"]["v"], i["question"]["a"], i["question"]["p"])
               for i in val_items}
    assert not (tri_train & tri_val), "train/val triple overlap at T=1"

    train_cs = comp.compile_all(train_items, angles)
    val_cs = comp.compile_all(val_items, angles)
    verbs = sorted(gen.VERBS)
    answer_of = (lambda cs: cs.answer)                     # noqa: E731

    # the spectator-inertness claim, MEASURED not assumed: at T = 1 the
    # global state factorises across the untouched wires, so removing them
    # cannot change the readout.
    m_ref = QuantumStoryModel("A2", verbs, 0)
    fwd_delta = check_forward_matches(m_ref, train_cs[0])
    with torch.no_grad():
        cs0 = train_cs[0]
        _i, psi_l, _q = traced_states(m_ref, cs0)
        wp = wire_purities(psi_l, cs0.K)
        acted = {w for _v, wa, wp_ in cs0.events for w in (wa, wp_)}
        spectator_purities = [wp[w] for w in range(cs0.K) if w not in acted]

    conditions = {}
    for label, arm, init in (("A2_uniform_init", "A2", "uniform"),
                             ("A2_normal_init", "A2", "normal"),
                             ("A1_uniform_init_control", "A1", "uniform")):
        per_seed = []
        for seed in range(args.t1_seeds):
            torch.manual_seed(seed)
            m = QuantumStoryModel(arm, verbs, seed)
            init_rec = {"policy": "uniform[0, 2pi) (committed default)",
                        "stream_seed": seed}
            if init == "normal":
                init_rec = reinit_normal(m, 990000 + seed)
            tr = trainer.train_with_trajectory(
                m, train_cs, {"val": val_cs}, answer_of, seed=seed,
                epochs=args.t1_epochs, lr=args.lr, batch_size=1,
                log_prefix="[T1 %s s%d] " % (label, seed), eval_every=5)
            rec = trainer.strip_state(tr)
            rec["seed"] = seed
            rec["init"] = init_rec
            rec["max_val_acc"] = max(a for a in tr["eval_acc"]["val"]
                                     if a is not None)
            per_seed.append(rec)
        finals = [r["final_eval_acc"]["val"] for r in per_seed]
        maxes = [r["max_val_acc"] for r in per_seed]
        conditions[label] = {
            "arm": arm, "init": init, "per_seed": per_seed,
            "final_val_acc": finals, "max_val_acc": maxes,
            "mean_final_val_acc": sum(finals) / len(finals),
            "mean_max_val_acc": sum(maxes) / len(maxes),
            "mean_final_train_acc": sum(r["final_clean_train_acc"]
                                        for r in per_seed) / len(per_seed),
        }

    a2u = conditions["A2_uniform_init"]
    a2n = conditions["A2_normal_init"]
    a1 = conditions["A1_uniform_init_control"]
    learns = bool(min(a2u["mean_final_val_acc"], a2n["mean_final_val_acc"])
                  >= 0.60)
    return {
        "diagnostic": "(a) T = 1 single-event Arabic block (doc 65 SS7.3a)",
        "not_a_verdict": NOT_A_VERDICT,
        "question": "does the two-effect readout learn thematic role when "
                    "DEPTH IS SWITCHED OFF? readout-soundness vs depth.",
        "construction": {
            "T": 1, "cast_K": 4, "n_train_items": len(train_cs),
            "n_val_items": len(val_cs),
            "verbs_used": HUMAN_PATIENT_SWAP_VERBS,
            "twin_rule": "yes-twin event (v, A, B); no-twin event (v, B, A); "
                         "identical question 'did A v B'; bag-identical, "
                         "answer flipped",
            "train_val_triples_disjoint": True,
            "n_train_triples": len(tri_train), "n_val_triples": len(tri_val),
            "read": "VAL accuracy ONLY -- doc 65 SS7.2 measured this family "
                    "memorising 200 items to ~0.98 with the labels "
                    "SCRAMBLED, so train accuracy is uninformative here",
            "spectator_wires_are_inert": {
                "measured_spectator_wire_purities": spectator_purities,
                "reading": "purity 1.0 to machine precision = the spectator "
                           "wires are in a product state with the rest, so "
                           "the readout's partial trace over them "
                           "contributes exactly 1 and they cannot carry "
                           "signal. Kept so cast size and partial-trace "
                           "structure mirror the real experiment.",
            },
            "traced_forward_matches_committed_readout_max_abs_delta":
                fwd_delta,
        },
        "config": {"epochs": args.t1_epochs, "lr": args.lr, "batch": 1,
                   "seeds": args.t1_seeds,
                   "optimizer": "Adam (full autodiff, no SPSA)"},
        "conditions": conditions,
        "descriptive_read": {
            "A2_mean_final_val_uniform_init": a2u["mean_final_val_acc"],
            "A2_mean_final_val_normal_init": a2n["mean_final_val_acc"],
            "A2_mean_max_val_uniform_init": a2u["mean_max_val_acc"],
            "A2_mean_max_val_normal_init": a2n["mean_max_val_acc"],
            "A1_control_mean_final_val": a1["mean_final_val_acc"],
            "A1_at_exact_chance": bool(
                all(abs(x - 0.5) < 1e-9 for x in a1["final_val_acc"])),
            "learns_at_T1_descriptive": learns,
            "descriptive_bar": "mean final val >= 0.60 under BOTH inits. "
                               "Declared in source before the run. This is a "
                               "READING AID, not a pre-registered criterion: "
                               "doc 65 SS7.3 bars this block from the "
                               "verdict.",
            "reading_key": (
                "A2 learns at T=1  => the two-effect readout CAN express "
                "thematic role; doc 65 SS7.2's failure is a DEPTH failure. "
                "A2 at chance at T=1 => the readout itself cannot express "
                "it at this design point, and depth is not the mechanism. "
                "A1 must be at exactly 0.5 (its blocks commute with SWAP, "
                "so the twins compile to the identical statevector); A1 "
                "above chance would mean THIS DIAGNOSTIC'S generator "
                "leaks."),
        },
    }


# ======================================================================
# diagnostic (b)
# ======================================================================
def run_purity(items, angles, args):
    by = comp.split_items(items)
    # H-honest condition of exp40r, reproduced (its states were never saved)
    rng = random.Random(41000)
    yes = [it for it in by["train"] if it["answer"] == "yes"]
    no = [it for it in by["train"] if it["answer"] == "no"]
    rng.shuffle(yes)
    rng.shuffle(no)
    train_items = yes[:args.purity_n // 2] + no[:args.purity_n // 2]
    rng.shuffle(train_items)
    train_cs = comp.compile_all(train_items, angles)
    verbs = sorted({ev["verb"] for it in items for ev in it["events"]})

    sweep = {}
    for T in T_SWEEP:
        sweep[T] = comp.compile_all(
            build_sweep_items(T, args.sweep_items, SEED_SWEEP + T), angles)

    # untrained reference, same architecture, committed init
    ref = QuantumStoryModel("A2", verbs, 0)
    untrained = {str(T): _stats([r["pair_purity_after_last_event"]
                                 for r in sweep_metrics(ref, sweep[T])])
                 for T in T_SWEEP}

    per_seed, states = [], []
    for seed in range(args.purity_seeds):
        torch.manual_seed(seed)
        m = QuantumStoryModel("A2", verbs, seed)
        tr = trainer.train_with_trajectory(
            m, train_cs, {}, lambda cs: cs.answer, seed=seed,
            epochs=args.purity_epochs, lr=args.lr, batch_size=1,
            log_prefix="[purity-train s%d] " % seed, eval_every=10)
        torch.save(m.state_dict(), "exp42r_diag_purity_A2_s%d.pt" % seed)
        rows_by_T, inv_max = {}, 0.0
        for T in T_SWEEP:
            rows = sweep_metrics(m, sweep[T])
            for r in rows:
                inv_max = max(inv_max,
                              abs(r["pair_purity_after_last_event"]
                                  - r["pair_purity_at_readout"]))
            rows_by_T[str(T)] = rows
        per_seed.append({
            "seed": seed,
            "train_final_acc": tr["final_clean_train_acc"],
            "train_max_acc": tr["max_clean_train_acc"],
            "epochs_run": tr["epochs_run"],
            "state_file": "exp42r_diag_purity_A2_s%d.pt" % seed,
            "purity_invariance_max_abs_delta": inv_max,
            "by_T": {T: {
                "pair_purity_after_last_event":
                    _stats([r["pair_purity_after_last_event"] for r in rr]),
                "pair_purity_at_readout":
                    _stats([r["pair_purity_at_readout"] for r in rr]),
                "pair_purity_after_intro":
                    _stats([r["pair_purity_after_intro"] for r in rr]),
                "wire_purity_queried":
                    _stats([r["wire_purity_q1"] for r in rr]
                           + [r["wire_purity_q2"] for r in rr]),
                "S_max_bits":
                    _stats([r["S_max_bits_after_last_event"] for r in rr]),
                "R_max": _stats([float(r["R_max_after_last_event"])
                                 for r in rr]),
            } for T, rr in rows_by_T.items()},
        })
        states.append(m)

    # the headline table: pair purity vs T, pooled over seeds
    table = []
    for T in T_SWEEP:
        vals = [s["by_T"][str(T)]["pair_purity_after_last_event"]["mean"]
                for s in per_seed]
        smax = [s["by_T"][str(T)]["S_max_bits"]["mean"] for s in per_seed]
        table.append({
            "T": T,
            "mean_pair_purity_trained": sum(vals) / len(vals),
            "per_seed_mean_pair_purity": vals,
            "mean_pair_purity_untrained_reference": untrained[str(T)]["mean"],
            "mean_S_max_bits_trained": sum(smax) / len(smax),
        })
    inv = max(s["purity_invariance_max_abs_delta"] for s in per_seed)
    return {
        "diagnostic": "(b) purity/entropy of the question-time reduced "
                      "2-wire state vs T (doc 65 SS7.3b)",
        "not_a_verdict": NOT_A_VERDICT,
        "reused_exp76_machinery": {
            "imported_unmodified": ["exp76_run.wire_purities",
                                    "exp76_run.schmidt_cuts",
                                    "exp76_run.SCHMIDT_EPS"],
            "written_here_because_exp76_has_no_2_wire_analogue":
                "pair_purity (Tr rho_2^2 of the queried pair), written in "
                "exp76_run.wire_purities' exact idiom",
            "schmidt_eps": SCHMIDT_EPS,
        },
        "checkpoints": {
            "exp40r_states_were_not_saved": True,
            "why": "exp40r_trainer.strip_state drops final_state and "
                   "exp40r_arabic_diag writes only the stripped record, so "
                   "no state_dict from that block exists on disk",
            "reproduced": "exp40r's H-honest condition (%d balanced train "
                          "items from the exp41r train split, %d epochs, lr "
                          "%.4f, batch 1, committed uniform init), states "
                          "saved this time"
                          % (args.purity_n, args.purity_epochs, args.lr),
        },
        "sweep_construction": {
            "T_values": list(T_SWEEP), "cast_K_held_fixed": SWEEP_K,
            "n_items_per_T": args.sweep_items, "seed_base": SEED_SWEEP,
            "why_fresh_items": "purity of a 2-wire marginal depends on how "
                               "many wires it can entangle with, so K is "
                               "held fixed and T is the only variable; and "
                               "test_a/test_b/test_c are doc 22 SS7's gate "
                               "sets, which a diagnostic must not touch",
        },
        "purity_invariance_under_question_inverse_prep": {
            "max_abs_delta_over_all_items_and_seeds": inv,
            "claim": "the question's inverse preparation is a unitary on the "
                     "queried pair ALONE, so it cannot change that pair's "
                     "purity; measured, not assumed",
            "bar": 1e-5,
            "bar_provenance": "the simulator is complex64 (torch_sv_sim."
                              "CDTYPE) and exp76_run records a purity floor "
                              "of ~1e-7 for it; 1e-5 is two decades above "
                              "that floor and was declared in source before "
                              "the run",
            "holds": bool(inv < 1e-5),
        },
        "config": {"n_train": len(train_cs), "epochs": args.purity_epochs,
                   "lr": args.lr, "batch": 1, "seeds": args.purity_seeds},
        "purity_vs_T_table": table,
        "per_seed": per_seed,
        "untrained_reference_by_T": untrained,
        "reading_key": (
            "Tr rho_2^2 = 1 -> the queried pair reaches the readout PURE and "
            "every bit of role information that got there is available to "
            "the two effects. Tr rho_2^2 -> 0.25 -> the pair is maximally "
            "mixed and NO trained effect pair can recover the answer. A fall "
            "toward 0.25 with T, together with a T = 1 block that learns, "
            "INSTRUMENTS doc 65 SS7.2's depth mechanism instead of inferring "
            "it."),
    }


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t1-pairs", type=int, default=100)
    ap.add_argument("--t1-seeds", type=int, default=5)
    ap.add_argument("--t1-epochs", type=int, default=60)
    ap.add_argument("--purity-n", type=int, default=200)
    ap.add_argument("--purity-seeds", type=int, default=3)
    ap.add_argument("--purity-epochs", type=int, default=60)
    ap.add_argument("--sweep-items", type=int, default=SWEEP_ITEMS)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--only", choices=("a", "b"), default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    t0 = time.time()
    items, meta, sha = load_exp41r()
    angles, emb_meta, emb_sha = comp.load_embeddings()
    print("[diag] exp41r sha=%s  embeddings sha=%s  torch %s"
          % (sha[:16], emb_sha[:16], torch.__version__), flush=True)

    payload = {
        "component": "exp42r diagnostics (doc 65 SS7.3)",
        "spec": "qnlp_private_docs/65_exp40_42_revival.md SS7.3",
        "NOT_A_VERDICT_FILE": NOT_A_VERDICT,
        "written_by": "exp42r_diagnostics.py",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "dataset": {"path": EXP41R_PATH, "sha256": sha,
                    "n_items": len(items), "seed": meta.get("seed")},
        "embeddings": {"path": comp.EMB_PATH, "sha256": emb_sha,
                       "n": len(angles)},
    }
    if args.only in (None, "a"):
        payload["diagnostic_a_T1"] = run_t1(angles, args)
    if args.only in (None, "b"):
        payload["diagnostic_b_purity_vs_T"] = run_purity(items, angles, args)
    payload["wall_time_sec"] = round(time.time() - t0, 1)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("\n=== exp42r DIAGNOSTICS (barred from the verdict) ===")
    if "diagnostic_a_T1" in payload:
        d = payload["diagnostic_a_T1"]["descriptive_read"]
        print("(a) T=1: A2 mean final val  uniform-init=%.4f  normal-init=%.4f"
              % (d["A2_mean_final_val_uniform_init"],
                 d["A2_mean_final_val_normal_init"]))
        print("    A2 mean MAX  val  uniform-init=%.4f  normal-init=%.4f"
              % (d["A2_mean_max_val_uniform_init"],
                 d["A2_mean_max_val_normal_init"]))
        print("    A1 control mean final val=%.4f (exact chance=%s)"
              % (d["A1_control_mean_final_val"], d["A1_at_exact_chance"]))
        print("    learns_at_T1_descriptive = %s"
              % d["learns_at_T1_descriptive"])
    if "diagnostic_b_purity_vs_T" in payload:
        b = payload["diagnostic_b_purity_vs_T"]
        print("(b) pair purity Tr rho_2^2 of the queried pair vs T "
              "(K=%d fixed):" % SWEEP_K)
        print("      T   trained   untrained   S_max(bits)")
        for row in b["purity_vs_T_table"]:
            print("    %3d   %.4f    %.4f      %.4f"
                  % (row["T"], row["mean_pair_purity_trained"],
                     row["mean_pair_purity_untrained_reference"],
                     row["mean_S_max_bits_trained"]))
        print("    purity invariance under question inverse-prep: "
              "max|delta| = %.3e"
              % b["purity_invariance_under_question_inverse_prep"]
                 ["max_abs_delta_over_all_items_and_seeds"])
    print("wrote %s (%.1f s)" % (args.out, payload["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
