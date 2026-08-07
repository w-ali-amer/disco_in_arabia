# -*- coding: utf-8 -*-
"""exp42_selftest.py -- mechanical smoke checks for the exp42 scaffolding.

Runs BEFORE the C5 gate opens (allowed: compilation, shapes, physics
identities, a few gradient steps). NO training, NO results file.

Checks:
  1. gate physics: exchange symmetry of the A1/A2 core ([E,SWAP]=0),
     unitarity of A2/A3 blocks, B1 core swap-commutation, Ry action;
  2. compiler: provenance-checked dataset, compile ALL 3040 items, wire
     counts K=4..12, forward sweep across K per split (norm preserved by
     the simulator's own assertion), values in [0,1];
  3. A1 negative-control identity: swap-twin pair -> identical readout
     values (exchange-symmetric blocks); A2 values differ (dressings);
  4. C3 order-invariance: SVO<->VSO re-render -> bitwise-identical
     statevector path for quantum and B1 (delta == 0.0); B2 input differs;
  5. autodiff: one story per arm, finite nonzero grads, loss decreases
     over a few Adam steps;
  6. C2 scaffold feasibility report (printed).
"""

import math
import random

import torch

import exp42_compiler as comp
import exp42_controls as ctl
import exp42_sim_adapter as sim
from exp42_baselines import (BagBaseline, ClassicalDisCoCirc, SeqBaseline,
                             _pair_swap, build_vocab)
from exp42_models import QuantumStoryModel


def check_gate_physics(verbs):
    # A1/A2 symmetric core commutes with SWAP.
    for th, ph in ((0.3, 1.1), (2.0, 4.4), (5.9, 0.01)):
        E = sim.cphase(th) @ sim.xx(ph)
        err = (sim.SWAP @ E @ sim.SWAP - E).abs().max().item()
        assert err < 1e-6, "core not exchange-symmetric: %g" % err
    # A2/A3 blocks unitary.
    for arm in ("A1", "A2", "A3"):
        m = QuantumStoryModel(arm, verbs, seed=0)
        U = m.verb_matrix(verbs[0])
        err = (U.conj().T @ U - torch.eye(4, dtype=sim.CDTYPE)) \
            .abs().max().item()
        assert err < 1e-5, "%s block not unitary: %g" % (arm, err)
    # B1 symmetric core commutes with the real pair swap.
    b1 = ClassicalDisCoCirc(verbs, seed=0)
    S = _pair_swap(2)
    E = torch.linalg.matrix_exp(0.7 * b1.G1 + 1.9 * b1.G2)
    err = (S @ E @ S - E).abs().max().item()
    assert err < 1e-6, "B1 core not exchange-symmetric: %g" % err
    # Ry action: Ry(pi)|0> = |1> up to sign.
    psi = sim.init_state(1)
    psi = sim.apply_gate(psi, sim.ry(math.pi), (0,))
    assert abs(psi[1].abs().item() - 1.0) < 1e-6
    print("[selftest] gate physics OK ([E,SWAP]=0 quantum+B1, unitarity, Ry)")


def check_compiler(items, angles):
    compiled = comp.compile_all(items, angles, with_tokens=True)
    ks = sorted({cs.K for cs in compiled})
    assert min(ks) >= 4 and max(ks) <= 12, "K out of range: %s" % ks
    for cs in compiled:
        assert len(cs.intro) == cs.K
        for v, wa, wp in cs.events:
            assert 0 <= wa < cs.K and 0 <= wp < cs.K and wa != wp
        vseq, q1, q2 = cs.question
        assert len(vseq) in (1, 2)
        assert 0 <= q1 < cs.K and 0 <= q2 < cs.K and q1 != q2
        assert cs.answer in (0, 1)
        assert cs.tokens, "empty token stream"
    print("[selftest] compiled all %d items; K range %d..%d"
          % (len(compiled), min(ks), max(ks)))
    return compiled


def forward_sweep(compiled, verbs):
    """A handful of stories from each split across the K range through the
    A2 forward pass -- the simulator asserts norm preservation on every
    gate; values must be probabilities."""
    model = QuantumStoryModel("A2", verbs, seed=1)
    by_split = {}
    for cs in compiled:
        by_split.setdefault(cs.split, {}).setdefault(cs.K, cs)
    n = 0
    with torch.no_grad():
        for split, by_k in sorted(by_split.items()):
            for k, cs in sorted(by_k.items()):
                vals = model.story_values(cs)
                for v in vals.tolist():
                    assert -1e-6 <= v <= 1.0 + 1e-6, \
                        "readout value not a probability: %r" % v
                n += 1
    print("[selftest] forward sweep OK: %d stories across splits, "
          "norms preserved, values in [0,1]" % n)


def check_a1_twin_chance(items, angles, verbs):
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], []).append(it)
    pair = next(p for p in by_pair.values()
                if len(p) == 2 and p[0]["answer"] != p[1]["answer"])
    cs_a = comp.compile_story(pair[0], angles)
    cs_b = comp.compile_story(pair[1], angles)
    a1 = QuantumStoryModel("A1", verbs, seed=3)
    a2 = QuantumStoryModel("A2", verbs, seed=3)
    with torch.no_grad():
        d1 = (a1.story_values(cs_a) - a1.story_values(cs_b)).abs().max()
        d2 = (a2.story_values(cs_a) - a2.story_values(cs_b)).abs().max()
    assert d1.item() < 5e-6, \
        "A1 distinguishes swap twins (%.2e) -- symmetry broken!" % d1.item()
    assert d2.item() > 1e-4, \
        "A2 does NOT distinguish swap twins (%.2e) -- dressing inert?" \
        % d2.item()
    print("[selftest] A1 twin identity OK (delta=%.1e); A2 twins differ "
          "(delta=%.1e) -- pair %s" % (d1.item(), d2.item(),
                                       pair[0]["pair_id"]))


def check_c3(items, angles, verbs, vocab):
    item = next(it for it in items if "SVO" in it["order_flags"]
                and "VSO" in it["order_flags"])
    struct = {"A2": QuantumStoryModel("A2", verbs, seed=5),
              "B1": ClassicalDisCoCirc(verbs, seed=5)}
    seq = SeqBaseline(vocab, seed=5)
    rep = ctl.c3_order_invariance(struct, seq, item, angles)
    assert rep["A2"]["machine_precision_identical"], rep
    assert rep["B1"]["machine_precision_identical"], rep
    assert rep["B2"]["input_tokens_differ"], rep
    print("[selftest] C3 OK on item %s: quantum/B1 statevalues identical "
          "(delta=0.0 over %d re-rendered sentences); B2 input differs "
          "(prediction_changed=%s untrained)"
          % (rep["item_id"], rep["n_sentences_rerendered"],
             rep["B2"]["prediction_changed"]))


def check_name_augmentation(compiled, vocab, meta):
    """Augmentation path sanity: a type-preserving name remap keeps the
    token count, changes some tokens, and B2's loss stays finite."""
    from exp42_baselines import remap_tokens, sample_name_mapping
    rng = random.Random(11)
    cs = next(c for c in compiled if c.split == "train")
    mapping = sample_name_mapping(meta["humans"], meta["objects"], rng)
    remapped = remap_tokens(cs.tokens, mapping)
    assert len(remapped) == len(cs.tokens)
    assert any(a != b for a, b in zip(remapped, cs.tokens)), \
        "augmentation mapping changed nothing"
    model = SeqBaseline(vocab, seed=11)
    loss, _ = model.story_loss(cs, mapping=mapping)
    assert math.isfinite(loss.item())
    print("[selftest] name-augmentation path OK (%d/%d tokens remapped)"
          % (sum(1 for a, b in zip(remapped, cs.tokens) if a != b),
             len(remapped)))


def check_autodiff(compiled, verbs, vocab):
    rng = random.Random(0)
    pool = [cs for cs in compiled if cs.split == "train"]
    arms = ("A1", "A2", "A3", "B1", "B2", "B2_generous", "B3")
    for arm in arms:
        if arm in QuantumStoryModel.ARMS:
            model = QuantumStoryModel(arm, verbs, seed=7)
        elif arm == "B1":
            model = ClassicalDisCoCirc(verbs, seed=7)
        elif arm == "B3":
            model = BagBaseline(vocab, seed=7)
        else:
            model = SeqBaseline(vocab, seed=7,
                                variant="matched" if arm == "B2"
                                else "generous")
        cs = pool[rng.randrange(len(pool))]
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        loss0, _ = model.story_loss(cs)
        loss0.backward()
        gn = math.sqrt(sum(p.grad.norm().item() ** 2
                           for p in model.parameters()
                           if p.grad is not None))
        assert math.isfinite(gn) and gn > 0.0, \
            "%s: bad first grad norm %r" % (arm, gn)
        opt.zero_grad()
        final = None
        for _ in range(40):
            opt.zero_grad()
            loss, _ = model.story_loss(cs)
            loss.backward()
            opt.step()
            final = loss.item()
        assert final < loss0.item(), \
            "%s: loss did not decrease (%.4f -> %.4f)" \
            % (arm, loss0.item(), final)
        print("[selftest] autodiff OK %-12s grad_norm=%.2e  loss %.4f -> "
              "%.4f  params=%d"
              % (arm, gn, loss0.item(), final,
                 model.param_counts()["total_trained"]))


def main():
    items, meta, sha = comp.load_dataset()
    angles, emb_meta, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    print("[selftest] dataset sha=%s.. embeddings sha=%s.. source=%s"
          % (sha[:12], emb_sha[:12], emb_meta["source"][:60]))

    check_gate_physics(verbs)
    compiled = check_compiler(items, angles)
    forward_sweep(compiled, verbs)
    check_a1_twin_chance(items, angles, verbs)
    vocab = build_vocab(cs.tokens for cs in compiled)
    check_c3(items, angles, verbs, vocab)
    check_name_augmentation(compiled, vocab, meta)
    check_autodiff(compiled, verbs, vocab)
    print("[selftest] C2 scaffold: %s"
          % ctl.c2_scaffold_report(items, verbs))
    print("[selftest] ALL CHECKS PASSED (mechanical only; training remains "
          "gated on results_exp40b.json harness_calibrated=true)")


if __name__ == "__main__":
    main()
