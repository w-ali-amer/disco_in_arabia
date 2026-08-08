"""B1 vs A3 INITIALISATION AUDIT -- training-free witness, NO pre-registered verdict.

Motivation (recorded before any remediation is run): exp43b Phase 1 found A3
meeting the L1 learnability gate in 4/8 configs (best val 0.77, test 0.78) while
B1 sat at chance in 8/8 (max train 0.559 over 60 epochs, curves flat).

exp43a Lemma 1 (verified to 6e-08 in results_exp43a_analysis_c.json) proves that
verb blocks commuting with the pair SWAP make twin stories compile to IDENTICAL
states, pinning ANY readout to exactly 50% on twins. B1's core is SWAP-commuting
by construction -- exp42_baselines._sym_generators docstring states
"[expm(aG1+bG2), SWAP] = 0". B1's only swap-sensitivity therefore comes from the
dressing kron(d0,d1) @ E @ kron(d2,d3), and that dressing is initialised at
identity + 0.1*randn (exp42_baselines.py:98-101). At exact identity d0 == d1 and
d2 == d3, so both Kronecker factors commute with SWAP and the whole verb matrix
does too. B1 therefore starts inside a 0.1-scale neighbourhood of the manifold on
which L1 -- a pure swap-twin rung -- is provably at chance. The A arms initialise
their dressings with uniform angles over [0, 2pi] (exp42_models.py:65), i.e. far
from that manifold.

This script MEASURES that at initialisation only. It trains nothing and decides
nothing. Output follows the results_exp43a_analysis*.json convention: witnesses
for the write-up, no verdict booleans.

Safe to run alongside exp43b_phase1: single-threaded, and intended to be launched
under nice(19).
"""
import json
import os
import time

import torch

torch.set_num_threads(1)

import exp42_compiler as comp                                   # noqa: E402
from exp42_baselines import ClassicalDisCoCirc, _pair_swap      # noqa: E402
from exp42_models import QuantumStoryModel                      # noqa: E402

RUNG = "L1"
SEEDS = (0, 1)
N_PAIRS = 200          # first N twin pairs of the train split
N_GRAD = 64            # stories in the fixed gradient batch
OUT = "results_exp43b_b1_init_audit.json"


def q(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"min": xs[0], "p25": xs[n // 4], "median": xs[n // 2],
            "p75": xs[(3 * n) // 4], "max": xs[-1], "n": n}


def rel_commutator(M, S):
    M = M.to(torch.complex128) if M.is_complex() else M.to(torch.float64)
    S = S.to(M.dtype)
    num = torch.linalg.matrix_norm(M @ S - S @ M).item()
    den = torch.linalg.matrix_norm(M).item()
    return num / max(den, 1e-12)


def main():
    t0 = time.time()
    angles, emb_meta, emb_sha = comp.load_embeddings()
    with open("stories_exp43b_%s.json" % RUNG, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    assert sha == meta["dataset_sha256"], "dataset hash mismatch"
    verbs = sorted(meta["verb_inventory"].keys())

    train = [i for i in items if i["split"] == "train"]
    by_pair = {}
    for it in train:
        by_pair.setdefault(it["pair_id"], []).append(it)
    pairs = [p for p in sorted(by_pair) if len(by_pair[p]) == 2][:N_PAIRS]

    compiled = {}
    for p in pairs:
        a, b = sorted(by_pair[p], key=lambda x: x["id"])
        compiled[p] = (comp.compile_story(a, angles),
                       comp.compile_story(b, angles))
    flat = [cs for p in pairs for cs in compiled[p]]
    n_opposite = sum(1 for p in pairs
                     if compiled[p][0].answer != compiled[p][1].answer)

    S = _pair_swap(2).to(torch.float64)
    out = {
        "experiment": "exp43b_b1_init_audit",
        "note": ("training-free witnesses for the exp43b Phase-1 write-up; "
                 "carries NO pre-registered verdicts (none exist for this "
                 "audit). Motivation and the Lemma-1 argument are in the "
                 "module docstring and in qnlp_private_docs/24."),
        "rung": RUNG,
        "dataset_sha256": sha,
        "embeddings_sha256": emb_sha,
        "n_twin_pairs_audited": len(pairs),
        "n_pairs_with_opposite_answers": n_opposite,
        "verbs": verbs,
        "torch_version": torch.__version__,
        "arms": {},
    }

    for arm in ("B1", "A3"):
        per_seed = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = (ClassicalDisCoCirc(verbs, seed, d_ref=2) if arm == "B1"
                     else QuantumStoryModel("A3", verbs, seed))

            comm = {v: rel_commutator(model.verb_matrix(v).detach(), S)
                    for v in verbs}

            deltas, collapsed, correct = [], 0, 0
            with torch.no_grad():
                for p in pairs:
                    ca, cb = compiled[p]
                    va = model.story_values(ca).detach()
                    vb = model.story_values(cb).detach()
                    va = va / va.sum()
                    vb = vb / vb.sum()
                    deltas.append(float((va - vb).abs().sum()))
                    pa = int(torch.argmax(va).item())
                    pb = int(torch.argmax(vb).item())
                    if pa == pb:
                        collapsed += 1
                    correct += int(pa == ca.answer) + int(pb == cb.answer)

            model.zero_grad()
            bloss = 0.0
            for cs in flat[:N_GRAD]:
                loss, _ = model.story_loss(cs)
                bloss = bloss + loss
            (bloss / N_GRAD).backward()
            gsq, gsq_dress = 0.0, 0.0
            for name, prm in model.named_parameters():
                if prm.grad is None:
                    continue
                g2 = float((prm.grad ** 2).sum())
                gsq += g2
                if name.startswith("dress"):
                    gsq_dress += g2

            per_seed[str(seed)] = {
                "rel_commutator_with_SWAP_per_verb":
                    {v: round(c, 6) for v, c in comm.items()},
                "rel_commutator_max": round(max(comm.values()), 6),
                "rel_commutator_mean":
                    round(sum(comm.values()) / len(comm), 6),
                "twin_value_L1_gap": q([round(d, 8) for d in deltas]),
                "frac_twin_pairs_predicting_identically":
                    collapsed / len(pairs),
                "init_accuracy_on_audited_items": correct / (2 * len(pairs)),
                "grad_global_norm_at_init": gsq ** 0.5,
                "grad_norm_dress_only_at_init": gsq_dress ** 0.5,
                "grad_frac_in_dress": (gsq_dress / gsq) if gsq > 0 else None,
            }
        out["arms"][arm] = per_seed

    out["wall_time_sec"] = round(time.time() - t0, 1)
    assert not os.path.exists(OUT), "refusing to overwrite %s (rule 3)" % OUT
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote %s in %.1fs" % (OUT, out["wall_time_sec"]))
    for arm, ps in out["arms"].items():
        for s, r in ps.items():
            print("%s s%s: comm_max=%.4g  twinL1_median=%.4g  "
                  "collapsed=%.3f  init_acc=%.3f  |g|=%.4g"
                  % (arm, s, r["rel_commutator_max"],
                     r["twin_value_L1_gap"]["median"],
                     r["frac_twin_pairs_predicting_identically"],
                     r["init_accuracy_on_audited_items"],
                     r["grad_global_norm_at_init"]))


if __name__ == "__main__":
    main()
