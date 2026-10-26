# -*- coding: utf-8 -*-
"""exp40r_smoke.py -- smoke battery with DESIGNED-TO-FAIL builds (doc 65 SS2.3).

Runs BEFORE the real exp40r C5 calibration. Every bar below is declared in
this file before any run; none is tuned after seeing a number.

The battery brackets the harness's learning power from both sides, so that a
later at-chance result is interpretable rather than ambiguous:

  S1  SCRAMBLED LABELS (designed-to-fail).  Train answers permuted, so the
      story->answer link is destroyed. The harness MUST NOT fit them.
      A harness that "learns" noise would make any positive meaningless.
      BAR: max clean train accuracy over all epochs < 0.75.

  S2  ONE-HOP EASY TASK (positive control / leak-sensitivity).  Width-2
      stories, introduction walks only, no turns/follows: the answer is a
      direct function of the two queried wires. The harness MUST learn this
      fast. This is the leak-detector calibration -- it proves the harness is
      powerful enough to FIND a trivially recoverable signal, so its
      at-chance behaviour on an honest dataset is evidence about the data and
      not about a dead optimizer.
      BAR: final clean train accuracy >= 0.95 within 20 epochs.

  S3  STORY-BLIND (leaking-question detector).  All events stripped; only the
      shared `person` introductions and the question remain. Every story then
      compiles to the IDENTICAL statevector on the queried wires, so the model
      can only emit a constant prediction and must score exactly 0.5 on a
      balanced set. Any deviation means the question encoding leaks the
      answer through something other than the story.
      BAR: accuracy == 0.5 exactly AND all story values identical to 1e-6.

S4 (wrong-order compile) needs a TRAINED model and therefore runs in the C5
stage, not here.

Writes exp40r_smoke_raw.json (a raw artifact, not a results file). The sole
writer of results_exp40r.json is exp40r_aggregate.py.
"""

import argparse
import json
import random
import sys
import time

import torch

import duneau_task_gen as taskgen
import exp40_mini_duneau as mini
import exp40r_trainer as trainer

# ---- bars, declared before any run -----------------------------------
S1_MAX_TRAIN_ACC_BAR = 0.75      # scrambled labels must stay below this
S2_MIN_TRAIN_ACC_BAR = 0.95      # easy one-hop must reach this
S3_VALUE_TOL = 1e-6              # story-blind: values identical to this

S1_N_TRAIN, S1_EPOCHS = 200, 15
S2_N_TRAIN, S2_EPOCHS = 120, 20
S3_N_ITEMS = 120

ANSWER_OF = lambda s: mini.LABELS[s["answer"]]        # noqa: E731


def balanced(stories, k, rng=None):
    """k stories, exactly balanced yes/no."""
    yes = [s for s in stories if s["answer"] == "yes"]
    no = [s for s in stories if s["answer"] == "no"]
    if rng is not None:
        rng.shuffle(yes)
        rng.shuffle(no)
    return yes[:k // 2] + no[:k // 2]


# ------------------------------------------------------------------ S1
def build_scrambled(stories, seed):
    """Permute the answer field across stories: story->answer link destroyed."""
    rng = random.Random(seed)
    out = [dict(s) for s in stories]
    answers = [s["answer"] for s in out]
    # Derangement-ish: reshuffle until at least 30% of labels actually moved.
    for _ in range(100):
        rng.shuffle(answers)
        moved = sum(1 for a, s in zip(answers, out) if a != s["answer"])
        if moved >= 0.3 * len(out):
            break
    for s, a in zip(out, answers):
        s["answer"] = a
    return out


# ------------------------------------------------------------------ S2
def build_easy_onehop(seed, n):
    """Width-2, introduction walks only: answer = f(the two walk directions).

    Uses the committed generator's own simulate() for ground truth, so the
    easy set cannot silently disagree with the task semantics.
    """
    rng = random.Random(seed)
    out = []
    want = True
    while len(out) < n:
        a, b = rng.sample(taskgen.NAMES, 2)
        da = rng.choice(taskgen.DIRECTIONS)
        db = rng.choice(taskgen.DIRECTIONS)
        events = [("walks", a, da), ("walks", b, db)]
        dirs = taskgen.simulate(events)
        ans = "yes" if dirs[a] == dirs[b] else "no"
        if (ans == "yes") != want:
            continue
        x, y = (a, b) if rng.random() < 0.5 else (b, a)
        out.append({"width": 2, "actors": [a, b],
                    "events": [list(e) for e in events],
                    "question": [x, y], "answer": ans, "n_ancilla": 0})
        want = not want
    return out


# ------------------------------------------------------------------ S3
def build_story_blind(stories):
    """Strip every event; keep cast, question and answer."""
    out = []
    for s in stories:
        t = dict(s)
        t["events"] = []
        t["n_ancilla"] = 0
        out.append(t)
    return out


def run_s1(data, seed):
    print("\n=== S1 scrambled labels (designed-to-fail) ===", flush=True)
    rng = random.Random(777)
    train = build_scrambled(balanced(data["train"], S1_N_TRAIN, rng), 555)
    torch.manual_seed(seed)
    model = mini.Exp40Model(seed)
    traj = trainer.train_with_trajectory(
        model, train, {}, ANSWER_OF, seed=seed, epochs=S1_EPOCHS, lr=0.005,
        batch_size=1, log_prefix="[S1] ")
    got = traj["max_clean_train_acc"]
    passed = got < S1_MAX_TRAIN_ACC_BAR
    print("[S1] max clean train acc = %.4f (bar < %.2f) -> %s"
          % (got, S1_MAX_TRAIN_ACC_BAR, "PASS" if passed else "FAIL"),
          flush=True)
    return {"name": "S1_scrambled_labels", "kind": "designed_to_fail",
            "bar": "max clean train acc < %.2f" % S1_MAX_TRAIN_ACC_BAR,
            "observed_max_clean_train_acc": got, "n_train": len(train),
            "epochs": S1_EPOCHS, "passed": bool(passed),
            "trajectory": trainer.strip_state(traj)}


def run_s2(seed):
    print("\n=== S2 easy one-hop (positive control) ===", flush=True)
    train = build_easy_onehop(4242, S2_N_TRAIN)
    torch.manual_seed(seed)
    model = mini.Exp40Model(seed)
    traj = trainer.train_with_trajectory(
        model, train, {}, ANSWER_OF, seed=seed, epochs=S2_EPOCHS, lr=0.005,
        batch_size=1, log_prefix="[S2] ", early_stop_train=1.0)
    got = traj["final_clean_train_acc"]
    passed = got >= S2_MIN_TRAIN_ACC_BAR
    print("[S2] final clean train acc = %.4f (bar >= %.2f) -> %s"
          % (got, S2_MIN_TRAIN_ACC_BAR, "PASS" if passed else "FAIL"),
          flush=True)
    return {"name": "S2_easy_onehop", "kind": "positive_control",
            "bar": "final clean train acc >= %.2f" % S2_MIN_TRAIN_ACC_BAR,
            "observed_final_clean_train_acc": got,
            "observed_max_clean_train_acc": traj["max_clean_train_acc"],
            "n_train": len(train), "epochs": S2_EPOCHS, "passed": bool(passed),
            "trajectory": trainer.strip_state(traj)}


def run_s3(data, seed):
    print("\n=== S3 story-blind (leaking-question detector) ===", flush=True)
    rng = random.Random(31337)
    items = build_story_blind(balanced(data["train"], S3_N_ITEMS, rng))
    torch.manual_seed(seed)
    model = mini.Exp40Model(seed)
    # No training needed: the claim is structural. Check BOTH an untrained
    # model and one given a few optimizer steps on the blind set (a leak
    # would let training move accuracy off 0.5).
    acc0 = trainer.evaluate(model, items, ANSWER_OF)
    with torch.no_grad():
        vals = torch.stack([model.story_values(s) for s in items])
    spread = (vals - vals[0]).abs().max().item()
    traj = trainer.train_with_trajectory(
        model, items, {}, ANSWER_OF, seed=seed, epochs=5, lr=0.005,
        batch_size=1, log_prefix="[S3] ")
    acc1 = traj["final_clean_train_acc"]
    n_yes = sum(1 for s in items if s["answer"] == "yes")
    balanced_ok = n_yes * 2 == len(items)
    passed = bool(balanced_ok and abs(acc0 - 0.5) < 1e-9
                  and abs(acc1 - 0.5) < 1e-9 and spread < S3_VALUE_TOL)
    print("[S3] untrained acc=%.6f  after-5-epochs acc=%.6f  value spread=%.3e"
          "  balanced=%s -> %s"
          % (acc0, acc1, spread, balanced_ok, "PASS" if passed else "FAIL"),
          flush=True)
    return {"name": "S3_story_blind", "kind": "leak_detector",
            "bar": "acc == 0.5 exactly (untrained AND after 5 epochs) and "
                   "all story values identical to %g" % S3_VALUE_TOL,
            "observed_untrained_acc": acc0, "observed_trained_acc": acc1,
            "observed_value_spread": spread, "balanced": balanced_ok,
            "n_items": len(items), "passed": passed,
            "trajectory": trainer.strip_state(traj)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=str, default="duneau_mini_data_b.json")
    ap.add_argument("--out", type=str, default="exp40r_smoke_raw.json")
    args = ap.parse_args()

    t0 = time.time()
    print("[smoke] torch %s  python %s"
          % (torch.__version__, sys.version.split()[0]), flush=True)
    mini.physics_selftest()
    data = mini.load_or_generate_data(args.data, 40)

    tests = [run_s1(data, args.seed), run_s2(args.seed), run_s3(data, args.seed)]
    all_passed = all(t["passed"] for t in tests)

    payload = {
        "stage": "exp40r smoke battery (doc 65 SS2.3 designed-to-fail builds)",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "dataset": {"path": args.data, "sha256": data["meta"]["sha256"]},
        "seed": args.seed,
        "tests": tests,
        "smoke_all_passed": all_passed,
        "wall_time_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print("\n=== smoke battery: %s ==="
          % ("ALL PASSED" if all_passed else "FAILURE PRESENT"), flush=True)
    for t in tests:
        print("  %-22s %s" % (t["name"], "PASS" if t["passed"] else "FAIL"))
    print("wrote %s (%.1f s)" % (args.out, payload["wall_time_sec"]))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
