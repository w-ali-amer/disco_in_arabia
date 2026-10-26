"""duneau_task_gen.py -- two-directional "following" story generator (exp40 mini-Duneau).

Re-implements the task generator of Duneau et al., arXiv:2409.08777 (QDisCoCirc
question answering) at mini scale, per the frozen spec in doc 22 SS9:

- Actors (2-12), directions {north, south}.
- Surface verbs: "X walks north", "X walks south", "X turns around",
  "X follows Y", "X goes in the opposite direction of Y".
- Semantic rewrites (hardcoded, as in the paper):
    goes-opposite(X, Y) := follows(X, Y) then turn-around(X).
- Ground truth by classical simulation: each actor carries a direction state;
  follows copies the followee's CURRENT direction; turn-around flips.
- Every story starts by walking every actor (so every actor ends with a
  definite direction). Walk events occur ONLY in this introduction block:
  a mid-story re-walk has overwrite semantics (direction := d), which is not
  realizable by a unitary on the actor's wire, and the paper's story dynamics
  after introduction are turns / follows / goes-opposite only.
  Question: "does X go in the same direction as Y?"
  Balanced yes/no; (X, Y) sampled among the actors.
- Splits by width: train/validA widths 2-8 (80/20 stratified per width),
  valid_comp widths 9-12. Sizes ~500 train+validA, ~300 valid_comp.
- follows + goes-opposite events per story are capped so that
  (actors + ancilla count) <= 18 qubits (one fresh ancilla per such event).
- Fully deterministic given a seed (python random.Random only).
"""

import argparse
import hashlib
import json
import random

DIRECTIONS = ("north", "south")

# Names are surface decoration only -- the model shares one `person` state.
NAMES = [
    "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Hugo",
    "Ivy", "Jack", "Kara", "Liam", "Mona", "Nina", "Omar", "Pia",
    "Quinn", "Rania", "Sami", "Tara",
]

MAX_QUBITS = 18

TRAIN_WIDTHS = list(range(2, 9))      # 2-8 actors
COMP_WIDTHS = list(range(9, 13))      # 9-12 actors
PER_WIDTH_TRAINPOOL = 72              # per width, split 58/14 -> 406 train, 98 validA
PER_WIDTH_COMP = 76                   # per width -> 304 valid_comp
# 1.1: walks restricted to the introduction block (mid-story re-walk has
# overwrite semantics, which no unitary verb can realize; paper dynamics
# after introduction are turns/follows/goes-opposite only).
GENERATOR_VERSION = "exp40-taskgen-1.1"


def flip(d):
    return "south" if d == "north" else "north"


def simulate(events):
    """Classical semantics of a story. Returns {actor: final_direction}.

    goes-opposite is evaluated with its rewrite semantics
    (follows then turn-around on the first actor), i.e. X ends up in the
    direction opposite to Y's CURRENT direction.
    """
    dirs = {}
    for ev in events:
        kind = ev[0]
        if kind == "walks":
            _, a, d = ev
            dirs[a] = d
        elif kind == "turns":
            _, a = ev
            assert a in dirs, "turn-around on actor with no direction"
            dirs[a] = flip(dirs[a])
        elif kind == "follows":
            _, a, b = ev
            assert b in dirs, "follows a followee with no direction"
            dirs[a] = dirs[b]
        elif kind == "opposite":
            _, a, b = ev
            assert b in dirs, "goes-opposite of an actor with no direction"
            dirs[a] = flip(dirs[b])
        else:
            raise ValueError("unknown event kind: %r" % (kind,))
    return dirs


def count_ancillas(events):
    """One fresh ancilla per follows / goes-opposite event."""
    return sum(1 for ev in events if ev[0] in ("follows", "opposite"))


def gen_story(rng, width, want_yes):
    """Generate one story of the given width whose question has the wanted label."""
    for _attempt in range(500):
        actors = rng.sample(NAMES, width)
        events = []
        intro_order = actors[:]
        rng.shuffle(intro_order)
        for a in intro_order:
            events.append(("walks", a, rng.choice(DIRECTIONS)))

        anc_cap = min(width, MAX_QUBITS - width)
        n_anc = 0
        n_extra = rng.randint(width, 2 * width)
        for _ in range(n_extra):
            # Post-introduction dynamics only: turns / follows / goes-opposite.
            # NO mid-story walks (overwrite semantics is not unitary).
            kinds = ["turns", "turns"]
            if n_anc < anc_cap:
                kinds += ["follows", "follows", "opposite"]
            k = rng.choice(kinds)
            if k == "turns":
                events.append(("turns", rng.choice(actors)))
            else:
                a, b = rng.sample(actors, 2)
                events.append((k, a, b))
                n_anc += 1

        dirs = simulate(events)
        assert set(dirs) == set(actors)
        assert width + n_anc <= MAX_QUBITS

        pairs = [(a, b) for i, a in enumerate(actors) for b in actors[i + 1:]]
        pool = [(a, b) for (a, b) in pairs if (dirs[a] == dirs[b]) == want_yes]
        if not pool:
            continue
        x, y = rng.choice(pool)
        if rng.random() < 0.5:
            x, y = y, x
        return {
            "width": width,
            "actors": actors,
            "events": [list(e) for e in events],
            "question": [x, y],
            "answer": "yes" if want_yes else "no",
            "n_ancilla": n_anc,
        }
    raise RuntimeError("could not generate story width=%d want_yes=%s" % (width, want_yes))


def generate_dataset(seed):
    """Deterministic dataset: dict with splits train / validA / valid_comp + meta."""
    rng = random.Random(seed)
    train, validA, valid_comp = [], [], []

    for w in TRAIN_WIDTHS:
        half = PER_WIDTH_TRAINPOOL // 2
        yes = [gen_story(rng, w, True) for _ in range(half)]
        no = [gen_story(rng, w, False) for _ in range(half)]
        cut = int(round(0.8 * half))  # 29 of 36 per class -> 58 train / 14 validA
        train += yes[:cut] + no[:cut]
        validA += yes[cut:] + no[cut:]

    for w in COMP_WIDTHS:
        half = PER_WIDTH_COMP // 2
        valid_comp += [gen_story(rng, w, True) for _ in range(half)]
        valid_comp += [gen_story(rng, w, False) for _ in range(half)]

    splits = {"train": train, "validA": validA, "valid_comp": valid_comp}
    payload = json.dumps(splits, sort_keys=True, separators=(",", ":"))
    data_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "meta": {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "sha256": data_hash,
            "sizes": {k: len(v) for k, v in splits.items()},
            "train_widths": TRAIN_WIDTHS,
            "comp_widths": COMP_WIDTHS,
            "max_qubits": MAX_QUBITS,
        },
        **splits,
    }


def _selftest():
    # Hand-checkable semantics.
    d = simulate([("walks", "A", "north"), ("walks", "B", "south")])
    assert d == {"A": "north", "B": "south"}
    d = simulate([("walks", "A", "north"), ("walks", "B", "south"), ("follows", "B", "A")])
    assert d == {"A": "north", "B": "north"}
    d = simulate([("walks", "A", "north"), ("walks", "B", "south"), ("opposite", "B", "A")])
    assert d == {"A": "north", "B": "south"}
    # goes-opposite rewrite == follows then turn on first actor.
    d2 = simulate([("walks", "A", "north"), ("walks", "B", "south"),
                   ("follows", "B", "A"), ("turns", "B")])
    assert d == d2
    d = simulate([("walks", "A", "north"), ("turns", "A")])
    assert d == {"A": "south"}
    # follows copies the CURRENT direction (order matters).
    d = simulate([("walks", "A", "north"), ("walks", "B", "south"),
                  ("follows", "B", "A"), ("turns", "A")])
    assert d == {"A": "south", "B": "north"}

    # Dataset invariants.
    ds = generate_dataset(40)
    ds2 = generate_dataset(40)
    assert ds["meta"]["sha256"] == ds2["meta"]["sha256"], "generator not deterministic"
    for split in ("train", "validA", "valid_comp"):
        stories = ds[split]
        n_yes = sum(1 for s in stories if s["answer"] == "yes")
        assert n_yes * 2 == len(stories), "split %s not balanced" % split
        for s in stories:
            dirs = simulate([tuple(e) for e in s["events"]])
            assert set(dirs) == set(s["actors"]), "actor without definite direction"
            x, y = s["question"]
            assert x != y and x in dirs and y in dirs
            truth = "yes" if dirs[x] == dirs[y] else "no"
            assert truth == s["answer"], "stored answer disagrees with simulation"
            assert s["n_ancilla"] == count_ancillas(s["events"])
            assert s["width"] + s["n_ancilla"] <= MAX_QUBITS
            # walks are introduction-only (first `width` events), v1.1
            intro, rest = s["events"][:s["width"]], s["events"][s["width"]:]
            assert all(e[0] == "walks" for e in intro)
            assert all(e[0] != "walks" for e in rest), "mid-story re-walk"
    widths = sorted({s["width"] for s in ds["train"]})
    assert widths == TRAIN_WIDTHS
    widths = sorted({s["width"] for s in ds["valid_comp"]})
    assert widths == COMP_WIDTHS
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--out", type=str, default="duneau_mini_data.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ds = _selftest()
        print("[duneau_task_gen] selftest OK  sha256=%s  sizes=%s"
              % (ds["meta"]["sha256"][:16], ds["meta"]["sizes"]))
        return

    ds = generate_dataset(args.seed)
    with open(args.out, "w") as f:
        json.dump(ds, f, indent=1)
    print("[duneau_task_gen] wrote %s  seed=%d  sha256=%s  sizes=%s"
          % (args.out, args.seed, ds["meta"]["sha256"][:16], ds["meta"]["sizes"]))


if __name__ == "__main__":
    main()
