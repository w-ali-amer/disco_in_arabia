# -*- coding: utf-8 -*-
"""exp95_data.py -- doc 90 (REGISTERED 1217f96): pro-drop CHAIN stories, compiler, encoding.

Story: intro (2 candidates, distinct classes/stems) ; T events [ثمّ, V_t+suffix(agent_t), O_t] (distinct verbs & objects) ;
question هل V_q(partial agr) X O_q؟  Label YES iff X is agent_q.  L = #{t' > q : agent_t' = X} (identical across twins).
Splits (pairs): train 1600 / val 200 / test_id 300 : T in {1,2}, L in {0,1};  test_oodl 300 : T in {3,4}, L in {2,3};
test_long0 300 : T in {3,4}, L = 0.   --corrupt: agent_q := agent_{q-1} (q >= 2), no twins (G-V designed-to-fail).
Reuses the exp94 lexicon and the ONE segmentation function (exp94_common).
"""
import argparse, random
import torch
import exp94_common as C

DRAWS = {"primary": 95000, "confirm": 95500, "smoke": 95900, "fresh": 95700,   # fresh: doc 92 (exp97), added 2026-09-27
         "e98p": 98000, "e98c": 98500}                                      # doc 93 (exp98), added 2026-09-28
SIZES = {"train": 1600, "val": 200, "test_id": 300, "test_oodl": 300, "test_long0": 300}


def plan(rng, T, L):
    """-> (q index, agents list, x) with exactly L X-events after q. X is candidate x."""
    for _ in range(1000):
        q = rng.randrange(T)
        after = T - 1 - q
        if after < L:
            continue
        x = rng.randrange(2)
        agents = [rng.randrange(2) for _ in range(T)]
        later = list(range(q + 1, T)); rng.shuffle(later)
        for i, t in enumerate(later):
            agents[t] = x if i < L else 1 - x
        return q, agents, x
    raise RuntimeError


def story(rng, T, L, corrupt=False):
    cls = rng.sample(C.CLASSES, 2); stems = rng.sample(C.STEMS, 2)
    verbs = rng.sample(C.VERBS, T); objs = rng.sample(C.OBJECTS, T)
    q, agents, x = plan(rng, T, L)
    if corrupt and q >= 1:
        agents[q] = agents[q - 1]
    intro = C.INTRO + [C.noun_surface(stems[0], cls[0]), C.noun_surface(stems[1], cls[1], conj=True)]
    def render(ag_q):
        ag = list(agents); ag[q] = ag_q
        ev = [[C.THEN, C.verb_surface(verbs[t], cls[ag[t]]), C.nfc(objs[t] + "َ")] for t in range(T)]
        qq = [C.Q, C.verb_surface(verbs[q], (cls[x][0], "sg")), C.noun_surface(stems[x], cls[x]), C.nfc(objs[q] + "َ"), "؟"]
        Lval = sum(1 for t in range(q + 1, T) if ag[t] == x)
        return {"intro": intro, "events": ev, "q": qq, "label": int(ag_q == x), "gold_agents": ag,
                "meta": {"T": T, "L": Lval, "q": q, "x": x, "B": sum(1 for t in range(q) if ag[t] == x),
                         "classes": ["%s-%s" % c for c in cls]}}
    if corrupt:
        return [render(agents[q])]
    return [render(x), render(1 - x)]


def split(rng, n, Ts, Ls, tag, corrupt=False):
    out = []
    for k in range(n):
        T = Ts[k % len(Ts)]
        Lopts = [l for l in Ls if l <= T - 1]
        L = Lopts[(k // len(Ts)) % len(Lopts)]
        for j, st in enumerate(story(rng, T, L, corrupt)):
            st["id"], st["pair"] = "%s|%04d|%d" % (tag, k, j), "%s|%04d" % (tag, k)
            out.append(st)
    return out


def build(draw, corrupt=False):
    rng = random.Random(DRAWS[draw] + (7 if corrupt else 0))
    d = {"draw": draw, "sizes": SIZES}
    d["train"] = split(rng, SIZES["train"], [1, 2], [0, 1], "train", corrupt)
    d["val"] = split(rng, SIZES["val"], [1, 2], [0, 1], "val", corrupt)
    d["test_id"] = split(rng, SIZES["test_id"], [1, 2], [0, 1], "tid", corrupt)
    d["test_oodl"] = split(rng, SIZES["test_oodl"], [3, 4], [2, 3], "oodl", corrupt)
    d["test_long0"] = split(rng, SIZES["test_long0"], [3, 4], [0], "long0", corrupt)
    fixed = random.Random(94000)                          # doc 89's intro-angle map, reused (doc 90 §2)
    d["intro_angles"] = {s: fixed.uniform(0, 3.141592653589793) for s in C.STEMS + C.OBJECTS}
    d["provenance"] = C.provenance()
    return d


def compile_story(st):
    assert st["intro"][:2] == C.INTRO
    cands = [C.segment_noun(t) for t in st["intro"][2:]]
    events = []
    for ev in st["events"]:
        assert ev[0] == C.THEN
        events.append((C.segment_verb(ev[1]), C.segment_noun(ev[2])[0]))
    qverb = C.segment_verb(st["q"][1])[0]                  # stem only; the question suffix is not compiled
    x_stem, qobj = C.segment_noun(st["q"][2])[0], C.segment_noun(st["q"][3])[0]
    q = [i for i, ((vs, _, _), o) in enumerate(events) if vs == qverb and o == qobj]
    assert len(q) == 1
    return {"cands": cands, "events": events, "q": q[0], "x": [c[0] for c in cands].index(x_stem)}


def encode(stories, angles):
    groups = {}
    for i, st in enumerate(stories):
        c = compile_story(st)
        groups.setdefault(len(c["events"]), []).append((i, c, st))
    out = {}
    for T, rows in groups.items():
        t = lambda xs, dt=torch.long: torch.tensor(xs, dtype=dt)
        out[T] = ({"K": 2, "T": T,
                   "cs": t([[C.STEMS.index(s) for s, _, _ in c["cands"]] for _, c, _ in rows]),
                   "cg": t([[C.NOUN_G.index(g) for _, g, _ in c["cands"]] for _, c, _ in rows]),
                   "cn": t([[C.NOUN_N.index(n) for _, _, n in c["cands"]] for _, c, _ in rows]),
                   "theta": t([[angles[s] for s, _, _ in c["cands"]] for _, c, _ in rows], torch.float32),
                   "v": t([[C.VERBS.index(vs) for (vs, _, _), _ in c["events"]] for _, c, _ in rows]),
                   "vg": t([[C.VERB_G.index(g) for (_, g, _), _ in c["events"]] for _, c, _ in rows]),
                   "vn": t([[C.VERB_N.index(n) for (_, _, n), _ in c["events"]] for _, c, _ in rows]),
                   "o": t([[C.OBJECTS.index(o) for _, o in c["events"]] for _, c, _ in rows]),
                   "theta_o": t([[angles[o] for _, o in c["events"]] for _, c, _ in rows], torch.float32),
                   "q": t([c["q"] for _, c, _ in rows]), "x": t([c["x"] for _, c, _ in rows]),
                   "gold": t([st["gold_agents"] for _, _, st in rows]),
                   "label": t([st["label"] for _, _, st in rows])}, [i for i, _, _ in rows])
    return out


def sub(b, j):
    return {k: (v[j] if torch.is_tensor(v) else v) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draw", required=True, choices=list(DRAWS)); a = ap.parse_args()
    d = build(a.draw)
    C.write_new_json("stories_exp95_%s.json" % a.draw, d)
    print({k: len(v) for k, v in d.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
