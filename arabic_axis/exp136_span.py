# -*- coding: utf-8 -*-
"""exp136_span.py -- doc 136 stage B (REGISTERED 4a25d61): span-dimension diagnostic for the Navascues-Vertesi
(1412.0924) rung.  For each operator subset (S2, S3) and word level (L1, L1+rhoM, L1+rhoM+MM), sample random
d-dimensional realisations (Haar pure states, general POVMs E_a = V^dag P_a V), form the moment matrix
Gamma_uw = tr(u^dag w) over the word list, and grow the real linear span of the (Re, Im) upper triangles by
Gram-Schmidt (relative tol 1e-9) until 50 consecutive samples add nothing.  Readout: dim(span at d) < dim(span at
4d) means the level "sees" the dimension.  numpy only.  Writes results_exp136_B.json (Rule 3).
"""
import hashlib, json, os, sys, time
import numpy as np
import exp94_common as C94

ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "results_exp136_B.json")
NSTATE = 6; TOL = 1e-9; PATIENCE = 50; SEED = 1361
SUBSETS = {"S2": {"nmeas": 3, "nout": 2, "d": (2, 8)}, "S3": {"nmeas": 3, "nout": 3, "d": (4, 16)}}
LEVELS = ("L1", "L1+rhoM", "L1+rhoM+MM")


def sample(rng, d, nmeas, nout):
    rho = []
    for _ in range(NSTATE):
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d); v /= np.linalg.norm(v); rho.append(np.outer(v, v.conj()))
    eff = []                                                                     # last outcome dropped: E_last = 1 - sum (same linear dependency at every d)
    for _ in range(nmeas):
        Z = rng.standard_normal((d * nout, d)) + 1j * rng.standard_normal((d * nout, d)); V, _ = np.linalg.qr(Z)
        E = [V[a * d:(a + 1) * d].conj().T @ V[a * d:(a + 1) * d] for a in range(nout)]
        assert np.allclose(sum(E), np.eye(d), atol=1e-10)
        eff += E[:-1]
    return rho, eff


def words(rho, eff, level, d):
    W = [np.eye(d, dtype=complex)] + rho + eff
    if level in ("L1+rhoM", "L1+rhoM+MM"):
        W += [r @ e for r in rho for e in eff]
    if level == "L1+rhoM+MM":
        W += [e @ f for e in eff for f in eff]
    return W


def moment_vec(W, iu):
    A = np.array([w.reshape(-1) for w in W]); G = A.conj() @ A.T               # G_uw = tr(u^dag w)
    up = G[iu]; return np.concatenate([up.real, up[iu[0] != iu[1]].imag])


def span_dim(d, nmeas, nout, level, rng, log):
    rho, eff = sample(rng, d, nmeas, nout); n = len(words(rho, eff, level, d)); iu = np.triu_indices(n)
    D = n * n; Q = np.zeros((D, D)); k = 0; stale = 0; ns = 0; acc_min = np.inf; rej_max = 0.0; t0 = time.time()
    while stale < PATIENCE and k < D:
        B = []
        for _ in range(PATIENCE):
            rho, eff = sample(rng, d, nmeas, nout); v = moment_vec(words(rho, eff, level, d), iu); B.append(v / np.linalg.norm(v))
        B = np.array(B).T; ns += B.shape[1]; added = 0
        for _ in range(2):                                                       # project twice (re-orthogonalisation)
            B = B - Q[:, :k] @ (Q[:, :k].T @ B)
        for j in range(B.shape[1]):                                              # sequential Gram-Schmidt within the batch
            r = B[:, j].copy()
            for _ in range(2):
                r = r - Q[:, k - added:k] @ (Q[:, k - added:k].T @ r)
            nr = np.linalg.norm(r)
            if nr > TOL and k < D:
                Q[:, k] = r / nr; k += 1; added += 1; acc_min = min(acc_min, nr)
            else:
                rej_max = max(rej_max, nr)
        stale = 0 if added else stale + PATIENCE
    out = {"d": d, "n_words": n, "ambient_real_dim": D, "span_dim": int(k), "samples": ns,
           "min_accepted_residual": float(acc_min), "max_rejected_residual": float(rej_max), "wall_s": time.time() - t0}
    log(out); return out


def main():
    assert not os.path.exists(OUT), "Rule 3"
    rng = np.random.default_rng(SEED); res = {"doc": "136 stage B", "registered": "4a25d61", "tol": TOL, "patience": PATIENCE,
        "seed": SEED, "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "cases": {}, "readout": {}}
    def log(o):
        print(json.dumps(o), flush=True)
    t0 = time.time()
    for sname, s in SUBSETS.items():
        for level in LEVELS:
            key = "%s|%s" % (sname, level); print("==", key, flush=True)
            r = {str(d): span_dim(d, s["nmeas"], s["nout"], level, rng, log) for d in s["d"]}
            lo, hi = (r[str(d)]["span_dim"] for d in s["d"])
            res["cases"][key] = r; res["readout"][key] = {"d": s["d"][0], "span_d": lo, "span_4d": hi, "sees_d": bool(lo < hi)}
            print("READOUT", key, res["readout"][key], flush=True)
    pred = {"S2|L1": False, "S3|L1": False, "S2|L1+rhoM": True}
    res["prediction"] = "L1 sees nothing; L1+rhoM sees d=2; d=4 first at L1+rhoM+MM or not at all (doc 136 B)"
    res["prediction_checks"] = {k: res["readout"][k]["sees_d"] == v for k, v in pred.items()}
    res["wall_s"] = time.time() - t0; res["provenance"] = C94.provenance()
    C94.write_new_json(OUT, res); print("DONE", json.dumps(res["readout"], indent=1), flush=True)


if __name__ == "__main__":
    main()
