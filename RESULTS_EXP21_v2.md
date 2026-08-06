# Exp21 series — trained lexical-content transport: final accounting v2 (2026-08-06)

> **TL;DR.** On leak-free splits the trained quantum encoder shows **no
> detectable transfer**: **0 of 5** splits significant under a correctly
> exchangeable sentence-level permutation test, mean held-out AUC **0.563**
> with a training effect of **−0.004**. The matched classical control **C2
> wins all five splits** (mean **0.975**). The one nominally-significant
> number, pooled p = **0.0287**, is an artefact of testing against 0.5 rather
> than against the permutation null: re-centred on each seed's own null it is
> **p = 0.232**. v1's "2/5 significant" was **leakage** — two verbatim-
> duplicated twin couples, landing in exactly the two splits v1 called
> significant.

**This supersedes RESULTS_EXP21.md.** v1 is kept for the record; every number
in it that changed is corrected here, and the reason for each change is stated.
The published paper does not depend on any exp21 number — these are post-paper
claims.

## Question (retitled)

**Can composition fixed by grammar + a tiny trained lexical encoder transport
*lexical content* into sentence-level geometry that generalizes to held-out
sentence pairs?**

v1 asked about "semantic geometry that generalizes from very little data".
That was too strong a title for what the experiment can measure. The encoder's
per-type PCA feature space is fit over the vocabulary of **all** 120 sentences,
and the words of the test sentences also occur in training sentences. Couples
are held out at the **sentence** level, never at the **vocabulary** level, so
the design is transductive in the words. What is actually under test is
whether fixed grammatical composition transports *already-seen lexical
content* into the geometry of *unseen word combinations*. Generalization to
novel words is untested and no claim is made about it.

## What changed since v1 (three defects, all of which favoured the positive)

- **D1 — duplicate twins (leakage).** `sentences.json:WordOrderMatched` has
  120 sentences but only **116 distinct sentence texts**. The multiset twin
  construction produced 60 couples, of which **2 are verbatim duplicates** of 2
  other couples. Splitting over couples let a training pair land in the test
  set *as the same sentences*.
- **D2 — non-exchangeable permutation null.** v1's `analyse()` pooled the 12
  twin fidelities and 60 same-order fidelities into one vector and permuted the
  A/B *label*. That null needs all 72 values to be exchangeable; they are not
  (60 within-order pairs are drawn from only 24 sentences, so each sentence
  sits in ~5 pairs). The null variance was understated.
- **D3 — no pooled inference.** "2/5 splits significant" is a vote count, not a
  test.

All three bias toward the positive, so correcting them can only weaken the
already-fragile v1 positive. **exp21c** (`exp21c_dedup_rerun.py`) re-runs the
identical protocol with all three fixed.

### D1 was not hypothetical, and it landed exactly where it mattered

Measured on the **original** 60-twin splits
(`results_exp21c.json :: original_split_leakage_diagnostic`):

| split seed | test couples verbatim already in train | v1 verdict |
|---|---|---|
| 42 | **1 of 12** | significant (p = 0.017) |
| 43 | 0 | not significant |
| 44 | **1 of 12** | significant (p = 0.004) |
| 45 | 0 | not significant |
| 46 | 0 | not significant |

The two leaking splits are **exactly** the two v1 reported as significant, and
the three clean splits are exactly the three that were not. Perfect rank
correspondence.

The duplicated couples dropped (keeping the first occurrence of each) were
couple #43 = sentences [86, 87] (البنت رسمت الصورة / رسمت البنت الصورة,
duplicate of couple #37 = [74, 75]) and couple #56 = sentences [112, 113]
(السائق يوقف السيارة / يوقف السائق السيارة, duplicate of couple #29 =
[58, 59]) → **58 unique couples over 116 distinct sentences**.

## Designs

- **21a**: 546 free symbol values, SPSA, 36/12/12 pair split. *(unchanged)*
- **21b-lite** *(v1, superseded)*: 91-parameter linear encoder (per-type PCA-6
  features → circuit parameters), least-squares init to the exact-AE point,
  SPSA 6000 iters, val-margin model selection, 5 split seeds over **60**
  couples at 36/12/12.
- **21c** *(this document)*: identical encoder, init, optimizer, iteration
  count and model-selection rule, re-run over the **58 de-duplicated** couples
  at **34/12/12**, seeds 42–46. Val and test are held at 12 couples so every
  test statistic keeps the exact shape it had in v1 (12 twin pairs, 60
  same-order pairs, 24 cross-order non-twin pairs, 24 test sentences) and the
  AUCs stay directly comparable; the 2 removed couples come out of train.
  Classical controls (C1 bag-of-vectors, C2 144-parameter linear metric on
  order-sensitive concatenation) re-run on the same de-duplicated splits.
  Code is reused from `exp21b_robust.py` by exec-slice; that file is unmodified.

### The corrected permutation test (D2 fix)

The null is now imposed on the **twin assignment**, not on the fidelity values.
A test split is 12 SVO states `s_1..s_12` and 12 VSO states `v_1..v_12` with
the true twin map being the identity. Draw a bijection `π` uniformly from
`S_12` and recompute `AUC({fid(s_m, v_π(m))}_m vs the same negatives)`, 10,000
times; `p = (#{AUC_π ≥ AUC_obs} + 1)/10001`.

Exchangeability holds because the randomization acts on the assignment while
the 24 sentence states and the negative set are held fixed: every dependence
created by a sentence appearing in ~5 within-order pairs is carried
identically into every replicate, and for the primary endpoint the negatives
(within-order pairs) are untouched by `π`, i.e. strictly ancillary.

A **derangement-restricted** variant (`π` with no fixed points) is reported
alongside as a **power sensitivity check, not as a better test**. Excluding
fixed points removes exactly those null draws that retain a true twin, which
shifts the reference distribution *down* and so makes the variant mildly
**anti-conservative by construction**. That is precisely why it is useful
here: it is biased in the direction that would help us reject, and it still
does not reject. It should not be quoted as the primary p-value.

## Results

### 21a (capacity control) — unchanged

Train AUC 0.9986 (p = 1e-4) — the composition CAN hold near-perfect twin
geometry. Held-out AUC 0.496 — unconstrained training memorizes. Noun-semantics
correlation destroyed (rho 0.083 → −0.005).

### 21c de-duplicated held-out test AUC (twin vs same-order), five splits

| split | PRE | POST | p (sentence-level) | p (derangement) | p (v1 pooled-label scheme, on dedup data) |
|-------|------|------|------|------|------|
| 42 | 0.486 | 0.515 | 0.528 | 0.530 | 0.435 |
| 43 | 0.601 | 0.632 | 0.211 | 0.175 | 0.078 |
| 44 | 0.556 | 0.601 | 0.571 | 0.576 | 0.135 |
| 45 | 0.617 | 0.551 | 0.190 | 0.158 | 0.291 |
| 46 | 0.574 | 0.513 | 0.685 | 0.696 | 0.455 |
| **mean ± sd** | **0.567 ± 0.051** | **0.563 ± 0.053** | **0/5 significant** | **0/5** | **0/5** |

**Mean change from training: −0.004 AUC.** v1 reported +0.046. Training no
longer improves held-out AUC at all; the point estimate is slightly negative.

Learning still happens: POST **train** AUC is 0.847–0.989 with sentence-level
p = 1e-4 on every split (the floor of a 10,000-draw test). The encoder fits
the 34 training couples nearly perfectly and transports none of it that is
detectable at this sample size.

### The A-vs-C endpoint (twin vs cross-order non-twin) agrees

A-vs-B compares twins against *same-order* pairs, so it can be moved by a
main effect of pair type. **A-vs-C** compares twins against *cross-order
non-twin* pairs, holding the order contrast fixed, and is therefore the
stricter meaning test:

| split | 42 | 43 | 44 | 45 | 46 | mean ± sd |
|---|---|---|---|---|---|---|
| PRE | 0.559 | 0.705 | 0.427 | 0.566 | 0.493 | 0.550 ± 0.103 |
| POST | 0.410 | 0.622 | 0.524 | 0.635 | 0.424 | 0.523 ± 0.106 |
| p (sentence-level) | 0.589 | 0.249 | 0.567 | 0.189 | 0.713 | **0/5 significant** |

Training moves this endpoint **down by 0.027**, and 0 of 5 splits are
significant (smallest p = 0.189; derangement variant also 0/5). The stricter
endpoint therefore points the same way as the primary one, slightly harder.
(For the A-vs-C null a permuted twin pair can coincide with a C-pair, which
makes that test mildly conservative; it is reported as supporting evidence,
not as the primary endpoint.)

### Corrected inference (D2, D3)

- **Per-seed, sentence-level: 0 of 5 splits significant at α = 0.05.** The
  smallest p is 0.190. At the Bonferroni level (α = 0.05/5 = 0.01), also 0 of 5.
  The derangement sensitivity check agrees (0/5, smallest 0.158).
- **Pooled across splits:** one-sample t of the 5 POST test AUCs vs 0.5,
  t(4) = 2.642, **one-sided p = 0.0287** (two-sided 0.0574).
- **Pooled, re-centred on the permutation null:** t(4) = 0.810,
  **one-sided p = 0.232**.
- Fisher combination of the 5 sentence-level p-values: chi2 = 9.580, df = 10,
  **p = 0.478**.

**The pooled p = 0.0287 does not resurrect the positive, and must not be read
as if it did.** Four reasons, in order of force:

1. **It tests a different null than the permutation test does.** The
   sentence-level permutation null is **not centred at 0.5**: its mean is
   0.484–0.615 across the five splits (mean **0.546**). Twin pairs are
   cross-order and the B negatives are same-order, so the two groups differ in
   mean fidelity for reasons that carry **no twin information** — a main effect
   of pair *type*, not of twin identity. A t-test against 0.5 charges that main
   effect to "transfer". The like-for-like pooled analogue of the permutation
   test is a one-sample t on each seed's **shift from its own null mean**, and
   that test is null: **t(4) = 0.810, one-sided p = 0.232**. This is the
   decisive point. (The same re-centring on PRE gives t(4) = 1.977, p = 0.060;
   re-centring on the derangement null gives p = 0.223.)
2. **Training has no detectable effect.** The paired within-split comparison —
   the right way to ask whether training did anything — is POST − PRE,
   t(4) = **−0.172, p = 0.872**, mean change −0.004. Whatever small
   above-chance structure exists was already present at the
   least-squares-to-exact-AE initialisation and 6000 SPSA steps do not move it.
3. **The 5 splits are not independent replicates.** They re-sample the *same*
   58 couples and 116 sentences, so the t-test's df = 4 is optimistic and its p
   is an *upper bound on the evidence for transfer* — the conservative
   direction for a negative claim, but it means 0.0287 overstates the case.
4. **Every properly-conditioned test disagrees.** 0/5 per-seed sentence-level,
   0/5 on the stricter A-vs-C endpoint, and Fisher p = 0.478.

*Colour, not evidence:* the untrained encoder also passes the uncentred pooled
test, and slightly more strongly (PRE p = 0.0214 vs POST p = 0.0287). Comparing
two non-independent p-values is not itself a test — the paired comparison in
reason 2 is the real version of that argument — but it is a useful sanity
check on the direction.

### Note on the size and direction of the D2 bias

The old pooled-label scheme was **not** uniformly anti-conservative. Recomputed
on the de-duplicated data, it gives p = 0.435 / 0.078 / 0.135 / 0.291 / 0.455
against the sentence-level 0.528 / 0.211 / 0.571 / 0.190 / 0.685 — larger on 4
splits, smaller on 1 (seed 45). It is the *wrong* null and its p-values should
not be used, but it should not be claimed that it inflated significance
everywhere.

**Consequently, D1 and D2 must be credited separately.** On the de-duplicated
data even the old, wrong permutation scheme yields **0/5 significant**. v1's
"2/5 significant" was therefore produced by the **leakage (D1)**, not by the
permutation scheme (D2). D2 is a correctness fix that changes the reported
p-values; it is not what created the v1 positive.

### Classical controls (same training couples, same loss, same optimizer, 144 params)

On the de-duplicated splits, C2 held-out meaning AUC:

| split | 42 | 43 | 44 | 45 | 46 | mean |
|---|---|---|---|---|---|---|
| C2 | 0.981 | 0.918 | 0.999 | 0.988 | 0.992 | **0.975** |
| C1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

C1 bag-of-vectors is 1.0 by construction and order-blind (twins are identical
vectors). **C2 mean 0.975 vs quantum POST mean 0.563** — a gap of 0.413 AUC at
matched data, matched loss, matched optimizer and 144 vs 91 parameters. C2 wins
on every individual split.

v1 reported C2 for seeds 43–46 only: **the seed-42 C2 run had crashed** with a
matmul shape error (size 24 vs 18 in `feats_concat`, caused by sentences with a
4th adjunct token), and v1 silently omitted it. The crash is fixed (truncation
to the first 3 tokens) and seed 42 has been replayed on the **original**
60-twin split for completeness: **C2 = 0.996, C1 = 1.000**
(`results_exp21b_seed42_c2.json`). At 0.996 it is the highest of the five v1
original-split C2 runs (0.996 / 0.960 / 0.994 / 0.989 / 0.982, from
`results_exp21b_seed42_c2.json` and `results_exp21b_seed4{3,4,5,6}.json`), so
its omission had made the classical control look marginally *weaker* than it
was. Note this 0.996 is an original-split number and is not comparable
cell-for-cell with the de-duplicated table above.

## Verdict (honest)

1. **The trained quantum encoder shows no detectable transfer.** After removing
   the duplicated twins, mean held-out AUC is 0.563 and training moves it by
   −0.004; **0 of 5 splits are significant** under a correctly-exchangeable
   sentence-level permutation test, the stricter A-vs-C endpoint is also 0/5
   and moves down, and the pooled signal vanishes once the test is asked
   against the permutation null rather than against 0.5 (p = 0.232) — with the
   paired POST−PRE test at p = 0.872 confirming training did nothing. v1's
   "small, fragile transfer effect (2/5 significant)" was an artefact of two
   verbatim-duplicated twin couples leaking train into test on exactly those
   two splits.
2. **The matched classical control decisively wins the meaning task**
   (0.975 vs 0.563, on every split). At 3–4 qubits with post-selected sentence
   states, the quantum pipeline offers no advantage on lexical-content
   transport. This is measured, not suspected, and now on leak-free splits.
3. Learning is real but local: train AUC 0.85–0.99 with p = 1e-4 on every
   split. The encoder has ample capacity to fit; none of that fit shows up in
   held-out geometry at a size this design could detect.
4. Combined with exp15c/17/18/19/20, the semantic-geometry program closes with
   a complete characterization — untrained: nothing (all ansatze, all encoders
   incl. provably exact AE); unconstrained trained: memorization; constrained
   trained: **no detectable transfer**, classically dominated. v1 called the
   last case "marginal fragile transfer"; that was too generous and is
   retracted.

Pre-registered verdict rule, evaluated in
`results_exp21c.json :: verdicts` (computed at runtime, not hardcoded):
`robust_transfer` = false, `c2_dominates` = true,
**`conclusion_unchanged` = true** — the direction of the v1 conclusion stands;
its strength was understated.

### Residual caveats

- **Transductive vocabulary** (see Question above): held out at the sentence
  level, not the word level. No claim about novel words.
- `order_eval` runs 5-fold CV over all 120 sentences regardless of split, so
  the 4 duplicate sentence texts still straddle its folds. This is v1's
  protocol, kept identical so the order-accuracy numbers remain comparable.
  Order accuracy is not the contested claim, but the 0.592–0.711 figures carry
  this caveat.
- The pooled t-test's non-independence is not repairable with 5 splits over 58
  couples; it is reported, not solved. This applies to the re-centred test too.
- Negative result at n = 58 couples: this rules out an effect of the size v1
  claimed, not an arbitrarily small one.
- Order-discrimination (5-fold CV, split-independent): PRE 0.642 on every
  split, POST 0.592–0.711 — it does not collapse, and on most splits it drifts
  slightly up rather than down (v1 read this as mild degradation on a smaller
  sample). Noun-level cosine alignment is not preserved: Spearman rho 0.050 →
  0.004–0.066.

## What survives untouched

The structural/topological program (L0 theorem, 64.9% entanglement result,
depth-stabilization effect with shuffle control), the analog resonance
window (V/Ω≈1, AUC 0.815 in device-model simulation), hardware access
(IBM Herons; Pasqal FRESNEL_SA1 via region="sa"), and the paper. None of
these depended on semantic geometry.

## Where this points

- Journal version gains a rigorous "limits of semantic geometry in
  near-term compositional QNLP" section — anti-hype, fully controlled,
  first of its kind for any language. The negative is now clean enough to
  publish as a negative: leak-free splits, an exchangeable null, and a pooled
  test that is reported *against* our own prior interest.
- Hardware stories proceed on the ROBUST results: L0/L1 architectural
  theorem on IBM; analog geometry experiment toward FRESNEL_SA1.
- The open theoretical door: derive which compositional structures provably
  transport word similarity (transport-theory track) — a math question,
  not a data question. exp21c sharpens it: the composition demonstrably *can*
  hold the geometry (train AUC 0.99) while showing **no transport at any
  effect size v1 claimed**, so the obstruction looks like the composition
  rather than encoder capacity.

## Provenance

Every number above traces to a committed JSON:

| source | contents |
|---|---|
| `results_exp21c.json` | dedup record, leakage diagnostic, per-seed PRE/POST train+test AUCs for both endpoints, all three permutation p-values, permutation-null means, order accuracy, noun rho, classical controls, pooled t, Fisher, `verdicts` |
| `results_exp21c_pooled_addendum.json` | re-centred pooled t, PRE pooled t, paired POST−PRE t, A-vs-C pooled summary (all derived from arrays stored in `results_exp21c.json`; separate file so that file is never overwritten) |
| `results_exp21b_seed42_c2.json` | seed-42 C2/C1 on the original 60-twin split |
| `results_exp21b_seed4{3,4,5,6}.json` | v1 original-split C2 values for seeds 43–46 |
| `results_exp21a.json` | 21a capacity control |

Reproduce with `qiskit_lambeq_env/bin/python3 exp21c_dedup_rerun.py`
(5 seeds × 6000 SPSA iters, 2050 s), then `exp21c_pooled_addendum.py`.
`verify_v2.py` re-checks every numeric token in this document against those
files. `exp21b_robust.py` and all `results_exp21b*.json` are unmodified.
