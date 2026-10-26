# -*- coding: utf-8 -*-
"""build_demo.py -- bundle demo/ into ONE self-contained index.html.

demo_engine_spec.md SS11 Gate Z: zero external requests, works offline. Every
asset (engine, matcher, sha256, app, and the three JSON payloads) is inlined, so
the page also survives a strict CSP that blocks fetch/XHR entirely.
"""
import json, os, sys

D = "demo"
PARITY = json.load(open(os.path.join(D, "parity_report.json"), encoding="utf-8"))
MODEL = json.load(open(os.path.join(D, "model.json"), encoding="utf-8"))
g = PARITY["gates"]

HEAD = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arabic QNLP — the verb-twin rung, running in your browser</title>
<style>
:root{--bg:#fbfaf8;--fg:#1a1a1a;--mut:#5c5c5c;--line:#e2ded8;--card:#fff;
      --acc:#8a5a2b;--warn:#a33;--t1:#2f6f4f;--t3:#a33;--hero:#7a4b9c}
@media (prefers-color-scheme:dark){:root{--bg:#151517;--fg:#ececec;--mut:#a0a0a0;
      --line:#2e2e33;--card:#1d1d21;--acc:#d09a5e;--t1:#63b98d;--t3:#e08585;--hero:#b98ede}}
:root[data-theme=dark]{--bg:#151517;--fg:#ececec;--mut:#a0a0a0;--line:#2e2e33;
      --card:#1d1d21;--acc:#d09a5e;--t1:#63b98d;--t3:#e08585;--hero:#b98ede}
:root[data-theme=light]{--bg:#fbfaf8;--fg:#1a1a1a;--mut:#5c5c5c;--line:#e2ded8;
      --card:#fff;--acc:#8a5a2b;--t1:#2f6f4f;--t3:#a33;--hero:#7a4b9c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:1.6rem;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:1.05rem;margin:34px 0 10px;letter-spacing:-.005em}
.sub{color:var(--mut);margin:0 0 20px}
.banner{border:1px solid var(--line);border-left:3px solid var(--acc);
        background:var(--card);padding:12px 14px;border-radius:6px;font-size:.87rem;color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}
textarea,input{width:100%;font:15px/1.7 ui-monospace,Menlo,monospace;background:transparent;
     color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:10px;direction:rtl;text-align:right}
textarea{min-height:84px;resize:vertical}
button{font:inherit;font-size:.87rem;background:transparent;color:var(--fg);
       border:1px solid var(--line);border-radius:5px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--acc)}
.row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.col{flex:1 1 280px;min-width:260px}
.big{font-size:1.9rem;font-weight:600;letter-spacing:-.02em}
.mono{font-family:ui-monospace,Menlo,monospace}
.note{font-size:.85rem;color:var(--mut);margin-top:8px}
.note.hero{color:var(--hero)}
.err{display:none;color:var(--warn);font-size:.87rem;margin-top:8px}
.bar{height:8px;background:var(--line);border-radius:99px;overflow:hidden;margin:8px 0}
.bar>div{height:100%;background:var(--acc)}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:500}
.scroll{overflow-x:auto}
.bloch{fill:none;stroke:var(--line);stroke-width:1.5}
.axis{stroke:var(--line);stroke-width:1;stroke-dasharray:3 3}
.spoke{stroke:var(--line);stroke-width:1}
.pt.t1{fill:var(--t1)} .pt.t3{fill:var(--t3)}
.lbl{font-size:9px;text-anchor:middle;fill:var(--mut)}
.lbl.t3{fill:var(--t3)}
.pole{font-size:10px;text-anchor:middle;fill:var(--mut)}
.grid{stroke:var(--line);stroke-width:1}
.ax{font-size:10px;fill:var(--mut)}
.ax.hero{fill:var(--hero)}
.curveline{fill:none;stroke:var(--acc);stroke-width:2}
.cp{fill:var(--acc)}
.cp.scale{fill:var(--hero);stroke:var(--bg);stroke-width:1.5}
.cp.sel{stroke:var(--fg);stroke-width:2}
.arrow{stroke:var(--hero);stroke-width:1.5;stroke-dasharray:4 3}
.arrowhead{fill:var(--hero)}
.ghost{fill:none;stroke:var(--hero);stroke-width:1.5;opacity:.6}
label.sw{display:flex;gap:10px;align-items:flex-start;font-size:.9rem;cursor:pointer}
label.sw>input{flex:0 0 auto;width:16px;height:16px;margin:3px 0 0}
label.sw>span{flex:1 1 auto}
code{font-family:ui-monospace,Menlo,monospace;font-size:.85em}
</style>"""

BODY = """<div class="wrap">
<h1>The verb-twin rung, running in your browser</h1>
<p class="sub">A trained quantum-circuit model for Arabic story questions — 120 parameters,
a 16-amplitude statevector, no server. Type a sentence and watch it run.</p>

<div class="banner"><b>Scope, permanently.</b> This is a <b>4-wire, 6-verb, one-rung toy</b>
trained on a synthetic generated corpus in simulation. It is not an Arabic NLP system, it is not
a parser, and no quantum advantage is claimed or measured anywhere. Its value is that it makes a
<b>negative</b> result visible: a benchmark that looks compositional is solved by machinery that
is meaning-blind.</div>

<h2>1 · Type a story</h2>
<div class="card">
  <div style="margin-bottom:8px"><span class="note">Verbs available (the only trained blocks):
    <span class="mono" id="verblist"></span></span></div>
  <textarea id="story" spellcheck="false"></textarea>
  <div style="height:8px"></div>
  <input id="question" spellcheck="false">
  <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
    <button id="run">Run</button><span id="presets"></span>
  </div>
  <div class="err" id="err"></div>
</div>

<div class="row">
  <div class="col card">
    <div class="note">Referent states — <b>exact</b>, not a projection</div>
    <div id="plot"></div>
    <div class="note">angular half-width <span class="mono" id="hw"></span> rad —
      the same statistic the research gates on.</div>
    <div class="note" id="tiernote" style="display:none;color:var(--t3)">
      <b>Red = not an embedding.</b> That word is outside the frozen vocabulary, so its angle is a
      deterministic <code>sha256</code> hash. It still runs — nouns are not trained parameters, they
      only set a rotation. Watching a hashed word work as well as a real one <i>is</i> the finding.</div>
  </div>
  <div class="col card">
    <div class="note">Model's answer</div>
    <div class="big" id="answer"></div>
    <div class="bar"><div id="pbar"></div></div>
    <div class="note">P(yes) <span class="mono" id="pyes"></span> ·
      P(no) <span class="mono" id="pno"></span></div>
    <div class="note"><b>Read this carefully.</b> The model is <b>geometry-sensitive and
      meaning-blind</b>. Changing a referent's angle can flip a single answer — that is geometry
      sensitivity, and it is exactly what the research measured. The claim that semantic
      <i>content</i> is inert is an <b>aggregate</b> one, established by the permutation controls
      below, never by any one sentence.</div>
  </div>
</div>

<h2>2 · The two diagnostics</h2>
<div class="card">
  <label class="sw"><input type="checkbox" id="tie">
    <span><b>Tie every verb to one block.</b> The model can no longer tell the six verbs apart.</span></label>
  <div style="height:10px"></div>
  <label class="sw"><input type="checkbox" id="swap">
    <span><b>Permute verb → block.</b> Each verb runs on a different verb's trained block — the
      pre-registered C1 dial-swap control.</span></label>
  <div style="height:14px"></div>
  <div class="note">Accuracy on the committed 300-item test set</div>
  <div class="big mono" id="agg"></div>
  <div class="note" id="aggn"></div>
  <div class="note" id="aggdelta"></div>
</div>

<h2>3 · What actually drives this model</h2>
<div class="card">
<p class="note" style="margin-top:0">Not what the words <i>mean</i> — how far apart their encoding angles sit.
Each dot is a <b>separately trained model</b>, 5 seeds, trained <i>at</i> that spread. Click any dot to load
those real weights and run them here. <b>There is no slider</b>: re-angling a model trained at one spread
would measure distribution shift, not this curve.</p>
<div id="curve"></div>
<div class="row" style="margin-top:6px">
  <div style="flex:0 0 auto" id="spreadplot"></div>
  <div class="col" id="spreadinfo"></div>
</div>
<p class="note"><b>The purple point.</b> That model started at half-width 2.40 — the worst encoding here,
barely above chance — with <b>one</b> extra learnable number scaling its input angles. It drove that number
to near zero, landing at half-width 0.12, at the top of a curve it never saw. Two of its five seeds chose a
<i>negative</i> scale, flipping the semantic axis end-for-end, and did just as well: only spread matters.</p>
<p class="note"><b>This is not our discovery, and saying so matters.</b> The effect is <b>quantum kernel
bandwidth</b> — Canatar, Peters, Pehlevan, Wild &amp; Shaydulin, <i>Bandwidth Enables Generalization in
Quantum Kernel Models</i> (arXiv:2206.06686, TMLR 2023): smaller bandwidth brings encoded states closer
together, reduces expressivity, and thereby enables generalisation. Its classical twin is the RBF kernel
width γ that every SVM user tunes. The learnable scalar is <b>trainable input scaling</b>, also standard.
We re-derived both from scratch before checking the literature. The curve above is a faithful reproduction
of a known phenomenon in a compositional Arabic setting — not a new one.</p>
<p class="note">Why meaning can&rsquo;t matter <i>here</i>: both twins of every pair share an identical cast and an
identical question, so any function of the nouns alone is constant within a pair and scores exactly 0.500.
On this rung the referent encoding cannot carry the answer. It can only condition the geometry.</p>
</div>

<h2>4 · What the encoding can actually carry</h2>
<div class="card">
<p class="note" style="margin-top:0">If content can&rsquo;t matter on the rung above, the obvious question is what
happens on one where it can. Each referent enters the circuit through <b>exactly one rotation</b>, so the
model&rsquo;s output in that referent&rsquo;s angle is provably <span class="mono">a + b·cos&nbsp;θ + c·sin&nbsp;θ</span>
— three numbers, measured to a residual of <span class="mono">1.4e-07</span>. That bounds how finely any
category can be read, and the bound is computable in closed form <i>before training</i>.</p>
<div class="scroll"><table>
<tr><th>category demand</th><th>derived ceiling</th><th>measured (held-out)</th></tr>
<tr><td>1 bit &mdash; 2 arcs</td><td class="mono">1.000</td><td class="mono">1.0000</td></tr>
<tr><td>2 bands/class &mdash; 4 arcs</td><td class="mono">0.875</td><td class="mono">0.6827</td></tr>
<tr><td>3 bands/class &mdash; 6 arcs</td><td class="mono">0.778</td><td class="mono">0.4807</td></tr>
</table></div>
<p class="note">The model binds a 1-bit category to a grammatical role <b>and transfers it to referents it never
saw in training</b> — the strongest standard test of role-filler binding. The ceiling was never violated across
60 runs. But as the category gets finer the measured value falls further below what is representable: on seen
nouns the model reaches <b>97%</b> of the ceiling, on novel fillers only <b>78%</b>. <b>The wall is transfer to
unseen fillers</b>, which is exactly where classical models fail too.</p>
<p class="note">Scope: the category here is <b>constructed</b>, not distributional — this is binding, not meaning.
No classical comparator is registered, and the ceiling is a property of the <b>encoding&rsquo;s function class</b>,
not of quantumness: a classical linear model on the same features hits the same bound by construction.</p>
</div>

<h2>5 · Provenance</h2>
<div class="card"><div class="scroll"><table>
<tr><th>shown</th><th>traces to</th></tr>
<tr><td>weights, 0.7267</td><td class="mono">results_exp51.json + exp51_ckpt/VT_A3_lr0.02_b8_s0.pt</td></tr>
<tr><td>tie-the-verbs 0.5000</td><td class="mono">results_exp51.json → tied_sentinel</td></tr>
<tr><td>encoding curve (§3)</td><td class="mono">results_exp55b.json, results_exp59.json</td></tr>\n<tr><td>capacity table (§4)</td><td class="mono">results_exp60.json, results_exp61.json</td></tr>
</table></div>
<p class="note">That last filename is not a typo. A scaffold rename missed one call, so exp57 wrote
its results to <code>results_exp55b.json</code>. Results files are never rewritten, so it keeps the
name it was born with and is cited by it. <b>Every number on this page was computed before it was
displayed, and none of it was chosen after seeing which way it came out.</b></p>
<p class="note">Browser engine verified against the PyTorch original:
<b>__PARITY__/300 identical predictions</b>, max value difference
<span class="mono">__MAXDIFF__</span> (the residual is float32-vs-float64 — the reference model runs
in complex64, this page in doubles). Template matcher round-trips the corpus
<b>9000/9000</b> sentences and <b>1800/1800</b> questions.</p>
</div>
</div>"""


def main():
    parts = []
    for f in ("engine.js", "matcher.js", "sha256.js", "compile.js"):
        parts.append(open(os.path.join(D, f), encoding="utf-8").read())
    payload = ""
    for var, f in (("__MODEL__", "model.json"), ("__VOCAB__", "vocab_core.json"),
                   ("__TESTSET__", "testset.json"),
                   ("__SPREAD__", "spread_models.json")):
        payload += "window.%s=%s;\n" % (
            var, open(os.path.join(D, f), encoding="utf-8").read())
    app = open(os.path.join(D, "app.js"), encoding="utf-8").read()

    verbs = MODEL["verbs"]
    presets = ""
    body = (BODY.replace("__PARITY__", g["P_parity_argmax"].split("/")[0])
                .replace("__MAXDIFF__", "%.1e" % g["P_max_value_diff"]))
    html = HEAD + "\n" + body + "\n<script>\n" + payload + "\n".join(parts) \
        + "\nwindow.VTModel=VTModel;window.Matcher=Matcher;" \
          "window.hashAngle=hashAngle;window.halfWidth=halfWidth;\n" \
        + app.replace("module.exports", "void 0 && module.exports") \
        + "\ndocument.getElementById('verblist').textContent=" \
          + json.dumps(" · ".join(verbs), ensure_ascii=False) + ";\n</script>"
    out = os.path.join(D, "index.html")
    open(out, "w", encoding="utf-8").write(html)
    print("[build] wrote %s (%.0f KB, %d verbs, self-contained)"
          % (out, len(html.encode()) / 1024, len(verbs)))
    for bad in ("fetch(", "XMLHttpRequest", "https://", "http://"):
        if bad in html:
            print("  [WARN] external-reference token found: %r" % bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
