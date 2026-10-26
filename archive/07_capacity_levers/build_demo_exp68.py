# -*- coding: utf-8 -*-
"""build_demo_exp68.py -- bundle the exp68 "width lever" assets into ONE
self-contained demo/index_exp68.html.

Same contract as build_demo.py (demo_engine_spec.md SS11 Gate Z: zero external
requests, works offline from file://), but a SEPARATE page and a SEPARATE
builder: build_demo.py, demo/index.html and demo/engine.js are committed
artefacts of the exp51 page and are neither modified nor rebuilt here.

Difference from build_demo.py, deliberately: this page hand-types NO numbers.
Every figure is written into the DOM by app_exp68.js from the two bundled
JSONs (model_exp68.json, parity_report_exp68.json), which in turn trace to
results_exp68.json and the exp68 checkpoints. The markup below carries prose
and structure only.

Refuses to build if the parity gate is not green.
"""
import json
import os
import sys

D = "demo"
OUT = os.path.join(D, "index_exp68.html")

HEAD = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The width lever — three encodings, one Arabic binding task</title>
<style>
:root{--bg:#fbfaf8;--fg:#1a1a1a;--mut:#5c5c5c;--line:#e2ded8;--card:#fff;
      --acc:#8a5a2b;--warn:#a33;--ok:#2f6f4f;--bad:#a33;--hero:#7a4b9c;--dim:#8b8b8b}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151517;--fg:#ececec;
      --mut:#a0a0a0;--line:#2e2e33;--card:#1d1d21;--acc:#d09a5e;--ok:#63b98d;--bad:#e08585;
      --hero:#b98ede;--dim:#77777d}}
:root[data-theme=dark]{--bg:#151517;--fg:#ececec;--mut:#a0a0a0;--line:#2e2e33;--card:#1d1d21;
      --acc:#d09a5e;--ok:#63b98d;--bad:#e08585;--hero:#b98ede;--dim:#77777d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:940px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:1.6rem;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:1.05rem;margin:34px 0 10px;letter-spacing:-.005em}
.sub{color:var(--mut);margin:0 0 20px}
.banner{border:1px solid var(--line);border-left:3px solid var(--acc);background:var(--card);
        padding:12px 14px;border-radius:6px;font-size:.87rem;color:var(--mut)}
.banner.gate{border-left-color:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}
.note{font-size:.85rem;color:var(--mut);margin-top:8px}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:.85em}
.strong{font-weight:600;color:var(--fg)}
select,button{font:inherit;font-size:.87rem;background:transparent;color:var(--fg);
       border:1px solid var(--line);border-radius:5px;padding:6px 10px;cursor:pointer}
button:hover,select:hover{border-color:var(--acc)}
select{max-width:100%}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:500}
table.num td{padding:4px 6px}
table.num td:first-child{width:58%}
table.num tr.live td{border-top:1px solid var(--line);color:var(--fg)}
.ar{font-size:1.05rem;font-family:"Geeza Pro","Noto Naskh Arabic","Segoe UI",serif;
    unicode-bidi:isolate}
.ar.big{font-size:1.35rem;line-height:2}
.ar.nb{color:var(--hero);font-weight:600}
.gl{font-size:.75rem;color:var(--mut);margin-inline-start:6px}
.sep{color:var(--dim);margin:0 10px}
.twins{display:flex;gap:14px;flex-wrap:wrap;margin-top:12px}
.twin{flex:1 1 260px;min-width:240px;border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.tlabel{font:.72rem/1.4 ui-monospace,Menlo,monospace;color:var(--dim)}
b.yes{color:var(--ok)} b.no{color:var(--bad)}
.arms{display:flex;gap:14px;flex-wrap:wrap;align-items:stretch}
.armcard{flex:1 1 280px;min-width:270px;background:var(--card);border:1px solid var(--line);
         border-radius:8px;padding:14px}
.ahead{display:flex;gap:8px;align-items:baseline}
.aname{font:600 .8rem/1 ui-monospace,Menlo,monospace;color:var(--acc);
       border:1px solid var(--line);border-radius:4px;padding:3px 6px}
.atitle{font-weight:600}
.cells{margin:10px 0}
.ans{border-inline-start:3px solid var(--line);padding:6px 0 6px 10px;margin:8px 0}
.ans.ok{border-inline-start-color:var(--ok)}
.ans.bad{border-inline-start-color:var(--bad)}
.alab{font-size:.72rem;color:var(--dim)}
.averd{font-size:1.15rem;font-weight:600}
.agold{font-size:.78rem;font-weight:400;color:var(--mut)}
.bar{height:6px;background:var(--line);border-radius:99px;overflow:hidden;margin:5px 0}
.bar>div{height:100%;background:var(--acc)}
#lever,#leverlab{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
#lever{font-size:1.5rem;font-weight:600;letter-spacing:-.02em}
#leverlab{font-size:.78rem;color:var(--mut);margin-top:2px}
#lever .step,#leverlab .step{flex:1 1 0;min-width:90px}
#lever .arrow,#leverlab .arrow{flex:0 0 auto;color:var(--dim);font-size:1rem}
.hero{color:var(--hero)}
.v{font-weight:600}
.v.true{color:var(--ok)} .v.false{color:var(--bad)}
.row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.col{flex:1 1 300px;min-width:280px}
code{font-family:ui-monospace,Menlo,monospace;font-size:.85em}

/* ---- data-viz palette: TWO categorical slots (the story's two referents).
   Validated all-pairs in both modes with the dataviz validator. Light-mode
   aqua is under 3:1 on white, so every mark in these hues carries a visible
   Arabic direct label (the relief rule). Category membership is the page's
   existing violet PLUS a star glyph PLUS a label -- never colour alone. ---- */
:root{--r0:#2a78d6;--r1:#1baf7a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--r0:#3987e5;--r1:#199e70}}
:root[data-theme=dark]{--r0:#3987e5;--r1:#199e70}

.axis{stroke:var(--line);stroke-width:1;stroke-dasharray:3 3}
.axis.solid{stroke-dasharray:none}
.tick{stroke:var(--line);stroke-width:1}
.ax{font-size:10px;fill:var(--mut)}
.note-svg{font-size:10px;fill:var(--dim)}
.pole{font-size:11px;fill:var(--mut)}
.dialring{fill:none;stroke:var(--line);stroke-width:1.5}
/* marks: identity by slot, membership by violet + star + label */
.pt{fill:var(--dim);transition:r .1s}
.pt.bear{fill:var(--hero)}
.pt.r0,.pt.r0.bear{fill:var(--r0)}
.pt.r1,.pt.r1.bear{fill:var(--r1)}
.pt.r0,.pt.r1{stroke:var(--card);stroke-width:2}
.stem{stroke:var(--hero);stroke-width:1.5}
.stem.r0{stroke:var(--r0)} .stem.r1{stroke:var(--r1)}
.spoke{stroke:var(--hero);stroke-width:1;opacity:.55}
.spoke.r0{stroke:var(--r0);opacity:1} .spoke.r1{stroke:var(--r1);opacity:1}
.arlbl{font-size:12px;fill:var(--hero);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.arlbl.r0{fill:var(--r0)} .arlbl.r1{fill:var(--r1)}
.arlbl.un.nonbear{fill:var(--mut)}
/* the thread: one noun, lit everywhere at once */
.thr{outline:none}
circle.thr{r:9;stroke:var(--fg);stroke-width:2}
text.thr{font-weight:700;text-decoration:underline}
line.thr,path.thr{stroke-width:3}
tr.thr>td{background:color-mix(in srgb,var(--fg) 8%,transparent)}
span.ar.thr,span.gl.thr{background:color-mix(in srgb,var(--fg) 12%,transparent);border-radius:3px}
[data-n]{cursor:pointer}
/* legend */
#legend{display:flex;align-items:center;flex-wrap:wrap;gap:4px;font-size:.85rem;margin-top:10px}
.key{width:11px;height:11px;border-radius:99px;display:inline-block;margin-inline-end:5px}
.key.r0{background:var(--r0)} .key.r1{background:var(--r1)}
.key.bearkey{background:var(--hero)} .key.unkey{background:var(--dim)}
/* encoding sub-panels */
.enc{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px}
.encpanel{flex:1 1 320px;min-width:300px}
.encpanel h3{font-size:.9rem;margin:0 0 2px;font-weight:600}
/* circuit diagram */
.armbtn{margin-inline-end:6px;margin-bottom:6px}
.armbtn.on{border-color:var(--acc);color:var(--acc);font-weight:600}
.armcard.picked{border-color:var(--acc)}
#circuits{display:flex;gap:16px;flex-wrap:wrap}
.twincol{flex:1 1 420px;min-width:340px;background:var(--card);border:1px solid var(--line);
         border-radius:8px;padding:14px}
.wire{stroke:var(--line);stroke-width:1.5}
.arwire{font-size:13px;fill:var(--hero);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.arwire.r0{fill:var(--r0)} .arwire.r1{fill:var(--r1)} .arwire.un{fill:var(--mut)}
.glwire{font-size:9px;fill:var(--dim)}
.gate{fill:none;stroke:var(--r0);stroke-width:1.5}
.gate.r1{stroke:var(--r1)} .gate.un{stroke:var(--line)}
.gtxt{font-size:10px;fill:var(--mut);font-family:ui-monospace,Menlo,monospace}
.lead{fill:none;stroke:var(--r0);stroke-width:2}
.lead.r1{stroke:var(--r1)} .lead.un{stroke:var(--line)}
.lead.cross{stroke-dasharray:none}
.verbblk{fill:color-mix(in srgb,var(--acc) 14%,transparent);stroke:var(--acc);stroke-width:1.5}
.qblk{fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 3}
.arverb{font-size:15px;fill:var(--acc);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.port{font-size:9px;fill:var(--mut)}
table.states{font-size:.78rem;margin-top:8px}
table.states td,table.states th{padding:5px 6px;vertical-align:middle}
td.wcell{white-space:nowrap}
.pbar{display:block;height:5px;background:var(--line);border-radius:99px;overflow:hidden;
      max-width:110px;margin-bottom:3px}
.pbar>i{display:block;height:100%;background:var(--hero)}
.pbar.r0>i{background:var(--r0)} .pbar.r1>i{background:var(--r1)}
.pnum{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;color:var(--mut);
      display:block}
.pur{font-size:.68rem;display:inline-block;margin-top:2px;padding:1px 5px;border-radius:99px;
     border:1px solid var(--line);color:var(--dim)}
.pur.isent{color:var(--acc);border-color:var(--acc)}
.mech{font-size:.8rem;color:var(--mut);margin-top:10px;padding:8px 10px;border-radius:6px;
      border:1px solid var(--line);border-inline-start:3px solid var(--line)}
.mech.undone{border-inline-start-color:var(--ok)}
.mech.residual{border-inline-start-color:var(--acc)}
</style>"""

BODY = """<div class="wrap">
<h1>The width lever</h1>
<p class="sub">Three trained circuits answer the same Arabic role-binding question side by side,
in your browser, with no server. They differ in one thing only: <b>how much of a word&rsquo;s
distributional vector the circuit is allowed to see</b>, and how many qubits it gets to see it on.</p>

<div class="banner"><b>Scope, permanently.</b> One task family, one semantic category, a
<b><span id="ninv"></span>-noun</b> human inventory, <span id="nref"></span> referents per story,
<span id="nverb"></span> verbs, constructed swap-twin stories,
<b>simulation only</b>. The held-out split is over <b>unseen pairings, not unseen nouns</b>, so
nothing here tests generalisation to words the model never met. <b>No quantum advantage is claimed
or measured</b>: the classical comparator on the identical features ties or exceeds every arm, and
it is shown below rather than omitted. The point of this page is a <b>capacity calculus and an
encoding-design result</b> &mdash; what a given encoding can carry, and what the circuit converts
into accuracy.</div>

<h2>1 &middot; The lever, in one line</h2>
<div class="card">
  <div id="lever"></div>
  <div id="leverlab"></div>
  <p class="note">Held-out means over 5 seeds. Two comparisons were registered in advance, and both
  are read against the same bar: the <b>register&rsquo;s lift over PC1-alone</b> clears it, and the
  <b>register&rsquo;s lift over the angle</b> does not. (The middle step, PC1-alone to the angle, was
  never a registered criterion &mdash; it is shown because it is the interesting part of the
  picture, not because it was tested.) <b>This may not be read as &ldquo;width works&rdquo;</b>: the registered
  reading is that width is a <b>reparameterisation win, not an information win</b>. The single-angle
  arm was already extracting nearly everything the two raw coordinates jointly carry about this
  category.</p>
</div>

<h2>2 &middot; Pick a story</h2>
<div class="card">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <select id="pick"></select><button id="rand">Random pair</button>
  </div>
  <p class="note" style="margin-top:12px">Cast &mdash; <span id="cast"></span></p>
  <p class="note">Question (identical for both twins): <span class="ar" dir="rtl" id="qsurface"></span></p>
  <p class="note" id="bearerline"></p>
  <div class="twins" id="twins"></div>
  <p class="note"><b>Why the pair matters.</b> The two twins share an identical cast, an identical
  verb and an identical question surface; only the grammatical roles are exchanged, and the gold
  answers are opposite. Any function of the nouns alone is therefore constant within a pair and
  scores exactly <span id="v1"></span> &mdash; measured, not assumed, on this dataset&rsquo;s
  <span id="v1n"></span> pairs. Whatever accuracy you see below has to come from <b>role
  binding</b>, not from recognising which words are present.</p>
</div>

<h2>3 &middot; From the Arabic word to the wire</h2>
<div class="card">
  <p class="note" style="margin-top:0">A noun enters this circuit by exactly one route:
  <b>word &rarr; its frozen word-vector &rarr; two principal coordinates &rarr; an angle
  &rarr; one <code>RY</code> rotation on a wire</b>. The three arms differ only in <i>which</i>
  of those coordinates they are allowed to use. These panels show the same
  <span id="ninv2"></span> nouns three times, once per arm&rsquo;s view of them. Hover any noun
  &mdash; here, in the story, or on a circuit wire &mdash; and it lights up everywhere.</p>
  <div id="legend"></div>
  <div class="enc">
    <div class="encpanel" style="flex:1 1 100%">
      <h3>F1 sees this &mdash; one number per noun</h3>
      <p class="note" style="margin-top:0">Every noun collapsed onto a single axis.</p>
      <div id="f1line"></div>
    </div>
    <div class="encpanel">
      <h3>F2 sees this &mdash; two numbers, one per qubit</h3>
      <p class="note" style="margin-top:0">The same nouns in the plane. The marked ones stop
      being interleaved: they occupy a region, and a region is something two angles can
      address.</p>
      <div id="plane"></div>
    </div>
    <div class="encpanel">
      <h3>FA sees this &mdash; the angle, on the actual Bloch circle</h3>
      <p class="note" style="margin-top:0"><code>RY(φ)|0&rang;</code> has Bloch vector
      (sin&nbsp;φ, 0, cos&nbsp;φ), so every prepared referent lies <b>exactly</b> on this
      great circle. The drawing is the state, not a projection of it.</p>
      <div id="dial"></div>
    </div>
  </div>
</div>

<h2>4 &middot; The width toggle &mdash; three arms, same story</h2>
<div class="arms" id="arms"></div>
<p class="note">The live answers come from <b>one val-selected seed per arm</b>, running in this
page. The committed mean above it is the <b>5-seed</b> figure of record &mdash; a different
quantity, labelled separately on purpose. Because the parity gate agrees with PyTorch on every one
of the bundled items, this page&rsquo;s accuracy for each arm is <i>exactly</i> that seed&rsquo;s
committed held-out number.</p>
<p class="note">Seed selection rule, stated rather than implied: <span class="mono" id="selrule"></span></p>

<h2>5 &middot; The sentence, compiled into a circuit</h2>
<div class="card">
  <div id="armpick"></div>
  <p class="note" id="circnote"></p>
</div>
<div id="circuits"></div>
<p class="note">Nothing in these two diagrams differs except the <b>leads into the trained
block</b>. Same nouns, same angles, same trained parameters, same question &mdash; the word
order is the whole input, and the crossing is the whole picture of it. The state readings are
computed by the same parity-gated forward pass that produces the answer, sampled at three
points; the &ldquo;pure&rdquo; badge means that wire is still a state of its own, and
<span class="mono">Tr ρ²</span> below 1 means it no longer is.</p>

<h2>6 &middot; The registered criteria, and what they returned</h2>
<div class="card"><div class="scroll"><table id="verdicts"></table></div>
<p class="note">Bar provenance: <span id="barprov"></span></p>
<p class="note">The margin criteria were fixed before the run, and both directions were written down
in advance. <b>width_registered_lifts</b> came back TRUE &mdash; the first lever in this
programme&rsquo;s three-lever ledger whose extra budget converts into trained accuracy in-circuit.
<b>width_adds_information</b> came back FALSE, and is recorded as descriptive only.</p>
</div>

<h2>7 &middot; The classical comparator &mdash; shown, not omitted</h2>
<div class="card"><div class="scroll"><table id="cmp"></table></div>
<p class="note"><b>Same features, classical solver, ties or exceeds everywhere.</b> This was
<b>pre-registered as the expected outcome</b>, not discovered afterwards: a linear model on the
identical per-referent features reaches the identical representational bound by construction, and
the product-state kernel this encoding induces factorises and is classically computable in linear
time. So no number on this page is a quantum result. What the comparison does buy is an arbiter:
where the circuit falls short of the comparator, the information was there and the <i>circuit</i>
failed to convert it &mdash; a trainability statement, which is a different and more useful claim
than &ldquo;quantum won&rdquo;.</p>
<p class="note">One qualifier, carried forward honestly: &ldquo;identical features&rdquo; means the
comparator is a <b>restriction</b> of the circuit&rsquo;s function class &mdash; the circuit carries
one form per verb and event direction, the comparator carries one overall. A comparator loss on a
cell is therefore not automatically a circuit win.</p>
</div>

<h2>8 &middot; The frozen constants behind every angle above</h2>
<div class="card">
  <div class="scroll"><table id="formulas"></table></div>
  <p class="note">These are <b>frozen constants</b>, baked into the bundle at build time from the
  committed artefacts. The page performs no PCA and fits nothing: every angle plotted in &sect;3
  and every <code>RY</code> in &sect;5 is a lookup in this table.</p>
</div>

<h2>9 &middot; The parity gate</h2>
<div class="banner gate" id="paritybanner"></div>
<div class="card"><div class="scroll"><table id="parity"></table></div>
<p class="note">This is a blocking gate, not a footnote: if any arm disagreed with PyTorch on any
item, this page would not have been built.</p>
</div>

<h2>10 &middot; Provenance</h2>
<div class="card"><div class="scroll"><table id="prov"></table></div>
<p class="note">Split: <span class="mono" id="splitnote"></span></p>
<p class="note"><b>Every number on this page was computed before it was displayed</b>, is written
into the page from a committed JSON rather than typed into the markup, and none of it was chosen
after seeing which way it came out. The criteria, their bars, and the reading of every branch were
registered in advance of the run.</p>
<p class="note"><b>What this is not.</b> Not an Arabic NLP system and not a parser &mdash; the
stories are generated from templates and the circuit consumes gold structure, never surface text.
Not a claim about the underlying word-vector model: the two components used here explain a small
share of a <span id="edim"></span>-dimensional vector&rsquo;s variance, and this page measures what those two carry,
not what the embedding carries. Not evidence about words the model never saw. And not, anywhere,
a quantum-advantage result.</p>
</div>
</div>"""


def main():
    pr_path = os.path.join(D, "parity_report_exp68.json")
    pr = json.load(open(pr_path, encoding="utf-8"))
    if not pr.get("ALL_PASSED"):
        print("[build] REFUSING: parity gate is not green (%s)" % pr_path)
        return 1

    payload = ""
    for var, f in (("__MODEL68__", "model_exp68.json"),
                   ("__TESTSET68__", "testset_exp68.json"),
                   ("__PARITY68__", "parity_report_exp68.json")):
        payload += "window.%s=%s;\n" % (
            var, open(os.path.join(D, f), encoding="utf-8").read())

    engine = open(os.path.join(D, "engine_exp68.js"), encoding="utf-8").read()
    engine = engine.replace("if (typeof module !== 'undefined' && module.exports) "
                            "{ module.exports = Exp68; }", "")
    app = open(os.path.join(D, "app_exp68.js"), encoding="utf-8").read()

    html = (HEAD + "\n" + BODY + "\n<script>\n" + payload + engine
            + "\n" + app + "\n</script>")
    open(OUT, "w", encoding="utf-8").write(html)

    n_arms = len(json.load(open(os.path.join(D, "model_exp68.json"),
                                encoding="utf-8"))["arm_order"])
    print("[build] wrote %s (%.0f KB, %d arms, %d items, self-contained)"
          % (OUT, len(html.encode()) / 1024, n_arms, pr["n_items"]))

    bad = [t for t in ("fetch(", "XMLHttpRequest", "https://", "http://",
                       "<link", "<img", "@import", "//cdn")
           if t in html]
    if bad:
        print("  [FAIL] external-reference token(s) found: %r" % bad)
        return 1
    print("[build] Gate Z: no external-reference token in the bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
