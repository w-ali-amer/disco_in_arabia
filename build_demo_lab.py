# -*- coding: utf-8 -*-
"""build_demo_lab.py -- bundle the exp69 language-lab assets into ONE
self-contained demo/index_lab.html.

Same contract as build_demo_exp68.py (demo_engine_spec.md SS11 Gate Z: zero
external requests, works offline from file://), but a SEPARATE page and a
SEPARATE builder: build_demo.py, build_demo_exp68.py, demo/index.html,
demo/index_exp68.html, demo/engine.js and demo/engine_exp68.js are committed
artefacts of the exp51 and exp68 pages and are neither modified nor rebuilt here.

As in build_demo_exp68.py, and deliberately: this page hand-types NO numbers.
Every figure is written into the DOM by app_lab.js from the five bundled JSONs
(model_exp69.json, testset_lab.json, featurizer_lab.json, parity_report_lab.json
and exp70_lab.json), which in turn trace to results_exp69.json,
exp69_predictions.json, diagnostic_kernel_factorisation.json, the exp69
checkpoints, and -- for the exp70 section -- results_exp70.json,
exp70_predictions.json, diagnostic_encoding_entanglement.json,
diagnostic_entangled_bounds.json and results_exp70_anchors.json. The markup
below carries prose and structure only.

THE exp70 SECTION SHIPS NUMBERS, NOT A MODEL. No exp70 checkpoint is bundled
and no exp70 circuit runs in the browser: the E-in encoding needs a CRY intro
the parity-gated engine does not implement, and shipping a live model the gate
has never seen is not an option. exp70 is therefore presented as committed
numbers of record, each cited to the public file it came from.

Refuses to build if any parity gate is red, if the exp70 bundle is missing or
ungated, or if any private-repository path reaches the page.
"""
import json
import os
import re
import sys

D = "demo"
OUT = os.path.join(D, "index_lab.html")

HEAD = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The language lab — compose an Arabic micro-story, watch it become a circuit</title>
<style>
:root{--bg:#fbfaf8;--fg:#1a1a1a;--mut:#5c5c5c;--line:#e2ded8;--card:#fff;
      --acc:#8a5a2b;--warn:#a33;--ok:#2f6f4f;--bad:#a33;--hero:#7a4b9c;--dim:#8b8b8b;
      --r0:#2a78d6;--r1:#1baf7a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151517;--fg:#ececec;
      --mut:#a0a0a0;--line:#2e2e33;--card:#1d1d21;--acc:#d09a5e;--ok:#63b98d;--bad:#e08585;
      --warn:#e08585;--hero:#b98ede;--dim:#77777d;--r0:#3987e5;--r1:#199e70}}
:root[data-theme=dark]{--bg:#151517;--fg:#ececec;--mut:#a0a0a0;--line:#2e2e33;--card:#1d1d21;
      --acc:#d09a5e;--ok:#63b98d;--bad:#e08585;--warn:#e08585;--hero:#b98ede;--dim:#77777d;
      --r0:#3987e5;--r1:#199e70}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:32px 20px 90px}
h1{font-size:1.7rem;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:1.1rem;margin:38px 0 10px;letter-spacing:-.005em}
h3{font-size:.94rem;margin:20px 0 4px;font-weight:600}
.sub{color:var(--mut);margin:0 0 20px}
.stage{font:.72rem/1 ui-monospace,Menlo,monospace;color:var(--acc);
       letter-spacing:.14em;text-transform:uppercase;margin:38px 0 2px}
.banner{border:1px solid var(--line);border-left:3px solid var(--acc);background:var(--card);
        padding:12px 14px;border-radius:6px;font-size:.87rem;color:var(--mut)}
.banner.gate{border-left-color:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}
.note{font-size:.85rem;color:var(--mut);margin-top:8px}
.note.tiny{font-size:.78rem}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:.85em}
.warn{color:var(--warn)}
.hero{color:var(--hero)}
b.yes{color:var(--ok)} b.no{color:var(--bad)}
.q{font-style:italic;color:var(--fg)}
select,button,input{font:inherit;font-size:.87rem;background:transparent;color:var(--fg);
       border:1px solid var(--line);border-radius:5px;padding:6px 10px;cursor:pointer}
input{cursor:text}
button:hover,select:hover{border-color:var(--acc)}
.seg{border-radius:5px}
.seg.on{border-color:var(--acc);color:var(--acc);font-weight:600;
        background:color-mix(in srgb,var(--acc) 10%,transparent)}
.seg.small{font-size:.78rem;padding:4px 8px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:500}
table.num td:first-child{width:50%}
tr.live>td{background:color-mix(in srgb,var(--acc) 9%,transparent)}
.ar{font-family:"Geeza Pro","Noto Naskh Arabic","Segoe UI",serif;unicode-bidi:isolate}
.ar.big{font-size:1.35rem;line-height:2}
.ar.sent{font-size:1rem;color:var(--acc);margin-inline-start:8px}
.ar.typed{font-size:1.1rem;width:100%;text-align:right}
.gl{font-size:.75rem;color:var(--mut);margin-inline-start:6px}
.tlabel{font:.7rem/1.4 ui-monospace,Menlo,monospace;color:var(--dim);text-transform:uppercase;
        letter-spacing:.06em}
.ctlrow{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:12px 0 6px}
.ctlrow.tight{margin:6px 0}
.ctllab{font:.7rem/1 ui-monospace,Menlo,monospace;color:var(--dim);text-transform:uppercase;
        letter-spacing:.06em;margin-inline-end:6px}
.modewhy{font-size:.83rem;color:var(--mut);border-inline-start:3px solid var(--acc);
         padding:8px 12px;margin:8px 0 4px;background:color-mix(in srgb,var(--acc) 5%,transparent);
         border-radius:0 6px 6px 0}
.refs{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.refcard{flex:1 1 260px;min-width:250px;border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.refcard select{width:100%}
.evrow{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0;
       padding:8px 10px;border:1px solid var(--line);border-radius:6px}
.ord{font:.68rem/1 ui-monospace,Menlo,monospace;color:var(--dim);margin-inline-start:10px;
     vertical-align:middle}
.story{margin-top:16px;padding:12px 14px;border:1px solid var(--line);border-radius:6px;
       background:color-mix(in srgb,var(--fg) 3%,transparent)}
.story .qline{color:var(--acc)}
.scope{font-size:.83rem;color:var(--mut);margin:8px 0 0;padding-inline-start:20px}
.scope li{margin-bottom:4px}
#circuits{display:flex;gap:16px;flex-wrap:wrap}
.twincol{flex:1 1 440px;min-width:330px;background:var(--card);border:1px solid var(--line);
         border-radius:8px;padding:14px}
.twincol.muted{opacity:.85;border-style:dashed}
.ahead{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.aname{font:600 .8rem/1 ui-monospace,Menlo,monospace;color:var(--acc);
       border:1px solid var(--line);border-radius:4px;padding:3px 6px}
.atitle{font-weight:600}
.ans{border-inline-start:3px solid var(--line);padding:6px 0 6px 10px;margin:10px 0}
.ans.ok{border-inline-start-color:var(--ok)}
.ans.bad{border-inline-start-color:var(--bad)}
.alab{font-size:.72rem;color:var(--dim)}
.averd{font-size:1.15rem;font-weight:600}
.agold{font-size:.78rem;font-weight:400;color:var(--mut)}
table.states{font-size:.76rem;margin-top:8px}
table.states td,table.states th{padding:4px 6px}
.stg{color:var(--mut)}
.pur{font-family:ui-monospace,Menlo,monospace;font-size:.92em;color:var(--dim)}
.pur.isent{color:var(--acc);font-weight:600}
.mech{font-size:.8rem;color:var(--mut);margin-top:10px;padding:8px 10px;border-radius:6px;
      border:1px solid var(--line);border-inline-start:3px solid var(--acc)}
.mech.undone{border-inline-start-color:var(--ok)}
.v{font-weight:600}
.v.true{color:var(--ok)} .v.false{color:var(--bad)}
.row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.col{flex:1 1 320px;min-width:300px}
code{font-family:ui-monospace,Menlo,monospace;font-size:.85em}
/* ---- SVG ---- */
.wire,.cwire{stroke:var(--line);stroke-width:1.6;fill:none}
.wire.r0,.swap.r0{stroke:var(--r0)} .wire.r1,.swap.r1{stroke:var(--r1)}
.swap{fill:none;stroke-width:2.2}
.swaplbl{font:9.5px ui-monospace,Menlo,monospace;fill:var(--acc)}
.wordbox{fill:none;stroke:var(--dim);stroke-width:1.2;rx:3}
.wordbox.verb{stroke:var(--acc);fill:color-mix(in srgb,var(--acc) 10%,transparent)}
.arword{font-size:12px;fill:var(--fg);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.leg{fill:none;stroke:var(--dim);stroke-width:1.1}
.legt{font:9px ui-monospace,Menlo,monospace;fill:var(--hero)}
.legt.s{fill:var(--acc);font-weight:700}
.swapx{stroke:var(--acc);stroke-width:2;fill:none}
.cup{fill:none;stroke:var(--dim);stroke-width:1.3}
.sleg{stroke:var(--acc);stroke-width:1.6}
.stateTri{fill:color-mix(in srgb,var(--r0) 22%,transparent);stroke:var(--r0);stroke-width:1.5}
.stateTri.r1{fill:color-mix(in srgb,var(--r1) 22%,transparent);stroke:var(--r1)}
.effTri{fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 3}
.verbblk{fill:color-mix(in srgb,var(--acc) 14%,transparent);stroke:var(--acc);stroke-width:1.5}
.qblk{fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 3}
.gate{fill:none;stroke:var(--dim);stroke-width:1.2}
.arverb{font-size:15px;fill:var(--acc);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.arverb.dim{fill:var(--mut)}
.arwire{font-size:13px;fill:var(--hero);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.arwire.r0{fill:var(--r0)} .arwire.r1{fill:var(--r1)}
.ar.qsurf{font-size:13px;fill:var(--acc);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.glwire,.note-svg{font-size:9.5px;fill:var(--dim)}
.gtxt{font:10px ui-monospace,Menlo,monospace;fill:var(--mut)}
.collbl{font:9.5px ui-monospace,Menlo,monospace;fill:var(--dim);text-transform:uppercase;
        letter-spacing:.08em}
.evlbl{font:9.5px ui-monospace,Menlo,monospace;fill:var(--dim)}
.port{font:9px ui-monospace,Menlo,monospace;fill:var(--mut)}
.ptype{font:10px ui-monospace,Menlo,monospace;fill:var(--hero)}
.dialring{fill:none;stroke:var(--line);stroke-width:1.5}
.idline{text-align:center;margin:10px 0 6px;font-size:1rem}
.arcband{fill:none;stroke:var(--acc);stroke-width:9;opacity:.30;stroke-linecap:butt}
.cut{stroke:var(--acc);stroke-width:1.6;stroke-dasharray:2 2}
.spoke{stroke:var(--dim);stroke-width:1;opacity:.6}
.spoke.bear{stroke:var(--hero);stroke-width:1.6;opacity:1}
.pt{fill:var(--dim)} .pt.bear{fill:var(--hero)}
.pt.live{stroke:var(--fg);stroke-width:2}
.arlbl{font-size:12px;fill:var(--fg);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.pole{font:10px ui-monospace,Menlo,monospace;fill:var(--mut)}

/* ============ SIMPLE / EXPERT MODE ============================
   Nothing is deleted by simple mode. Expert mode is byte-for-byte the page
   that shipped; simple mode hides the expert sections with CSS and shows a
   three-step guided walkthrough in their place. */
:root[data-lab=simple] .expert-only{display:none}
:root[data-lab=expert] .simple-only{display:none}
.modebar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0 22px;
         padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.modebar .lab{font-size:.8rem;color:var(--mut);margin-inline-end:2px}
.modebtn{font-size:.9rem;padding:7px 14px;border-radius:6px}
.modebtn.on{border-color:var(--acc);color:var(--acc);font-weight:600;
            box-shadow:inset 0 0 0 1px var(--acc)}
.modebar .hint{font-size:.78rem;color:var(--dim);flex:1 1 200px}

/* the three guided steps */
.step{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:16px 18px;margin:14px 0}
.stephead{display:flex;gap:10px;align-items:baseline;margin-bottom:4px}
.stepnum{font:700 .8rem/1 ui-monospace,Menlo,monospace;color:var(--bg);background:var(--acc);
         border-radius:99px;padding:6px 9px}
.steptitle{font-size:1.06rem;font-weight:600}
.steptitle .ar{font-size:1.06rem}
.narr{font-size:.92rem;color:var(--mut);margin:2px 0 12px}

/* presets */
.presets{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.preset{flex:1 1 210px;min-width:200px;text-align:start;padding:11px 13px;border-radius:8px;
        line-height:1.45;white-space:normal}
.preset.on{border-color:var(--acc)}
.preset .pt{display:block;font-size:1.15rem;margin-bottom:4px;
            font-family:"Geeza Pro","Noto Naskh Arabic",serif;direction:rtl;text-align:right}
.preset .pg{display:block;font-size:.78rem;color:var(--mut)}

/* the big friendly swap */
.swapwrap{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}
.swapbtn{font-size:1rem;font-weight:600;padding:11px 18px;border-radius:8px;
         border-color:var(--acc);color:var(--acc)}
.swapbtn:hover{background:color-mix(in srgb,var(--acc) 12%,transparent)}
.punch{flex:1 1 240px;font-size:.85rem;color:var(--mut)}
.punch b{color:var(--fg)}

/* glossary chips */
.chip{position:relative;border-bottom:1.5px dotted var(--acc);cursor:help;color:var(--fg)}
.chip>.chipdef{position:absolute;left:50%;transform:translateX(-50%);bottom:calc(100% + 8px);
  width:250px;background:var(--card);color:var(--fg);border:1px solid var(--acc);
  border-radius:7px;padding:9px 11px;font-size:.79rem;line-height:1.5;font-weight:400;
  box-shadow:0 6px 22px rgba(0,0,0,.32);opacity:0;visibility:hidden;transition:opacity .12s;
  z-index:40;text-align:start;direction:ltr;pointer-events:none}
.chip:hover>.chipdef,.chip:focus>.chipdef,.chip:focus-within>.chipdef{opacity:1;visibility:visible}
@media (max-width:560px){.chip>.chipdef{width:200px}}

/* the simple circuit + wow panel */
.simplecirc{margin:6px 0 4px}
.swire{stroke:var(--line);stroke-width:2}
.sar{font-size:15px;font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.sar.r0{fill:var(--r0)} .sar.r1{fill:var(--r1)}
.sgl{font-size:9.5px;fill:var(--dim)}
.sprep{fill:none;stroke-width:2}
.sprep.r0{stroke:var(--r0)} .sprep.r1{stroke:var(--r1)}
.sptxt{font-size:9.5px;fill:var(--mut)}
.slead{fill:none;stroke-width:2.5}
.slead.r0{stroke:var(--r0)} .slead.r1{stroke:var(--r1)}
.sverb{fill:color-mix(in srgb,var(--acc) 16%,transparent);stroke:var(--acc);stroke-width:2}
.sverbtxt{font-size:17px;fill:var(--acc);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.sport{font-size:9.5px;fill:var(--mut)}
.scol{font-size:10px;fill:var(--dim)}

.meterwrap{margin-top:14px;padding:14px 16px;border:1px solid var(--line);border-radius:9px;
           border-inline-start:3px solid var(--hero)}
.meterhead{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
           flex-wrap:wrap;font-size:.9rem;font-weight:600}
.metertrack{height:16px;border-radius:99px;background:var(--line);overflow:hidden;margin:9px 0 6px}
.meterfill{height:100%;width:0;border-radius:99px;
           background:linear-gradient(90deg,var(--r0),var(--hero));transition:width .05s linear}
.metercap{font-size:.83rem;color:var(--mut)}
.meterval{font:600 1.05rem/1 ui-monospace,Menlo,monospace;color:var(--hero)}
.meterstages{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;font-size:.76rem;color:var(--dim)}
.meterstages span{border:1px solid var(--line);border-radius:99px;padding:2px 9px}

/* simple answer */
.simpans{margin-top:14px;padding:14px 16px;border:1px solid var(--line);border-radius:9px}
.simpans.ok{border-inline-start:3px solid var(--ok)}
.simpans.bad{border-inline-start:3px solid var(--bad)}
.bigverd{font-size:2rem;font-weight:700;letter-spacing:-.02em}
.conftrack{height:13px;border-radius:99px;background:var(--line);overflow:hidden;margin:9px 0 5px;
           display:flex}
.confyes{height:100%;background:var(--ok)}
.confno{height:100%;background:var(--bad);opacity:.55}
.conflab{display:flex;justify-content:space-between;font-size:.78rem;color:var(--mut)}
.goldline{font-size:.85rem;color:var(--mut);margin-top:9px}

/* the always-visible honest line in simple mode */
.honestsimple{border:1px solid var(--line);border-left:3px solid var(--ok);background:var(--card);
              padding:13px 15px;border-radius:8px;margin:20px 0;font-size:.9rem;color:var(--mut)}
.honestsimple b{color:var(--fg)}

/* ---- exp70: the entangled-encoding panel and its staircase ----------------
   Same visual grammar as .meterwrap (a bordered inset with an accent rule) so
   the new panel reads as a sibling of the tangle meter rather than a new page. */
.e70{margin-top:16px;padding:15px 17px;border:1px solid var(--line);border-radius:9px;
     border-inline-start:3px solid var(--acc)}
.e70head{font-size:1.02rem;font-weight:600;margin-bottom:2px;
         display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.e70head .ar{font-size:1.02rem;color:var(--acc)}
.e70 p{margin:9px 0;font-size:.9rem;color:var(--mut)}
.e70 p b{color:var(--fg)}
.e70 .tag{font:.68rem/1 ui-monospace,Menlo,monospace;color:var(--dim);
          border:1px solid var(--line);border-radius:99px;padding:3px 9px;
          text-transform:uppercase;letter-spacing:.07em}
.e70 ul{margin:8px 0;padding-inline-start:20px;font-size:.9rem;color:var(--mut)}
.e70 ul li{margin-bottom:6px}
.e70 ul li b{color:var(--fg)}
.stairbox{margin:14px 0 2px}
.stairtop{fill:none;stroke:var(--acc);stroke-width:2.4;stroke-linejoin:round}
.stairbar{fill:color-mix(in srgb,var(--acc) 13%,transparent);stroke:var(--acc);stroke-width:1.2}
.stairbar.worst{fill:color-mix(in srgb,var(--bad) 20%,transparent);stroke:var(--bad)}
.stairval{font:600 11.5px ui-monospace,Menlo,monospace;fill:var(--fg)}
.stairval.worst{fill:var(--bad)}
.stairlbl{font-size:10.5px;fill:var(--mut)}
.stairar{font-size:12px;fill:var(--acc);font-family:"Geeza Pro","Noto Naskh Arabic",serif}
.stairax{stroke:var(--line);stroke-width:1}
.staircap{font-size:.85rem;color:var(--mut);margin-top:6px}
.staircap b{color:var(--fg)}
</style>"""

BODY = """<div class="wrap">
<h1>The language lab</h1>
<p class="sub">Compose your own Arabic micro-story. Watch it become a string diagram, watch the
diagram compile into a quantum circuit, and watch <span id="narms"></span> trained models &mdash; the real
checkpoints from this programme&rsquo;s <span class="mono">exp69</span> run &mdash; answer a
question about it, in your browser, with no server. Then read why the accuracy any of them can
reach was <b>calculable before training started</b>, and which quantum machinery is actually
doing the work.</p>

<div class="modebar">
  <span class="lab">How much detail?</span>
  <button id="modesimple" class="modebtn">Simple</button>
  <button id="modeexpert" class="modebtn">Expert</button>
  <span class="hint" id="modehint"></span>
</div>

<div class="simple-only">
  <div class="honestsimple" id="honestsimple"></div>

  <div class="step">
    <div class="stephead"><span class="stepnum">1</span>
      <span class="steptitle">Build a tiny story &mdash; who did what to whom?
        <span class="ar" dir="rtl">اِبْنِ قصة صغيرة</span></span></div>
    <p class="narr" id="narr1"></p>
    <div id="simplepresets" class="presets"></div>
    <div id="simplecompose"></div>
    <div class="swapwrap">
      <button id="swapbtn" class="swapbtn"></button>
      <span class="punch" id="swappunch"></span>
    </div>
    <div id="simplestory"></div>
  </div>

  <div class="step">
    <div class="stephead"><span class="stepnum">2</span>
      <span class="steptitle">Watch it become a machine
        <span class="ar" dir="rtl">شاهدها تصير آلة</span></span></div>
    <p class="narr" id="narr2"></p>
    <div class="scroll"><div id="simplecircuit" class="simplecirc"></div></div>
    <div class="meterwrap">
      <div class="meterhead"><span id="meterlabel"></span><span class="meterval" id="metervalue"></span></div>
      <div class="note tiny" id="meterstage" style="margin:0 0 2px"></div>
      <div class="metertrack"><div class="meterfill" id="meterfill"></div></div>
      <p class="metercap" id="metercap"></p>
      <div class="meterstages" id="meterstages"></div>
    </div>
    <div class="e70" id="e70simple"></div>
  </div>

  <div class="step">
    <div class="stephead"><span class="stepnum">3</span>
      <span class="steptitle">Ask it a question
        <span class="ar" dir="rtl">اسألها سؤالاً</span></span></div>
    <p class="narr" id="narr3"></p>
    <div id="simplequestion"></div>
    <div id="simpleanswer"></div>
    <p class="note" id="simplefoot"></p>
  </div>

  <div class="step">
    <div class="stephead"><span class="stepnum">?</span>
      <span class="steptitle">Words this page used</span></div>
    <p class="narr">Every one of these has a plain-language meaning. Hover or tap any dotted
    word anywhere on this page to see it again.</p>
    <div id="glossarylist"></div>
  </div>
</div>

<div class="expert-only">
<div class="banner"><b>Scope, permanently.</b> <span id="ncat"></span> task families, <span id="ncat2"></span> semantic categories, a
<b><span id="ninv"></span>-noun</b> human inventory, <span id="nref"></span> referents per story,
<span id="nverb"></span> verbs in <span id="nslots"></span> disjoint slots, constructed swap-twin stories,
<b>simulation only</b>. The held-out split is over <b>unseen pairings, not unseen nouns</b>.
<b>No quantum advantage is claimed or measured anywhere on this page</b>: the classical comparator
on the identical features ties or exceeds every arm, and it is shown in Stage 4 rather than
omitted. The value on offer is a <b>capacity calculus, a compositional structure, and an
interpretable mechanism</b> &mdash; not speed.
<div id="scopelist"></div></div>

<p class="stage">Stage 1</p>
<h2>Compose a story</h2>
<div class="card" id="composer"></div>

<p class="stage">Stage 2</p>
<h2>The string diagram</h2>
<div class="card">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px">
    <button id="typetoggle" class="seg small">show pregroup types</button>
    <span class="note tiny" style="margin:0">Referent wires run top to bottom. Arabic reads
    right to left, so wire&nbsp;0 is drawn on the right; nothing in the circuit depends on that.</span>
  </div>
  <div class="scroll"><div id="diagram"></div></div>
  <p class="note" id="diagnote"></p>
</div>

<p class="stage">Stage 3</p>
<h2>The circuit, live</h2>
<div id="circuits"></div>
<p class="note"><b>Read the two kinds of number apart.</b> The verdict inside each panel is
<i>this trained model, on your composed input</i> &mdash; and your input is almost certainly in no
held-out set, so it is <b>not</b> an accuracy claim and nothing on this page treats it as one. The
committed <span id="nseeds"></span>-seed held-out figures printed under each panel are the numbers
of record. Because the parity gate agrees with PyTorch on every one of the <span id="nitems"></span> bundled
held-out items for every arm, the circuit
running above is <i>the</i> circuit, not a re-implementation of it.</p>
<p class="note">Seed selection rule, stated rather than implied: <span class="mono" id="selrule"></span></p>

<p class="stage">Stage 4 &middot; part 1</p>
<h2>Why the ceiling is calculable before you train anything</h2>
<div class="card">
  <div class="row">
    <div class="col"><div id="circle"></div></div>
    <div class="col"><div class="scroll"><div id="ceilingnums"></div></div></div>
  </div>
  <p class="note" id="ceilingscope"></p>
</div>
<div class="card">
  <div class="scroll"><div id="boundtable"></div></div>
  <p class="note" id="boundnote"></p>
  <p class="note">Those bounds are the <b>maximum-feasible-subsystem</b> problem for this
  architecture: given a fixed set of referent feature vectors and a readout that is linear in them,
  how many twins can any parameter setting get right at once? It is a geometry question, it is
  solved by search rather than by gradient descent, and it is answerable before a single training
  step exists. That is the calculus this programme is really offering.</p>
</div>

<p class="stage">Stage 4 &middot; part 2</p>
<h2>What quantum machinery this actually uses</h2>
<div class="card" id="quantum"></div>

<p class="stage">Stage 4 &middot; part 3</p>
<h2>The honest close</h2>
<div class="card">
  <h3>The kernel factorises, so this is classically simulable in linear time</h3>
  <p class="idline" id="kernelid"></p>
  <div class="scroll"><div id="kernel"></div></div>
  <p class="note" id="kernelnote"></p>
</div>
<div class="card">
  <h3>The classical comparator &mdash; shown, not omitted</h3>
  <div class="scroll"><div id="cmp"></div></div>
  <p class="note" id="cmpnote"></p>
</div>
<div class="card">
  <h3>So what is the contribution?</h3>
  <p class="note" id="close"></p>
</div>

<h2>The registered criteria, and what they returned</h2>
<div class="card"><div class="scroll"><div id="verdicts"></div></div>
<p class="note">Bar provenance: <span id="barprov"></span></p>
<p class="note" id="outcome"></p>
</div>

<p class="stage">exp70 &middot; the fourth lever</p>
<h2>Entangled encodings &mdash; when the word arrives already tangled
  <span class="ar" dir="rtl">الترميز المتشابك</span></h2>
<div class="banner" id="e70banner"></div>
<p class="note" id="e70scope"></p>

<div class="card">
  <h3>The form, and the closed forms that come with it</h3>
  <div id="e70form"></div>
  <div class="scroll"><div id="e70span"></div></div>
  <p class="note" id="e70formnote"></p>
</div>

<div class="card">
  <h3>The dead zone &mdash; why the registered angle map had to be replaced before a single arm ran</h3>
  <div id="e70deadtext"></div>
  <div class="scroll"><div id="e70dead"></div></div>
  <p class="note" id="e70deadnote"></p>
</div>

<div class="card">
  <h3>The cross-referent arm &mdash; a null that was <i>required</i>, and proved before it was measured</h3>
  <div id="e70clifford"></div>
</div>

<div class="card">
  <h3>The ten arms</h3>
  <div class="scroll"><div id="e70arms"></div></div>
  <p class="note" id="e70armsnote"></p>
  <div class="scroll"><div id="e70cmp"></div></div>
  <p class="note" id="e70cmpnote"></p>
</div>

<div class="card">
  <h3>The composition-fragility ladder</h3>
  <div class="scroll"><div id="e70ladderexpert"></div></div>
  <p class="note" id="e70laddernote"></p>
</div>

<div class="card">
  <h3>The eight one-sided criteria, and the one directional call</h3>
  <div class="scroll"><div id="e70bools"></div></div>
  <div class="scroll"><div id="e70pairs"></div></div>
  <p class="note" id="e70call"></p>
</div>

<div class="card">
  <h3>The replication anchors, and what this section is rendered from</h3>
  <div class="scroll"><div id="e70anchors"></div></div>
  <p class="note" id="e70anchnote"></p>
  <div class="scroll"><div id="e70prov"></div></div>
  <p class="note" id="e70provnote"></p>
</div>

<h2>The featuriser gate &mdash; what lets you type a word at all</h2>
<div class="banner gate" id="featbanner"></div>
<div class="card"><div class="scroll"><div id="novel"></div></div>
<p class="note">These <span id="nnovel"></span> words were <b>registered in the export script</b>,
before the gate ran, so the list could not be tuned to its own result. <span id="nfaail"></span> are
<span class="ar">فاعِل</span>-pattern and <span id="nnotfaail"></span> are not; the list includes forms carrying
tatweel and forms carrying diacritics, so the page&rsquo;s input hygiene is exercised end to end.
None of them is in the trained inventory. The featuriser is fitted on the <span id="ninv2"></span>
inventory nouns only &mdash; a typed word&rsquo;s unseen character-position tokens are simply
dropped, exactly as <span class="mono">CountVectorizer.transform</span> drops them, and the count
of dropped tokens is shown live in Stage 1 so you can see how far outside the fit you have gone.</p>
<p class="note"><b class="warn">Do not over-read that table.</b> The angles in it cluster by
pattern, and you will see it. That is an <b>observation on <span id="nnovel2"></span> words labelled by eye for this
gate</b> &mdash; not a measurement. It was never registered, no criterion reads it, and it is
<b>not</b> evidence that the featuriser generalises the wazn to unseen vocabulary. The committed
wazn results (<span class="mono">R1-M</span>, <span class="mono">R2-M</span>) are held out over
unseen <b>pairings of the <span id="ninv3"></span> inventory nouns</b>, never over unseen nouns. This table exists for
exactly one purpose: to prove the browser computes the <i>same</i> angle Python does.</p>
</div>

<h2>The parity gates</h2>
<div class="banner gate" id="paritybanner"></div>
<div class="card"><div class="scroll"><div id="parity"></div></div>
<p class="note">This is a blocking gate, not a footnote: if any arm disagreed with PyTorch on any
item, or if the browser&rsquo;s <span class="mono">Tr &rho;&sup2;</span> disagreed with the
experiment&rsquo;s own <span class="mono">forward_traced</span>, this page would not have been
built.</p>
<div class="scroll"><div id="qasym"></div></div>
<p class="note">The last table is doc&nbsp;55 &sect;19.4&rsquo;s mechanism signature, re-measured
here as a gate: the readout-versus-post-event-1 purity gap must be <b>smaller when you query the
last event than when you query the first</b>. It was derived from non-commutativity and written
down before the run.</p>
</div>

<h2>The frozen constants behind every semantic angle</h2>
<div class="card"><p class="note" id="constnote"></p></div>

<h2>Provenance</h2>
<div class="card"><div class="scroll"><div id="prov"></div></div>
<p class="note">Split: <span class="mono" id="splitnote"></span></p>
<p class="note"><b>Every number on this page was computed before it was displayed</b>, is written
into the page from a committed JSON or by a parity-gated engine rather than typed into the markup,
and none of it was chosen after seeing which way it came out. The criteria, their bars, and the
reading of every branch were registered in advance of the run; the capacity bounds were computed
before the data existed.</p>
<p class="note"><b>What this is not.</b> Not an Arabic NLP system and not a parser &mdash; the
stories are generated from templates and the circuit consumes gold structure, never surface text.
Not a claim about the underlying word-vector model: the <span id="ncomp"></span> principal components used here explain a small
share of a <span id="edim"></span>-dimensional vector&rsquo;s variance. Not evidence about
discourse-position resolution, anaphora, or multi-hop reasoning &mdash; the verb-indexed query
resolves by verb identity, because the <span id="nslots2"></span> verb slots are disjoint. And not, anywhere, a
quantum-advantage result.</p>
</div>
</div>
</div>"""


def main():
    pr_path = os.path.join(D, "parity_report_lab.json")
    pr = json.load(open(pr_path, encoding="utf-8"))
    if not pr.get("ALL_PASSED"):
        print("[build] REFUSING: parity gates are not all green (%s)" % pr_path)
        return 1
    for k in ("F_featurizer", "C_arc_calculus"):
        flag = "F_passed" if k == "F_featurizer" else "C_passed"
        if not pr[k][flag]:
            print("[build] REFUSING: %s is red" % k)
            return 1
    for a, g in pr["arms"].items():
        if not g["ALL_PASSED"]:
            print("[build] REFUSING: arm %s is red" % a)
            return 1

    e70_path = os.path.join(D, "exp70_lab.json")
    if not os.path.exists(e70_path):
        print("[build] REFUSING: %s is missing -- run export_demo_lab_exp70.py"
              % e70_path)
        return 1
    e70 = json.load(open(e70_path, encoding="utf-8"))
    # exp70 is rendered from committed numbers only: no exp70 checkpoint is
    # bundled and no exp70 circuit runs in the browser, so there is nothing for
    # the circuit parity gate to certify -- but the export's OWN gates must have
    # run, and the bundle must carry the source shas it claims to render from.
    if len(e70.get("sources", [])) != 5 or not all(
            len(s.get("sha256", "")) == 64 for s in e70["sources"]):
        print("[build] REFUSING: the exp70 bundle does not carry its five "
              "source shas")
        return 1
    if e70["e_in"]["closed_form_recheck"]["max_abs_error_eps_map"] != 0.0:
        print("[build] REFUSING: the exp70 concurrence closed-form recheck is "
              "not exact")
        return 1

    # A demo page cites PUBLIC artefacts. The exp69 payloads carry the design
    # doc as a path inside a PRIVATE repository (model_exp69.json .spec and
    # .provenance.design_doc, which the provenance table prints verbatim, and
    # parity_report_lab.json .spec). Rewritten to a doc citation HERE, in the
    # bundle only -- the committed JSONs on disk are read-only to this script and
    # are not modified. The reference survives; the path does not.
    def publicise(src):
        src = re.sub(r"qnlp_private_docs/(\d+)_[A-Za-z0-9_]*\.md", r"doc \1", src)
        return re.sub(r"qnlp_private_docs[^\"\\\s]*", "the registered design doc",
                      src)

    payload = ""
    for var, f in (("__MODEL69__", "model_exp69.json"),
                   ("__TESTSET_LAB__", "testset_lab.json"),
                   ("__FEATURIZER_LAB__", "featurizer_lab.json"),
                   ("__PARITY_LAB__", "parity_report_lab.json"),
                   ("__EXP70_LAB__", "exp70_lab.json")):
        payload += "window.%s=%s;\n" % (
            var, publicise(open(os.path.join(D, f), encoding="utf-8").read()))

    def strip_export(src, name):
        return src.replace("if (typeof module !== 'undefined' && module.exports) "
                           "{ module.exports = %s; }" % name, "")

    engine = strip_export(open(os.path.join(D, "engine_lab.js"),
                               encoding="utf-8").read(), "Lab")
    feat = strip_export(open(os.path.join(D, "featurizer_lab.js"),
                             encoding="utf-8").read(), "FeaturizerLab")
    app = open(os.path.join(D, "app_lab.js"), encoding="utf-8").read()

    html = (HEAD + "\n" + BODY + "\n<script>\n" + payload + feat + "\n" + engine
            + "\n" + app + "\n</script>")
    open(OUT, "w", encoding="utf-8").write(html)

    model = json.load(open(os.path.join(D, "model_exp69.json"), encoding="utf-8"))
    ts = json.load(open(os.path.join(D, "testset_lab.json"), encoding="utf-8"))
    print("[build] wrote %s (%.0f KB, %d arms, %d bundled held-out items, "
          "self-contained)"
          % (OUT, len(html.encode()) / 1024, len(model["arm_order"]),
             sum(s["n_items"] for s in ts["sets"].values())))

    # Gate P: a demo page cites PUBLIC artefacts only. The exp70 sources name a
    # design doc that lives in a private repository; the exporter rewrites those
    # citations, and this is the second, independent check on the finished page.
    priv = [t for t in ("qnlp_private_docs", "/Users/", "/home/") if t in html]
    if priv:
        print("  [FAIL] private or local path in the bundle: %r" % priv)
        return 1
    print("[build] Gate P: no private-repo or local filesystem path on the page")

    bad = [t for t in ("fetch(", "XMLHttpRequest", "https://", "http://",
                       "<link", "<img", "@import", "//cdn", "importScripts")
           if t in html]
    # the SVG namespace URIs are the one legitimate occurrence of "http://" --
    # they are identifiers, never fetched. Checked explicitly rather than waved
    # through: no other external-reference token may appear.
    if bad == ["http://"]:
        others = [ln for ln in html.split("\n")
                  if "http://" in ln and "www.w3.org" not in ln]
        if not others:
            bad = []
    if bad:
        print("  [FAIL] external-reference token(s) found: %r" % bad)
        return 1
    print("[build] Gate Z: no external-reference token in the bundle "
          "(SVG/XHTML namespace URIs excepted, verified line by line)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
