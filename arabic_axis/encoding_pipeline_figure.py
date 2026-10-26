# -*- coding: utf-8 -*-
"""encoding_pipeline_figure.py -- fig_encoding_pipeline: how an Arabic sentence
and its word encodings get loaded onto a quantum circuit.

Reads only COMMITTED artefacts; writes fig_encoding_pipeline.svg / .png and
fig_encoding_pipeline_data.json (the table view: every displayed number with the
file it came from). Nothing else in the repo is touched.

FORM (dataviz step 1). Two of the five panels have a chart's job -- panel (b)
"where does this word sit in the plane" and panel (d) "what magnitude does each
word carry" -- and both are position-on-a-plane questions over 24 points, so both
are scatters, (d) with a magnitude channel. The other three panels are not charts
at all (dataviz `choosing-a-form`: sometimes the answer is not a chart): a
sentence with roles, a gate-level encoding diagram, and a circuit schematic are
DIAGRAMS, drawn as such, with the numbers direct-labelled.

COLOUR (dataviz steps 2-3). Exactly two categorical hues are spent, and they buy
IDENTITY of the two story referents, held constant across panels (a), (b), (c):
slot 1 blue = the man, slot 2 orange = the researcher. Colour follows the entity,
never the panel. Panel (d) needs MAGNITUDE, so it uses the sequential blue ramp
(one hue, light -> dark) and marks the two referents with neutral ink rings +
direct Arabic labels instead of their identity hues -- a magnitude fill and an
identity fill cannot share a mark. Status of an encoding variant ("registered but
not yet run") is carried by a dashed border and a written label, never by a
colour.

    node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light --pairs all
    [PASS] lightness band | [PASS] chroma floor
    [PASS] CVD separation      worst all-pairs #eb6834<->#2a78d6 dE 24.7 (protan)
    [PASS] normal-vision floor worst all-pairs dE 33.6
    [PASS] contrast vs surface both >= 3:1

MODE. A paper figure: one selected light theme, painted explicitly, so it renders
identically on a light page, a dark page and in print. Not auto-flipped.

ARABIC. matplotlib does no complex-script shaping, so this module does it:
Unicode presentation forms (FE70-FEFC) are derived from `unicodedata`
decompositions -- never hand-typed -- joined by the standard dual-joining /
right-joining rule with the lam-alef ligature, then reversed into visual order
for the pure-RTL runs used here. `svg.fonttype` is left at matplotlib's default
'path', so the SVG carries outlines: a browser cannot re-apply bidi to text that
is already geometry.

FAMILY. Same tokens, headline/subhead/footer structure, dpi and data-file
contract as `capacity_validation_figure.py` and fig_arc_budget.
"""
import json
import math
import os
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.font_manager import FontProperties                # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- committed inputs -----------------------------------------------------
F_DIAG = "diagnostic_encoding_entanglement.json"
F_EMB = "exp42_noun_embeddings.json"
F_STORIES = "stories_exp69_QA.json"
F_STORIES_B = "stories_exp69_QB.json"       # glosses only (committed manifest)
F_RESULTS = "results_exp69.json"
F_KERNEL = "diagnostic_kernel_factorisation.json"

SVG_OUT = "fig_encoding_pipeline.svg"
PNG_OUT = "fig_encoding_pipeline.png"
JSON_OUT = "fig_encoding_pipeline_data.json"

ITEM_ID = "train_p2_00002_a"        # the one story the figure walks through
TWIN_ID = "train_p2_00002_b"
R1 = "الرجل"                         # slot 1 -- the bearer in Part A
R2 = "الباحث"                        # slot 2 -- the non-bearer

# ---- design tokens (dataviz reference palette, light surface) -------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"                       # categorical slot 1 -- blue
S2 = "#eb6834"                       # categorical slot 2 -- orange
WASH = "#f0efec"                     # neutral highlight wash
# sequential blue ramp, steps 100 -> 700 of the reference palette
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)

AR_FONT = "/System/Library/Fonts/GeezaPro.ttc"
EPS = math.pi / 6.0

# =========================================================== Arabic shaping
_FORMS, _LIG = {}, {}
for _cp in range(0xFE70, 0xFEFD):
    _dec = unicodedata.decomposition(chr(_cp))
    if not _dec.startswith("<"):
        continue
    _form = _dec[1:_dec.index(">")]
    _base = "".join(chr(int(p, 16)) for p in _dec[_dec.index(">") + 1:].split())
    if len(_base) == 1:
        _FORMS.setdefault(_base, {})[_form] = chr(_cp)
    elif len(_base) == 2:
        _LIG.setdefault(_base, {})[_form] = chr(_cp)


def _dual(c):
    """c connects to the letter that FOLLOWS it (dual-joining)."""
    return c in _FORMS and "initial" in _FORMS[c]


def _joins_right(c):
    """c connects to the letter that PRECEDES it (has a final form)."""
    return c in _FORMS and "final" in _FORMS[c]


def shape(text):
    """Logical Arabic -> presentation forms, in logical order."""
    units, i = [], 0
    while i < len(text):
        pair = text[i:i + 2]
        if pair in _LIG:
            units.append(pair)
            i += 2
        else:
            units.append(text[i])
            i += 1
    out = []
    for k, u in enumerate(units):
        prev = units[k - 1][-1] if k > 0 else ""
        nxt = units[k + 1][0] if k + 1 < len(units) else ""
        pj, nj = _dual(prev), _joins_right(nxt)
        if u in _LIG:
            f = _LIG[u]
            out.append(f["final"] if pj and "final" in f else f["isolated"])
            continue
        f = _FORMS.get(u)
        if not f:
            out.append(u)
        elif pj and nj and "medial" in f:
            out.append(f["medial"])
        elif pj and "final" in f:
            out.append(f["final"])
        elif nj and "initial" in f:
            out.append(f["initial"])
        else:
            out.append(f.get("isolated", u))
    return "".join(out)


def ar(text):
    """Shaped and reversed into visual order -- pure-RTL runs only."""
    return shape(text)[::-1]


def arfp(size):
    return FontProperties(fname=AR_FONT, size=size)


# ================================================================ the maps
def theta_registered(p):
    """exp68 F2 convention: theta = (p + 1) * pi   (instrument_sb_bounds
    SCALINGS['full'], DEFAULT_SCALING)."""
    return (p + 1.0) * math.pi


def theta_eps(p):
    """doc 56 eps-map: theta = p * (pi - 2 eps) / 2 + pi/2, eps = pi/6."""
    return p * (math.pi - 2 * EPS) / 2.0 + math.pi / 2.0


def concurrence(t1, t2):
    """Closed form for RY(t1) on q0 then CRY(t2) q0->q1."""
    return abs(math.sin(t1)) * abs(math.sin(t2 / 2.0))


def load():
    j = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
    diag, emb, stories = j(F_DIAG), j(F_EMB), j(F_STORIES)
    res, kern = j(F_RESULTS), j(F_KERNEL)
    item = next(i for i in stories["items"] if i["id"] == ITEM_ID)
    twin = next(i for i in stories["items"] if i["id"] == TWIN_ID)
    # English glosses live in the two committed generator manifests: Part A
    # glosses its own bearers, Part B glosses the wazn bearers. Both are
    # committed; neither is retyped here.
    gl = dict(stories["meta"]["category"]["glosses"])
    gl.update(j(F_STORIES_B)["meta"]["category"]["glosses"])
    return diag, emb, stories, res, kern, item, twin, gl


def per_noun_rows(diag):
    """Recompute every angle from the committed p1/p2 and ASSERT the committed
    concurrences reproduce exactly -- the figure never displays a number it
    could not re-derive."""
    rows = {}
    for n, v in diag["per_noun"].items():
        p1, p2 = v["p1"], v["p2"]
        tr1, tr2 = theta_registered(p1), theta_registered(p2)
        te1, te2 = theta_eps(p1), theta_eps(p2)
        cr, ce = concurrence(tr1, tr2), concurrence(te1, te2)
        assert abs(cr - v["concurrence_registered_map"]) < 1e-12, n
        assert abs(ce - v["concurrence_eps_map"]) < 1e-12, n
        rows[n] = {"p1": p1, "p2": p2,
                   "theta_registered": [tr1, tr2],
                   "theta_eps": [te1, te2],
                   "concurrence_registered_map": cr,
                   "concurrence_eps_map": ce,
                   "frozen_angle_FA": v["frozen_angle"]}
    return rows


# ============================================================ draw helpers
def txt(ax, x, y, s, size=8.0, color=INK_2, ha="left", va="center",
        weight="normal", ls=1.45, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, linespacing=ls, transform=ax.transAxes,
                   **kw)


def artxt(ax, x, y, s, size=13.0, color=INK, ha="right", va="center", **kw):
    return ax.text(x, y, ar(s), fontproperties=arfp(size), color=color,
                   ha=ha, va=va, transform=ax.transAxes, **kw)


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def rule(ax, y, x0=0.0, x1=1.0, color=GRID, lw=0.9):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, transform=ax.transAxes,
            clip_on=False, zorder=1)


def wire(ax, x0, x1, y, color=AXIS, lw=1.1):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, transform=ax.transAxes,
            zorder=2, solid_capstyle="butt")


def gatebox(ax, xc, y, w, h, label, size=7.6, fc=SURFACE, ec=INK_2, lw=1.1,
            color=INK, dashed=False, weight="normal"):
    p = FancyBboxPatch((xc - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.006,rounding_size=0.012",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle=(0, (2.6, 2.0)) if dashed else "-",
                       transform=ax.transAxes, zorder=3, clip_on=False)
    ax.add_patch(p)
    ax.text(xc, y, label, fontsize=size, color=color, ha="center", va="center",
            fontweight=weight, transform=ax.transAxes, zorder=4)
    return p


def word_widths(fig, ax, words, size):
    """Rendered width of each shaped word, in axes-x fraction."""
    r = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    out = []
    for w in words:
        t = ax.text(0.5, -3.0, ar(w), fontproperties=arfp(size),
                    transform=ax.transAxes)
        bb = t.get_window_extent(renderer=r)
        x0 = inv.transform((bb.x0, bb.y0))[0]
        x1 = inv.transform((bb.x1, bb.y0))[0]
        t.remove()
        out.append(x1 - x0)
    return out


def rtl_line(fig, ax, x_right, y, words, size, colors, gap=0.028):
    """Lay Arabic words right-to-left; return each word's (centre, width)."""
    ws = word_widths(fig, ax, words, size)
    placed, x = [], x_right
    for w, wd, c in zip(words, ws, colors):
        artxt(ax, x, y, w, size=size, color=c, ha="right")
        placed.append((w, x - wd / 2.0, wd))
        x -= wd + gap
    return placed


# ============================================================ panel (a)
def panel_a(fig, ax, item, twin, stories, gloss, sha):
    blank(ax)

    txt(ax, 0.0, 0.985, "CAST  —  two referents, one category bearer", 7.6,
        MUTED, weight="bold")
    for i, (n, col, role) in enumerate([
            (R1, S1, "%s  ·  BEARER of the non-occupational category"
             % gloss[R1]),
            (R2, S2, "%s  ·  non-bearer" % gloss[R2])]):
        y = 0.930 - i * 0.058
        artxt(ax, 0.155, y, n, size=14.5, color=col)
        txt(ax, 0.185, y, role, 8.0, INK_2)

    rule(ax, 0.836)

    ev = item["events"]
    qv = item["queried_verb"]
    marks = []
    for i, (e, of) in enumerate(zip(ev, item["order_flags"])):
        top = 0.790 - i * 0.180
        txt(ax, 0.0, top, "EVENT %d" % (i + 1), 7.6, MUTED, weight="bold")
        txt(ax, 0.200, top, "surface order %s" % of, 7.6, MUTED)
        words = item["sentences"][i].split()
        roles = {e["agent"]: "agent", e["patient"]: "patient",
                 e["verb"]: "verb"}
        cols = [S1 if w == R1 else S2 if w == R2 else INK for w in words]
        placed = rtl_line(fig, ax, 0.985, top - 0.062, words, 15.0, cols)
        for (w, xc, wd) in placed:
            lab = roles[w]
            if w == qv:
                lab = "verb\nQUERIED"
                marks.append((xc, wd, top - 0.062))
            txt(ax, xc, top - 0.118, lab, 7.0,
                INK if w == qv else MUTED, ha="center", va="top",
                weight="bold" if w == qv else "normal", ls=1.35)

    rule(ax, 0.437)

    txt(ax, 0.0, 0.393, "QUESTION", 7.6, MUTED, weight="bold")
    txt(ax, 0.200, 0.393,
        "names one of the two verbs  —  here q = %d" % item["query_index"],
        7.6, MUTED)
    qwords = item["question"]["surface"].split()
    qcols = [INK if w == qv else INK_2 for w in qwords]
    qplaced = rtl_line(fig, ax, 0.985, 0.328, qwords, 13.5, qcols, gap=0.022)
    for (w, xc, wd) in qplaced:
        if w == qv:
            marks.append((xc, wd, 0.328))
    txt(ax, 0.985, 0.272,
        "“the one who lifted — is he the marked one?”", 7.8, MUTED, ha="right")

    for (xc, wd, y) in marks:                      # queried-verb highlight
        ax.add_patch(Rectangle((xc - wd / 2 - 0.012, y - 0.040),
                               wd + 0.024, 0.080, facecolor=WASH,
                               edgecolor="none", transform=ax.transAxes,
                               zorder=0, clip_on=False))

    txt(ax, 0.0, 0.202, "ANSWER", 7.6, MUTED, weight="bold")
    artxt(ax, 0.230, 0.202, "نعم", size=13.0, color=INK, ha="right")
    txt(ax, 0.255, 0.202,
        "%s  —  the agent of event %d is the %s, a bearer"
        % (item["answer"], item["query_index"], gloss[R1]), 8.0, INK)

    txt(ax, 0.0, 0.172,
        "Twin item %s flips agent and patient inside\n"
        "event %d only — same cast, same word multiset, same question\n"
        "surface, opposite label (“%s”). Any cast-only function scores 0.500."
        % (TWIN_ID, item["query_index"], twin["answer"]),
        7.4, INK_2, va="top")

    txt(ax, 0.0, 0.078,
        "%s  ·  item %s  ·  split %s  ·  T = %d\n"
        "category %s  ·  dataset sha256\n%s"
        % (F_STORIES, ITEM_ID, item["split"], item["T"],
           stories["meta"]["category"]["key"], sha),
        6.4, MUTED, va="top")


# ============================================================ panel (b)
def panel_b(fig, ax, rows, notes_xy):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    ax.set_xlim(-1.10, 0.10)
    ax.set_ylim(-1.10, 1.15)
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    others = [n for n in rows if n not in (R1, R2)]
    ax.scatter([rows[n]["p1"] for n in others], [rows[n]["p2"] for n in others],
               s=26, facecolors=MUTED, edgecolors=SURFACE, linewidths=0.8,
               alpha=0.55, zorder=2)
    for n, c in ((R1, S1), (R2, S2)):
        ax.scatter([rows[n]["p1"]], [rows[n]["p2"]], s=104, facecolors=c,
                   edgecolors=SURFACE, linewidths=1.6, zorder=4)
    # direct labels placed into free space -- (b) and (d) use the same offsets
    ax.text(rows[R1]["p1"] + 0.060, rows[R1]["p2"] + 0.005, ar(R1),
            fontproperties=arfp(13.0), color=S1, ha="left", va="center",
            zorder=5)
    ax.text(rows[R2]["p1"] + 0.030, rows[R2]["p2"] + 0.150, ar(R2),
            fontproperties=arfp(13.0), color=S2, ha="left", va="center",
            zorder=5)
    ax.set_xlabel("p₁   first principal coordinate", color=INK_2, fontsize=8.6,
                  labelpad=5)
    ax.set_ylabel("p₂   second principal coordinate", color=INK_2, fontsize=8.6,
                  labelpad=4)

    x0, y0 = notes_xy
    fig.text(x0, y0,
             "All 24 human referents. Each word's 300-dim\n"
             "AraVec vector, PCA over all 44 referents,\n"
             "sign-pinned, each axis divided by its own\n"
             "inventory maximum:\n"
             "      p_k(r) = PC_k(r) / max_r′ |PC_k(r′)|",
             color=INK_2, fontsize=7.2, va="top", ha="left", linespacing=1.6)
    fig.text(x0, y0 - 0.114,
             "the man        p = (%+.4f, %+.4f)\n"
             "the researcher p = (%+.4f, %+.4f)\n"
             "                    ↑ p₁ at the inventory extreme"
             % (rows[R1]["p1"], rows[R1]["p2"], rows[R2]["p1"], rows[R2]["p2"]),
             color=INK, fontsize=6.9, va="top", ha="left", linespacing=1.7,
             family="monospace")
    fig.text(x0, y0 - 0.190,
             "The single-angle map in (c) is atan2(y, x) on\n"
             "the frozen UNIT-NORMED twin of this plane\n"
             "(exp42 vec2), not atan2(p₂, p₁); the two\n"
             "register maps do use p as plotted.\n"
             "p: diagnostic_encoding_entanglement.json\n"
             "vectors: exp42_noun_embeddings.json",
             color=MUTED, fontsize=6.9, va="top", ha="left", linespacing=1.6)


# ============================================================ panel (c)
def panel_c(fig, ax, rows, dial_ax):
    blank(ax)
    r1, r2 = rows[R1], rows[R2]

    def reglabel(y0, y1, col):
        artxt(ax, 0.100, (y0 + y1) / 2.0, R1, size=10.0, color=col)
        for k, y in enumerate((y0, y1)):
            txt(ax, 0.140, y, "q%d" % k, 6.4, MUTED, ha="right")

    # ---------------------------------------------------- variant 1 : FA
    txt(ax, 0.0, 0.975, "1   single angle  ·  FA", 8.6, INK, weight="bold")
    txt(ax, 0.0, 0.945,
        "φ(r) = atan2(y, x) on the frozen exp42 vec2\n"
        "one wire per referent\n"
        "as run: exp69 arms A0 / R1-A / R2-A", 7.0, MUTED, va="top")
    for i, (n, col, phi) in enumerate([(R1, S1, r1["frozen_angle_FA"]),
                                       (R2, S2, r2["frozen_angle_FA"])]):
        y = 0.812 - i * 0.053
        artxt(ax, 0.130, y, n, size=10.5, color=col)
        wire(ax, 0.148, 0.560, y)
        gatebox(ax, 0.310, y, 0.180, 0.034, "RY(%+.4f)" % phi, size=7.2,
                ec=col, color=INK)

    # ------------------------------------- variant 2 : product register (F2)
    txt(ax, 0.0, 0.700, "2   product register  ·  F2", 8.6, INK, weight="bold")
    txt(ax, 0.0, 0.670,
        "θ_k = (p_k + 1)·π      two wires per referent,\n"
        "one RY each, no link      as run: arms R1-W / R2-W",
        7.0, MUTED, va="top")
    yp = [0.572, 0.520]
    reglabel(yp[0], yp[1], S1)
    for i, (y, th) in enumerate(zip(yp, r1["theta_registered"])):
        wire(ax, 0.148, 0.640, y)
        gatebox(ax, 0.310, y, 0.180, 0.034, "RY(%.4f)" % th, size=7.2, ec=S1)
        txt(ax, 0.655, y, ["θ₁", "θ₂"][i], 7.2, MUTED)
    txt(ax, 0.0, 0.468,
        "The researcher's p₁ = −1.0000 sends θ₁ to %.4f: that\n"
        "RY does nothing, and its register's entanglement under\n"
        "this map is %.1e — a normalisation extreme landing on\n"
        "a sin dead zone. Structural, not semantic."
        % (r2["theta_registered"][0], r2["concurrence_registered_map"]),
        7.0, INK_2, va="top")

    # ------------------------------- variant 3 : entangled register (E-in)
    txt(ax, 0.0, 0.318, "3   entangled register  ·  E-in", 8.6, INK,
        weight="bold")
    txt(ax, 0.0, 0.288,
        "θ_k = p_k(π − 2ε)/2 + π/2,  ε = π/6  →  θ ∈ [π/6, 5π/6]\n"
        "PRE-REGISTERED for doc 56 — NOT YET RUN",
        7.0, MUTED, va="top")
    ye = [0.190, 0.138]
    te = r1["theta_eps"]
    reglabel(ye[0], ye[1], S1)
    for y in ye:
        wire(ax, 0.148, 0.640, y)
    gatebox(ax, 0.262, ye[0], 0.170, 0.034, "RY(%.4f)" % te[0], size=7.2,
            ec=S1, dashed=True)
    ax.plot([0.470, 0.470], [ye[0], ye[1]], color=INK_2, lw=1.2,
            transform=ax.transAxes, zorder=2)
    ax.plot([0.470], [ye[0]], marker="o", ms=5.4, color=INK_2,
            transform=ax.transAxes, zorder=4)
    gatebox(ax, 0.470, ye[1], 0.170, 0.034, "RY(%.4f)" % te[1], size=7.2,
            ec=S1, dashed=True)
    txt(ax, 0.655, ye[0], "control", 6.9, MUTED)
    txt(ax, 0.655, ye[1], "controlled-RY", 6.9, MUTED)
    txt(ax, 0.0, 0.090,
        "C(r) = |sin θ₁| · |sin(θ₂/2)|\n"
        "    the man %.4f    ·    the researcher %.4f\n"
        "doc 56 runs E-in against an ε-map PRODUCT TWIN — same map, no\n"
        "entangler — so the contrast measures the entangler and not the\n"
        "angle map (diagnostic, design_consequence)."
        % (r1["concurrence_eps_map"], r2["concurrence_eps_map"]),
        7.0, INK_2, va="top")

    # ------------------------------------------------------------- the dial
    dial_ax.set_facecolor(SURFACE)
    dial_ax.set_xlim(-1.35, 1.35)
    dial_ax.set_ylim(-1.35, 1.35)
    dial_ax.set_aspect("equal")
    dial_ax.axis("off")
    th = [i * math.pi / 180.0 for i in range(361)]
    dial_ax.plot([math.cos(t) for t in th], [math.sin(t) for t in th],
                 color=AXIS, lw=1.0)
    dial_ax.plot([-1.15, 1.15], [0, 0], color=GRID, lw=0.8)
    dial_ax.plot([0, 0], [-1.15, 1.15], color=GRID, lw=0.8)
    for n, c in ((R1, S1), (R2, S2)):
        a = rows[n]["frozen_angle_FA"]
        dial_ax.annotate("", xy=(math.cos(a), math.sin(a)), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6,
                                         shrinkA=0, shrinkB=0))
    dial_ax.text(0, 1.32, "the φ dial", fontsize=6.8, color=MUTED,
                 ha="center", va="bottom")
    dial_ax.text(1.26, -0.20, "0", fontsize=6.4, color=MUTED, ha="center")
    dial_ax.text(-1.26, -0.20, "π", fontsize=6.4, color=MUTED, ha="center")


# ============================================================ panel (d)
def panel_d(fig, ax, cax, rows, diag, notes_xy):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    ax.set_xlim(-1.10, 0.10)
    ax.set_ylim(-1.10, 1.15)
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    names = list(rows)
    vals = [rows[n]["concurrence_eps_map"] for n in names]
    norm = Normalize(vmin=0.0, vmax=1.0)
    ax.scatter([rows[n]["p1"] for n in names], [rows[n]["p2"] for n in names],
               c=vals, cmap=SEQ, norm=norm, s=92, edgecolors=INK_2,
               linewidths=0.6, zorder=3)
    for n, dx, dy in ((R1, 0.090, 0.0), (R2, 0.050, 0.160)):
        ax.scatter([rows[n]["p1"]], [rows[n]["p2"]], s=210, facecolors="none",
                   edgecolors=INK, linewidths=1.3, zorder=4)
        ax.text(rows[n]["p1"] + dx, rows[n]["p2"] + dy, ar(n),
                fontproperties=arfp(12.5), color=INK, ha="left", va="center",
                zorder=5)
    ax.set_xlabel("p₁   first principal coordinate", color=INK_2, fontsize=8.6,
                  labelpad=5)
    ax.set_ylabel("p₂   second principal coordinate", color=INK_2, fontsize=8.6,
                  labelpad=4)

    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cax.tick_params(colors=MUTED, labelsize=7.0, length=2.5, width=0.8)
    for lbl in cax.get_xticklabels():
        lbl.set_color(INK_2)
    cax.set_title("E-in concurrence of the referent's own register  ·  ε-map",
                  fontsize=7.0, color=INK_2, pad=3)

    e = diag["finding"]["eps_map"]
    r = diag["finding"]["registered_map"]
    lo = min(rows, key=lambda n: rows[n]["concurrence_eps_map"])
    hi = max(rows, key=lambda n: rows[n]["concurrence_eps_map"])
    x0, y0 = notes_xy
    fig.text(x0, y0,
             "What sets a word's entanglement? Only where it sits:\n"
             "   C(r) = |sin θ₁| · |sin(θ₂/2)|,  θ from the ε-map",
             color=INK_2, fontsize=7.2, va="top", ha="left", linespacing=1.6)
    fig.text(x0, y0 - 0.056,
             "over the 24 humans  min %.4f   median %.4f\n"
             "                    max %.4f   none below 0.05\n"
             "the man        %.4f  — the inventory median\n"
             "the researcher %.4f  — 2nd lowest of 24"
             % (e["min"], e["median"], e["max"],
                rows[R1]["concurrence_eps_map"],
                rows[R2]["concurrence_eps_map"]),
             color=INK, fontsize=6.7, va="top", ha="left", linespacing=1.8,
             family="monospace")
    fig.text(x0, y0 - 0.164,
             "Under the registered map θ = (p+1)π, %d of 24 fall below\n"
             "0.05 (min %.1e): normalisation extremes land on sin dead\n"
             "zones — which is why doc 56 registers the ε-map, whose\n"
             "floor keeps sin θ₁ ≥ 0.5.\n"
             "Source: diagnostic_encoding_entanglement.json"
             % (r["n_below_0.05"], r["min"]),
             color=MUTED, fontsize=6.9, va="top", ha="left", linespacing=1.6)
    return lo, hi


# ============================================================ panel (e)
def panel_e(fig, ax, item, purity, n_params):
    blank(ax)
    ev, qv = item["events"], item["queried_verb"]
    yy = [0.905, 0.855, 0.775, 0.725]                 # 2 registers x 2 wires
    xs = {"intro": 0.235, "e1": 0.395, "e2": 0.545, "iq": 0.695, "ro": 0.855}

    txt(ax, 0.0, 0.985,
        "the register encoding as run (4 wires, %d trained parameters)"
        % n_params, 7.8, INK_2)

    for i, (n, col) in enumerate(((R1, S1), (R2, S2))):
        artxt(ax, 0.090, (yy[2 * i] + yy[2 * i + 1]) / 2.0, n, size=11.5,
              color=col)
        for k in (0, 1):
            y = yy[2 * i + k]
            txt(ax, 0.115, y, "q%d" % k, 6.6, MUTED, ha="right")
            wire(ax, 0.125, 0.935, y)

    for i, y in enumerate(yy):                        # 1. intro
        gatebox(ax, xs["intro"], y, 0.072, 0.030, "RY", size=6.8,
                ec=S1 if i < 2 else S2)
    for key, lab in (("e1", ev[0]["verb"]), ("e2", ev[1]["verb"])):
        gatebox(ax, xs[key], (yy[0] + yy[3]) / 2.0, 0.100, 0.230, "", ec=INK_2)
        ax.text(xs[key], (yy[0] + yy[3]) / 2.0 + 0.018, ar(lab),
                fontproperties=arfp(12.5), color=INK, ha="center", va="center",
                transform=ax.transAxes, zorder=4)
        ax.text(xs[key], (yy[0] + yy[3]) / 2.0 - 0.040, "U", fontsize=8.0,
                color=INK_2, ha="center", va="center", style="italic",
                transform=ax.transAxes, zorder=4)
    gatebox(ax, xs["iq"], (yy[0] + yy[3]) / 2.0, 0.100, 0.230, "", ec=INK_2,
            dashed=True)
    ax.text(xs["iq"], (yy[0] + yy[3]) / 2.0 + 0.018, ar(qv),
            fontproperties=arfp(12.5), color=INK, ha="center", va="center",
            transform=ax.transAxes, zorder=4)
    ax.text(xs["iq"], (yy[0] + yy[3]) / 2.0 - 0.040, "U †", fontsize=8.0,
            color=INK_2, ha="center", va="center", style="italic",
            transform=ax.transAxes, zorder=4)
    gatebox(ax, xs["ro"], (yy[0] + yy[3]) / 2.0, 0.110, 0.230,
            "", ec=INK_2)
    ax.text(xs["ro"], (yy[0] + yy[3]) / 2.0 + 0.030, "Q⁺†", fontsize=8.0,
            color=INK, ha="center", va="center", transform=ax.transAxes,
            zorder=4)
    ax.text(xs["ro"], (yy[0] + yy[3]) / 2.0 - 0.005, "Q⁻†", fontsize=8.0,
            color=INK, ha="center", va="center", transform=ax.transAxes,
            zorder=4)
    ax.text(xs["ro"], (yy[0] + yy[3]) / 2.0 - 0.048, "⟨0000|", fontsize=7.4,
            color=INK_2, ha="center", va="center", transform=ax.transAxes,
            zorder=4)

    for i, (k, lab) in enumerate((("intro", "1"), ("e1", "2"), ("e2", "3"),
                                  ("iq", "4"), ("ro", "5"))):
        txt(ax, xs[k], 0.955, lab, 7.6, INK, ha="center", weight="bold")

    # ---- the measured purity trace, aligned under its stage
    rule(ax, 0.672)
    txt(ax, 0.0, 0.648, "mean wire purity per stage  (measured, descriptive)",
        6.9, MUTED, va="center")
    xro = (xs["iq"] + xs["ro"]) / 2.0
    trace = [(xs["intro"], purity["after_intro"]),
             (xs["e1"], purity["after_event1"]),
             (xs["e2"], purity["after_event2"]),
             (xro, purity["at_readout"])]
    for x, v in trace:
        txt(ax, x, 0.600, "%.4f" % v, 7.6, INK, ha="center", weight="bold")
    for a, b in zip(trace, trace[1:]):
        txt(ax, (a[0] + b[0]) / 2.0, 0.600, "→", 8.0, MUTED, ha="center")
    rule(ax, 0.562)

    lines = [
        ("1   intro — one RY per wire, applied once, before event 1. Each "
         "referent is\n      a product of two pure single-qubit states, purity "
         "%.4f." % purity["after_intro"]),
        ("2   event 1 (order %s) — the verb's block acts on the agent and "
         "patient registers\n      TOGETHER; binding shows up as loss of "
         "single-wire purity: %.4f → %.4f."
         % (item["order_flags"][0], purity["after_intro"],
            purity["after_event1"])),
        ("3   event 2 (order %s) — the same six verb blocks, reused, on the "
         "same\n      persistent referent wires. Purity falls to %.4f."
         % (item["order_flags"][1], purity["after_event2"])),
        ("4   inverse question block — the question names one verb, so THAT "
         "verb's own\n      block is applied as its adjoint on the queried "
         "wires: the question undoes\n      its own verb, and purity returns "
         "to %.4f — mean |gap| %.4f at\n      q = 2, against %.4f at q = 1."
         % (purity["at_readout"], purity["gap_q2"], purity["gap_q1"])),
        ("5   readout — two shared trained effects Q⁺, Q⁻ applied as adjoints, "
         "then the\n      |0000⟩ probability on the queried wires, softmax over "
         "log p. Binary: yes / no."),
    ]
    y = 0.525
    for ln in lines:
        txt(ax, 0.0, y, ln, 7.0, INK_2, va="top", ls=1.45)
        y -= 0.0270 * (ln.count("\n") + 1) + 0.013

    txt(ax, 0.0, 0.008,
        "purity: results_exp69.json  purity_contrast_doc55_SS19_4 · R2-W ·\n"
        "q2_correct (n = 134) — DESCRIPTIVE, no criterion reads it.\n"
        "circuit of record: exp68_run.F2Model.story_values (register arms) /\n"
        "exp42_models.QuantumStoryModel.story_values (angle arms).\n"
        "n_params: results_exp69.json arms.R2-W.n_params",
        6.2, MUTED, va="bottom", ls=1.5)


# ================================================================== assemble
def render():
    diag, emb, stories, res, kern, item, twin, gloss = load()
    rows = per_noun_rows(diag)
    pc = res["purity_contrast_doc55_SS19_4"]["R2-W"]
    purity = dict(pc["cells"]["q2_correct"]["mean_wire_purity"])
    purity["gap_q1"] = pc["readout_minus_after_event1_abs"]["q1"]["mean_abs_gap"]
    purity["gap_q2"] = pc["readout_minus_after_event1_abs"]["q2"]["mean_abs_gap"]
    n_params = res["arms"]["R2-W"]["n_params"]

    fig = plt.figure(figsize=(19.6, 7.4))
    fig.patch.set_facecolor(SURFACE)

    fig.text(0.012, 0.987,
             "From an Arabic sentence to a quantum circuit", color=INK,
             fontsize=16.5, fontweight="bold", va="top", ha="left")
    fig.text(0.012, 0.951,
             "One committed two-event item, carried through the whole encoding "
             "path: grammar → the word's coordinates → rotation angles on wires "
             "→ the circuit that answers the question.\n"
             "Every number on this figure is read from a committed file and "
             "re-derived from its formula before it is drawn; the file is named "
             "under the panel that shows it.",
             color=INK_2, fontsize=9.0, va="top", ha="left", linespacing=1.5)

    X = {"a": 0.020, "b": 0.226, "c": 0.388, "d": 0.614, "e": 0.776}
    W = {"a": 0.176, "b": 0.132, "c": 0.196, "d": 0.132, "e": 0.218}
    TITLES = {
        "a": ("(a)  the sentence", "grammar-level: who did what, in which "
              "event, and what is asked"),
        "b": ("(b)  word → coordinates", "each word is a point in its own "
              "distributional plane"),
        "c": ("(c)  coordinates → angles → wires", "three encodings of the "
              "same two coordinates"),
        "d": ("(d)  the entanglement dial", "the word's position sets its "
              "own entanglement"),
        "e": ("(e)  the circuit runs", "prepare, compose, ask, read out"),
    }
    for k in "abcde":
        fig.text(X[k], 0.900, TITLES[k][0], color=INK, fontsize=11.2,
                 fontweight="bold", va="top", ha="left")
        fig.text(X[k], 0.868, TITLES[k][1], color=MUTED, fontsize=7.8,
                 va="top", ha="left")
    for k in "abcd":
        fig.text(X[k] + W[k] + 0.015, 0.500, "→", color=AXIS, fontsize=17,
                 ha="center", va="center")

    BOT, HGT = 0.135, 0.700
    ax_a = fig.add_axes((X["a"], BOT, W["a"], HGT))
    ax_c = fig.add_axes((X["c"], BOT, W["c"], HGT))
    ax_e = fig.add_axes((X["e"], BOT, W["e"], HGT))
    sq = W["b"] * 19.6 / 7.4
    ax_b = fig.add_axes((X["b"] + 0.014, 0.845 - sq, W["b"] - 0.014, sq))
    ax_d = fig.add_axes((X["d"] + 0.014, 0.845 - sq, W["d"] - 0.014, sq))
    cax = fig.add_axes((X["d"] + 0.014, 0.845 - sq - 0.092, W["d"] - 0.014,
                        0.015))
    dial = fig.add_axes((X["c"] + W["c"] - 0.046, 0.632, 0.044,
                         0.044 * 19.6 / 7.4))

    panel_a(fig, ax_a, item, twin, stories, gloss,
            res["datasets"]["A"]["sha256"])
    panel_b(fig, ax_b, rows, (X["b"], 0.845 - sq - 0.065))
    panel_c(fig, ax_c, rows, dial)
    lo, hi = panel_d(fig, ax_d, cax, rows, diag,
                     (X["d"], 0.845 - sq - 0.132))
    panel_e(fig, ax_e, item, purity, n_params)

    fig.text(0.012, 0.107,
             "HONEST LIMIT.  Every encoding drawn here as RUN is a PRODUCT-state "
             "encoding, and a product encoding is classically simulable: the "
             "kernel factorises exactly, ⟨ψ(θ)|ψ(θ′)⟩ = Π_i cos((θ_i − θ′_i)/2), "
             "checked over 200 random pairs at\n"
             "n = 1, 2, 3, 5, 8, 12 qubits to %.2e (complex64 statevector, the "
             "simulator of record) and %.2e (float64 closed form) — "
             "diagnostic_kernel_factorisation.json.  What the pipeline buys is "
             "the compositional\n"
             "calculus and a mechanism you can read off the wires stage by "
             "stage, not speed.  The entangled variant E-in in (c) is a doc 56 "
             "PRE-REGISTRATION: it has not been run, and no accuracy is claimed "
             "for it here.  No accuracy from\n"
             "exp69 is shown on this figure at all — it is a mechanism figure. "
             " Simulation throughout; no quantum-advantage reading is licensed. "
             " Angles and concurrences are recomputed here from the committed "
             "p₁, p₂ and asserted equal to the\n"
             "committed values to < 1e-12.  Table view — every displayed number "
             "with its source file: %s"
             % (kern["max_abs_deviation_sim_c64_overall"],
                kern["max_abs_deviation_float64_overall"], JSON_OUT),
             color=MUTED, fontsize=6.9, va="top", ha="left", linespacing=1.6)

    fig.savefig(os.path.join(HERE, SVG_OUT), format="svg", facecolor=SURFACE,
                dpi=300)
    fig.savefig(os.path.join(HERE, PNG_OUT), format="png", facecolor=SURFACE,
                dpi=300)
    plt.close(fig)
    write_data(diag, emb, stories, res, kern, item, twin, rows, purity,
               n_params, lo, hi, gloss)


def write_data(diag, emb, stories, res, kern, item, twin, rows, purity,
               n_params, lo, hi, gloss):
    e = diag["finding"]["eps_map"]
    rmap = diag["finding"]["registered_map"]
    payload = {
        "figure": SVG_OUT,
        "produced_by": os.path.basename(__file__),
        "what_is_plotted": (
            "a five-panel left-to-right walk of ONE committed exp69 item "
            "through the encoding pipeline: (a) the Arabic two-event text with "
            "roles and the queried verb, (b) the two story referents among all "
            "24 human referents in the normalised PC1-PC2 plane, (c) the three "
            "referent encodings of the same two coordinates (FA single angle, "
            "F2 product register, E-in entangled register), (d) all 24 "
            "referents coloured by the E-in concurrence under the eps-map, "
            "(e) the circuit as run, with the measured mean-wire-purity trace"),
        "encoding": {
            "panel_b_x": "p1", "panel_b_y": "p2",
            "panel_b_colour": "identity only -- 2 categorical slots for the two "
                              "story referents, all other referents muted grey",
            "panel_c_status": "dashed border + written label = pre-registered "
                              "and NOT YET RUN; solid = as run at exp69",
            "panel_d_colour": "sequential single-hue blue ramp, concurrence "
                              "0 -> 1 under the eps-map",
            "panel_d_identity": "the two story referents carry a neutral ink "
                                "ring and a direct Arabic label, NOT their "
                                "categorical hue -- a magnitude fill and an "
                                "identity fill cannot share a mark",
            "panel_e_annotation": "measured mean wire purity per stage",
            "panel_d_mark_edge": "every dot carries a 0.6px INK_2 hairline "
                                 "edge. This is the contrast relief for the "
                                 "light end of the sequential ramp on a near-"
                                 "white surface, and it is what keeps "
                                 "overlapping dots separable; it is not a "
                                 "border used in place of a gap",
            "anti_patterns_checked": "dataviz references/anti-patterns.md, "
                                     "read against the finished figure: no "
                                     "dual axis, colour follows the entity "
                                     "across panels, 2 categorical hues, "
                                     "palette validated by script, one-hue "
                                     "sequential with a scale legend, no "
                                     "status colours, hairline solid grid, "
                                     "selective direct labels (never a number "
                                     "on every point), table view shipped, no "
                                     "texture, monospace only where numbers "
                                     "align in columns",
        },
        "palette": {
            "surface": SURFACE,
            "referents": {R1: S1, R2: S2},
            "sequential": {"hue": "blue", "steps_100_to_700": SEQ_STEPS},
            "validator": ('node scripts/validate_palette.js "#2a78d6,#eb6834" '
                          "--mode light --pairs all -> ALL CHECKS PASS "
                          "(CVD dE 24.7 protan, normal-vision dE 33.6, both "
                          "slots >= 3:1 on the surface)"),
        },
        "sources": {
            "angles_coordinates_concurrences": F_DIAG,
            "frozen_vectors_and_FA_angles": F_EMB,
            "story_item": F_STORIES,
            "purity_trace_and_parameter_count": F_RESULTS,
            "kernel_factorisation_check": F_KERNEL,
            "dataset_sha256_of_story_file":
                res["datasets"]["A"]["sha256"],
            "embeddings_sha256": emb["embeddings_sha256"],
        },
        "panel_a_story": {
            "source": F_STORIES, "item_id": ITEM_ID, "twin_id": TWIN_ID,
            "split": item["split"], "T": item["T"],
            "qtype": item["qtype"], "query_index": item["query_index"],
            "queried_verb": item["queried_verb"],
            "category": stories["meta"]["category"]["key"],
            "category_gloss": stories["meta"]["category"]["gloss"],
            "cast": [r["name"] for r in item["referents"]],
            "bearer_in_cast": R1, "nonbearer_in_cast": R2,
            "glosses_shown": {R1: gloss[R1], R2: gloss[R2]},
            "gloss_source": "%s / %s meta.category.glosses (committed "
                            "generator manifests)" % (F_STORIES, F_STORIES_B),
            "events": [{"verb": ev["verb"], "agent": ev["agent"],
                        "patient": ev["patient"], "surface_order": of,
                        "sentence": s}
                       for ev, of, s in zip(item["events"],
                                            item["order_flags"],
                                            item["sentences"])],
            "question_surface": item["question"]["surface"],
            "answer": item["answer"],
            "twin_answer": twin["answer"],
            "twin_construction": "agent and patient exchanged INSIDE the "
                                 "queried event only; the other event is "
                                 "byte-identical and the question surface is "
                                 "identical, so any function of the cast alone "
                                 "scores exactly 0.500 (doc 55 SS4.2)",
        },
        "panel_b_coordinates": {
            "formula": "p_k(r) = PC_k(r) / max_{r' in the 24 humans} |PC_k(r')|"
                       "  in [-1, 1]; PCA over all 44 referents, sign-pinned "
                       "(doc 55 SS4.4)",
            "source": F_DIAG + " (per_noun p1, p2)",
            "n_points": len(rows),
            "story_referents": {n: {"p1": rows[n]["p1"], "p2": rows[n]["p2"]}
                                for n in (R1, R2)},
            "FA_provenance_note": "the FA angle shown in panel (c) is "
                                  "atan2(y, x) on the frozen UNIT-NORMED exp42 "
                                  "vec2 plane, not atan2(p2, p1) on the plane "
                                  "plotted here; the two register maps do use "
                                  "p as plotted",
        },
        "panel_c_encodings": [
            {"variant": "FA -- single angle, one wire per referent",
             "status": "as run at exp69 (arms A0, R1-A, R2-A)",
             "formula": "phi(r) = atan2(y, x) on the frozen exp42 vec2",
             "source": F_EMB + " angle (identical to " + F_DIAG +
                       " per_noun.frozen_angle)",
             "values": {n: rows[n]["frozen_angle_FA"] for n in (R1, R2)}},
            {"variant": "F2 -- product register, two wires, two RY, no link",
             "status": "as run at exp69 (arms R1-W, R2-W)",
             "formula": "theta_k = (p_k + 1) * pi   "
                        "(instrument_sb_bounds.SCALINGS['full'])",
             "source": "recomputed here from " + F_DIAG + " per_noun p1, p2",
             "values": {n: {"theta_1": rows[n]["theta_registered"][0],
                            "theta_2": rows[n]["theta_registered"][1],
                            "concurrence": rows[n][
                                "concurrence_registered_map"]}
                        for n in (R1, R2)},
             "dead_zone_note": "%s has p1 = %.10f, so theta_1 = %.2e and its "
                               "concurrence under this map is %.2e -- a "
                               "normalisation extreme on a sin dead zone, "
                               "structural, not semantic"
                               % (R2, rows[R2]["p1"],
                                  rows[R2]["theta_registered"][0],
                                  rows[R2]["concurrence_registered_map"])},
            {"variant": "E-in -- entangled register, RY on q0 + "
                        "controlled-RY q0->q1",
             "status": "PRE-REGISTERED for doc 56 -- NOT YET RUN; no accuracy "
                       "is claimed for it on this figure",
             "formula": "theta_k = p_k * (pi - 2 eps) / 2 + pi/2, eps = pi/6 "
                        "-> theta in [pi/6, 5pi/6]",
             "state": diag["e_in_form"],
             "source": F_DIAG + " (eps_map; concurrences recomputed here and "
                                "asserted equal to the committed values to "
                                "< 1e-12)",
             "values": {n: {"theta_1": rows[n]["theta_eps"][0],
                            "theta_2": rows[n]["theta_eps"][1],
                            "concurrence": rows[n]["concurrence_eps_map"]}
                        for n in (R1, R2)},
             "design_consequence": diag["design_consequence"]},
        ],
        "panel_d_entanglement": {
            "quantity": "concurrence of one referent's own two-qubit register",
            "closed_form": "C(r) = |sin theta_1| * |sin(theta_2 / 2)|",
            "map": diag["eps_map"],
            "source": F_DIAG,
            "n_points": len(rows),
            "eps_map_stats": {"min": e["min"], "median": e["median"],
                              "max": e["max"],
                              "n_below_0.05": e["n_below_0.05"],
                              "argmin": lo, "argmax": hi},
            "registered_map_stats": {"min": rmap["min"],
                                     "median": rmap["median"],
                                     "max": rmap["max"],
                                     "n_below_0.05": rmap["n_below_0.05"],
                                     "mechanism": rmap["mechanism"]},
            "story_referents": {
                n: {"concurrence_eps_map": rows[n]["concurrence_eps_map"],
                    "rank_of_24_ascending": sorted(
                        rows, key=lambda m: rows[m]["concurrence_eps_map"]
                    ).index(n) + 1}
                for n in (R1, R2)},
            "all_24": {n: {"p1": rows[n]["p1"], "p2": rows[n]["p2"],
                           "concurrence_eps_map": rows[n][
                               "concurrence_eps_map"],
                           "concurrence_registered_map": rows[n][
                               "concurrence_registered_map"],
                           "theta_eps": rows[n]["theta_eps"],
                           "theta_registered": rows[n]["theta_registered"],
                           "frozen_angle_FA": rows[n]["frozen_angle_FA"]}
                       for n in rows},
        },
        "panel_e_circuit": {
            "arm_drawn": "R2-W (T = 2, F2 register, 4 wires)",
            "n_params": n_params,
            "n_params_source": F_RESULTS + " arms.R2-W.n_params",
            "stages": [
                "1 intro -- RY(theta_1) on wire 2w and RY(theta_2) on wire "
                "2w+1, once per referent, before event 1",
                "2 event 1 block -- the verb's trained 4-wire block on the "
                "agent register followed by the patient register",
                "3 event 2 block -- the same six blocks, reused, on the same "
                "persistent referent wires",
                "4 inverse question block -- block(queried verb) applied as "
                "its adjoint on the queried wires",
                "5 readout -- two shared trained effects Q_pos, Q_neg applied "
                "as adjoints, then P(queried wires == |0000>), softmax over "
                "log p; binary",
            ],
            "code_of_record": "exp68_run.F2Model.story_values (register arms) "
                              "/ exp42_models.QuantumStoryModel.story_values "
                              "(angle arms)",
            "purity_trace": {
                "quantity": "mean single-wire purity Tr rho^2 over the 4 wires",
                "source": F_RESULTS +
                          " purity_contrast_doc55_SS19_4.R2-W.cells."
                          "q2_correct.mean_wire_purity (n = 134)",
                "status": "DESCRIPTIVE -- doc 55 SS18.3 / SS19.4; no criterion "
                          "reads it",
                "after_intro": purity["after_intro"],
                "after_event1": purity["after_event1"],
                "after_event2": purity["after_event2"],
                "at_readout": purity["at_readout"],
                "mean_abs_gap_readout_minus_after_event1_q1": purity["gap_q1"],
                "mean_abs_gap_readout_minus_after_event1_q2": purity["gap_q2"],
                "gap_source": F_RESULTS + " purity_contrast_doc55_SS19_4.R2-W."
                                          "readout_minus_after_event1_abs "
                                          "(n = 150 per cell)",
            },
        },
        "footer_numbers": {
            "kernel_factorisation_claim": kern["claim"],
            "max_abs_deviation_sim_complex64":
                kern["max_abs_deviation_sim_c64_overall"],
            "max_abs_deviation_float64":
                kern["max_abs_deviation_float64_overall"],
            "pairs_per_row": 200,
            "n_qubits_rows": [r["n_qubits"] for r in kern["rows"]],
            "source": F_KERNEL,
        },
        "caveats_shipped_on_the_figure": [
            "product-state encodings are classically simulable; the kernel "
            "factorisation check is committed and quoted on the figure",
            "E-in is pre-registered for doc 56 and has NOT been run; no "
            "accuracy is claimed for it",
            "no exp69 accuracy is displayed on this figure at all -- it is a "
            "mechanism figure, and the results figure is "
            "fig_capacity_validation",
            "the purity trace is DESCRIPTIVE; no registered criterion reads it",
            "the FA angle is atan2 on the frozen unit-normed vec2 plane, not "
            "on the p-plane drawn in panels (b) and (d)",
            "simulation throughout; no quantum-advantage reading is licensed",
        ],
        "arabic_rendering": {
            "method": "presentation forms FE70-FEFC derived from unicodedata "
                      "decompositions, standard dual/right joining with the "
                      "lam-alef ligature, then reversed into visual order",
            "font": AR_FONT,
            "svg_fonttype": matplotlib.rcParams["svg.fonttype"],
        },
    }
    with open(os.path.join(HERE, JSON_OUT), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    render()
    print("wrote %s, %s, %s" % (SVG_OUT, PNG_OUT, JSON_OUT))
