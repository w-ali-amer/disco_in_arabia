# -*- coding: utf-8 -*-
"""
generate_figures.py
-------------------
Creates all three figures for the QNLP.ai 2026 submission.

  Figure 1 : SVO vs VSO pregroup string diagrams + IQP circuit schematics
  Figure 2 : Word-order benchmark results – bar chart (all methods)
  Figure 3 : Learning curves – AraVec vs QFM L1

Output:   figures/fig1_pregroup_circuits.pdf  (.png)
          figures/fig2_wordorder_results.pdf   (.png)
          figures/fig3_learning_curves.pdf     (.png)

Run inside the venv:
  source qiskit_lambeq_env/bin/activate
  python generate_figures.py
"""

import matplotlib
matplotlib.use('Agg')          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import json
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
FIG_DIR = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    'chance':    '#d62728',   # red   – L0 / below-chance result
    'main':      '#2ca02c',   # green – L1 main result
    'quantum':   '#1f77b4',   # blue  – other quantum
    'classical': '#7f7f7f',   # grey  – classical baselines
    'arabert':   '#17becf',   # teal  – AraBERT
    'swap':      '#ff7f0e',   # orange – SWAP gate highlight
    'wire':      '#222222',   # near-black for diagram wires
    'box':       '#f0f0f0',   # light grey for word boxes
    'cup':       '#444444',   # cup / arc colour
}

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — PREGROUP DIAGRAMS + IQP CIRCUIT SCHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

def _word_box(ax, x0, x1, y0, y1, label, sublabel='', fc=C['box']):
    """Draw a labelled word box."""
    rect = FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                          boxstyle='round,pad=0.05',
                          facecolor=fc, edgecolor='#555555', linewidth=1.2)
    ax.add_patch(rect)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    ax.text(cx, cy + 0.12*(y1-y0), label,
            ha='center', va='center', fontsize=9, fontweight='bold')
    if sublabel:
        ax.text(cx, cy - 0.22*(y1-y0), sublabel,
                ha='center', va='center', fontsize=7.5, color='#555555',
                style='italic')


def _wire(ax, x, y_top, y_bot, **kw):
    """Draw a vertical wire segment."""
    defaults = dict(color=C['wire'], lw=1.8, solid_capstyle='round')
    defaults.update(kw)
    ax.plot([x, x], [y_top, y_bot], **defaults)


def _cup(ax, x1, x2, y_top, depth=None, color=C['cup'], lw=1.8):
    """Draw a pregroup cup (downward parabolic arc) between x1 and x2."""
    if depth is None:
        depth = abs(x2 - x1) * 0.45
    cx = (x1 + x2) / 2
    t  = np.linspace(0, 1, 200)
    # Cubic Bezier: goes down then back up
    p0 = np.array([x1, y_top])
    p1 = np.array([x1, y_top - depth])
    p2 = np.array([x2, y_top - depth])
    p3 = np.array([x2, y_top])
    pts = (np.outer((1-t)**3, p0) + np.outer(3*(1-t)**2*t, p1) +
           np.outer(3*(1-t)*t**2, p2) + np.outer(t**3, p3))
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle='round')


def _type_label(ax, x, y, label, color='#333333', size=8.5):
    ax.text(x, y, label, ha='center', va='center',
            fontsize=size, color=color,
            bbox=dict(fc='white', ec='none', pad=1))


def _output_label(ax, x, y, label='$s$', size=11):
    ax.text(x, y, label, ha='center', va='center',
            fontsize=size, color=C['main'], fontweight='bold')


def _swap_symbol(ax, x1, x2, y_top, y_bot, lw=2.2):
    """Draw SWAP as two crossing wires (string diagram style)."""
    mid_y = (y_top + y_bot) / 2
    gap   = 0.08 * abs(x2 - x1)  # gap to show over/under

    # Wire 1: x1 → x2  (goes over)
    ax.plot([x1, x2], [y_top, y_bot], color=C['swap'], lw=lw+0.5,
            solid_capstyle='round', zorder=4)

    # Wire 2: x2 → x1  (goes under – drawn in two segments with white gap)
    # Upper segment
    t_break_top = 0.5 + gap / abs(y_bot - y_top)
    t_break_bot = 0.5 - gap / abs(y_bot - y_top)
    break_x_top = x2 + (x1 - x2) * t_break_top
    break_x_bot = x2 + (x1 - x2) * t_break_bot
    break_y_top = y_top + (y_bot - y_top) * t_break_top
    break_y_bot = y_top + (y_bot - y_top) * t_break_bot

    ax.plot([x2, break_x_top], [y_top, break_y_top],
            color=C['swap'], lw=lw, solid_capstyle='round', zorder=3)
    ax.plot([break_x_bot, x1], [break_y_bot, y_bot],
            color=C['swap'], lw=lw, solid_capstyle='round', zorder=3)

    # "SWAP" label
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, mid_y + 0.25, 'SWAP', ha='center', va='bottom',
            fontsize=7.5, color=C['swap'], fontweight='bold',
            bbox=dict(fc='#fff8f0', ec=C['swap'], pad=2, boxstyle='round,pad=0.2'))


def _draw_svo_diagram(ax):
    """SVO pregroup string diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # ── word boxes (y=8–10) ──────────────────────────────────────────────────
    _word_box(ax, 0.2, 2.3, 8.0, 9.8, 'al-ṭālib',  'Subject')
    _word_box(ax, 2.8, 7.2, 8.0, 9.8, 'kataba',     'Verb  (SVO type)')
    _word_box(ax, 7.7, 9.8, 8.0, 9.8, 'al-dars',    'Object')

    # ── type labels under boxes ──────────────────────────────────────────────
    _type_label(ax, 1.25, 7.7,  r'$n$')
    _type_label(ax, 3.8,  7.7,  r'$n^r$')
    _type_label(ax, 5.0,  7.7,  r'$s$',   color=C['main'])
    _type_label(ax, 6.2,  7.7,  r'$n^l$')
    _type_label(ax, 8.75, 7.7,  r'$n$')

    # ── wires from boxes down to cups ────────────────────────────────────────
    wire_xs = [1.25, 3.8, 5.0, 6.2, 8.75]
    for x in wire_xs:
        _wire(ax, x, 7.7, 5.5)

    # ── cups  ────────────────────────────────────────────────────────────────
    _cup(ax, 1.25, 3.8, 5.5, depth=1.5)   # n ─── n^r
    _cup(ax, 6.2,  8.75, 5.5, depth=1.5)  # n^l ─ n

    # ── output wire (s) ──────────────────────────────────────────────────────
    _wire(ax, 5.0, 5.5, 2.5)
    _output_label(ax, 5.0, 2.1)

    ax.set_title('SVO — standard reduction', fontsize=10, pad=4)

    # annotation: no swap
    ax.text(5.0, 6.5, 'no SWAP needed', ha='center', va='center',
            fontsize=7.5, color='#888888', style='italic')


def _draw_vso_diagram(ax):
    """VSO pregroup string diagram with Swap."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # ── word boxes ────────────────────────────────────────────────────────────
    _word_box(ax, 0.2, 5.2, 8.0, 9.8, 'kataba',     'Verb  (VSO type)')
    _word_box(ax, 5.6, 7.4, 8.0, 9.8, 'al-ṭālib',  'Subject')
    _word_box(ax, 7.8, 9.8, 8.0, 9.8, 'al-dars',    'Object')

    # ── type labels  ─────────────────────────────────────────────────────────
    _type_label(ax, 1.2,  7.7, r'$s$',   color=C['main'])
    _type_label(ax, 2.8,  7.7, r'$n^l$')
    _type_label(ax, 4.6,  7.7, r'$n^l$')
    _type_label(ax, 6.5,  7.7, r'$n$')
    _type_label(ax, 8.8,  7.7, r'$n$')

    # ── wires from boxes down to SWAP level  ──────────────────────────────────
    top_xs   = [1.2, 2.8, 4.6, 6.5, 8.8]
    swap_y_t = 6.2   # wire enters swap
    swap_y_b = 4.8   # wire exits  swap

    for x in top_xs:
        _wire(ax, x, 7.7, swap_y_t)

    # ── SWAP between x=4.6 (n^l) and x=6.5 (n[subject]) ─────────────────────
    _wire(ax, 1.2, swap_y_t, swap_y_b)   # s wire: straight through
    _wire(ax, 2.8, swap_y_t, swap_y_b)   # n^l wire: straight through
    _wire(ax, 8.8, swap_y_t, swap_y_b)   # n[obj] wire: straight through

    _swap_symbol(ax, 4.6, 6.5, swap_y_t, swap_y_b)

    # ── After SWAP: wire 4.6 now carries n[subject], wire 6.5 carries n^l ────
    # Wires from SWAP bottom down to cup level
    post_xs = [1.2, 2.8, 4.6, 6.5, 8.8]   # same x, but swapped content
    cup_y   = 4.8
    for x in post_xs:
        _wire(ax, x, swap_y_b, 3.6)

    # Type labels after swap
    _type_label(ax, 1.2,  4.2, r'$s$',   color=C['main'])
    _type_label(ax, 2.8,  4.2, r'$n^l$')
    _type_label(ax, 4.6,  4.2, r'$n$',   color='#555555')
    _type_label(ax, 6.5,  4.2, r'$n^l$')
    _type_label(ax, 8.8,  4.2, r'$n$',   color='#555555')

    # ── cups  ────────────────────────────────────────────────────────────────
    _cup(ax, 2.8,  4.6, 3.6, depth=1.2)   # n^l ─── n(subject)
    _cup(ax, 6.5,  8.8, 3.6, depth=1.2)   # n^l ─── n(object)

    # ── output wire ──────────────────────────────────────────────────────────
    _wire(ax, 1.2, 3.6, 2.5)
    _output_label(ax, 1.2, 2.1)

    ax.set_title('VSO — Swap required', fontsize=10, pad=4,
                 color=C['swap'])

    # annotation
    ax.text(5.5, 5.4, 'SWAP is the unique\nminimal reduction',
            ha='center', va='center', fontsize=7.5,
            color=C['swap'], style='italic')


# ── IQP circuit schematics ────────────────────────────────────────────────────

def _draw_circuit_svo(ax):
    """Simplified IQP L1 circuit for SVO sentence."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 3.3)
    ax.axis('off')

    qubit_ys  = [2.5, 1.5, 0.5]
    labels    = [r'$q_\mathrm{subj}$', r'$q_s$', r'$q_\mathrm{obj}$']
    line_kw   = dict(color='#333333', lw=1.5, solid_capstyle='round')

    # Qubit lines
    for y in qubit_ys:
        ax.plot([0.5, 9.5], [y, y], **line_kw)

    # Labels
    for y, lbl in zip(qubit_ys, labels):
        ax.text(0.3, y, lbl, ha='right', va='center', fontsize=8)

    # H gates  (x=1.5)
    for y in qubit_ys:
        _gate_box(ax, 1.5, y, 'H', fc='#ddeeff')

    # Rz gates (x=3.0)
    for y in qubit_ys:
        _gate_box(ax, 3.0, y, r'$R_z$', fc='#eeffdd', w=0.7)

    # CZ gates:  subj↔s  at x=5.5,  s↔obj at x=7.0
    _cz_gate(ax, 5.5, qubit_ys[0], qubit_ys[1])
    _cz_gate(ax, 7.0, qubit_ys[1], qubit_ys[2])

    # Measurement
    for y in qubit_ys:
        _measure(ax, 9.2, y)

    ax.set_title('SVO circuit  (IQP L1)', fontsize=9, pad=3)
    ax.text(6.25, 0.05, '2 CZ gates', ha='center', va='bottom',
            fontsize=7.5, color='#666666')


def _draw_circuit_vso(ax):
    """Simplified IQP L1 circuit for VSO sentence (with SWAP)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 3.3)
    ax.axis('off')

    qubit_ys = [2.5, 1.5, 0.5]
    labels   = [r'$q_s$', r'$q_\mathrm{subj}$', r'$q_\mathrm{obj}$']
    line_kw  = dict(color='#333333', lw=1.5, solid_capstyle='round')

    for y in qubit_ys:
        ax.plot([0.5, 9.5], [y, y], **line_kw)

    for y, lbl in zip(qubit_ys, labels):
        ax.text(0.3, y, lbl, ha='right', va='center', fontsize=8)

    # H gates
    for y in qubit_ys:
        _gate_box(ax, 1.5, y, 'H', fc='#ddeeff')

    # SWAP gate  (x=3.0) between q_subj and q_obj
    _swap_gate(ax, 3.0, qubit_ys[1], qubit_ys[2])

    # Rz gates  (x=4.5)
    for y in qubit_ys:
        _gate_box(ax, 4.5, y, r'$R_z$', fc='#eeffdd', w=0.7)

    # CZ: only one CZ (fewer than SVO after compilation)  at x=6.5
    _cz_gate(ax, 6.5, qubit_ys[0], qubit_ys[1])

    # Measurement
    for y in qubit_ys:
        _measure(ax, 9.2, y)

    ax.set_title('VSO circuit  (IQP L1, post-compilation)', fontsize=9, pad=3,
                 color=C['swap'])
    ax.text(6.5, 0.05, '1 CZ gate  ← different topology',
            ha='center', va='bottom', fontsize=7.5, color=C['swap'])


def _gate_box(ax, x, y, label, fc='#eeeeee', w=0.55, h=0.55):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.04',
                          facecolor=fc, edgecolor='#555555', lw=1.0, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=7.5, zorder=4)


def _cz_gate(ax, x, y1, y2):
    """Draw a CZ gate (two filled dots connected by vertical line)."""
    ax.plot([x, x], [y1, y2], color='#333333', lw=1.5, zorder=2)
    for y in [y1, y2]:
        ax.plot(x, y, 'o', color='#333333', ms=7, zorder=3)


def _swap_gate(ax, x, y1, y2):
    """Draw a SWAP gate (two × symbols connected by vertical line)."""
    ax.plot([x, x], [y1, y2], color=C['swap'], lw=1.8, zorder=2)
    sz = 0.18
    for y in [y1, y2]:
        ax.plot([x-sz, x+sz], [y-sz, y+sz], color=C['swap'], lw=2.0, zorder=3)
        ax.plot([x-sz, x+sz], [y+sz, y-sz], color=C['swap'], lw=2.0, zorder=3)
    ax.text(x, (y1+y2)/2, 'SWAP', ha='left', va='center',
            fontsize=7, color=C['swap'], fontweight='bold',
            bbox=dict(fc='#fff8f0', ec='none', pad=1))


def _measure(ax, x, y, size=0.5):
    """Draw a simple measurement symbol (arc + arrow)."""
    theta = np.linspace(np.pi, 0, 60)
    mx = x + size/2 * np.cos(theta)
    my = y - size/4 + size/2 * np.sin(theta) * 0.6
    ax.plot(mx, my, color='#555555', lw=1.0, zorder=3)
    ax.annotate('', xy=(x + size/2 * 0.6, y + size/4 * 0.5),
                xytext=(x, y - size/4 + 0.05),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
                zorder=3)


def make_figure1():
    fig = plt.figure(figsize=(13, 9))

    # 2×2 grid: top row = string diagrams, bottom row = circuits
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.15,
                          height_ratios=[1.3, 0.8])

    ax_svo_diag = fig.add_subplot(gs[0, 0])
    ax_vso_diag = fig.add_subplot(gs[0, 1])
    ax_svo_circ = fig.add_subplot(gs[1, 0])
    ax_vso_circ = fig.add_subplot(gs[1, 1])

    _draw_svo_diagram(ax_svo_diag)
    _draw_vso_diagram(ax_vso_diag)
    _draw_circuit_svo(ax_svo_circ)
    _draw_circuit_vso(ax_vso_circ)

    # Column headers
    fig.text(0.26, 0.97,
             'SVO: "al-ṭālib kataba al-dars"\n(The student wrote the lesson)',
             ha='center', va='top', fontsize=10, fontweight='bold',
             color='#333333')
    fig.text(0.74, 0.97,
             'VSO: "kataba al-ṭālib al-dars"\n(Wrote the student the lesson)',
             ha='center', va='top', fontsize=10, fontweight='bold',
             color=C['swap'])

    # Divider labels
    fig.text(0.01, 0.82, 'Pregroup\ndiagram', ha='left', va='center',
             fontsize=9, color='#555555', rotation=90)
    fig.text(0.01, 0.33, 'IQP L1\ncircuit', ha='left', va='center',
             fontsize=9, color='#555555', rotation=90)

    # Key message
    fig.text(0.5, 0.01,
             'Same three words — different pregroup type assignment for the verb → '
             'structurally distinct circuits.\n'
             'The VSO Swap is derivable by necessity from the pregroup type mathematics; '
             'it produces different gate topology regardless of vocabulary.',
             ha='center', va='bottom', fontsize=8, color='#444444', style='italic',
             wrap=True)

    fig.savefig(FIG_DIR / 'fig1_pregroup_circuits.pdf')
    fig.savefig(FIG_DIR / 'fig1_pregroup_circuits.png')
    print("  ✓ Figure 1 saved.")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — WORD ORDER RESULTS BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure2():
    # Data — all verified against taskA_wordorder.json
    methods = [
        'AraVec\n(bag-of-words)',
        'Topology-only\n(0 params)',
        'QFM IQP L0\n(no entanglement)',
        'SPSA IQP L1\n(full quantum training)',
        'QFM IQP L2\n(2 entangling layers)',
        'QFM IQP L1\n(1 entangling layer)',
        'AraBERT\n(frozen CLS)',
        'AraBERT\n(fine-tuned)',
    ]
    accs = [12.8, 35.8, 50.0, 53.8, 61.8, 64.9, 86.1, 100.0]
    # Topology-only: raw 35.8% (symmetric = 64.2%, same as L1 — note in annotation)
    # CI for QFM L1: [62.8, 66.3]
    ci_lo = [None, None, None, None, None, 62.8, None, None]
    ci_hi = [None, None, None, None, None, 66.3, None, None]

    colors = [
        C['chance'],     # AraVec — below chance
        C['classical'],  # Topology-only
        C['chance'],     # QFM L0 — chance
        C['quantum'],    # SPSA
        C['quantum'],    # QFM L2
        C['main'],       # QFM L1 — main result
        C['arabert'],    # AraBERT frozen
        C['arabert'],    # AraBERT fine-tuned
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    y_pos = np.arange(len(methods))
    bars  = ax.barh(y_pos, accs, color=colors, height=0.62,
                    edgecolor='white', linewidth=0.5, alpha=0.88)

    # Error bar for QFM L1 (index 5)
    idx = 5
    ax.errorbar(accs[idx], y_pos[idx],
                xerr=[[accs[idx]-ci_lo[idx]], [ci_hi[idx]-accs[idx]]],
                fmt='none', color='#1a6e1a', capsize=4, lw=2, capthick=2,
                zorder=5)

    # Chance line
    ax.axvline(50, color='#999999', lw=1.2, ls='--', alpha=0.8, zorder=1)
    ax.text(50.5, -0.7, 'chance (50%)', va='top', ha='left',
            fontsize=8, color='#888888', style='italic')

    # Value labels on bars
    for i, (acc, bar) in enumerate(zip(accs, bars)):
        x_label = acc + 1.5
        if acc > 90:
            x_label = acc - 2
            ha = 'right'
            color = 'white'
        else:
            ha = 'left'
            color = '#333333'
        ax.text(x_label, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', ha=ha,
                fontsize=8.5, color=color, fontweight='bold')

    # Key annotations
    # AraVec: below chance
    ax.annotate('vocabulary fails on\nmatched pairs',
                xy=(12.8, y_pos[0]),
                xytext=(22, y_pos[0] + 0.7),
                fontsize=7.5, color=C['chance'],
                arrowprops=dict(arrowstyle='->', color=C['chance'],
                                connectionstyle='arc3,rad=0.3', lw=1.2))

    # Topology-only: note symmetric correction
    ax.text(37.5, y_pos[1],
            'raw 35.8%\n(sym: 64.2%)',
            va='center', ha='left', fontsize=7, color='#555555', style='italic')

    # L0: theorem
    ax.annotate('theorem:\nzero-variance\nconstant output',
                xy=(50.0, y_pos[2]),
                xytext=(58, y_pos[2] + 1.0),
                fontsize=7.5, color=C['chance'], style='italic',
                arrowprops=dict(arrowstyle='->', color=C['chance'],
                                connectionstyle='arc3,rad=-0.3', lw=1.2))

    # L1: main result
    ax.annotate('+15pp from\nentanglement',
                xy=(64.9, y_pos[5]),
                xytext=(72, y_pos[5] + 0.9),
                fontsize=8, color=C['main'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['main'],
                                connectionstyle='arc3,rad=-0.2', lw=1.5))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=8.5)
    ax.set_xlabel('Accuracy (%)', fontsize=10)
    ax.set_xlim(0, 112)
    ax.set_ylim(-1.2, len(methods) - 0.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.tick_params(axis='x', labelsize=8.5)

    ax.set_title(
        'Word Order Classification: SVO vs VSO (120 matched pairs)\n'
        'Vocabulary fully controlled — any model exceeding 50% uses structural information',
        fontsize=10, pad=8)

    # Legend
    legend_patches = [
        mpatches.Patch(color=C['chance'],    label='at/below chance — order-blind'),
        mpatches.Patch(color=C['quantum'],   label='quantum (other depths)'),
        mpatches.Patch(color=C['main'],      label='QFM L1 — main result'),
        mpatches.Patch(color=C['classical'], label='topology-only'),
        mpatches.Patch(color=C['arabert'],   label='AraBERT (classical upper bound)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right',
              fontsize=7.5, framealpha=0.9, ncol=1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_wordorder_results.pdf')
    fig.savefig(FIG_DIR / 'fig2_wordorder_results.png')
    print("  ✓ Figure 2 saved.")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — LEARNING CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure3():
    # Data — verified against learning_curves.json
    n_vals = [5, 10, 20, 40]

    aravec_mean = [0.460, 0.419, 0.340, 0.185]
    aravec_std  = [0.007, 0.010, 0.028, 0.032]

    qfm_l1_mean = [0.563, 0.582, 0.600, 0.668]
    qfm_l1_std  = [0.090, 0.063, 0.052, 0.058]

    arabert_mean = [0.643, 0.681, 0.686, 0.810]
    arabert_std  = [0.089, 0.042, 0.082, 0.051]

    # Convert to percent
    def pct(lst): return [v * 100 for v in lst]
    def pct_err(lst): return [v * 100 for v in lst]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    lw = 2.2
    ms = 8

    # AraVec — red, dashed
    ax.plot(n_vals, pct(aravec_mean), 'o--',
            color=C['chance'], lw=lw, ms=ms, label='AraVec (bag-of-words)',
            zorder=4)
    ax.fill_between(n_vals,
                    [m - s for m, s in zip(pct(aravec_mean), pct_err(aravec_std))],
                    [m + s for m, s in zip(pct(aravec_mean), pct_err(aravec_std))],
                    color=C['chance'], alpha=0.12)

    # QFM L1 — green, solid
    ax.plot(n_vals, pct(qfm_l1_mean), 's-',
            color=C['main'], lw=lw, ms=ms, label='QFM IQP L1 (main result)',
            zorder=4)
    ax.fill_between(n_vals,
                    [m - s for m, s in zip(pct(qfm_l1_mean), pct_err(qfm_l1_std))],
                    [m + s for m, s in zip(pct(qfm_l1_mean), pct_err(qfm_l1_std))],
                    color=C['main'], alpha=0.12)

    # AraBERT frozen — teal, dotted
    ax.plot(n_vals, pct(arabert_mean), '^:',
            color=C['arabert'], lw=lw, ms=ms, label='AraBERT (frozen CLS)',
            zorder=3, alpha=0.7)
    ax.fill_between(n_vals,
                    [m - s for m, s in zip(pct(arabert_mean), pct_err(arabert_std))],
                    [m + s for m, s in zip(pct(arabert_mean), pct_err(arabert_std))],
                    color=C['arabert'], alpha=0.08)

    # Chance line
    ax.axhline(50, color='#aaaaaa', lw=1.2, ls='--', alpha=0.8, zorder=1)
    ax.text(40.5, 50.8, 'chance', ha='right', va='bottom',
            fontsize=8, color='#999999', style='italic')

    # Arrows showing divergence
    ax.annotate('', xy=(40, pct(aravec_mean)[-1]),
                xytext=(40, pct(aravec_mean)[-2]),
                arrowprops=dict(arrowstyle='->', color=C['chance'], lw=1.5))
    ax.annotate('', xy=(40, pct(qfm_l1_mean)[-1]),
                xytext=(40, pct(qfm_l1_mean)[-2]),
                arrowprops=dict(arrowstyle='->', color=C['main'], lw=1.5))

    # End-point value labels
    ax.text(40.5, pct(aravec_mean)[-1],  f'{pct(aravec_mean)[-1]:.1f}%',
            va='center', ha='left', fontsize=8.5, color=C['chance'],
            fontweight='bold')
    ax.text(40.5, pct(qfm_l1_mean)[-1], f'{pct(qfm_l1_mean)[-1]:.1f}%',
            va='center', ha='left', fontsize=8.5, color=C['main'],
            fontweight='bold')

    # Divergence annotation
    ax.annotate('trajectories\ndiverge',
                xy=(20, (pct(aravec_mean)[2] + pct(qfm_l1_mean)[2]) / 2),
                xytext=(12, 53),
                fontsize=8, color='#555555', style='italic',
                arrowprops=dict(arrowstyle='->', color='#888888',
                                connectionstyle='arc3,rad=0.3', lw=1.0))

    # AraVec annotation
    ax.text(7, pct(aravec_mean)[0] + 2,
            'more data → worse\n(spurious anti-correlation)',
            ha='center', va='bottom', fontsize=7.5, color=C['chance'],
            style='italic', multialignment='center')

    # QFM annotation
    ax.text(25, pct(qfm_l1_mean)[2] + 4,
            'more data → better\n(structural encoding)',
            ha='center', va='bottom', fontsize=7.5, color=C['main'],
            style='italic', multialignment='center')

    ax.set_xlabel('Training examples per class (N)', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_xticks(n_vals)
    ax.set_xlim(3, 48)
    ax.set_ylim(5, 90)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.tick_params(labelsize=9)

    ax.set_title(
        'Learning Curves: Matched-pair Word Order (SVO vs VSO)\n'
        'AraVec degrades with more data; QFM L1 improves — opposite trajectories',
        fontsize=10, pad=8)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_learning_curves.pdf')
    fig.savefig(FIG_DIR / 'fig3_learning_curves.png')
    print("  ✓ Figure 3 saved.")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating figures …")
    make_figure1()
    make_figure2()
    make_figure3()
    print(f"\nAll figures written to  {FIG_DIR.resolve()}/")
    print("Files:")
    for f in sorted(FIG_DIR.iterdir()):
        kb = f.stat().st_size // 1024
        print(f"  {f.name:45s} {kb:5d} KB")
