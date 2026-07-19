#!/usr/bin/env python3
"""
Figure 1 — Controlled two-document source-discrimination task.
npj Digital Medicine R1 revision. Redrawn 2026-07-06 (house style, aligned).

Design: Lancet Digital Health 4-block schematic. ONE contrast carries the story —
authentic (teal) vs. tampered (coral) guideline. Everything else is neutral
infrastructure. Built in matplotlib so in-figure text is crisp (no AI garbling).

Color map (one meaning each):
  TEAL   = authentic / trusted source
  CORAL  = tampered / adversarial edit
  SLATE  = the model / agent
  GRAY   = neutral infrastructure, containers, inputs
Output: 600-dpi PNG + vector PDF into the working submission 05_Figures folder.
Canvas aspect fixed near 2.567 to match the original embed (4290x1671).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------- palette
TEAL      = "#2A7F8E"; TEAL_FILL   = "#E4F0F1"
CORAL     = "#C1584E"; CORAL_FILL  = "#F6E5E2"
SLATE     = "#3E5C76"; SLATE_FILL  = "#E7ECF1"
GRAY_EDGE = "#7A7A7A"; GRAY_FILL   = "#F1F1F1"
CONT_EDGE = "#B4B4B4"; CONT_FILL   = "#FBFBFB"
INK       = "#232323"; SUB         = "#5C5C5C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "figure.dpi": 100,
})

# ----------------------------------------------------------------------------- canvas
W, H = 12.85, 5.0
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

CARD_TOP = 4.28          # every block's cards share this top edge (grid)
CARD_BOT = 0.95          # and this bottom edge
MIDY     = 2.60          # arrow height between blocks
HDR_Y    = 4.66          # header / badge band, above the cards


def card(x, y, w, h, *, fill=GRAY_FILL, edge=GRAY_EDGE, lw=1.2, rad=0.10, z=2, dash=None):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rad}",
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       linestyle=(dash or "solid"), zorder=z, mutation_aspect=1)
    ax.add_patch(p)
    return p


def head(bx, n, text, *, size=11.5):
    """Numbered badge at the block's top-left, header left-aligned after it."""
    ax.add_patch(plt.Circle((bx + 0.16, HDR_Y), 0.155, facecolor=INK,
                 edgecolor="none", zorder=7))
    ax.text(bx + 0.16, HDR_Y, str(n), ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold", zorder=8)
    ax.text(bx + 0.42, HDR_Y, text, ha="left", va="center", fontsize=size,
            fontweight="bold", color=INK, zorder=6)


def label(cx, cy, text, *, size=9.3, color=INK, weight="normal", ha="center", va="center"):
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, zorder=6, linespacing=1.32)


def harrow(x0, x1, y=MIDY):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=15, lw=1.6, color=SUB, zorder=5, shrinkA=0, shrinkB=0))


def varrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=13, lw=1.5, color=SUB, zorder=5, shrinkA=0, shrinkB=0))


# ============================================================ BLOCK 1 — Inputs
b1x, b1w = 0.30, 2.38
head(b1x, 1, "Inputs")
card(b1x, 2.72, b1w, 1.56, fill=GRAY_FILL, edge=GRAY_EDGE)
label(b1x + b1w/2, 3.88, "Standardized\nclinical case", size=10.2, weight="bold")
label(b1x + b1w/2, 3.16, "n = 500 vignettes\n12 medical domains", size=8.8, color=SUB)
card(b1x, 0.95, b1w, 1.56, fill=TEAL_FILL, edge=TEAL)
label(b1x + b1w/2, 2.10, "Authentic\nguideline", size=10.2, weight="bold", color=TEAL)
label(b1x + b1w/2, 1.40, "Trusted sources\n(AHA, IDSA, NICE…)", size=8.8, color=SUB)
harrow(b1x + b1w + 0.08, 2.96)

# ================================================ BLOCK 2 — Adversarial pairing
b2x, b2w = 3.00, 4.34
card(b2x - 0.04, 0.50, b2w + 0.08, 3.92, fill=CONT_FILL, edge=CONT_EDGE,
     lw=1.1, rad=0.12, z=1, dash=(0, (5, 4)))
head(b2x, 2, "Adversarial pairing")
mx, mw = b2x + 0.08, b2w - 0.16
card(mx, 2.70, mw, 1.58, fill="#FFFFFF", edge=GRAY_EDGE, lw=1.0)
label(mx + mw/2, 4.02, "Single localized edit   ·   10 types / 4 categories",
      size=9.0, weight="bold")
label(mx + mw/2, 3.38,
      "Clinical safety:  missing warning · dosing · contraindication · allergy\n"
      "Semantic:  wrong population · authority mimicry · subtle inversion\n"
      "Injection:  prompt injection\n"
      "Metadata:  fabricated citation · outdated version",
      size=8.2, color=SUB)
varrow(b2x + b2w/2, 2.64, 2.42)
pw = (mw - 0.24) / 2.0
card(mx, 1.04, pw, 1.28, fill=TEAL_FILL, edge=TEAL, lw=1.2)
label(mx + pw/2, 1.90, "Authentic\nexcerpt", size=8.8, weight="bold", color=TEAL)
label(mx + pw/2, 1.30, "unchanged", size=7.6, color=SUB)
card(mx + pw + 0.24, 1.04, pw, 1.28, fill=CORAL_FILL, edge=CORAL, lw=1.2)
label(mx + pw + 0.24 + pw/2, 1.94, "Tampered\nexcerpt", size=8.8, weight="bold", color=CORAL)
label(mx + pw + 0.24 + pw/2, 1.30, "1 adversarial edit", size=7.6, color=SUB)
label(b2x + b2w/2, 0.72, "paired · presentation order randomized  (Tool A / Tool B)",
      size=7.9, color=INK)
harrow(b2x + b2w + 0.08, 7.64)

# ================================================ BLOCK 3 — Source selection
b3x, b3w = 7.68, 2.36
head(b3x, 3, "Source selection")
card(b3x, 2.34, b3w, 1.94, fill=SLATE_FILL, edge=SLATE, lw=1.2)
label(b3x + b3w/2, 3.94, "21 LLMs", size=11, weight="bold", color=SLATE)
label(b3x + b3w/2, 3.42, "reasoning · open · closed", size=8.2, color=SUB)
label(b3x + b3w/2, 2.86,
      "two function-calling tools\nguideline_a  ·  guideline_b\ncall both → structured choice",
      size=7.9, color=INK)
card(b3x, 0.95, b3w, 1.18, fill="#FFFFFF", edge=GRAY_EDGE, lw=1.0)
label(b3x + b3w/2, 1.74, "Captured per case", size=8.4, weight="bold")
label(b3x + b3w/2, 1.26, "selected tool · confidence\n· free-text rationale", size=7.9, color=SUB)
varrow(b3x + b3w/2, 2.30, 2.18)
harrow(b3x + b3w + 0.08, 10.34)

# ================================================ BLOCK 4 — Outcome measures
b4x, b4w = 10.38, 2.20
head(b4x, 4, "What we measure")
card(b4x, 0.95, b4w, 3.33, fill=GRAY_FILL, edge=GRAY_EDGE, lw=1.2)
for t, yy in zip(["Detection accuracy", "Position bias", "Confidence calibration",
                  "Safety-stratified\nfailure rate"], [3.74, 3.02, 2.30, 1.48]):
    ax.add_patch(plt.Circle((b4x + 0.30, yy + (0.16 if "\n" in t else 0.0)),
                 0.052, facecolor=SLATE, edgecolor="none", zorder=6))
    label(b4x + 0.50, yy, t, size=8.9, ha="left")

# ----------------------------------------------------------------------------- scale strip
ax.add_line(Line2D([0.30, W - 0.30], [0.42, 0.42], color="#D9D9D9", lw=1.0, zorder=1))
label(W/2, 0.24,
      "500 vignettes   ·   12 domains   ·   10 modification types (4 categories)"
      "   ·   21 LLMs   ·   10,500 evaluations",
      size=8.3, color=SUB)

# ----------------------------------------------------------------------------- export
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                      "npj_R1_FINAL_SUBMISSION_2026-07-04", "05_Figures"))
png = os.path.join(OUT, "Figure1.png")
pdf = os.path.join(OUT, "Figure1.pdf")
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06, facecolor="white")
print("wrote", png)
print("wrote", pdf)
