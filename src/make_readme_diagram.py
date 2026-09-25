"""Static, professional architecture diagram (PNG, not animated) -- every
stage is always fully colored and complete, so there is no "paused
mid-animation" frame that can look unfinished on GitHub or in a screenshot.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT_DIR = "docs/assets"
PALETTE = ["#2E5EAA", "#D98E04", "#2E8B57", "#6A4C93", "#C0392B"]

STAGES = [
    ("Data Cleaning", "Hidden-missing unmask,\nrare-med consolidation"),
    ("EDA", "30+ charts,\nstatistical tests"),
    ("Feature Engineering", "Patient-grouped split,\nencoding, selection"),
    ("Model Training", "LightGBM, tuned,\ncalibrated, OOF threshold"),
    ("Live Serving", "Results dashboard +\nreal-time Case Checker"),
]

def tint(hex_color, alpha=0.12):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * (1 - alpha)); g = int(g + (255 - g) * (1 - alpha)); b = int(b + (255 - b) * (1 - alpha))
    return f"#{r:02X}{g:02X}{b:02X}"

N = len(STAGES)
X = [i * 2.5 for i in range(N)]
Y = 0.0
BOX_W, BOX_H = 2.05, 1.15

os.makedirs(OUT_DIR, exist_ok=True)
fig, ax = plt.subplots(figsize=(16.5, 5.0))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(-1.3, X[-1] + 1.3); ax.set_ylim(-1.3, 1.9)
ax.axis("off")

ax.text((X[0] + X[-1]) / 2, 1.55, "Hospital Readmission Prediction", ha="center", va="center",
        fontsize=17, fontweight="bold", color="#111827")
ax.text((X[0] + X[-1]) / 2, 1.18, "End-to-End Pipeline", ha="center", va="center",
        fontsize=11, color="#6B7280", style="italic")

for i in range(N - 1):
    arrow = FancyArrowPatch(
        (X[i] + BOX_W / 2 + 0.06, Y), (X[i + 1] - BOX_W / 2 - 0.06, Y),
        arrowstyle="-|>", mutation_scale=22, linewidth=2.4,
        color="#9AA3AF", zorder=1, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)

for i, (title, sub) in enumerate(STAGES):
    color = PALETTE[i]
    shadow = mpatches.FancyBboxPatch(
        (X[i] - BOX_W/2 + 0.05, Y - BOX_H/2 - 0.06), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.16", linewidth=0, facecolor="#00000012", zorder=2,
    )
    ax.add_patch(shadow)
    box = mpatches.FancyBboxPatch(
        (X[i] - BOX_W/2, Y - BOX_H/2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.16", linewidth=2.2,
        edgecolor=color, facecolor=tint(color), zorder=3,
    )
    ax.add_patch(box)

    badge = mpatches.Circle((X[i] - BOX_W/2 + 0.28, Y + BOX_H/2 - 0.24), 0.19,
                            facecolor=color, edgecolor="white", linewidth=1.6, zorder=5)
    ax.add_patch(badge)
    ax.text(X[i] - BOX_W/2 + 0.28, Y + BOX_H/2 - 0.24, str(i + 1), ha="center", va="center",
           fontsize=11, fontweight="bold", color="white", zorder=6)

    ax.text(X[i], Y + 0.20, title, ha="center", va="center", fontsize=11.8,
           fontweight="bold", color="#111827", zorder=4)
    ax.text(X[i], Y - 0.24, sub, ha="center", va="center", fontsize=8.3,
           color="#4B5563", zorder=4, linespacing=1.5)

ax.text((X[0] + X[-1]) / 2, -1.05,
        "Every prediction is traceable to the same trained, calibrated pipeline.",
        ha="center", va="center", fontsize=11.5, color="#4B5563", style="italic")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT_DIR}/architecture.png")
