"""Animated architecture diagram, v2 -- deliberately different visual style
from the horizontal-box-row diagrams used in other projects: a vertical
zig-zag timeline with a curved connecting path and a rotating progress ring
around the active stage number.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

OUT_DIR = "docs/assets"
PALETTE = {"blue": "#2E5EAA", "green": "#2E8B57", "orange": "#D98E04", "purple": "#6A4C93", "red": "#C0392B"}
DIM_EDGE = "#DDE1E7"; DIM_TEXT = "#A3AAB5"; ON_TEXT = "#1F2937"; BG_OFF = "#FBFCFD"; SHADOW = "#E9ECEF"

def ease_in_out(t): return t * t * (3 - 2 * t)

def tint(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * (1 - alpha)); g = int(g + (255 - g) * (1 - alpha)); b = int(b + (255 - b) * (1 - alpha))
    return f"#{r:02X}{g:02X}{b:02X}"

STAGES = [
    ("Data Cleaning", "hidden-missing unmask, rare-med consolidation", PALETTE["blue"]),
    ("EDA", "30+ charts, statistical tests", PALETTE["orange"]),
    ("Feature Engineering", "patient-grouped split, encoding, selection", PALETTE["green"]),
    ("Model Training", "LightGBM, tuned, OOF threshold", PALETTE["purple"]),
    ("Evaluation", "SHAP, calibration, test-set report", PALETTE["red"]),
]
CAPTIONS = [
    "Cleaning raw data: unmasking hidden-missing codes, consolidating rare medications...",
    "Exploring patterns: 30+ charts, statistical significance tests...",
    "Building features: patient-grouped split, encoding, mutual-information selection...",
    "Training LightGBM: 5-fold group CV, hyperparameter search, OOF threshold...",
    "Evaluating once on the held-out test set: SHAP, calibration, error analysis.",
]

N = len(STAGES)
# Zig-zag vertical positions: alternate left / right down the page
Y = [-(i * 1.75) for i in range(N)]
X = [-2.4 if i % 2 == 0 else 2.4 for i in range(N)]
NODE_R = 0.85


def build_figure():
    fig, ax = plt.subplots(figsize=(9, 12))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(Y[-1] - 1.8, 2.0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(0, 1.4, "Hospital Readmission Prediction", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="#111827")
    ax.text(0, 0.95, "Pipeline Flow", ha="center", va="center",
            fontsize=10, color="#6B7280", style="italic")

    # Curved connector path between consecutive nodes (smooth S-curve)
    curve_lines = []
    for i in range(N - 1):
        xs = np.linspace(X[i], X[i + 1], 60)
        t = (xs - X[i]) / (X[i + 1] - X[i])
        ys = Y[i] + (Y[i + 1] - Y[i]) * (t ** 2 * (3 - 2 * t))
        line, = ax.plot(xs, ys, color=DIM_EDGE, linewidth=2.4, zorder=1, solid_capstyle="round")
        curve_lines.append((line, xs, ys))

    nodes = []
    for i, (title, sub, color) in enumerate(STAGES):
        shadow = plt.Circle((X[i] + 0.05, Y[i] - 0.06), NODE_R, color=SHADOW, zorder=2, alpha=0.7)
        ax.add_patch(shadow)
        glow = plt.Circle((X[i], Y[i]), NODE_R + 0.14, color=color, zorder=2, alpha=0.0)
        ax.add_patch(glow)
        circle = plt.Circle((X[i], Y[i]), NODE_R, facecolor=BG_OFF, edgecolor=DIM_EDGE, linewidth=2.2, zorder=3)
        ax.add_patch(circle)
        num_txt = ax.text(X[i], Y[i], f"{i+1}", ha="center", va="center", fontsize=20,
                          fontweight="bold", color=DIM_TEXT, zorder=5)
        ring = mpatches.Wedge((X[i], Y[i]), NODE_R + 0.22, 90, 90, width=0.09,
                              facecolor=color, edgecolor="none", zorder=4, alpha=0.0)
        ax.add_patch(ring)

        label_x = X[i] + (1.35 if X[i] < 0 else -1.35)
        ha = "left" if X[i] < 0 else "right"
        title_txt = ax.text(label_x, Y[i] + 0.13, title, ha=ha, va="center", fontsize=11.5,
                            color=DIM_TEXT, fontweight="normal", zorder=5)
        sub_txt = ax.text(label_x, Y[i] - 0.20, sub, ha=ha, va="center", fontsize=7.8,
                          color=DIM_TEXT, zorder=5, wrap=True)

        nodes.append(dict(circle=circle, glow=glow, ring=ring, num=num_txt,
                          title=title_txt, sub=sub_txt, color=color, shadow=shadow))

    caption = ax.text(0, Y[-1] - 1.35, "", ha="center", va="center", fontsize=10.5,
                      color="#4B5563", style="italic")

    fig.tight_layout()
    return fig, ax, nodes, curve_lines, caption


TRAVEL_FRAMES = 20
HOLD_FRAMES = 14


def make_animation():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax, nodes, curve_lines, caption = build_figure()

    pulse = ax.scatter([], [], s=0, color=PALETTE["blue"], zorder=6, edgecolors="white", linewidths=1.3)

    def reset_all():
        for nd in nodes:
            nd["circle"].set_edgecolor(DIM_EDGE); nd["circle"].set_facecolor(BG_OFF)
            nd["glow"].set_alpha(0.0)
            nd["ring"].set_theta1(90); nd["ring"].set_theta2(90); nd["ring"].set_alpha(0.0)
            nd["num"].set_color(DIM_TEXT); nd["num"].set_fontweight("bold")
            nd["title"].set_color(DIM_TEXT); nd["title"].set_fontweight("normal")
            nd["sub"].set_color(DIM_TEXT)
        for line, xs, ys in curve_lines:
            line.set_color(DIM_EDGE); line.set_linewidth(2.4)
        pulse.set_sizes([0])
        caption.set_text("")

    def activate_node(i):
        nd = nodes[i]
        nd["circle"].set_edgecolor(nd["color"]); nd["circle"].set_facecolor(tint(nd["color"]))
        nd["num"].set_color(nd["color"])
        nd["title"].set_color(ON_TEXT); nd["title"].set_fontweight("bold")
        nd["sub"].set_color("#4B5563")
        nd["glow"].set_alpha(0.14)

    def spin_ring(i, frac):
        nd = nodes[i]
        nd["ring"].set_alpha(0.9)
        theta_end = 90 - 360 * frac
        nd["ring"].set_theta1(theta_end); nd["ring"].set_theta2(90)
        nd["ring"].set_facecolor(nd["color"])

    per_stage = TRAVEL_FRAMES + HOLD_FRAMES
    total = per_stage * N + 24

    def update(frame):
        f = frame % (total + 16)
        if f == 0:
            reset_all()
        if f >= total:
            pulse.set_sizes([0])
            caption.set_text("\u2713 Test AUC 0.667 \u2014 reported once, honestly.")
            return []

        stage_idx = min(f // per_stage, N - 1)
        within = f % per_stage

        if within < TRAVEL_FRAMES and stage_idx > 0:
            line, xs, ys = curve_lines[stage_idx - 1]
            t = ease_in_out(within / TRAVEL_FRAMES)
            idx = int(t * (len(xs) - 1))
            cx, cy = xs[idx], ys[idx]
            pulse.set_offsets([[cx, cy]]); pulse.set_sizes([150])
            line.set_color(PALETTE["blue"]); line.set_linewidth(3.0)
            caption.set_text(CAPTIONS[stage_idx - 1])
        else:
            activate_node(stage_idx)
            hold_within = within - TRAVEL_FRAMES if within >= TRAVEL_FRAMES else 0
            frac = min(hold_within / HOLD_FRAMES, 1.0)
            spin_ring(stage_idx, frac)
            pulse.set_sizes([0])
            caption.set_text(CAPTIONS[stage_idx])
        return []

    anim = FuncAnimation(fig, update, frames=total + 40, interval=85, blit=False)
    out_path = f"{OUT_DIR}/architecture.gif"
    anim.save(out_path, writer=PillowWriter(fps=1000 / 85))
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    make_animation()


if __name__ == "__main__":
    main()
