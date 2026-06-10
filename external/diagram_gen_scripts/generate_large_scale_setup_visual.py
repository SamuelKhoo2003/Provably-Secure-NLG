"""
Generate a report-ready visual for the large-scale model-based setup.

Outputs:
  - large_scale_setup_visual.pdf
  - large_scale_setup_visual.png
  - large_scale_setup_visual.svg

Run:
  python generate_large_scale_setup_visual.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).resolve().parent

# Colours are deliberately muted for report readability.
COLORS = {
    "data": "#EAF3F8",
    "model": "#F1ECE2",
    "grid": "#EEF5EA",
    "baseline": "#F7E8E8",
    "milp": "#EDE8F7",
    "note": "#F7F7F7",
    "edge": "#333333",
    "accent": "#4C78A8",
    "dark": "#222222",
}


def add_box(ax, x, y, w, h, text, facecolor, fontsize=9, weight="normal", radius=0.025, lw=1.1):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=COLORS["edge"],
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, wrap=True)
    return box


def arrow(ax, x1, y1, x2, y2, text=None, rad=0.0, lw=1.2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=COLORS["edge"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.018, text, ha="center", va="bottom", fontsize=7.5)
    return arr


def draw_token_grid(ax, x, y, w, h, n_rows=4, n_cols=6):
    """Draw a prompt-token grid with tiny shard-vote stacks inside cells."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=COLORS["edge"], linewidth=1.1))
    cell_w = w / n_cols
    cell_h = h / n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            cx = x + c * cell_w
            cy = y + (n_rows - 1 - r) * cell_h
            fill = "#F9FBF9" if (r + c) % 2 == 0 else "#EEF5EA"
            ax.add_patch(Rectangle((cx, cy), cell_w, cell_h, facecolor=fill, edgecolor="#C9C9C9", linewidth=0.55))
            # Three tiny shard votes inside each cell
            for s in range(3):
                sx = cx + 0.012 + s * 0.016
                sy = cy + 0.012
                ax.add_patch(Rectangle((sx, sy), 0.010, 0.030, facecolor=["#4C78A8", "#72B7B2", "#F58518"][(r+c+s) % 3], edgecolor="none", alpha=0.9))
    # Labels
    for c in range(n_cols):
        ax.text(x + (c + 0.5) * cell_w, y + h + 0.015, f"l={c+1}", ha="center", va="bottom", fontsize=7)
    for r in range(n_rows):
        ax.text(x - 0.015, y + (n_rows - r - 0.5) * cell_h, f"i={r+1}", ha="right", va="center", fontsize=7)
    ax.text(x + w / 2, y - 0.030, "Shard-indexed tensor X[i,l,k]", ha="center", va="top", fontsize=8.5, weight="bold")


def draw_vote_bar(ax, x, y, w, h):
    labels = ["dictionary", "get_definitions", "detect"]
    vals = [0.66, 0.20, 0.14]
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    for idx, (lab, val, col) in enumerate(zip(labels, vals, colors)):
        yy = y + h - (idx + 1) * h / 3 + 0.012
        ax.add_patch(Rectangle((x, yy), w * val, 0.030, facecolor=col, edgecolor="none"))
        ax.add_patch(Rectangle((x, yy), w, 0.030, facecolor="none", edgecolor="#BBBBBB", linewidth=0.5))
        ax.text(x + w + 0.010, yy + 0.015, lab, ha="left", va="center", fontsize=7)
    ax.text(x + w / 2, y - 0.020, "aggregate tool votes", ha="center", va="top", fontsize=8.5, weight="bold")


def make_visual():
    fig, ax = plt.subplots(figsize=(15.5, 8.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Large-Scale Model-Based Certification Interface", ha="center", va="top", fontsize=18, weight="bold")
    ax.text(0.5, 0.925, "From shard adapters and vote vectors to count-based baselines and shard-aware row-column MILPs", ha="center", va="top", fontsize=10.5)

    # Top pipeline
    add_box(ax, 0.035, 0.790, 0.125, 0.080, "Toucan\ntool-use data", COLORS["data"], weight="bold")
    add_box(ax, 0.195, 0.790, 0.135, 0.080, "Disjoint\ntraining shards", COLORS["data"], weight="bold")
    add_box(ax, 0.365, 0.790, 0.150, 0.080, "K=500 LoRA\nshard adapters", COLORS["model"], weight="bold")
    add_box(ax, 0.550, 0.790, 0.165, 0.080, "Sequential\nadapter inference", COLORS["model"], weight="bold")
    add_box(ax, 0.755, 0.790, 0.190, 0.080, "JSONL vote vectors\nper test prompt", COLORS["note"], weight="bold")
    arrow(ax, 0.160, 0.830, 0.195, 0.830)
    arrow(ax, 0.330, 0.830, 0.365, 0.830)
    arrow(ax, 0.515, 0.830, 0.550, 0.830)
    arrow(ax, 0.715, 0.830, 0.755, 0.830)

    # Vote vector contents
    add_box(ax, 0.065, 0.640, 0.245, 0.090, "final_output[k]\none tool-call vote per shard", COLORS["baseline"], fontsize=9, weight="bold")
    add_box(ax, 0.365, 0.640, 0.260, 0.090, "token_vote_matrix[k,l]\none token sequence per shard", COLORS["grid"], fontsize=9, weight="bold")
    add_box(ax, 0.680, 0.640, 0.235, 0.090, "horizon filter H\nkeep prompts with no None padding", COLORS["note"], fontsize=9, weight="bold")
    arrow(ax, 0.850, 0.790, 0.188, 0.730, rad=0.18, text="stored fields")
    arrow(ax, 0.850, 0.790, 0.495, 0.730, rad=0.05)
    arrow(ax, 0.495, 0.640, 0.680, 0.685)

    # Middle visual interface
    ax.text(0.055, 0.585, "A. Count-based final-output interface", fontsize=11.5, weight="bold", ha="left")
    draw_vote_bar(ax, 0.080, 0.455, 0.180, 0.105)
    add_box(ax, 0.060, 0.350, 0.225, 0.070, "No shard identities needed\nonly aggregate vote counts", COLORS["baseline"], fontsize=8.5)

    ax.text(0.390, 0.585, "B. Shard-aware token-grid interface", fontsize=11.5, weight="bold", ha="left")
    draw_token_grid(ax, 0.390, 0.355, 0.315, 0.205, n_rows=4, n_cols=6)
    add_box(ax, 0.735, 0.390, 0.205, 0.100, "Shared poisoned-shard\nallocation a[k]\nwith sum a[k] <= B", COLORS["milp"], fontsize=9, weight="bold")
    arrow(ax, 0.705, 0.455, 0.735, 0.440)

    # Methods bottom
    ax.text(0.052, 0.290, "Certification outputs", fontsize=12.5, weight="bold", ha="left")
    add_box(ax, 0.055, 0.190, 0.205, 0.075, "Final-tool DPA\nstability baseline", COLORS["baseline"], fontsize=8.8)
    add_box(ax, 0.285, 0.190, 0.205, 0.075, "Aggregate TPA\nfinal-tool validity", COLORS["baseline"], fontsize=8.8)
    add_box(ax, 0.515, 0.190, 0.205, 0.075, "Token-grid DPA\ndiagnostic", COLORS["grid"], fontsize=8.8)
    add_box(ax, 0.745, 0.190, 0.205, 0.075, "Joint row-column MILP\nshard-aware certificate", COLORS["milp"], fontsize=8.8)
    arrow(ax, 0.170, 0.350, 0.157, 0.265)
    arrow(ax, 0.170, 0.350, 0.388, 0.265, rad=-0.15)
    arrow(ax, 0.540, 0.355, 0.617, 0.265)
    arrow(ax, 0.838, 0.390, 0.848, 0.265)

    # Small matrix table
    x0, y0, table_w, table_h = 0.070, 0.055, 0.860, 0.090
    ax.add_patch(Rectangle((x0, y0), table_w, table_h, facecolor="white", edgecolor=COLORS["edge"], linewidth=1.0))
    headers = ["Interface", "Input information", "Stability question", "Validity question", "Main limitation"]
    row1 = ["Final-output", "tool counts", "preserve tool class", "block observed tool target", "ignores token/shard structure"]
    row2 = ["Token grid", "X[i,l,k]", "protect any retained token", "force all active target tokens", "requires rectangular H horizon"]
    col_fracs = [0.16, 0.22, 0.22, 0.23, 0.17]
    xs = [x0]
    for f in col_fracs:
        xs.append(xs[-1] + table_w * f)
    for xx in xs[1:-1]:
        ax.add_line(Line2D([xx, xx], [y0, y0 + table_h], color="#BBBBBB", linewidth=0.7))
    for yy in [y0 + table_h * 2/3, y0 + table_h * 1/3]:
        ax.add_line(Line2D([x0, x0 + table_w], [yy, yy], color="#BBBBBB", linewidth=0.7))
    for c, htxt in enumerate(headers):
        ax.text((xs[c]+xs[c+1])/2, y0 + table_h * 5/6, htxt, ha="center", va="center", fontsize=7.4, weight="bold")
    for r, row in enumerate([row1, row2]):
        yy = y0 + table_h * (0.5 if r == 0 else 1/6)
        for c, val in enumerate(row):
            ax.text((xs[c]+xs[c+1])/2, yy, val, ha="center", va="center", fontsize=7.2)

    ax.text(0.500, 0.018, "Use as a setup figure near Section 5.4 or before the large-scale results. Replace example labels/counts with exact run metadata if needed.", ha="center", va="bottom", fontsize=8, style="italic")

    fig.tight_layout(pad=0.3)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT_DIR / f"large_scale_setup_visual.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_visual()
    print("Wrote large_scale_setup_visual.pdf, .png, .svg")
