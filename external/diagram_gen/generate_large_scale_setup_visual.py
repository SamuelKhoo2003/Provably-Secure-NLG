"""Generate the large-scale setup diagram beside this script."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUT_BASENAME = "large_scale_setup_visual"
OUTPUT_DIR = Path(__file__).resolve().parent

FIG_W = 24
FIG_H = 11.2
DPI = 220

# Pastel palette
COLORS = {
    "blue_fill": "#dfe8ef",
    "blue_edge": "#2b5797",
    "beige_fill": "#efe8d8",
    "lav_fill": "#e8e3f2",
    "pink_fill": "#f1e3e3",
    "green_fill": "#e6eee2",
    "green_edge": "#3f6b2a",
    "grey_fill": "#f2f2f2",
    "black": "#222222",
    "dark_blue": "#4c78a8",
    "teal": "#72b7b2",
    "orange": "#f58518",
    "grid_line": "#bdbdbd",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def rounded_box(ax, x, y, w, h, text, facecolor="#ffffff", edgecolor="#222222",
                fontsize=12, weight="normal", ha="center", va="center",
                round_size=0.012, lw=1.3, text_color="#222222",
                linespacing=1.2, zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={round_size}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        fontsize=fontsize,
        fontweight=weight,
        ha=ha,
        va=va,
        color=text_color,
        linespacing=linespacing,
        zorder=zorder + 1
    )
    return box


def simple_arrow(ax, p1, p2, lw=1.5, color="#222222", mutation=18, zorder=3):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        zorder=zorder,
        shrinkA=0,
        shrinkB=0
    )
    ax.add_patch(arr)
    return arr


def orthogonal_connector(ax, p_start, p_mid1=None, p_mid2=None, p_end=None,
                         lw=1.4, color="#222222", arrow_end=True, zorder=3):
    """
    Draws a polyline connector with an optional arrow at the end.
    """
    points = [p_start]
    if p_mid1 is not None:
        points.append(p_mid1)
    if p_mid2 is not None:
        points.append(p_mid2)
    if p_end is not None:
        points.append(p_end)

    for i in range(len(points) - 2):
        a, b = points[i], points[i + 1]
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], linewidth=lw, color=color, zorder=zorder))

    if len(points) >= 2:
        a, b = points[-2], points[-1]
        if arrow_end:
            simple_arrow(ax, a, b, lw=lw, color=color, mutation=16, zorder=zorder)
        else:
            ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], linewidth=lw, color=color, zorder=zorder))


def panel(ax, x, y, w, h, title, edgecolor, title_color, facecolor="#ffffff", lw=1.4):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=1
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.03,
        title,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=title_color,
        zorder=2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    return patch


def draw_bar_stack(ax, x, y, total_w, bar_h, fill_frac, fill_color, label):
    ax.add_patch(Rectangle((x, y), total_w, bar_h, facecolor="white", edgecolor="#c8c8c8", linewidth=1.0))
    ax.add_patch(Rectangle((x, y), total_w * fill_frac, bar_h, facecolor=fill_color, edgecolor="none"))
    ax.text(x + total_w + 0.01, y + bar_h / 2, label, ha="left", va="center", fontsize=11, color=COLORS["black"])


def draw_token_icon(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#c8c8c8", linewidth=0.9))
    pad_x = w * 0.12
    pad_y = h * 0.18
    bw = w * 0.12
    gap = w * 0.08
    heights = [0.60, 0.42, 0.72]
    colors = [COLORS["dark_blue"], COLORS["teal"], COLORS["orange"]]
    lefts = [
        x + pad_x,
        x + pad_x + bw + gap,
        x + pad_x + 2 * (bw + gap)
    ]
    for lx, hh, cc in zip(lefts, heights, colors):
        ax.add_patch(
            Rectangle(
                (lx, y + pad_y),
                bw,
                h * hh,
                facecolor=cc,
                edgecolor="none"
            )
        )


def draw_dynamic_grid(ax, x, y, w, h):
    """
    Draws the dynamic shard-aware token grid with i=1..N and l=1..L
    using visible sample cells and ellipses.
    """
    # Title area handled by panel
    grid_x = x + 0.04
    grid_y = y + 0.045
    grid_w = w - 0.07
    grid_h = h - 0.105

    # Layout for displayed sample cells
    token_w = grid_w * 0.11
    token_h = grid_h * 0.16

    col_xs = [
        grid_x + grid_w * 0.14,
        grid_x + grid_w * 0.28,
        grid_x + grid_w * 0.42,
        grid_x + grid_w * 0.67,
        grid_x + grid_w * 0.81,
        grid_x + grid_w * 0.95 - token_w
    ]
    row_ys = [
        grid_y + grid_h * 0.73,
        grid_y + grid_h * 0.50,
        grid_y + grid_h * 0.08
    ]

    # Column labels
    ax.text(col_xs[0] + token_w / 2, grid_y + grid_h * 0.96, "l=1", ha="center", va="bottom",
            fontsize=12, color=COLORS["black"])
    ax.text((col_xs[2] + col_xs[3]) / 2 + token_w / 2, grid_y + grid_h * 0.96, "⋯", ha="center", va="bottom",
            fontsize=20, color=COLORS["black"])
    ax.text(col_xs[-1] + token_w / 2, grid_y + grid_h * 0.96, "l=L", ha="center", va="bottom",
            fontsize=12, color=COLORS["black"])

    # Row labels
    ax.text(grid_x + grid_w * 0.04, row_ys[0] + token_h / 2, "i=1", ha="center", va="center",
            fontsize=12, color=COLORS["black"])
    ax.text(grid_x + grid_w * 0.04, grid_y + grid_h * 0.47, "⋮", ha="center", va="center",
            fontsize=24, color=COLORS["black"])
    ax.text(grid_x + grid_w * 0.04, row_ys[-1] + token_h / 2, "i=N", ha="center", va="center",
            fontsize=12, color=COLORS["black"])

    # Draw sample cells
    for yy in row_ys:
        for xx in col_xs[:3]:
            draw_token_icon(ax, xx, yy, token_w, token_h)
        for xx in col_xs[3:]:
            draw_token_icon(ax, xx, yy, token_w, token_h)

    # Horizontal ellipses between shown columns
    mid_gap_x = (col_xs[2] + token_w + col_xs[3]) / 2
    for yy in row_ys:
        ax.text(mid_gap_x, yy + token_h / 2, "⋯", ha="center", va="center", fontsize=22, color=COLORS["black"])

    # Vertical ellipses between shown rows
    for xx in [col_xs[0], col_xs[1], col_xs[2], col_xs[3], col_xs[4], col_xs[5]]:
        ax.text(xx + token_w / 2, grid_y + grid_h * 0.39, "⋮", ha="center", va="center",
                fontsize=20, color=COLORS["black"])

    # Caption
    ax.text(
        x + w / 2,
        y + 0.02,
        "Shard-indexed tensor X[i,l,k]",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["black"]
    )


# --------------------------------------------------
# Main figure
# --------------------------------------------------

def generate_figure():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.965,
        "Large-Scale Model-Based Certification Interface",
        ha="center", va="center",
        fontsize=21, fontweight="bold", color=COLORS["black"]
    )
    ax.text(
        0.5, 0.925,
        "From shard adapters and vote vectors to count-based baselines and shard-aware row-column MILPs",
        ha="center", va="center",
        fontsize=13.5, color=COLORS["black"]
    )

    # --------------------------------------------------
    # Top pipeline
    # --------------------------------------------------

    top_y = 0.80
    top_h = 0.08
    box_w = 0.14
    gap = 0.018

    x1 = 0.03
    x2 = x1 + box_w + gap
    x3 = x2 + box_w + gap
    x4 = x3 + box_w + gap
    x5 = x4 + box_w + gap

    b1 = rounded_box(ax, x1, top_y, box_w, top_h, "Toucan\ntool-use data",
                     facecolor=COLORS["blue_fill"], edgecolor=COLORS["black"], fontsize=13, weight="bold")
    b2 = rounded_box(ax, x2, top_y, box_w, top_h, "Disjoint\ntraining shards",
                     facecolor=COLORS["blue_fill"], edgecolor=COLORS["black"], fontsize=13, weight="bold")
    b3 = rounded_box(ax, x3, top_y, box_w, top_h, "K=500 LoRA\nshard adapters",
                     facecolor=COLORS["beige_fill"], edgecolor=COLORS["black"], fontsize=13, weight="bold")
    b4 = rounded_box(ax, x4, top_y, box_w, top_h, "Sequential\nadapter inference",
                     facecolor=COLORS["beige_fill"], edgecolor=COLORS["black"], fontsize=13, weight="bold")
    b5 = rounded_box(ax, x5, top_y, box_w, top_h, "JSONL vote vectors\nper test prompt",
                     facecolor=COLORS["lav_fill"], edgecolor=COLORS["black"], fontsize=13, weight="bold")

    simple_arrow(ax, (x1 + box_w, top_y + top_h / 2), (x2, top_y + top_h / 2))
    simple_arrow(ax, (x2 + box_w, top_y + top_h / 2), (x3, top_y + top_h / 2))
    simple_arrow(ax, (x3 + box_w, top_y + top_h / 2), (x4, top_y + top_h / 2))
    simple_arrow(ax, (x4 + box_w, top_y + top_h / 2), (x5, top_y + top_h / 2))

    # Stored fields boxes
    sf_y = 0.66
    sf_h = 0.075

    final_box = rounded_box(
        ax, 0.205, sf_y, 0.16, sf_h,
        "final_output[k]\none tool-call vote per shard",
        facecolor=COLORS["pink_fill"], edgecolor=COLORS["black"], fontsize=11.5, weight="bold"
    )
    token_box = rounded_box(
        ax, 0.44, sf_y, 0.21, sf_h,
        "token_vote_matrix[k,l]\none token sequence per shard",
        facecolor=COLORS["green_fill"], edgecolor=COLORS["black"], fontsize=11.5, weight="bold"
    )
    horizon_box = rounded_box(
        ax, 0.69, sf_y, 0.255, sf_h,
        "horizon filter H\nshortest non-None shard prefix reaches H",
        facecolor=COLORS["grey_fill"], edgecolor=COLORS["black"], fontsize=10.8, weight="bold"
    )

    # Stored fields label
    ax.text(0.53, 0.775, "stored fields", ha="center", va="center",
            fontsize=12, style="italic", color=COLORS["black"])

    # Clean branch from JSONL box to stored fields
    stem_top = (x5 + box_w * 0.06, top_y)
    stem_mid = (x5 + box_w * 0.06, 0.765)
    left_split = (0.285, 0.765)
    mid_split = (0.545, 0.765)

    # main stem down from JSONL region
    ax.add_line(Line2D([stem_top[0], stem_mid[0]], [stem_top[1], stem_mid[1]], color=COLORS["black"], linewidth=1.4))
    # horizontal split line
    ax.add_line(Line2D([left_split[0], stem_mid[0]], [stem_mid[1], stem_mid[1]], color=COLORS["black"], linewidth=1.4))

    # final_output down arrow
    orthogonal_connector(
        ax,
        p_start=(left_split[0], stem_mid[1]),
        p_mid1=(left_split[0], sf_y + sf_h + 0.01),
        p_end=(0.285, sf_y + sf_h),
        arrow_end=True
    )

    # token matrix down arrow
    orthogonal_connector(
        ax,
        p_start=(mid_split[0], stem_mid[1]),
        p_mid1=(mid_split[0], sf_y + sf_h + 0.01),
        p_end=(0.545, sf_y + sf_h),
        arrow_end=True
    )

    # token matrix to horizon filter
    simple_arrow(ax, (0.65, sf_y + sf_h / 2), (0.69, sf_y + sf_h / 2))

    # --------------------------------------------------
    # Main panels
    # --------------------------------------------------

    left_panel = panel(
        ax, 0.06, 0.365, 0.30, 0.27,
        "A. Count-based final-output interface",
        edgecolor=COLORS["blue_edge"],
        title_color="#173b76",
        facecolor="white"
    )

    right_panel = panel(
        ax, 0.385, 0.365, 0.31, 0.27,
        "B. Shard-aware token-grid interface",
        edgecolor=COLORS["green_edge"],
        title_color="#284d17",
        facecolor="white"
    )

    alloc_box = rounded_box(
        ax, 0.72, 0.395, 0.225, 0.11,
        "Shared poisoned-shard\nallocation a[k]\nwith sum a[k] <= B",
        facecolor=COLORS["lav_fill"], edgecolor=COLORS["black"], fontsize=11.2, weight="bold"
    )

    # Down arrows from stored fields into panels
    simple_arrow(ax, (0.285, sf_y), (0.285, 0.635))
    simple_arrow(ax, (0.545, sf_y), (0.545, 0.635))

    # Left panel contents
    bar_x = 0.075
    bar_y_top = 0.545
    bar_w = 0.19
    bar_h = 0.028
    gap_y = 0.036

    draw_bar_stack(ax, bar_x, bar_y_top, bar_w, bar_h, 0.66, COLORS["dark_blue"], "dictionary")
    draw_bar_stack(ax, bar_x, bar_y_top - gap_y, bar_w, bar_h, 0.22, COLORS["teal"], "get_definitions")
    draw_bar_stack(ax, bar_x, bar_y_top - 2 * gap_y, bar_w, bar_h, 0.15, COLORS["orange"], "detect")

    ax.text(0.19, 0.46, "aggregate tool votes", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COLORS["black"])

    rounded_box(
        ax, 0.09, 0.38, 0.20, 0.055,
        "No shard identities needed\nonly aggregate vote counts",
        facecolor=COLORS["pink_fill"], edgecolor=COLORS["black"], fontsize=10.2
    )

    # Right panel grid
    draw_dynamic_grid(ax, 0.395, 0.375, 0.29, 0.245)

    # Arrow from grid panel to allocation box
    simple_arrow(ax, (0.695, 0.495), (0.72, 0.495))

    # --------------------------------------------------
    # Certification outputs
    # --------------------------------------------------

    ax.text(
        0.5,
        0.322,
        "Certification outputs",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=COLORS["black"],
        zorder=6,
    )

    out_y = 0.205
    out_h = 0.075
    out_w = 0.19
    out_gap = 0.035

    out1_x = 0.05
    out2_x = out1_x + out_w + out_gap
    out3_x = out2_x + out_w + out_gap
    out4_x = out3_x + out_w + out_gap

    out1 = rounded_box(ax, out1_x, out_y, out_w, out_h,
                       "Final-tool DPA\nstability baseline",
                       facecolor=COLORS["pink_fill"], edgecolor=COLORS["black"], fontsize=12)
    out2 = rounded_box(ax, out2_x, out_y, out_w, out_h,
                       "Aggregate TPA\nfinal-tool validity",
                       facecolor=COLORS["pink_fill"], edgecolor=COLORS["black"], fontsize=12)
    out3 = rounded_box(ax, out3_x, out_y, out_w, out_h,
                       "Token-grid DPA\ndiagnostic",
                       facecolor=COLORS["green_fill"], edgecolor=COLORS["black"], fontsize=12)
    out4 = rounded_box(ax, out4_x, out_y, out_w, out_h,
                       "Joint row-column MILP\nshard-aware certificate",
                       facecolor=COLORS["lav_fill"], edgecolor=COLORS["black"], fontsize=12)

    # Route the branches through whitespace and terminate arrows at box edges.
    left_panel_center_bottom = (0.21, 0.365)
    split_y_left = 0.295
    ax.add_line(Line2D([left_panel_center_bottom[0], left_panel_center_bottom[0]],
                       [left_panel_center_bottom[1], split_y_left], color=COLORS["black"], linewidth=1.4))
    ax.add_line(Line2D([out1_x + out_w / 2, out2_x + out_w / 2],
                       [split_y_left, split_y_left], color=COLORS["black"], linewidth=1.4))
    simple_arrow(ax, (out1_x + out_w / 2, split_y_left), (out1_x + out_w / 2, out_y + out_h))
    simple_arrow(ax, (out2_x + out_w / 2, split_y_left), (out2_x + out_w / 2, out_y + out_h))

    # Offset this stem from the centered heading, then branch to both
    # token-grid certification outputs.
    right_panel_center_bottom = (0.62, 0.365)
    split_y_right = 0.295
    ax.add_line(Line2D([right_panel_center_bottom[0], right_panel_center_bottom[0]],
                       [right_panel_center_bottom[1], split_y_right], color=COLORS["black"], linewidth=1.4))
    grid_out3_x = out3_x + out_w / 2 - 0.012
    grid_out4_x = out4_x + out_w / 2 - 0.012
    ax.add_line(Line2D([grid_out3_x, grid_out4_x],
                       [split_y_right, split_y_right], color=COLORS["black"], linewidth=1.4))
    simple_arrow(ax, (grid_out3_x, split_y_right), (grid_out3_x, out_y + out_h))
    simple_arrow(ax, (grid_out4_x, split_y_right), (grid_out4_x, out_y + out_h))

    # Keep the allocation input visibly separate from the token-grid branch.
    allocation_x = 0.72 + 0.225 / 2
    simple_arrow(ax, (allocation_x, 0.395), (allocation_x, out_y + out_h))

    return fig


def main():
    """Render the diagram into ``external/diagram_gen``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = generate_figure()
    output_paths = [
        OUTPUT_DIR / f"{OUT_BASENAME}.png",
        OUTPUT_DIR / f"{OUT_BASENAME}.pdf",
        OUTPUT_DIR / f"{OUT_BASENAME}.svg",
    ]
    fig.savefig(
        output_paths[0],
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    for output_path in output_paths[1:]:
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved diagram files:")
    for output_path in output_paths:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
