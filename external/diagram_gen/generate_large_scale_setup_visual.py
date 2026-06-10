from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D


OUT_BASENAME = "large_scale_setup_visual"
OUTPUT_DIR = Path(__file__).resolve().parent

FIG_W = 24
FIG_H = 11
DPI = 220

COLORS = {
    "blue_fill": "#dfe8ef",
    "blue_edge": "#365b8c",
    "beige_fill": "#efe8d8",
    "lav_fill": "#e7e1f3",
    "pink_fill": "#f3e7e7",
    "green_fill": "#e8efe4",
    "green_edge": "#4c6e38",
    "grey_fill": "#f5f5f5",
    "black": "#222222",
    "dark_blue": "#4c78a8",
    "teal": "#72b7b2",
    "orange": "#f58518",
    "light_border": "#bdbdbd",
}


# --------------------------------------------------
# basic drawing helpers
# --------------------------------------------------

def rounded_box(
    ax,
    x, y, w, h,
    text,
    facecolor="white",
    edgecolor="#222222",
    fontsize=12,
    weight="normal",
    text_color="#222222",
    lw=1.4,
    round_size=0.012,
    zorder=2,
    linespacing=1.2,
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={round_size}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        linespacing=linespacing,
        zorder=zorder + 1,
    )
    return patch


def arrow(ax, p1, p2, lw=1.5, color="#222222", mutation=18, zorder=3):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def polyline_with_arrow(ax, points, lw=1.4, color="#222222", zorder=3):
    """
    Draw a polyline with an arrow only on the final segment.
    """
    if len(points) < 2:
        return

    for i in range(len(points) - 2):
        a = points[i]
        b = points[i + 1]
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], linewidth=lw, color=color, zorder=zorder))

    arrow(ax, points[-2], points[-1], lw=lw, color=color, mutation=16, zorder=zorder)


def panel(ax, x, y, w, h, title, edgecolor, title_color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h - 0.022,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=title_color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        zorder=3,
    )
    return patch


def draw_bar(ax, x, y, total_w, bar_h, frac, fill_color, label):
    ax.add_patch(Rectangle((x, y), total_w, bar_h, facecolor="white",
                           edgecolor=COLORS["light_border"], linewidth=1.0, zorder=2))
    ax.add_patch(Rectangle((x, y), total_w * frac, bar_h, facecolor=fill_color,
                           edgecolor="none", zorder=3))
    ax.text(x + total_w + 0.008, y + bar_h / 2, label,
            ha="left", va="center", fontsize=11.5, color=COLORS["black"], zorder=4)


def draw_token_icon(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white",
                           edgecolor=COLORS["light_border"], linewidth=0.9, zorder=2))
    pad_x = w * 0.10
    pad_y = h * 0.16
    bw = w * 0.12
    gap = w * 0.085

    heights = [0.62, 0.44, 0.74]
    colors = [COLORS["dark_blue"], COLORS["teal"], COLORS["orange"]]

    lefts = [
        x + pad_x,
        x + pad_x + bw + gap,
        x + pad_x + 2 * (bw + gap),
    ]

    for lx, hh, cc in zip(lefts, heights, colors):
        ax.add_patch(Rectangle((lx, y + pad_y), bw, h * hh,
                               facecolor=cc, edgecolor="none", zorder=3))


def draw_dynamic_grid(ax, x, y, w, h):
    """
    A grid visually similar to the fixed PNG grid, but with dynamic continuation.
    Shown columns are
        l=1, l=2, l=3, ..., l=L-2, l=L-1, l=L
    Shown rows are
        i=1, i=2, ..., i=N-1, i=N
    """
    # content region inside the green panel
    gx = x + 0.03
    gy = y + 0.045
    gw = w - 0.05
    gh = h - 0.10

    # row label area and cell spacing
    row_label_w = 0.038 * w
    top_label_h = 0.028 * h

    cell_w = gw * 0.13
    cell_h = gh * 0.17

    # 3 visible columns, ellipsis, 3 visible columns
    col_xs = [
        gx + row_label_w + 0.012,
        gx + row_label_w + 0.012 + cell_w * 1.08,
        gx + row_label_w + 0.012 + cell_w * 2.16,
        gx + row_label_w + 0.012 + cell_w * 4.15,
        gx + row_label_w + 0.012 + cell_w * 5.23,
        gx + row_label_w + 0.012 + cell_w * 6.31,
    ]

    row_ys = [
        gy + gh * 0.70,
        gy + gh * 0.48,
        gy + gh * 0.20,
        gy + gh * 0.02,
    ]

    # column labels
    col_labels = ["l=1", "l=2", "l=3", "l=L-2", "l=L-1", "l=L"]
    for cx, lab in zip(col_xs, col_labels):
        ax.text(cx + cell_w / 2, gy + gh * 0.90, lab,
                ha="center", va="bottom", fontsize=11.5, color=COLORS["black"], zorder=4)

    # ellipsis label between left and right groups
    x_gap_mid = (col_xs[2] + cell_w + col_xs[3]) / 2
    ax.text(x_gap_mid, gy + gh * 0.90, "⋯",
            ha="center", va="bottom", fontsize=18, color=COLORS["black"], zorder=4)

    # row labels
    row_labels = ["i=1", "i=2", "i=N-1", "i=N"]
    for ry, lab in zip(row_ys, row_labels):
        ax.text(gx + row_label_w * 0.35, ry + cell_h / 2, lab,
                ha="center", va="center", fontsize=11.5, color=COLORS["black"], zorder=4)

    y_gap_mid = (row_ys[1] + row_ys[2] + cell_h) / 2 - cell_h / 2
    ax.text(gx + row_label_w * 0.35, y_gap_mid + cell_h / 2, "⋮",
            ha="center", va="center", fontsize=18, color=COLORS["black"], zorder=4)

    # draw visible token cells
    for ry in row_ys:
        for cx in col_xs:
            draw_token_icon(ax, cx, ry, cell_w, cell_h)

    # horizontal ellipses between groups
    for ry in row_ys:
        ax.text(x_gap_mid, ry + cell_h / 2, "⋯",
                ha="center", va="center", fontsize=18, color=COLORS["black"], zorder=4)

    # vertical ellipses between upper and lower row groups
    for cx in col_xs:
        ax.text(cx + cell_w / 2, y_gap_mid + cell_h / 2, "⋮",
                ha="center", va="center", fontsize=18, color=COLORS["black"], zorder=4)

    ax.text(
        x + w / 2,
        y + 0.018,
        "Shard-indexed tensor X[i,l,k]",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["black"],
        zorder=4,
    )


def draw_table(ax, x, y, w, h):
    headers = [
        "Interface",
        "Input information",
        "Stability question",
        "Validity question",
        "Main limitation",
    ]
    rows = [
        ["Final-output", "tool counts", "preserve tool class",
         "block observed tool target", "ignores token/shard structure"],
        ["Token grid", "X[i,l,k]", "protect any retained token",
         "force all active target tokens", "requires rectangular H horizon"],
    ]

    col_fracs = [0.17, 0.20, 0.20, 0.22, 0.21]
    total = sum(col_fracs)
    col_fracs = [c / total for c in col_fracs]

    ax.add_patch(Rectangle((x, y), w, h, facecolor="white",
                           edgecolor=COLORS["black"], linewidth=1.2, zorder=1))

    n_rows = 1 + len(rows)
    row_h = h / n_rows

    # horizontal lines
    for i in range(1, n_rows):
        yy = y + i * row_h
        ax.add_line(Line2D([x, x + w], [yy, yy],
                           linewidth=1.0, color=COLORS["black"], zorder=2))

    # vertical lines
    running_x = x
    xs = [x]
    for frac in col_fracs:
        running_x += w * frac
        xs.append(running_x)

    for xx in xs[1:-1]:
        ax.add_line(Line2D([xx, xx], [y, y + h],
                           linewidth=1.0, color=COLORS["black"], zorder=2))

    # header
    left = x
    for head, frac in zip(headers, col_fracs):
        cw = w * frac
        ax.text(left + cw / 2, y + h - row_h / 2,
                head, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=COLORS["black"], zorder=3)
        left += cw

    # rows
    for r, row in enumerate(rows):
        left = x
        cy = y + h - (r + 1.5) * row_h
        for txt, frac in zip(row, col_fracs):
            cw = w * frac
            ax.text(left + cw / 2, cy, txt,
                    ha="center", va="center", fontsize=10.2, color=COLORS["black"], zorder=3)
            left += cw


# --------------------------------------------------
# main figure
# --------------------------------------------------

def generate_figure():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # title
    ax.text(
        0.5, 0.965,
        "Large-Scale Model-Based Certification Interface",
        ha="center", va="center",
        fontsize=22, fontweight="bold", color=COLORS["black"]
    )
    ax.text(
        0.5, 0.925,
        "From shard adapters and vote vectors to count-based baselines and shard-aware row-column MILPs",
        ha="center", va="center",
        fontsize=14, color=COLORS["black"]
    )

    # --------------------------------------------------
    # top pipeline
    # --------------------------------------------------

    top_y = 0.80
    top_h = 0.075
    box_w = 0.12
    gap = 0.025

    x1 = 0.03
    x2 = x1 + box_w + gap
    x3 = x2 + box_w + gap
    x4 = x3 + box_w + gap
    x5 = x4 + box_w + gap

    rounded_box(ax, x1, top_y, box_w, top_h,
                "Toucan\ntool-use data",
                facecolor=COLORS["blue_fill"], edgecolor=COLORS["black"],
                fontsize=13, weight="bold")
    rounded_box(ax, x2, top_y, box_w, top_h,
                "Disjoint\ntraining shards",
                facecolor=COLORS["blue_fill"], edgecolor=COLORS["black"],
                fontsize=13, weight="bold")
    rounded_box(ax, x3, top_y, box_w, top_h,
                "K=500 LoRA\nshard adapters",
                facecolor=COLORS["beige_fill"], edgecolor=COLORS["black"],
                fontsize=13, weight="bold")
    rounded_box(ax, x4, top_y, box_w, top_h,
                "Sequential\nadapter inference",
                facecolor=COLORS["beige_fill"], edgecolor=COLORS["black"],
                fontsize=13, weight="bold")
    rounded_box(ax, x5, top_y, box_w, top_h,
                "JSONL vote vectors\nper test prompt",
                facecolor=COLORS["lav_fill"], edgecolor=COLORS["black"],
                fontsize=13, weight="bold")

    arrow(ax, (x1 + box_w, top_y + top_h / 2), (x2, top_y + top_h / 2))
    arrow(ax, (x2 + box_w, top_y + top_h / 2), (x3, top_y + top_h / 2))
    arrow(ax, (x3 + box_w, top_y + top_h / 2), (x4, top_y + top_h / 2))
    arrow(ax, (x4 + box_w, top_y + top_h / 2), (x5, top_y + top_h / 2))

    # stored fields row
    sf_y = 0.655
    sf_h = 0.07

    final_x, final_w = 0.17, 0.14
    token_x, token_w = 0.37, 0.18
    horiz_x, horiz_w = 0.60, 0.16

    rounded_box(ax, final_x, sf_y, final_w, sf_h,
                "final_output[k]\none tool-call vote per shard",
                facecolor=COLORS["pink_fill"], edgecolor=COLORS["black"],
                fontsize=11.5, weight="bold")
    rounded_box(ax, token_x, sf_y, token_w, sf_h,
                "token_vote_matrix[k,l]\none token sequence per shard",
                facecolor=COLORS["green_fill"], edgecolor=COLORS["black"],
                fontsize=11.5, weight="bold")
    rounded_box(ax, horiz_x, sf_y, horiz_w, sf_h,
                "horizon filter H\nkeep prompts with no None padding",
                facecolor=COLORS["grey_fill"], edgecolor=COLORS["black"],
                fontsize=11.5, weight="bold")

    # a dedicated routing lane under the top pipeline
    route_y = 0.76
    start_x = x5 + box_w * 0.50

    # main routing line from JSONL down and across
    ax.add_line(Line2D([start_x, start_x], [top_y, route_y], color=COLORS["black"], linewidth=1.4, zorder=2))
    ax.add_line(Line2D([start_x, final_x + final_w / 2], [route_y, route_y], color=COLORS["black"], linewidth=1.4, zorder=2))

    # label
    ax.text((start_x + final_x + final_w / 2) / 2, route_y + 0.005,
            "stored fields",
            ha="center", va="bottom",
            fontsize=12, style="italic", color=COLORS["black"], zorder=3)

    # branch to final_output
    polyline_with_arrow(ax, [
        (final_x + final_w / 2, route_y),
        (final_x + final_w / 2, sf_y + sf_h),
    ])

    # branch to token_vote_matrix
    branch_x = token_x + token_w / 2
    ax.add_line(Line2D([branch_x, branch_x], [route_y, sf_y + sf_h + 0.01], color=COLORS["black"], linewidth=1.4, zorder=2))
    arrow(ax, (branch_x, sf_y + sf_h + 0.01), (branch_x, sf_y + sf_h), lw=1.4)

    # token to horizon
    arrow(ax, (token_x + token_w, sf_y + sf_h / 2), (horiz_x, sf_y + sf_h / 2), lw=1.4)

    # --------------------------------------------------
    # main panels
    # --------------------------------------------------

    left_px, left_py, left_pw, left_ph = 0.045, 0.365, 0.29, 0.27
    right_px, right_py, right_pw, right_ph = 0.365, 0.365, 0.29, 0.27
    alloc_x, alloc_y, alloc_w, alloc_h = 0.70, 0.44, 0.16, 0.10

    panel(ax, left_px, left_py, left_pw, left_ph,
          "A. Count-based final-output interface",
          edgecolor=COLORS["blue_edge"],
          title_color="#173b76")

    panel(ax, right_px, right_py, right_pw, right_ph,
          "B. Shard-aware token-grid interface",
          edgecolor=COLORS["green_edge"],
          title_color="#284d17")

    rounded_box(ax, alloc_x, alloc_y, alloc_w, alloc_h,
                "Shared poisoned-shard\nallocation a[k]\nwith sum a[k] <= B",
                facecolor=COLORS["lav_fill"], edgecolor=COLORS["black"],
                fontsize=12.5, weight="bold")

    # clean down arrows from stored field boxes to panels
    arrow(ax, (final_x + final_w / 2, sf_y), (final_x + final_w / 2, left_py + left_ph), lw=1.4)
    arrow(ax, (token_x + token_w / 2, sf_y), (token_x + token_w / 2, right_py + right_ph), lw=1.4)

    # left panel content
    bar_x = left_px + 0.02
    bar_y_top = left_py + 0.185
    bar_w = 0.16
    bar_h = 0.027
    bar_gap = 0.040

    draw_bar(ax, bar_x, bar_y_top, bar_w, bar_h, 0.68, COLORS["dark_blue"], "dictionary")
    draw_bar(ax, bar_x, bar_y_top - bar_gap, bar_w, bar_h, 0.21, COLORS["teal"], "get_definitions")
    draw_bar(ax, bar_x, bar_y_top - 2 * bar_gap, bar_w, bar_h, 0.14, COLORS["orange"], "detect")

    ax.text(left_px + left_pw / 2, left_py + 0.095,
            "aggregate tool votes",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=COLORS["black"], zorder=4)

    rounded_box(ax, left_px + 0.03, left_py + 0.015, 0.17, 0.055,
                "No shard identities needed\nonly aggregate vote counts",
                facecolor=COLORS["pink_fill"], edgecolor=COLORS["black"],
                fontsize=10.5)

    # right panel content
    draw_dynamic_grid(ax, right_px + 0.015, right_py + 0.015, right_pw - 0.03, right_ph - 0.03)

    # grid to allocation arrow
    arrow(ax, (right_px + right_pw, right_py + right_ph / 2), (alloc_x, alloc_y + alloc_h / 2), lw=1.4)

    # --------------------------------------------------
    # certification outputs
    # --------------------------------------------------

    ax.text(0.5, 0.325, "Certification outputs",
            ha="center", va="center",
            fontsize=16.5, fontweight="bold", color=COLORS["black"])

    out_y = 0.225
    out_h = 0.075
    out_w = 0.155
    out_gap = 0.02

    out1_x = 0.045
    out2_x = out1_x + out_w + out_gap
    out3_x = out2_x + out_w + out_gap
    out4_x = out3_x + out_w + out_gap

    rounded_box(ax, out1_x, out_y, out_w, out_h,
                "Final-tool DPA\nstability baseline",
                facecolor=COLORS["pink_fill"], edgecolor=COLORS["black"],
                fontsize=12)
    rounded_box(ax, out2_x, out_y, out_w, out_h,
                "Aggregate TPA\nfinal-tool validity",
                facecolor=COLORS["pink_fill"], edgecolor=COLORS["black"],
                fontsize=12)
    rounded_box(ax, out3_x, out_y, out_w, out_h,
                "Token-grid DPA\ndiagnostic",
                facecolor=COLORS["green_fill"], edgecolor=COLORS["black"],
                fontsize=12)
    rounded_box(ax, out4_x, out_y, out_w, out_h,
                "Joint row-column MILP\nshard-aware certificate",
                facecolor=COLORS["lav_fill"], edgecolor=COLORS["black"],
                fontsize=12)

    # left panel to first two outputs
    left_drop_x = left_px + left_pw / 2
    split_y = 0.30
    ax.add_line(Line2D([left_drop_x, left_drop_x], [left_py, split_y],
                       color=COLORS["black"], linewidth=1.4, zorder=2))
    ax.add_line(Line2D([out1_x + out_w / 2, out2_x + out_w / 2], [split_y, split_y],
                       color=COLORS["black"], linewidth=1.4, zorder=2))
    arrow(ax, (out1_x + out_w / 2, split_y), (out1_x + out_w / 2, out_y + out_h), lw=1.4)
    arrow(ax, (out2_x + out_w / 2, split_y), (out2_x + out_w / 2, out_y + out_h), lw=1.4)

    # right panel to third output
    grid_drop_x = right_px + right_pw / 2
    ax.add_line(Line2D([grid_drop_x, grid_drop_x], [right_py, split_y],
                       color=COLORS["black"], linewidth=1.4, zorder=2))
    arrow(ax, (out3_x + out_w / 2, split_y), (out3_x + out_w / 2, out_y + out_h), lw=1.4)
    ax.add_line(Line2D([grid_drop_x, out3_x + out_w / 2], [split_y, split_y],
                       color=COLORS["black"], linewidth=1.4, zorder=2))

    # allocation to fourth output
    ax.add_line(Line2D([alloc_x + alloc_w / 2, alloc_x + alloc_w / 2], [alloc_y, split_y],
                       color=COLORS["black"], linewidth=1.4, zorder=2))
    ax.add_line(Line2D([alloc_x + alloc_w / 2, out4_x + out_w / 2], [split_y, split_y],
                       color=COLORS["black"], linewidth=1.4, zorder=2))
    arrow(ax, (out4_x + out_w / 2, split_y), (out4_x + out_w / 2, out_y + out_h), lw=1.4)

    # --------------------------------------------------
    # table and footnote
    # --------------------------------------------------

    draw_table(ax, 0.045, 0.085, 0.81, 0.10)

    ax.text(
        0.45, 0.045,
        "Use as a setup figure near Section 5.4 or before the large-scale results. "
        "Replace example labels/counts with exact run metadata if needed.",
        ha="center", va="center",
        fontsize=10.5, style="italic", color=COLORS["black"]
    )

    return fig


def main():
    fig = generate_figure()

    png_path = OUTPUT_DIR / f"{OUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUT_BASENAME}.pdf"
    svg_path = OUTPUT_DIR / f"{OUT_BASENAME}.svg"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()