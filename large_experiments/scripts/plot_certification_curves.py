#!/usr/bin/env python3
"""Plot full-scale certification budget curves.

The script consumes one or more result directories produced by
large_experiments/scripts/certify_vote_vectors_runner.py.

Each result directory should contain:
- budget_curve_summary.csv
- summary.json

Plots are written as publication-ready PDF files.

Example:
python large_experiments/scripts/plot_certification_curves.py \
  --inputs large_experiments/outputs/certification/1b_full_targets2/H015 \
           large_experiments/outputs/certification/1b_full_targets2/H020 \
  --labels "1B full H=15" "1B full H=20" \
  --output-dir large_experiments/outputs/certification/plots
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "dpa_final_tool_stability": "DPA final-tool stability",
    "joint_row_column_stability_milp": "Joint row-column stability MILP",
    "aggregate_tpa_final_tool_validity": "Aggregate TPA final-tool validity",
    "dpa_token_grid_weakest_token_stability_diagnostic": (
        "DPA token-grid weakest-token stability diagnostic"
    ),
    "dpa_max_target_token_validity_diagnostic": (
        "DPA max-target-token validity diagnostic"
    ),
    "joint_row_column_validity_milp": "Joint row-column validity MILP",
}

METHOD_ORDER = [
    "dpa_final_tool_stability",
    "joint_row_column_stability_milp",
    "aggregate_tpa_final_tool_validity",
    "joint_row_column_validity_milp",
    "dpa_token_grid_weakest_token_stability_diagnostic",
    "dpa_max_target_token_validity_diagnostic",
]

METHOD_STYLES = {
    "dpa_final_tool_stability": {
        "linestyle": ":",
        "marker": "o",
        "markerfacecolor": "white",
        "zorder": 4,
    },
    "aggregate_tpa_final_tool_validity": {
        "linestyle": "--",
        "marker": "^",
        "markerfacecolor": "white",
        "zorder": 3,
    },
    "dpa_token_grid_weakest_token_stability_diagnostic": {
        "linestyle": "-.",
        "marker": "v",
        "markerfacecolor": "white",
        "zorder": 3,
    },
    "dpa_max_target_token_validity_diagnostic": {
        "linestyle": ":",
        "marker": "s",
        "markerfacecolor": "white",
        "zorder": 4,
    },
    "joint_row_column_stability_milp": {
        "linestyle": "-",
        "marker": "D",
        "zorder": 2,
    },
    "joint_row_column_validity_milp": {
        "linestyle": "-",
        "marker": "P",
        "zorder": 2,
    },
}

MILP_METHODS = {
    "joint_row_column_stability_milp",
    "joint_row_column_validity_milp",
}

LEGEND_FONT_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot certification budget curves from budget_curve_summary.csv files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more result directories containing budget_curve_summary.csv.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels, one per input directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write PDF plots and merged CSV.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Full-scale certification",
        help="Prefix used in plot titles.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Optional prefix added to every generated PDF and CSV filename.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive windows after saving plots.",
    )
    parser.add_argument(
        "--milp-only",
        action="store_true",
        help="Plot only the joint row-column stability and validity MILPs.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in (0, len(args.inputs)):
        parser.error("--labels must be omitted or have exactly one label per --inputs entry")
    if "/" in args.filename_prefix or "\\" in args.filename_prefix:
        parser.error("--filename-prefix must not contain path separators")

    return args


def load_summary(path: Path) -> dict[str, Any]:
    """Load optional run metadata from ``summary.json``."""
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open() as handle:
        return json.load(handle)


def default_label(path: Path, summary: dict[str, Any]) -> str:
    """Build a concise display label from a result path and its metadata."""
    name = summary.get("name", path.parent.name)
    horizon = summary.get("horizon")
    prompts = summary.get("num_prompts")
    if horizon is not None and prompts is not None:
        return f"{name} H={horizon} N={prompts}"
    if horizon is not None:
        return f"{name} H={horizon}"
    return str(path)


def load_run(path: Path, label: str | None) -> pd.DataFrame:
    """Load and validate one certification run for plotting."""
    csv_path = path / "budget_curve_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    summary = load_summary(path)
    run_label = label or default_label(path, summary)

    df = pd.read_csv(csv_path)
    required = {"budget", "method", "objective_mode", "num_prompts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    df["method"] = df["method"].astype(str)
    unknown_methods = sorted(set(df["method"]) - set(METHOD_LABELS))
    if unknown_methods:
        raise ValueError(
            f"{csv_path} contains unknown method names: {unknown_methods}. "
            "Regenerate this run with the current "
            "certify_vote_vectors_runner.py, or add the method explicitly "
            "to METHOD_LABELS if it is intentional."
        )

    if "certified_fraction" not in df.columns:
        df["certified_fraction"] = pd.NA
    if "certified_fraction_lower_bound" not in df.columns:
        df["certified_fraction_lower_bound"] = pd.NA

    certified_fraction = pd.to_numeric(
        df["certified_fraction"],
        errors="coerce",
    )
    certified_fraction_lower_bound = pd.to_numeric(
        df["certified_fraction_lower_bound"],
        errors="coerce",
    )
    is_milp = df["objective_mode"].eq("fixed_budget_adversarial_success")
    df["plot_fraction"] = certified_fraction.where(
        ~is_milp,
        certified_fraction_lower_bound,
    )
    df["certified_percent"] = 100.0 * df["plot_fraction"]

    df["run_label"] = run_label
    df["run_name"] = summary.get("name", path.parent.name)
    df["horizon"] = summary.get("horizon", pd.NA)
    df["num_retained_prompts"] = summary.get("num_prompts", pd.NA)
    df["num_total_rows_read"] = summary.get("num_total_rows_read", pd.NA)
    df["num_rows_filtered_shorter_than_horizon"] = summary.get(
        "num_rows_filtered_shorter_than_horizon", pd.NA
    )
    df["padding_policy"] = summary.get("padding_policy", "")
    df["horizon_filter_basis"] = summary.get("horizon_filter_basis", "")

    def family(method: str) -> str:
        if "stability" in method:
            return "stability"
        if "validity" in method:
            return "validity"
        return "other"

    df["family"] = df["method"].map(family)
    df["method_label"] = df["method"].map(lambda m: METHOD_LABELS.get(m, m))

    return df


def method_sort_key(method: str) -> tuple[int, str]:
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def method_plot_style(method: str) -> dict[str, Any]:
    return {
        "linewidth": 1.9,
        "markersize": 5,
        "markeredgewidth": 1.2,
        **METHOD_STYLES.get(method, {"linestyle": "-", "marker": "o"}),
    }


def dynamic_certified_fraction_ylim(values_percent: pd.Series) -> tuple[float, float]:
    """Return a five-point-rounded lower bound and a fixed 105% upper bound."""
    finite_values = [
        float(value)
        for value in values_percent
        if pd.notna(value) and math.isfinite(float(value))
    ]
    if not finite_values:
        return 0, 105
    ymin = 5 * math.floor(min(finite_values) / 5)
    return max(0, ymin), 105


def prefixed_filename(filename: str, prefix: str) -> str:
    normalized = prefix.strip().rstrip("_-")
    return f"{normalized}_{filename}" if normalized else filename


def save_pdf(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    filename_prefix: str,
) -> None:
    """Save a figure as a prefixed PDF inside the selected output directory."""
    path = output_dir / prefixed_filename(filename, filename_prefix)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Wrote {path}")


def plot_certified_fraction_curves(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    filename: str,
    filename_prefix: str,
    title: str,
    figsize: tuple[float, float],
) -> None:
    """Render and save certified-fraction curves for the supplied rows."""
    fig, ax = plt.subplots(figsize=figsize)
    methods = sorted(df["method"].unique(), key=method_sort_key)
    run_labels = list(dict.fromkeys(df["run_label"].tolist()))

    for run_label in run_labels:
        run_df = df[df["run_label"].eq(run_label)]
        for method in methods:
            method_df = run_df[run_df["method"].eq(method)].sort_values("budget")
            if method_df.empty:
                continue
            method_label = METHOD_LABELS[method]
            label = method_label if len(run_labels) == 1 else f"{run_label} · {method_label}"
            ax.plot(
                method_df["budget"],
                method_df["certified_percent"],
                label=label,
                **method_plot_style(method),
            )

    ax.set_xlabel("Poisoned shard budget B")
    ax.set_ylabel("Certified fraction (%)")
    ax.set_ylim(*dynamic_certified_fraction_ylim(df["certified_percent"]))
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    save_pdf(fig, output_dir, filename, filename_prefix)
    plt.close(fig)


def plot_family(
    df: pd.DataFrame,
    family: str,
    output_dir: Path,
    title_prefix: str,
    filename_prefix: str,
) -> None:
    subset = df[df["family"].eq(family)].copy()
    subset = subset.dropna(subset=["certified_percent"])
    if subset.empty:
        print(f"No rows for {family}, skipping")
        return

    plot_certified_fraction_curves(
        subset,
        output_dir=output_dir,
        filename=f"{family}_budget_curve.pdf",
        filename_prefix=filename_prefix,
        title=f"{title_prefix} {family} budget curve",
        figsize=(8.5, 5.0),
    )


def plot_all_methods(
    df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
    filename_prefix: str,
) -> None:
    subset = df.dropna(subset=["certified_percent"]).copy()
    if subset.empty:
        print("No plot-ready rows, skipping combined plot")
        return

    plot_certified_fraction_curves(
        subset,
        output_dir=output_dir,
        filename="all_methods_budget_curve.pdf",
        filename_prefix=filename_prefix,
        title=f"{title_prefix} all methods",
        figsize=(9.5, 5.6),
    )


def write_report(
    df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str,
) -> None:
    rows = []
    for (run_label, method), group in df.dropna(subset=["plot_fraction"]).groupby(["run_label", "method"]):
        group = group.sort_values("budget")
        rows.append(
            {
                "run_label": run_label,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "num_prompts": int(group["num_prompts"].dropna().iloc[0]) if group["num_prompts"].notna().any() else "",
                "min_certified_fraction": group["plot_fraction"].min(),
                "mean_certified_fraction": group["plot_fraction"].mean(),
                "final_budget": int(group["budget"].max()),
                "final_certified_fraction": group.loc[group["budget"].idxmax(), "plot_fraction"],
            }
        )
    report = pd.DataFrame(rows)
    path = output_dir / prefixed_filename(
        "method_comparison_summary.csv",
        filename_prefix,
    )
    report.to_csv(path, index=False)
    print(f"Wrote {path}")


def warn_solver_statuses(df: pd.DataFrame) -> None:
    if "solver_status" not in df.columns:
        return
    status_df = df[df["objective_mode"].eq("fixed_budget_adversarial_success")].copy()
    if status_df.empty:
        return
    bad = status_df[status_df["solver_status"].notna() & ~status_df["solver_status"].eq("OPTIMAL")]
    if bad.empty:
        return
    print("Warning: some MILP rows are not OPTIMAL")
    print(bad[["run_label", "budget", "method", "solver_status", "mip_gap"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels if args.labels else [None] * len(args.inputs)
    frames = [load_run(path, label) for path, label in zip(args.inputs, labels)]
    df = pd.concat(frames, ignore_index=True)
    if args.milp_only:
        df = df[df["method"].isin(MILP_METHODS)].copy()

    merged_path = args.output_dir / prefixed_filename(
        "plot_ready_budget_curves.csv",
        args.filename_prefix,
    )
    df.to_csv(merged_path, index=False)
    print(f"Wrote {merged_path}")

    warn_solver_statuses(df)
    write_report(df, args.output_dir, args.filename_prefix)

    plot_family(
        df,
        "stability",
        args.output_dir,
        args.title_prefix,
        args.filename_prefix,
    )
    plot_family(
        df,
        "validity",
        args.output_dir,
        args.title_prefix,
        args.filename_prefix,
    )
    plot_all_methods(
        df,
        args.output_dir,
        args.title_prefix,
        args.filename_prefix,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
