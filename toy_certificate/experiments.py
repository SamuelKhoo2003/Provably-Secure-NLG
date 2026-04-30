from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .data import ToyData, generate_toy_votes, stability_margins
from .milp import (
    CertificateResult,
    solve_col_stability,
    solve_col_validity,
    solve_row_col_stability,
    solve_row_col_validity,
    solve_row_stability,
    solve_row_validity,
)


def run_sanity(
    K: int = 7,
    N: int = 3,
    L: int = 4,
    T: int = 5,
    delta_stab: float = 0.2,
    delta_val: float = 0.2,
    target_bias: float = 0.2,
    seed: int = 0,
    show_grid: bool = False,
    save_dir: str | None = None,
) -> list[CertificateResult]:
    data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta_stab, delta_val=delta_val, target_bias=target_bias, seed=seed)
    _print_instance_summary(data, K, N, L, T, delta_stab, delta_val, target_bias, seed)
    if show_grid:
        print_console_grid(data)
    if save_dir is not None:
        save_instance_plots(data, Path(save_dir), K=K, N=N, L=L, T=T, delta_stab=delta_stab, delta_val=delta_val, target_bias=target_bias, seed=seed)
    results = solve_default_certificates(data, T)
    print_certificate_table(results)
    return results


def solve_default_certificates(data: ToyData, T: int) -> list[CertificateResult]:
    q_all_rows = data.stab_votes.shape[1]
    return [
        solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence),
        solve_col_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence),
        solve_row_col_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, definition="any_cell"),
        solve_row_col_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, definition="full_row"),
        solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence),
        solve_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, definition="full_column"),
        solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1),
        solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=q_all_rows),
    ]


def sweep_delta(K: int, N: int, L: int, T: int, deltas: Iterable[float], seed: int) -> None:
    rows = []
    for delta in deltas:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta, delta_val=delta, seed=seed)
        rows.append(
            [
                delta,
                solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence).B_star,
                solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["delta", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def sweep_length(K: int, N: int, lengths: Iterable[int], T: int, delta: float, seed: int) -> None:
    rows = []
    for L in lengths:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta, delta_val=delta, seed=seed)
        rows.append(
            [
                L,
                solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence).B_star,
                solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["L", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def sweep_prompts(K: int, prompts: Iterable[int], L: int, T: int, delta: float, seed: int) -> None:
    rows = []
    for N in prompts:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta, delta_val=delta, seed=seed)
        rows.append(
            [
                N,
                solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence).B_star,
                solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["N", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def visualize_instance(K: int, N: int, L: int, T: int, delta_stab: float, delta_val: float, target_bias: float, seed: int, save_dir: str) -> None:
    data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta_stab, delta_val=delta_val, target_bias=target_bias, seed=seed)
    _print_instance_summary(data, K, N, L, T, delta_stab, delta_val, target_bias, seed)
    print_console_grid(data)
    save_instance_plots(data, Path(save_dir), K=K, N=N, L=L, T=T, delta_stab=delta_stab, delta_val=delta_val, target_bias=target_bias, seed=seed)


def benchmark_scale(
    Ks: Iterable[int],
    Ns: Iterable[int],
    Ls: Iterable[int],
    Ts: Iterable[int],
    deltas: Iterable[float],
    target_bias: float,
    seed: int,
    save_dir: str,
) -> list[dict[str, object]]:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for K in Ks:
        for N in Ns:
            for L in Ls:
                for T in Ts:
                    for delta in deltas:
                        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta, delta_val=delta, target_bias=target_bias, seed=seed)
                        results = _solve_benchmark_certificates(data, T)
                        row = {
                            "K": K,
                            "N": N,
                            "L": L,
                            "T": T,
                            "delta": delta,
                            "delta_stab": delta,
                            "delta_val": delta,
                            "target_bias": target_bias,
                            "seed": seed,
                        }
                        for result in results:
                            metric_name = "row_col_validity_qN" if result.name == f"row_col_validity_q{N}" else result.name
                            row[metric_name] = result.B_star
                        row.update(compute_reference_baselines(data))
                        rows.append(row)
                        print(
                            "bench "
                            f"K={K} N={N} L={L} T={T} delta={delta}: "
                            + ", ".join(f"{result.name}={result.B_star}" for result in results)
                        )

    csv_path = output_dir / "benchmark_results.csv"
    _write_rows_csv(csv_path, rows)
    save_benchmark_plots(rows, output_dir)
    print()
    print(f"Wrote benchmark CSV: {csv_path}")
    print(f"Wrote benchmark plots under: {output_dir}")
    return rows


def plot_benchmark_csv(csv_path: str, save_dir: str | None = None) -> list[dict[str, object]]:
    path = Path(csv_path)
    output_dir = Path(save_dir) if save_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows_csv(path)
    save_benchmark_plots(rows, output_dir)
    print(f"Wrote benchmark plots under: {output_dir}")
    return rows


def print_console_grid(data: ToyData) -> None:
    """Print a compact per-cell grid and shard vote layers."""
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
    winner_counts = np.take_along_axis(data.stab_counts, data.clean_pred[:, :, None], axis=2)[:, :, 0]
    target_counts = np.take_along_axis(data.val_counts, data.target[:, :, None], axis=2)[:, :, 0]

    print()
    print("cell grid: pred->target | runner | stab_margin | stab_winner_count | val_target_count")
    for i in range(data.clean_pred.shape[0]):
        cells = []
        for j in range(data.clean_pred.shape[1]):
            cells.append(
                f"{int(data.clean_pred[i, j])}->{int(data.target[i, j])}"
                f" r{int(data.runner_up[i, j])}"
                f" m{int(margins[i, j])}"
                f" w{int(winner_counts[i, j])}"
                f" h{int(target_counts[i, j])}"
            )
        print(f"row {i:02d}: " + " | ".join(cells))

    max_layers_to_print = 20
    print()
    print("stability shard vote layers stab_votes[k, row, col]:")
    for k in range(min(data.votes.shape[0], max_layers_to_print)):
        print(f"k={k:02d}")
        print(data.stab_votes[k])
    if data.votes.shape[0] > max_layers_to_print:
        print(f"... omitted {data.votes.shape[0] - max_layers_to_print} shard layers")

    print()
    print("validity shard vote layers val_votes[k, row, col]:")
    for k in range(min(data.val_votes.shape[0], max_layers_to_print)):
        print(f"k={k:02d}")
        print(data.val_votes[k])
    if data.val_votes.shape[0] > max_layers_to_print:
        print(f"... omitted {data.val_votes.shape[0] - max_layers_to_print} shard layers")


def save_instance_plots(
    data: ToyData, output_dir: Path, K: int, N: int, L: int, T: int, delta_stab: float, delta_val: float, target_bias: float, seed: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
    winner_counts = np.take_along_axis(data.stab_counts, data.clean_pred[:, :, None], axis=2)[:, :, 0]
    target_counts = np.take_along_axis(data.val_counts, data.target[:, :, None], axis=2)[:, :, 0]
    disagreement = 1.0 - (winner_counts / K)

    title_suffix = f"K={K}, N={N}, L={L}, T={T}, delta_stab={delta_stab}, delta_val={delta_val}, target_bias={target_bias}, seed={seed}"
    _save_heatmap_svg(data.clean_pred, output_dir / "clean_predictions.svg", "Clean predictions | " + title_suffix)
    _save_heatmap_svg(data.target, output_dir / "harmful_targets.svg", "Harmful targets | " + title_suffix)
    _save_heatmap_svg(margins, output_dir / "stability_margins.svg", "Winner vs runner-up margins | " + title_suffix)
    _save_heatmap_svg(winner_counts, output_dir / "winner_counts.svg", "Winner vote counts | " + title_suffix)
    _save_heatmap_svg(target_counts, output_dir / "validity_target_counts.svg", "Target validity vote counts | " + title_suffix)
    _save_heatmap_svg(disagreement, output_dir / "stability_disagreement_rate.svg", "Stability cell disagreement rate | " + title_suffix, fmt=".2f")

    print()
    print(f"Wrote instance plots under: {output_dir}")


def save_benchmark_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return

    axis_names = ["K", "N", "L", "T", "delta"]
    for axis_name in axis_names:
        _save_focused_plot(
            rows,
            output_dir / f"focused_stability_by_{axis_name}.svg",
            title=f"Row-column stability comparison by {axis_name}",
            axis_name=axis_name,
            metrics={
                "shared MILP: full row": "row_col_stability_full_row",
                "naive DPA: independent full row": "naive_dpa_stability_full_row",
                "PHD ref: any cell margin": "phd_ref_stability_any_cell",
            },
        )
        _save_focused_plot(
            rows,
            output_dir / f"focused_validity_q1_by_{axis_name}.svg",
            title=f"Row-column validity q1 comparison by {axis_name}",
            axis_name=axis_name,
            metrics={
                "shared MILP: full target row": "row_col_validity_q1",
                "naive DPA: independent target row": "naive_dpa_validity_q1",
                "PHD ref: any target cell": "phd_ref_validity_any_cell",
            },
        )
        _save_focused_plot(
            rows,
            output_dir / f"focused_validity_qN_by_{axis_name}.svg",
            title=f"Row-column validity qN comparison by {axis_name}",
            axis_name=axis_name,
            metrics={
                "shared MILP: all rows": "row_col_validity_qN",
                "naive DPA: independent all rows": "naive_dpa_validity_qN",
            },
        )


def print_certificate_table(results: list[CertificateResult]) -> None:
    print()
    print(f"{'Certificate':34} B*   Status")
    print("-" * 52)
    for result in results:
        b_star = "NA" if result.B_star is None else str(result.B_star)
        print(f"{result.name:34} {b_star:>2}   {result.status_name}")


def _print_instance_summary(data: ToyData, K: int, N: int, L: int, T: int, delta_stab: float, delta_val: float, target_bias: float, seed: int) -> None:
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
    print(f"K={K}, N={N}, L={L}, T={T}, delta_stab={delta_stab}, delta_val={delta_val}, target_bias={target_bias}, seed={seed}")
    print()
    print("clean predictions:")
    print(data.clean_pred)
    print()
    print("harmful targets:")
    print(data.target)
    print()
    print("winner-vs-runner-up margins:")
    print(margins)


def _print_sweep_table(headers: list[str], rows: list[list[object]]) -> None:
    widths = [max(len(str(header)), *(len(str(row[idx])) for row in rows)) for idx, header in enumerate(headers)]
    header_line = " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers))
    print(header_line)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def _solve_benchmark_certificates(data: ToyData, T: int) -> list[CertificateResult]:
    N = data.stab_votes.shape[1]
    return [
        solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence),
        solve_col_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence),
        solve_row_col_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, definition="any_cell"),
        solve_row_col_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, definition="full_row"),
        solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence),
        solve_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, definition="full_column"),
        solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1),
        solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N),
    ]


def compute_reference_baselines(data: ToyData) -> dict[str, int]:
    stability_cell_budgets = _cell_stability_budgets(data)
    validity_cell_budgets = _cell_validity_budgets(data)
    return {
        # Naive DPA: solve each token independently, then add independent token costs.
        "naive_dpa_stability_full_row": int(np.min(stability_cell_budgets.sum(axis=1))),
        "naive_dpa_validity_q1": int(np.min(validity_cell_budgets.sum(axis=1))),
        "naive_dpa_validity_qN": int(validity_cell_budgets.sum()),
        # PHD-style reference: single-cell majority-margin baselines.
        # These are intentionally weaker/easier objectives than full sequence certificates.
        "phd_ref_stability_any_cell": int(np.min(_phd_margin_stability_budgets(data))),
        "phd_ref_validity_any_cell": int(np.min(validity_cell_budgets)),
    }


def _cell_stability_budgets(data: ToyData) -> np.ndarray:
    K, N, L = data.stab_votes.shape
    budgets = np.zeros((N, L), dtype=np.int64)
    for i in range(N):
        for j in range(L):
            w = int(data.clean_pred[i, j])
            c = int(data.runner_up[i, j])
            deficit = int(data.stab_counts[i, j, w] - data.stab_counts[i, j, c])
            contributions = [
                int(data.influence[k, i, j]) * (int(data.stab_votes[k, i, j] != c) + int(data.stab_votes[k, i, j] == w)) for k in range(K)
            ]
            budgets[i, j] = _min_budget_from_contributions(deficit, contributions)
    return budgets


def _cell_validity_budgets(data: ToyData) -> np.ndarray:
    K, N, L = data.val_votes.shape
    T = data.val_counts.shape[2]
    budgets = np.zeros((N, L), dtype=np.int64)
    for i in range(N):
        for j in range(L):
            h = int(data.target[i, j])
            target_count = int(data.val_counts[i, j, h])
            budget_candidates = []
            for c in range(T):
                if c == h:
                    continue
                deficit = int(data.val_counts[i, j, c]) - target_count
                contributions = [
                    int(data.influence[k, i, j]) * (int(data.val_votes[k, i, j] != h) + int(data.val_votes[k, i, j] == c)) for k in range(K)
                ]
                budget_candidates.append(_min_budget_from_contributions(deficit, contributions))
            budgets[i, j] = max(budget_candidates)
    return budgets


def _phd_margin_stability_budgets(data: ToyData) -> np.ndarray:
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
    return ((margins + 1) // 2).astype(np.int64)


def _min_budget_from_contributions(deficit: int, contributions: list[int]) -> int:
    if deficit <= 0:
        return 0
    running = 0
    for budget, contribution in enumerate(sorted(contributions, reverse=True), start=1):
        running += contribution
        if running >= deficit:
            return budget
    return len(contributions)


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_focused_plot(rows: list[dict[str, object]], path: Path, title: str, axis_name: str, metrics: dict[str, str]) -> None:
    series = {}
    for label, metric in metrics.items():
        xs, ys = [], []
        grouped: dict[float, list[float]] = {}
        for row in rows:
            value = row.get(metric)
            if value in {None, ""}:
                continue
            grouped.setdefault(float(row[axis_name]), []).append(float(value))
        for x in sorted(grouped):
            xs.append(x)
            ys.append(float(np.mean(grouped[x])))
        if xs:
            series[label] = (xs, ys)
    _save_line_plot_svg(path, title, axis_name, "Mean poison budget B*", series)


def _read_rows_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {"K", "N", "L", "T", "seed"}:
                    parsed[key] = int(value)
                elif key == "delta" or _looks_numeric(value):
                    parsed[key] = float(value) if "." in value else int(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _certificate_metric_names(rows: list[dict[str, object]]) -> list[str]:
    parameter_names = {"K", "N", "L", "T", "delta", "seed"}
    metrics = []
    for row in rows:
        for key, value in row.items():
            if key in parameter_names or value in {None, ""}:
                continue
            if key not in metrics:
                metrics.append(key)
    return metrics


def _looks_numeric(value: object) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _save_heatmap_svg(matrix: np.ndarray, path: Path, title: str, fmt: str = ".0f") -> None:
    rows, cols = matrix.shape
    cell = 54
    left = 70
    top = 58
    width = left + cols * cell + 24
    height = top + rows * cell + 56
    values = matrix.astype(float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">{_xml_escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">token column j</text>',
        f'<text x="16" y="{top + rows * cell / 2:.1f}" transform="rotate(-90 16 {top + rows * cell / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">prompt row i</text>',
    ]
    for j in range(cols):
        svg.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 10}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{j}</text>')
    for i in range(rows):
        svg.append(f'<text x="{left - 12}" y="{top + i * cell + cell / 2 + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">{i}</text>')
        for j in range(cols):
            value = float(values[i, j])
            color = _heat_color(value, vmin, vmax)
            x = left + j * cell
            y = top + i * cell
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            svg.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#ffffff">{format(value, fmt)}</text>'
            )
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def _save_line_plot_svg(path: Path, title: str, x_label: str, y_label: str, series: dict[str, tuple[list[float], list[float]]]) -> None:
    width = 1080
    height = max(540, 430 + 22 * len(series))
    left, right, top, bottom = 72, 260, 52, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_x = [x for xs, _ in series.values() for x in xs]
    all_y = [y for _, ys in series.values() for y in ys]
    if not all_x or not all_y:
        return
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(0.0, min(all_y)), max(all_y)
    if xmin == xmax:
        xmin -= 1
        xmax += 1
    if ymin == ymax:
        ymax += 1
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be123c", "#4d7c0f", "#9333ea", "#475569"]

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#111827">{_xml_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{_xml_escape(x_label)}</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{_xml_escape(y_label)}</text>',
    ]
    for tick in range(5):
        y_value = ymin + (ymax - ymin) * tick / 4
        y = sy(y_value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">{y_value:.1f}</text>')
    for tick in range(5):
        x_value = xmin + (xmax - xmin) * tick / 4
        x = sx(x_value)
        svg.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{x_value:.2g}</text>')

    for idx, (name, (xs, ys)) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in zip(xs, ys):
            svg.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        legend_y = top + 18 + idx * 20
        svg.append(f'<rect x="{left + plot_w + 24}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{left + plot_w + 42}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{_xml_escape(name)}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def _heat_color(value: float, vmin: float, vmax: float) -> str:
    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    start = np.array([37, 99, 235], dtype=float)
    mid = np.array([20, 184, 166], dtype=float)
    end = np.array([234, 179, 8], dtype=float)
    if t < 0.5:
        rgb = start + (mid - start) * (t / 0.5)
    else:
        rgb = mid + (end - mid) * ((t - 0.5) / 0.5)
    return "#" + "".join(f"{int(round(c)):02x}" for c in rgb)


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _parse_float_list(value: str) -> list[float]:
    if isinstance(value, list):
        return value
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    if isinstance(value, list):
        return value
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toy row/column certificate experiments.")
    parser.add_argument("command", choices=["sanity", "visualize", "benchmark", "plot-csv", "sweep-delta", "sweep-length", "sweep-prompts"])
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--delta-stab", type=float, default=None)
    parser.add_argument("--delta-val", type=float, default=None)
    parser.add_argument("--target-bias", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deltas", type=_parse_float_list, default=[0.0, 0.2, 0.4])
    parser.add_argument("--Ks", type=_parse_int_list, default=[5, 7, 9])
    parser.add_argument("--Ns", type=_parse_int_list, default=[2, 3, 5])
    parser.add_argument("--lengths", type=_parse_int_list, default=[2, 4])
    parser.add_argument("--prompts", type=_parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--Ts", type=_parse_int_list, default=[3, 5])
    parser.add_argument("--save-dir", default="toy_results")
    parser.add_argument("--csv", default="toy_results/benchmark_large/benchmark_results.csv")
    parser.add_argument("--show-grid", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    delta_stab = args.delta if args.delta_stab is None else args.delta_stab
    delta_val = args.delta if args.delta_val is None else args.delta_val
    if args.command == "sanity":
        run_sanity(
            K=args.K,
            N=args.N,
            L=args.L,
            T=args.T,
            delta_stab=delta_stab,
            delta_val=delta_val,
            target_bias=args.target_bias,
            seed=args.seed,
            show_grid=args.show_grid,
            save_dir=args.save_dir if args.show_grid else None,
        )
    elif args.command == "visualize":
        visualize_instance(
            K=args.K,
            N=args.N,
            L=args.L,
            T=args.T,
            delta_stab=delta_stab,
            delta_val=delta_val,
            target_bias=args.target_bias,
            seed=args.seed,
            save_dir=args.save_dir,
        )
    elif args.command == "benchmark":
        benchmark_scale(Ks=args.Ks, Ns=args.Ns, Ls=args.lengths, Ts=args.Ts, deltas=args.deltas, target_bias=args.target_bias, seed=args.seed, save_dir=args.save_dir)
    elif args.command == "plot-csv":
        plot_benchmark_csv(args.csv, save_dir=args.save_dir)
    elif args.command == "sweep-delta":
        sweep_delta(K=args.K, N=args.N, L=args.L, T=args.T, deltas=args.deltas, seed=args.seed)
    elif args.command == "sweep-length":
        sweep_length(K=args.K, N=args.N, lengths=args.lengths, T=args.T, delta=args.delta, seed=args.seed)
    elif args.command == "sweep-prompts":
        sweep_prompts(K=args.K, prompts=args.prompts, L=args.L, T=args.T, delta=args.delta, seed=args.seed)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
