from __future__ import annotations

import argparse
from collections.abc import Iterable

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


def run_sanity(K: int = 7, N: int = 3, L: int = 4, T: int = 5, delta: float = 0.2, seed: int = 0) -> list[CertificateResult]:
    data = generate_toy_votes(K=K, N=N, L=L, T=T, delta=delta, seed=seed)
    _print_instance_summary(data, K, N, L, T, delta, seed)
    results = solve_default_certificates(data, T)
    print_certificate_table(results)
    return results


def solve_default_certificates(data: ToyData, T: int) -> list[CertificateResult]:
    q_all_rows = data.votes.shape[1]
    return [
        solve_row_stability(data.votes, data.clean_counts, data.clean_pred, data.runner_up),
        solve_col_stability(data.votes, data.clean_counts, data.clean_pred, data.runner_up),
        solve_row_col_stability(data.votes, data.clean_counts, data.clean_pred, data.runner_up, definition="any_cell"),
        solve_row_col_stability(data.votes, data.clean_counts, data.clean_pred, data.runner_up, definition="full_row"),
        solve_row_validity(data.votes, data.clean_counts, data.target, T),
        solve_col_validity(data.votes, data.clean_counts, data.target, T, definition="full_column"),
        solve_row_col_validity(data.votes, data.clean_counts, data.target, T, q_rows=1),
        solve_row_col_validity(data.votes, data.clean_counts, data.target, T, q_rows=q_all_rows),
    ]


def sweep_delta(K: int, N: int, L: int, T: int, deltas: Iterable[float], seed: int) -> None:
    rows = []
    for delta in deltas:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta=delta, seed=seed)
        rows.append(
            [
                delta,
                solve_row_stability(data.votes, data.clean_counts, data.clean_pred, data.runner_up).B_star,
                solve_row_validity(data.votes, data.clean_counts, data.target, T).B_star,
                solve_row_col_validity(data.votes, data.clean_counts, data.target, T, q_rows=1).B_star,
                solve_row_col_validity(data.votes, data.clean_counts, data.target, T, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["delta", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def sweep_length(K: int, N: int, lengths: Iterable[int], T: int, delta: float, seed: int) -> None:
    rows = []
    for L in lengths:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta=delta, seed=seed)
        rows.append(
            [
                L,
                solve_row_stability(data.votes, data.clean_counts, data.clean_pred, data.runner_up).B_star,
                solve_row_validity(data.votes, data.clean_counts, data.target, T).B_star,
                solve_row_col_validity(data.votes, data.clean_counts, data.target, T, q_rows=1).B_star,
                solve_row_col_validity(data.votes, data.clean_counts, data.target, T, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["L", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def sweep_prompts(K: int, prompts: Iterable[int], L: int, T: int, delta: float, seed: int) -> None:
    rows = []
    for N in prompts:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta=delta, seed=seed)
        rows.append(
            [
                N,
                solve_row_stability(data.votes, data.clean_counts, data.clean_pred, data.runner_up).B_star,
                solve_row_validity(data.votes, data.clean_counts, data.target, T).B_star,
                solve_row_col_validity(data.votes, data.clean_counts, data.target, T, q_rows=1).B_star,
                solve_row_col_validity(data.votes, data.clean_counts, data.target, T, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["N", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def print_certificate_table(results: list[CertificateResult]) -> None:
    print()
    print(f"{'Certificate':34} B*   Status")
    print("-" * 52)
    for result in results:
        b_star = "NA" if result.B_star is None else str(result.B_star)
        print(f"{result.name:34} {b_star:>2}   {result.status_name}")


def _print_instance_summary(data: ToyData, K: int, N: int, L: int, T: int, delta: float, seed: int) -> None:
    margins = stability_margins(data.clean_counts, data.clean_pred, data.runner_up)
    print(f"K={K}, N={N}, L={L}, T={T}, delta={delta}, seed={seed}")
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


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toy row/column certificate experiments.")
    parser.add_argument("command", choices=["sanity", "sweep-delta", "sweep-length", "sweep-prompts"])
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deltas", type=_parse_float_list, default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--lengths", type=_parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--prompts", type=_parse_int_list, default=[1, 2, 4, 8])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "sanity":
        run_sanity(K=args.K, N=args.N, L=args.L, T=args.T, delta=args.delta, seed=args.seed)
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
