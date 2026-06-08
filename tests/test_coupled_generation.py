from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from gurobipy import GurobiError

from toy_experiments.data import compute_counts, generate_toy_votes, generate_validity_demo_votes, slice_toy_data
from toy_experiments.experiments import (
    _is_standard_size_benchmark_csv,
    _metric_series,
    _fixed_denominator_budget_series,
    _aggregate_budget_rows_across_seeds,
    _aggregate_result_rows_across_seeds,
    _summative_validity_stat_lines,
    benchmark_scale,
    load_experiment_config,
)
from toy_experiments.milp import solve_row_col_validity, solve_structured_stability


class SliceToyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.master = generate_toy_votes(K=9, N=5, L=6, T=7, seed=13)

    def test_n_slice_is_master_prefix(self) -> None:
        sliced = slice_toy_data(self.master, K=9, N=3, L=6, T=7)
        for name in ["stab_votes", "val_votes", "influence"]:
            np.testing.assert_array_equal(getattr(sliced, name), getattr(self.master, name)[:, :3, :])
        for name in ["stab_counts", "val_counts"]:
            np.testing.assert_array_equal(getattr(sliced, name), getattr(self.master, name)[:3, :, :])
        for name in ["clean_pred", "runner_up", "target", "base_token", "val_base"]:
            np.testing.assert_array_equal(getattr(sliced, name), getattr(self.master, name)[:3, :])

    def test_l_slice_is_master_prefix(self) -> None:
        sliced = slice_toy_data(self.master, K=9, N=5, L=4, T=7)
        for name in ["stab_votes", "val_votes", "influence"]:
            np.testing.assert_array_equal(getattr(sliced, name), getattr(self.master, name)[:, :, :4])
        for name in ["stab_counts", "val_counts"]:
            np.testing.assert_array_equal(getattr(sliced, name), getattr(self.master, name)[:, :4, :])
        for name in ["clean_pred", "runner_up", "target", "base_token", "val_base"]:
            np.testing.assert_array_equal(getattr(sliced, name), getattr(self.master, name)[:, :4])

    def test_k_slice_recomputes_counts(self) -> None:
        sliced = slice_toy_data(self.master, K=4, N=5, L=6, T=7)
        np.testing.assert_array_equal(sliced.stab_votes, self.master.stab_votes[:4])
        np.testing.assert_array_equal(sliced.val_votes, self.master.val_votes[:4])
        np.testing.assert_array_equal(sliced.influence, self.master.influence[:4])
        np.testing.assert_array_equal(sliced.stab_counts, compute_counts(self.master.stab_votes[:4], 7))
        np.testing.assert_array_equal(sliced.val_counts, compute_counts(self.master.val_votes[:4], 7))
        self.assertFalse(np.array_equal(sliced.stab_counts, self.master.stab_counts))

    def test_t_slice_projects_removed_candidates_and_recomputes(self) -> None:
        sliced = slice_toy_data(self.master, K=9, N=5, L=6, T=4)
        self.assertEqual(sliced.stab_counts.shape, (5, 6, 4))
        self.assertEqual(sliced.val_counts.shape, (5, 6, 4))
        self.assertLess(int(sliced.stab_votes.max()), 4)
        self.assertLess(int(sliced.val_votes.max()), 4)
        self.assertLess(int(sliced.target.max()), 4)
        np.testing.assert_array_equal(sliced.stab_counts, compute_counts(sliced.stab_votes, 4))
        np.testing.assert_array_equal(sliced.val_counts, compute_counts(sliced.val_votes, 4))

    def test_slice_does_not_alias_master(self) -> None:
        sliced = slice_toy_data(self.master, K=4, N=3, L=2, T=7)
        original = int(self.master.stab_votes[0, 0, 0])
        sliced.stab_votes[0, 0, 0] = (original + 1) % 7
        self.assertEqual(int(self.master.stab_votes[0, 0, 0]), original)
        self.assertTrue(sliced.metadata["coupled_generation"])

    def test_invalid_slice_dimensions_are_rejected(self) -> None:
        for kwargs in [
            {"K": 10, "N": 5, "L": 6, "T": 7},
            {"K": 9, "N": 6, "L": 6, "T": 7},
            {"K": 9, "N": 5, "L": 7, "T": 7},
            {"K": 9, "N": 5, "L": 6, "T": 8},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                slice_toy_data(self.master, **kwargs)


class CoupledBenchmarkTests(unittest.TestCase):
    def test_master_generator_called_once_per_distribution_group(self) -> None:
        real_generator = generate_toy_votes
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch(
                "toy_experiments.experiments.generate_toy_votes",
                side_effect=real_generator,
            ) as generator:
                rows = benchmark_scale(
                    Ks=[8, 16],
                    Ns=[2, 4],
                    Ls=[3, 5],
                    Ts=[5],
                    delta_stabs=[0.2],
                    delta_vals=[0.2],
                    target_biases=[0.3],
                    influence_mode="dense",
                    seed=0,
                    save_dir=output_dir,
                    make_budget_curves=False,
                    make_stability_objectives=False,
                    make_validity_objectives=False,
                    make_stability_budget_curves=False,
                    make_validity_budget_curves=False,
                )
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["coupled_generation"] for row in rows))
        self.assertTrue(all(row["K_master"] == 16 for row in rows))
        self.assertTrue(all(row["N_master"] == 4 for row in rows))
        self.assertTrue(all(row["L_master"] == 5 for row in rows))

    def test_existing_config_requires_no_coupling_option(self) -> None:
        config = load_experiment_config("toy_experiments/configs/small.yaml")
        self.assertNotIn("coupled_generation", config)
        self.assertNotIn("prefix_coupled_lengths", config)
        self.assertEqual(config["seeds"], [0])

    def test_sweep_configs_use_seed_replicates(self) -> None:
        for sweep in ["K", "N", "L", "degenerate"]:
            config = load_experiment_config(f"toy_experiments/configs/sweep_{sweep}.yaml")
            self.assertEqual(config["seeds"], [0, 10, 20, 30, 40])

    def test_legacy_single_seed_config_remains_supported(self) -> None:
        config = load_experiment_config("toy_experiments/configs/validity_demo.yaml")
        self.assertEqual(config["seeds"], [0])

    def test_master_generator_called_once_per_seed_and_distribution_group(self) -> None:
        real_generator = generate_toy_votes
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch(
                "toy_experiments.experiments.generate_toy_votes",
                side_effect=real_generator,
            ) as generator:
                rows = benchmark_scale(
                    Ks=[8, 16],
                    Ns=[2],
                    Ls=[3],
                    Ts=[5],
                    delta_stabs=[0.2],
                    delta_vals=[0.2],
                    target_biases=[0.3],
                    influence_mode="dense",
                    seed=[0, 25, 50],
                    save_dir=output_dir,
                    make_budget_curves=False,
                    make_stability_objectives=False,
                    make_validity_objectives=False,
                    make_stability_budget_curves=False,
                    make_validity_budget_curves=False,
                )
        self.assertEqual(generator.call_count, 3)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["seed"] for row in rows}, {0, 25, 50})

    def test_budget_curve_aggregation_averages_seed_groups(self) -> None:
        rows = [
            {"seed": 0, "K": 8, "budget": 0, "certified_fraction": 1.0},
            {"seed": 0, "K": 8, "budget": 1, "certified_fraction": 0.5},
            {"seed": 25, "K": 8, "budget": 0, "certified_fraction": 0.5},
            {"seed": 25, "K": 8, "budget": 1, "certified_fraction": 0.0},
        ]
        series, diagnostic = _fixed_denominator_budget_series(rows)
        self.assertIsNone(diagnostic)
        self.assertEqual(series, ([0.0, 1.0], [0.75, 0.25]))

    def test_seed_aggregate_csv_rows_include_mean_and_range(self) -> None:
        result_rows = [
            {"K": 8, "N": 2, "L": 3, "T": 5, "seed": 0, "row_col_val_qN": 4},
            {"K": 8, "N": 2, "L": 3, "T": 5, "seed": 25, "row_col_val_qN": 8},
            {"K": 8, "N": 2, "L": 3, "T": 5, "seed": 50, "row_col_val_qN": 6},
        ]
        aggregate = _aggregate_result_rows_across_seeds(result_rows)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["seed_count"], 3)
        self.assertEqual(aggregate[0]["seed_values"], "0,25,50")
        self.assertEqual(aggregate[0]["row_col_val_qN"], 6.0)
        self.assertEqual(aggregate[0]["row_col_val_qN_min"], 4.0)
        self.assertEqual(aggregate[0]["row_col_val_qN_max"], 8.0)
        self.assertEqual(aggregate[0]["row_col_val_qN_minus"], 2.0)
        self.assertEqual(aggregate[0]["row_col_val_qN_plus"], 2.0)

        budget_rows = [
            {"K": 8, "seed": 0, "budget": 1, "method": "M", "objective": "O", "certified_fraction": 0.2},
            {"K": 8, "seed": 25, "budget": 1, "method": "M", "objective": "O", "certified_fraction": 0.8},
        ]
        budget_aggregate = _aggregate_budget_rows_across_seeds(budget_rows)
        self.assertEqual(budget_aggregate[0]["certified_fraction"], 0.5)
        self.assertEqual(budget_aggregate[0]["certified_fraction_min"], 0.2)
        self.assertEqual(budget_aggregate[0]["certified_fraction_max"], 0.8)
        self.assertAlmostEqual(budget_aggregate[0]["certified_fraction_minus"], 0.3)
        self.assertAlmostEqual(budget_aggregate[0]["certified_fraction_plus"], 0.3)

    def test_validity_demo_k_expansion_is_reported_by_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "validity.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: test",
                        "generator: validity_demo",
                        "K_values: [4]",
                        "N_values: [1]",
                        "L_values: [3]",
                        "T_values: [8]",
                        "group_size: 4",
                        "overlap: 1",
                        "target_gap: 4",
                        "seed: 0",
                        "budget_max: 4",
                        "objective_family: validity_only",
                        "make_budget_curves: false",
                        "solver:",
                        "  gurobi_threads: 0",
                        f"output_dir: {directory}/results",
                    ]
                )
            )
            config = load_experiment_config(config_path)
        self.assertEqual(config["Ks"], [4])

    def test_summative_validity_report_is_limited_to_size_presets(self) -> None:
        for size in ["small", "medium", "large"]:
            path = Path("toy_experiments") / "outputs" / size / "results" / "benchmark_results.csv"
            self.assertTrue(_is_standard_size_benchmark_csv(path))
        self.assertFalse(
            _is_standard_size_benchmark_csv(
                Path("toy_experiments/outputs/validity_demo/results/benchmark_results.csv")
            )
        )

    def test_summative_validity_series_uses_additive_dpa_and_shared_milp(self) -> None:
        rows = [
            {"K": 8, "independent_val_sequence_qN": 12, "row_col_val_qN": 4},
            {"K": 8, "independent_val_sequence_qN": 16, "row_col_val_qN": 6},
            {"K": 16, "independent_val_sequence_qN": 20, "row_col_val_qN": 8},
        ]
        series, skipped = _metric_series(
            rows,
            "K",
            {
                "Summative DPA validity": "independent_val_sequence_qN",
                "Shared MILP validity": "row_col_val_qN",
            },
        )
        self.assertEqual(skipped, [])
        self.assertEqual(series["Summative DPA validity"], ([8.0, 16.0], [14.0, 20.0]))
        self.assertEqual(series["Shared MILP validity"], ([8.0, 16.0], [5.0, 8.0]))
        stats = _summative_validity_stat_lines(rows)
        self.assertIn("Overall mean summative DPA budget: 17", stats)
        self.assertIn("Overall mean shared MILP budget: 6.5", stats)
        self.assertIn("Mean additive overestimate: 10.5 budget units", stats)
        self.assertIn("Mean summative-DPA/shared-MILP ratio: 2.61538x", stats)

    def test_heterogeneous_validity_groups_survive_required_k_prefixes(self) -> None:
        master = generate_validity_demo_votes(
            L=3,
            group_size=8,
            target_gap=8,
            overlap=5,
            N=2,
            T=8,
            K=24,
            distribution="heterogeneous",
            num_competitor_max=4,
            target_count_max=2,
            competitor_gap_min=2,
            competitor_gap_max=4,
            competitor_jitter=1,
            minimum_requested_K=8,
        )
        for length, actual_k in [(1, 8), (2, 11), (3, 14)]:
            sliced = slice_toy_data(master, K=actual_k, N=2, L=length, T=8)
            np.testing.assert_array_equal(
                sliced.influence.sum(axis=0),
                np.full((2, length), 8, dtype=np.int64),
            )


class CoupledMonotonicityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.master = generate_validity_demo_votes(
            L=3,
            group_size=2,
            target_gap=2,
            overlap=1,
            N=1,
            T=4,
            K=4,
        )

    def test_full_sequence_validity_is_nondecreasing_with_l(self) -> None:
        budgets = []
        try:
            for length in [1, 2, 3]:
                data = slice_toy_data(self.master, K=4, N=1, L=length, T=4)
                result = solve_row_col_validity(
                    data.val_votes,
                    data.val_counts,
                    data.target,
                    4,
                    data.influence,
                    q_rows=1,
                    gurobi_threads=1,
                )
                self.assertIsNotNone(result.B_star)
                budgets.append(result.B_star)
        except GurobiError as exc:
            self.skipTest(f"Gurobi unavailable: {exc}")
        self.assertEqual(budgets, sorted(budgets))

    def test_any_token_stability_is_nonincreasing_with_l(self) -> None:
        budgets = []
        try:
            for length in [1, 2, 3]:
                data = slice_toy_data(self.master, K=4, N=1, L=length, T=4)
                result = solve_structured_stability(
                    data.stab_votes,
                    data.stab_counts,
                    data.clean_pred,
                    data.influence,
                    q_rows=1,
                    r_cols=1,
                    gurobi_threads=1,
                )
                self.assertIsNotNone(result.B_star)
                budgets.append(result.B_star)
        except GurobiError as exc:
            self.skipTest(f"Gurobi unavailable: {exc}")
        self.assertEqual(budgets, sorted(budgets, reverse=True))


if __name__ == "__main__":
    unittest.main()
