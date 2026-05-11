import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from toy_certificate.data import generate_toy_votes
from toy_certificate import experiments
from toy_certificate.experiments import compute_reference_baselines, compute_structured_stability_grid, compute_validity_q_curve
from toy_certificate.milp import solve_row_col_validity, solve_structured_stability


class ToyGenerationTests(unittest.TestCase):
    def test_generation_uses_separate_stability_and_validity_votes(self):
        data = generate_toy_votes(K=5, N=2, L=3, T=4, delta_stab=0.1, delta_val=0.3, target_bias=0.2, seed=7)

        self.assertEqual(data.stab_votes.shape, (5, 2, 3))
        self.assertEqual(data.val_votes.shape, (5, 2, 3))
        self.assertEqual(data.stab_counts.shape, (2, 3, 4))
        self.assertEqual(data.val_counts.shape, (2, 3, 4))
        self.assertTrue((data.target != data.clean_pred).all())

    def test_reference_baselines_use_spec_column_names(self):
        data = generate_toy_votes(K=5, N=2, L=3, T=4, seed=3)
        baselines = compute_reference_baselines(data)

        expected = {
            "raw_dpa_stab_min_cell",
            "dpa_stab_cell_min",
            "dpa_stab_row_radius_q1",
            "dpa_stab_row_radius_qN",
            "dpa_val_cell_min",
            "dpa_val_row_weak_q1",
            "dpa_val_row_weak_qN",
            "raw_dpa_val_min_cell",
            "independent_stab_full_row_q1",
            "independent_stab_full_row_qN",
            "independent_stab_qN_rL",
            "independent_val_sequence_q1",
            "independent_val_sequence_qN",
            "independent_val_q1",
            "independent_val_qN",
            "phrase_dpa_val_q1",
            "phrase_dpa_val_qN",
            "phrase_independent_val_q1",
            "phrase_independent_val_qN",
        }
        self.assertTrue(expected.issubset(baselines))
        self.assertGreaterEqual(baselines["dpa_stab_row_radius_qN"], baselines["dpa_stab_row_radius_q1"])
        self.assertGreaterEqual(baselines["dpa_val_row_weak_qN"], baselines["dpa_val_row_weak_q1"])
        self.assertGreaterEqual(baselines["phrase_dpa_val_qN"], baselines["phrase_dpa_val_q1"])
        self.assertGreaterEqual(baselines["phrase_independent_val_qN"], baselines["phrase_dpa_val_qN"])


class ExperimentCliTests(unittest.TestCase):
    def test_benchmark_does_not_make_plots_by_default(self):
        parser = experiments.build_parser()

        args = parser.parse_args(["benchmark"])
        self.assertFalse(args.make_plots)

        args = parser.parse_args(["benchmark", "--make-plots"])
        self.assertTrue(args.make_plots)

    def test_metric_names_use_qN_and_rL_aliases_for_current_shape(self):
        self.assertEqual(experiments._csv_metric_name("row_col_stability_q1_r1", N=4, L=5), "row_col_stab_q1_r1")
        self.assertEqual(experiments._csv_metric_name("row_col_stability_q4_r5", N=4, L=5), "row_col_stab_qN_rL")
        self.assertEqual(experiments._csv_metric_name("row_col_validity_q4", N=4, L=5), "row_col_val_qN")

    def test_degenerate_shape_fills_qN_and_rL_columns(self):
        row = {
            "N": 1,
            "L": 1,
            "row_col_stab_q1_r1": 7,
            "row_col_val_q1": 5,
        }

        experiments._fill_degenerate_corner_columns(row)

        self.assertEqual(row["row_col_stab_q1_rL"], 7)
        self.assertEqual(row["row_col_stab_qN_r1"], 7)
        self.assertEqual(row["row_col_stab_qN_rL"], 7)
        self.assertEqual(row["row_col_val_qN"], 5)

    def test_legacy_csv_columns_are_copied_when_reading_rows(self):
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "legacy.csv"
            csv_path.write_text(
                "K,N,L,T,seed,row_col_stability_any_cell,row_col_validity_qN,phd_ref_validity_any_cell\n"
                "5,2,3,4,0,2,6,1\n"
            )

            rows = experiments._read_rows_csv(csv_path)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["K"], 5)
        self.assertEqual(row["row_col_stab_q1_r1"], 2)
        self.assertEqual(row["row_col_val_qN"], 6)
        self.assertEqual(row["dpa_val_cell_min"], 1)
        self.assertEqual(row["dpa_val_row_weak_q1"], 1)

    def test_plot_csv_regenerates_expected_svg_files_from_existing_rows(self):
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            csv_path = base / "benchmark_results.csv"
            csv_path.write_text(
                "K,N,L,T,delta_stab,delta_val,target_bias,seed,influence_mode,"
                "dpa_val_row_weak_q1,independent_val_sequence_q1,phrase_dpa_val_q1,"
                "row_col_val_q1,row_col_val_qN,dpa_stab_row_radius_q1,dpa_stab_row_radius_qN,"
                "row_col_stab_q1_r1,row_col_stab_q1_rL,row_col_stab_qN_r1,row_col_stab_qN_rL,"
                "independent_stab_full_row_qN,dpa_val_row_weak_qN\n"
                "5,2,2,3,0.2,0.2,0.2,0,dense,"
                "1,3,2,2,4,1,2,1,2,2,4,6,2\n"
            )

            rows = experiments.plot_benchmark_csv(str(csv_path), save_dir=str(base))

            self.assertEqual(len(rows), 1)
            self.assertTrue((base / "validity_one_prompt_by_L.svg").exists())
            self.assertTrue((base / "validity_all_prompts_by_L.svg").exists())
            self.assertTrue((base / "stability_one_prompt_by_L.svg").exists())
            self.assertTrue((base / "stability_all_prompts_by_L.svg").exists())
            self.assertTrue((base / "validity_scaling_by_L.svg").exists())
            self.assertTrue((base / "validity_independent_overestimate_by_L.svg").exists())
            self.assertTrue((base / "stability_structured_by_L.svg").exists())
            self.assertTrue((base / "stability_independent_overestimate_by_L.svg").exists())
            self.assertTrue((base / "validity_bias_sweep.svg").exists())
            one_prompt_validity_svg = (base / "validity_one_prompt_by_L.svg").read_text()
            validity_diagnostic_svg = (base / "validity_independent_overestimate_by_L.svg").read_text()
            one_prompt_stability_svg = (base / "stability_one_prompt_by_L.svg").read_text()
            diagnostic_svg = (base / "stability_independent_overestimate_by_L.svg").read_text()
            self.assertIn("shared MILP full sequence", one_prompt_validity_svg)
            self.assertIn("independent full sequence", one_prompt_validity_svg)
            self.assertNotIn("q1", one_prompt_validity_svg)
            self.assertIn("Independent composition overestimate", validity_diagnostic_svg)
            self.assertIn("shared MILP: full sequence", one_prompt_stability_svg)
            self.assertNotIn("qN", one_prompt_stability_svg)
            self.assertIn("independent overestimate factor", diagnostic_svg)
            self.assertFalse((base / "monotonicity_violations.csv").exists())

    def test_check_script_does_not_run_benchmark_or_plot_csv(self):
        script = Path("scripts/check.sh").read_text()

        self.assertIn("compileall toy_certificate tests", script)
        self.assertIn("unittest discover", script)
        self.assertIn("toy_certificate.experiments visualize", script)
        self.assertNotIn("toy_certificate.experiments benchmark", script)
        self.assertNotIn("toy_certificate.experiments plot-csv", script)

    def test_short_scripts_have_expected_roles(self):
        data_script = Path("scripts/data.sh").read_text()
        plot_script = Path("scripts/plot.sh").read_text()
        benchmark_script = Path("scripts/benchmark.sh").read_text()

        self.assertIn("toy_certificate.experiments benchmark", data_script)
        self.assertIn("toy_certificate.experiments plot-csv", plot_script)
        self.assertIn("scripts/data.sh", benchmark_script)
        self.assertIn("scripts/plot.sh", benchmark_script)


class GurobiBackedTests(unittest.TestCase):
    def test_stability_checks_all_competitors_not_only_runner_up(self):
        votes = np.array([[[1]], [[1]], [[1]]])
        counts = np.array([[[1, 0, 0, 0]]])
        clean_pred = np.array([[0]])
        runner_up = np.array([[1]])
        influence = np.ones((3, 1, 1), dtype=int)

        try:
            result = solve_structured_stability(votes, counts, clean_pred, runner_up, influence, q_rows=1, r_cols=1)
        except Exception as exc:
            if "Gurobi" in str(exc) or "license" in str(exc).lower():
                self.skipTest(f"Gurobi unavailable: {exc}")
            raise

        self.assertEqual(result.B_star, 1)
        self.assertTrue(result.is_optimal)
        self.assertIn((0, 0), result.attacked_cells)
        self.assertIsNotNone(result.lower_bound)
        self.assertIsNotNone(result.upper_bound)

        runner_result = solve_structured_stability(votes, counts, clean_pred, runner_up, influence, q_rows=1, r_cols=1, competitor_mode="runner_up")
        self.assertTrue(runner_result.B_star is None or runner_result.B_star >= result.B_star)

    def test_runner_up_stability_mode_works_and_invalid_mode_fails(self):
        votes = np.array([[[0]], [[1]], [[2]]])
        counts = np.array([[[2, 1, 0]]])
        clean_pred = np.array([[0]])
        runner_up = np.array([[1]])
        influence = np.ones((3, 1, 1), dtype=int)

        try:
            all_result = solve_structured_stability(votes, counts, clean_pred, runner_up, influence, q_rows=1, r_cols=1, competitor_mode="all")
            runner_result = solve_structured_stability(votes, counts, clean_pred, runner_up, influence, q_rows=1, r_cols=1, competitor_mode="runner_up")
        except Exception as exc:
            if "Gurobi" in str(exc) or "license" in str(exc).lower():
                self.skipTest(f"Gurobi unavailable: {exc}")
            raise

        self.assertEqual(runner_result.B_star, 1)
        if all_result.is_optimal and runner_result.is_optimal:
            self.assertGreaterEqual(runner_result.B_star, all_result.B_star)
        with self.assertRaises(ValueError):
            solve_structured_stability(votes, counts, clean_pred, runner_up, influence, q_rows=1, r_cols=1, competitor_mode="bad")

    def test_structured_stability_grid_and_validity_curve_shapes(self):
        data = generate_toy_votes(K=5, N=2, L=2, T=3, delta_stab=0.2, delta_val=0.2, target_bias=0.2, seed=0)
        try:
            stability_grid = compute_structured_stability_grid(data)
            q_curve = compute_validity_q_curve(data, T=3)
        except Exception as exc:
            if "Gurobi" in str(exc) or "license" in str(exc).lower():
                self.skipTest(f"Gurobi unavailable: {exc}")
            raise

        self.assertEqual(stability_grid.shape, (2, 2))
        self.assertEqual(len(q_curve), 2)
        self.assertGreaterEqual(q_curve[1], q_curve[0])

    def test_structured_objective_monotonicity(self):
        data = generate_toy_votes(K=5, N=2, L=2, T=3, delta_stab=0.2, delta_val=0.2, target_bias=0.2, seed=11)
        try:
            stab_q1_r1 = solve_structured_stability(
                data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=1, r_cols=1
            ).B_star
            stab_q1_rL = solve_structured_stability(
                data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=1, r_cols=2
            ).B_star
            stab_qN_r1 = solve_structured_stability(
                data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=2, r_cols=1
            ).B_star
            stab_qN_rL = solve_structured_stability(
                data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=2, r_cols=2
            ).B_star
            val_q1 = solve_row_col_validity(data.val_votes, data.val_counts, data.target, T=3, influence=data.influence, q_rows=1).B_star
            val_qN = solve_row_col_validity(data.val_votes, data.val_counts, data.target, T=3, influence=data.influence, q_rows=2).B_star
        except Exception as exc:
            if "Gurobi" in str(exc) or "license" in str(exc).lower():
                self.skipTest(f"Gurobi unavailable: {exc}")
            raise

        self.assertLessEqual(stab_q1_r1, stab_q1_rL)
        self.assertLessEqual(stab_q1_r1, stab_qN_r1)
        self.assertLessEqual(stab_q1_rL, stab_qN_rL)
        self.assertLessEqual(stab_qN_r1, stab_qN_rL)
        self.assertLessEqual(val_q1, val_qN)


if __name__ == "__main__":
    unittest.main()
