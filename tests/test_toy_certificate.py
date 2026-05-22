import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from toy_certificate.data import generate_toy_votes
from toy_certificate import experiments
from toy_certificate.experiments import (
    aggregate_tpa_sequence_baselines,
    certified_fraction_from_radii,
    compute_reference_baselines,
    compute_horizon_curve_rows,
    compute_radius_derived_budget_curve_rows,
    compute_structured_stability_grid,
    compute_validity_q_curve,
    prefix_horizons_from_token_radii,
    targeted_partition_radius,
)
from toy_certificate.milp import maximize_attacked_rows_stability, solve_row_col_validity, solve_structured_stability


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
            "tpa_val_cell_min",
            "tpa_val_sequence_q1",
            "tpa_val_sequence_qN",
            "tpa_val_sequence_mean",
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
        self.assertGreaterEqual(baselines["tpa_val_sequence_qN"], baselines["tpa_val_sequence_q1"])
        self.assertGreaterEqual(baselines["phrase_dpa_val_qN"], baselines["phrase_dpa_val_q1"])
        self.assertGreaterEqual(baselines["phrase_independent_val_qN"], baselines["phrase_dpa_val_qN"])


class TargetedPartitionBaselineTests(unittest.TestCase):
    def test_targeted_partition_radius_already_tied_or_winning(self):
        self.assertEqual(targeted_partition_radius(np.array([3, 2, 1]), target=0), 0)
        self.assertEqual(targeted_partition_radius(np.array([2, 2, 1]), target=1), 0)

    def test_targeted_partition_radius_one_behind_leader(self):
        self.assertEqual(targeted_partition_radius(np.array([3, 2, 0]), target=1), 1)

    def test_targeted_partition_radius_far_behind_multiple_competitors(self):
        self.assertEqual(targeted_partition_radius(np.array([5, 5, 1]), target=2), 3)

    def test_targeted_partition_radius_multiple_competitors_tied_above_target(self):
        self.assertEqual(targeted_partition_radius(np.array([4, 4, 2]), target=2), 2)

    def test_targeted_partition_radius_zero_votes_and_small_vocab(self):
        self.assertEqual(targeted_partition_radius(np.array([0, 0, 0]), target=1), 0)
        self.assertEqual(targeted_partition_radius(np.array([0]), target=0), 0)
        self.assertEqual(targeted_partition_radius(np.array([0, 2]), target=0), 1)

    def test_targeted_partition_radius_validates_inputs(self):
        with self.assertRaises(ValueError):
            targeted_partition_radius(np.array([[1, 2]]), target=0)
        with self.assertRaises(ValueError):
            targeted_partition_radius(np.array([1, 2]), target=2)
        with self.assertRaises(ValueError):
            targeted_partition_radius(np.array([1, -1]), target=0)

    def test_tpa_sequence_aggregation_uses_max_token_per_row(self):
        summary = aggregate_tpa_sequence_baselines(np.array([[1, 4, 2], [3, 0, 2], [5, 1, 1]]))

        self.assertEqual(summary["tpa_val_sequence_q1"], 3)
        self.assertEqual(summary["tpa_val_sequence_qN"], 5)
        self.assertAlmostEqual(summary["tpa_val_sequence_mean"], 4.0)


class BudgetCurveHelperTests(unittest.TestCase):
    def test_certified_fraction_from_radii_uses_strict_inequality(self):
        rows = certified_fraction_from_radii(np.array([1, 3, 5]), budgets=[0, 1, 3, 5])

        fractions = {row["budget"]: row["certified_fraction"] for row in rows}
        self.assertEqual(fractions[0], 1.0)
        self.assertEqual(fractions[1], 2 / 3)
        self.assertEqual(fractions[3], 1 / 3)
        self.assertEqual(fractions[5], 0.0)

    def test_certified_fraction_treats_zero_and_unknown_radii_conservatively(self):
        rows = certified_fraction_from_radii(np.array([0, np.nan, np.inf, 2]), budgets=[0, 1, 2])

        by_budget = {row["budget"]: row for row in rows}
        self.assertEqual(by_budget[0]["certified_fraction"], 1 / 4)
        self.assertEqual(by_budget[1]["certified_fraction"], 1 / 4)
        self.assertEqual(by_budget[2]["certified_fraction"], 0.0)
        self.assertEqual(by_budget[0]["num_known"], 2)
        self.assertEqual(by_budget[0]["num_unknown"], 2)
        self.assertEqual(by_budget[0]["num_total"], 4)

    def test_prefix_horizon_stops_at_first_uncertified_token(self):
        token_radii = np.array([[2, 4, 1]])

        self.assertEqual(prefix_horizons_from_token_radii(token_radii, budget=0).tolist(), [3])
        self.assertEqual(prefix_horizons_from_token_radii(token_radii, budget=1).tolist(), [2])
        self.assertEqual(prefix_horizons_from_token_radii(token_radii, budget=2).tolist(), [0])

    def test_new_long_format_curve_rows_have_expected_columns(self):
        data = generate_toy_votes(K=4, N=2, L=2, T=4, delta_stab=0.2, delta_val=0.2, target_bias=0.3, seed=5)
        metadata = {
            "seed": 5,
            "K": 4,
            "N": 2,
            "L": 2,
            "T": 4,
            "delta_stab": 0.2,
            "delta_val": 0.2,
            "target_bias": 0.3,
            "influence_mode": "dense",
            "stability_competitor_mode": "runner_up",
        }

        try:
            budget_rows = compute_radius_derived_budget_curve_rows(data, T=4, budgets=[0, 1], metadata=metadata, stability_competitor_mode="runner_up")
            horizon_rows = compute_horizon_curve_rows(data, budgets=[0, 1], metadata=metadata)
        except Exception as exc:
            if "Gurobi" in str(exc) or "license" in str(exc).lower():
                self.skipTest(f"Gurobi unavailable: {exc}")
            raise

        self.assertTrue(
            {
                "seed",
                "K",
                "N",
                "L",
                "T",
                "delta_stab",
                "delta_val",
                "target_bias",
                "influence_mode",
                "stability_competitor_mode",
                "budget",
                "method",
                "objective",
                "curve_type",
                "certified_fraction",
                "attacked_fraction",
                "mean_radius",
                "median_radius",
                "min_radius",
                "max_radius",
                "num_certified",
                "num_known",
                "num_unknown",
                "num_total",
            }.issubset(budget_rows[0])
        )
        self.assertTrue(
            {
                "seed",
                "K",
                "N",
                "L",
                "T",
                "delta_stab",
                "delta_val",
                "target_bias",
                "budget",
                "method",
                "mean_horizon",
                "median_horizon",
                "min_horizon",
                "max_horizon",
                "max_possible_horizon",
                "certified_fraction_full_horizon",
            }.issubset(horizon_rows[0])
        )

    def test_benchmark_writes_new_curve_csvs(self):
        with TemporaryDirectory() as tmp_dir:
            try:
                experiments.benchmark_scale(
                    Ks=[4],
                    Ns=[2],
                    Ls=[2],
                    Ts=[4],
                    delta_stabs=[0.2],
                    delta_vals=[0.2],
                    target_biases=[0.3],
                    influence_mode="dense",
                    stability_competitor_mode="runner_up",
                    seed=0,
                    save_dir=tmp_dir,
                    budget_max=1,
                    make_budget_curves=True,
                    make_damage_curves=True,
                    make_horizon_curves=True,
                )
            except Exception as exc:
                if "Gurobi" in str(exc) or "license" in str(exc).lower():
                    self.skipTest(f"Gurobi unavailable: {exc}")
                raise

            self.assertTrue((Path(tmp_dir) / "benchmark_budget_curves.csv").exists())
            self.assertTrue((Path(tmp_dir) / "benchmark_damage_curves.csv").exists())
            self.assertTrue((Path(tmp_dir) / "benchmark_horizons.csv").exists())

            budget_header = (Path(tmp_dir) / "benchmark_budget_curves.csv").read_text().splitlines()[0].split(",")
            damage_header = (Path(tmp_dir) / "benchmark_damage_curves.csv").read_text().splitlines()[0].split(",")
            horizon_header = (Path(tmp_dir) / "benchmark_horizons.csv").read_text().splitlines()[0].split(",")

        self.assertIn("certified_fraction", budget_header)
        self.assertIn("curve_type", damage_header)
        self.assertIn("status_name", damage_header)
        self.assertIn("is_optimal", damage_header)
        self.assertIn("lower_bound", damage_header)
        self.assertIn("upper_bound", damage_header)
        self.assertIn("mip_gap", damage_header)
        self.assertIn("runtime_sec", damage_header)
        self.assertIn("certified_fraction_is_exact", damage_header)
        self.assertIn("mean_horizon", horizon_header)


class ExperimentCliTests(unittest.TestCase):
    def test_benchmark_does_not_make_plots_by_default(self):
        parser = experiments.build_parser()

        args = parser.parse_args(["benchmark"])
        self.assertFalse(args.make_plots)
        self.assertEqual(args.budget_max, 15)
        self.assertTrue(args.make_budget_curves)
        self.assertTrue(args.make_damage_curves)
        self.assertTrue(args.make_horizon_curves)

        args = parser.parse_args(["benchmark", "--make-plots"])
        self.assertTrue(args.make_plots)

        args = parser.parse_args(["benchmark", "--budget-max", "3", "--no-make-damage-curves"])
        self.assertEqual(args.budget_max, 3)
        self.assertFalse(args.make_damage_curves)

        args = parser.parse_args(["audit-curves", "--csv-dir", "toy_results/medium_benchmark"])
        self.assertEqual(args.command, "audit-curves")
        self.assertEqual(args.csv_dir, "toy_results/medium_benchmark")

    def test_small_benchmark_preset_is_bounded(self):
        preset = experiments._benchmark_preset("small")

        instance_count = len(preset["Ks"]) * len(preset["Ns"]) * len(preset["lengths"]) * len(preset["Ts"]) * len(preset["delta_stabs"]) * len(preset["delta_vals"]) * len(preset["target_biases"])

        self.assertEqual(instance_count, 36)
        self.assertEqual(experiments._benchmark_preset("smoke")["Ks"], [8])
        self.assertEqual(experiments._benchmark_preset("smoke")["Ns"], [2])
        self.assertEqual(experiments._benchmark_preset("smoke")["lengths"], [6])
        self.assertEqual(experiments._benchmark_preset("smoke")["Ts"], [4])

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

    def test_csv_reader_handles_scientific_notation(self):
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "benchmark_results.csv"
            csv_path.write_text("K,N,L,T,seed,runtime_gurobi_total,mip_gap\n5,2,3,4,0,1e-06,2E-05\n")

            rows = experiments._read_rows_csv(csv_path)

        self.assertEqual(rows[0]["runtime_gurobi_total"], 1e-06)
        self.assertEqual(rows[0]["mip_gap"], 2e-05)

    def test_csv_reader_parses_boolean_solver_metadata(self):
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "benchmark_damage_curves.csv"
            csv_path.write_text("is_optimal,certified_fraction_is_exact,certified_fraction\nTrue,False,0.5\n")

            rows = experiments._read_rows_csv(csv_path)

        self.assertIs(rows[0]["is_optimal"], True)
        self.assertIs(rows[0]["certified_fraction_is_exact"], False)
        self.assertEqual(rows[0]["certified_fraction"], 0.5)

    def test_damage_curve_failed_solve_keeps_unknown_fractions_blank(self):
        result = experiments.DamageResult(
            name="failed",
            budget=2,
            selected_poisoned_shards=[],
            attacked_cells=[],
            attacked_rows=None,
            status=0,
            status_name="NO_SOLUTION",
            objective_value=None,
        )

        row = experiments._damage_curve_row({}, result, objective="stability_one_token_per_prompt", num_rows=4)

        self.assertEqual(row["max_attacked_rows"], "")
        self.assertEqual(row["attacked_fraction"], "")
        self.assertEqual(row["certified_fraction"], "")
        self.assertIs(row["certified_fraction_is_exact"], False)

    def test_damage_curve_nonoptimal_feasible_row_is_marked_as_bound(self):
        result = experiments.DamageResult(
            name="feasible",
            budget=2,
            selected_poisoned_shards=[0, 1],
            attacked_cells=[(0, 0)],
            attacked_rows=[0],
            status=0,
            status_name="TIME_LIMIT",
            objective_value=1.0,
            is_optimal=False,
        )

        row = experiments._damage_curve_row({}, result, objective="stability_one_token_per_prompt", num_rows=4)

        self.assertEqual(row["certified_fraction"], 0.75)
        self.assertIs(row["certified_fraction_is_exact"], False)
        self.assertEqual(row["bound_type"], "feasible_attacked_lower_bound")

    def test_direct_damage_rows_use_damage_solver_source(self):
        calls = []

        def fake_stability(*args, budget, row_requirement, competitor_mode):
            calls.append(("stability", budget, row_requirement, competitor_mode))
            return experiments.DamageResult(
                name="fake",
                budget=budget,
                selected_poisoned_shards=[],
                attacked_cells=[],
                attacked_rows=[0],
                status=0,
                status_name="OPTIMAL",
                objective_value=1.0,
                is_optimal=True,
            )

        def fake_validity(*args, budget, row_requirement):
            calls.append(("validity", budget, row_requirement))
            return experiments.DamageResult(
                name="fake",
                budget=budget,
                selected_poisoned_shards=[],
                attacked_cells=[],
                attacked_rows=[0],
                status=0,
                status_name="OPTIMAL",
                objective_value=1.0,
                is_optimal=True,
            )

        old_stability = experiments.maximize_attacked_rows_stability
        old_validity = experiments.maximize_attacked_rows_validity
        try:
            experiments.maximize_attacked_rows_stability = fake_stability
            experiments.maximize_attacked_rows_validity = fake_validity
            data = generate_toy_votes(K=4, N=2, L=2, T=3, seed=0)
            rows = experiments.compute_direct_damage_curve_rows(data, T=3, budgets=[0, 1], metadata={}, stability_competitor_mode="runner_up")
        finally:
            experiments.maximize_attacked_rows_stability = old_stability
            experiments.maximize_attacked_rows_validity = old_validity

        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["curve_type"] == "direct_damage_milp" for row in rows))
        self.assertTrue(all(row["method"] == "Shared MILP" for row in rows))
        self.assertEqual(len(calls), 6)

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

            stdout = StringIO()
            with redirect_stdout(stdout):
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
            self.assertIn("Shared MILP full sequence", one_prompt_validity_svg)
            self.assertIn("Independent full sequence", one_prompt_validity_svg)
            self.assertIn("Atomic phrase aggregation", one_prompt_validity_svg)
            self.assertNotIn("phrase-DPA", one_prompt_validity_svg)
            self.assertNotIn("q1", one_prompt_validity_svg)
            self.assertIn("tpa_val_sequence_q1", stdout.getvalue())
            self.assertIn("Independent composition overestimate", validity_diagnostic_svg)
            self.assertIn("Shared MILP one prompt, full sequence", one_prompt_stability_svg)
            self.assertNotIn("qN", one_prompt_stability_svg)
            self.assertIn("independent overestimate factor", diagnostic_svg)
            self.assertFalse((base / "monotonicity_violations.csv").exists())

    def test_budget_curve_plot_labels_and_audit_sources(self):
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            (base / "benchmark_budget_curves.csv").write_text(
                "seed,K,N,L,T,delta_stab,delta_val,target_bias,influence_mode,stability_competitor_mode,budget,method,objective,curve_type,certified_fraction,attacked_fraction,mean_radius,median_radius,min_radius,max_radius,num_certified,num_known,num_unknown,num_total\n"
                "0,4,2,2,3,0.2,0.2,0.3,dense,runner_up,0,DPA token margin,full_response_stable_against_any_token_change,radius_derived,1.0,0.0,2,2,2,2,2,2,0,2\n"
                "0,4,2,2,3,0.2,0.2,0.3,dense,runner_up,0,TPA max-token sequence,validity_full_harmful_sequence_per_prompt,radius_derived,0.5,0.5,1,1,1,1,1,2,0,2\n"
                "0,4,2,2,3,0.2,0.2,0.3,dense,runner_up,0,Shared MILP,validity_full_harmful_sequence_per_prompt,radius_derived,0.5,0.5,1,1,1,1,1,2,0,2\n"
            )
            (base / "benchmark_damage_curves.csv").write_text(
                "seed,K,N,L,T,delta_stab,delta_val,target_bias,influence_mode,stability_competitor_mode,budget,method,objective,curve_type,max_attacked_rows,max_attacked_cells,attacked_fraction,certified_fraction,status_name,is_optimal,certified_fraction_is_exact,bound_type,objective_value,lower_bound,upper_bound,mip_gap,runtime_sec\n"
                "0,4,2,2,3,0.2,0.2,0.3,dense,runner_up,0,Shared MILP,stability_one_token_per_prompt,direct_damage_milp,0,0,0.0,1.0,OPTIMAL,True,True,exact,0,0,0,0,0.01\n"
                "0,4,2,2,3,0.2,0.2,0.3,dense,runner_up,0,Shared MILP,stability_full_sequence_per_prompt,direct_damage_milp,0,0,0.0,1.0,OPTIMAL,True,True,exact,0,0,0,0,0.01\n"
                "0,4,2,2,3,0.2,0.2,0.3,dense,runner_up,0,Shared MILP,validity_full_harmful_sequence_per_prompt,direct_damage_milp,1,2,0.5,0.5,OPTIMAL,True,True,exact,1,1,1,0,0.01\n"
            )
            (base / "benchmark_horizons.csv").write_text(
                "seed,K,N,L,T,delta_stab,delta_val,target_bias,influence_mode,stability_competitor_mode,budget,method,mean_horizon,median_horizon,min_horizon,max_horizon,max_possible_horizon,certified_fraction_full_horizon\n"
                "0,4,2,2,3,0.2,0.2,0.3,dense,runner_up,0,DPA stability horizon,2,2,2,2,2,1.0\n"
                "0,4,2,2,3,0.2,0.2,0.3,dense,runner_up,0,TPA validity horizon,1,1,1,1,2,0.0\n"
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                experiments.save_budget_curve_plots(base)
                diagnostics = experiments.audit_curve_csvs(str(base))

            stability_svg = (base / "certified_fraction_stability_by_budget.svg").read_text()
            validity_svg = (base / "certified_fraction_validity_by_budget.svg").read_text()
            self.assertIn("DPA weakest token, radius-derived", stability_svg)
            self.assertIn("Shared one-token-per-prompt, direct MILP", stability_svg)
            self.assertIn("Shared full-sequence-per-prompt, direct MILP", stability_svg)
            self.assertIn("TPA max-token sequence, radius-derived", validity_svg)
            self.assertIn("Shared full sequence, radius-derived", validity_svg)
            self.assertIn("Shared full sequence, direct MILP", validity_svg)
            self.assertIn("source=benchmark_budget_curves.csv", stdout.getvalue())
            self.assertIn("source=benchmark_damage_curves.csv", stdout.getvalue())
            self.assertTrue(any(item["check"] == "identical_series" for item in diagnostics))

    def test_check_script_does_not_run_benchmark_or_plot_csv(self):
        script = Path("scripts/check.sh").read_text()

        self.assertIn("compileall toy_certificate tests", script)
        self.assertIn("unittest discover", script)
        self.assertIn("toy_certificate.experiments visualize", script)
        self.assertNotIn("toy_certificate.experiments benchmark", script)
        self.assertNotIn("toy_certificate.experiments plot-csv", script)
        self.assertIn("toy_results/smoke/instance", script)
        self.assertIn("runner_up", script)
        self.assertIn('--N "${VIS_N:-8}"', script)
        self.assertIn('--L "${VIS_L:-8}"', script)

    def test_short_scripts_have_expected_roles(self):
        data_script = Path("scripts/data.sh").read_text()
        plot_script = Path("scripts/plot.sh").read_text()
        benchmark_script = Path("scripts/benchmark.sh").read_text()

        self.assertIn("toy_certificate.experiments benchmark", data_script)
        self.assertIn("toy_certificate.experiments plot-csv", plot_script)
        self.assertIn("scripts/data.sh", benchmark_script)
        self.assertIn("scripts/plot.sh", benchmark_script)
        self.assertIn("toy_results/small_benchmark", data_script)
        self.assertIn("--preset", data_script)
        self.assertIn("--delta-stabs", data_script)
        self.assertIn("--delta-vals", data_script)
        self.assertIn("--target-biases", data_script)
        self.assertIn("--budget-max", data_script)
        self.assertIn("MAKE_BUDGET_CURVES", data_script)
        self.assertIn("MAKE_DAMAGE_CURVES", data_script)
        self.assertIn("MAKE_HORIZON_CURVES", data_script)
        self.assertIn("TOY_RESULTS_DIR", plot_script)
        self.assertIn("find \"$TOY_RESULTS_DIR\"", plot_script)
        self.assertIn("toy_results/small_benchmark", benchmark_script)


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

    def test_direct_damage_stability_is_monotone_in_budget(self):
        data = generate_toy_votes(K=4, N=2, L=2, T=3, delta_stab=0.2, delta_val=0.2, target_bias=0.2, seed=17)
        try:
            b0 = maximize_attacked_rows_stability(
                data.stab_votes,
                data.stab_counts,
                data.clean_pred,
                data.runner_up,
                data.influence,
                budget=0,
                row_requirement="any_token",
                competitor_mode="runner_up",
            )
            b1 = maximize_attacked_rows_stability(
                data.stab_votes,
                data.stab_counts,
                data.clean_pred,
                data.runner_up,
                data.influence,
                budget=1,
                row_requirement="any_token",
                competitor_mode="runner_up",
            )
            b2 = maximize_attacked_rows_stability(
                data.stab_votes,
                data.stab_counts,
                data.clean_pred,
                data.runner_up,
                data.influence,
                budget=2,
                row_requirement="any_token",
                competitor_mode="runner_up",
            )
        except Exception as exc:
            if "Gurobi" in str(exc) or "license" in str(exc).lower():
                self.skipTest(f"Gurobi unavailable: {exc}")
            raise

        attacked = [b0.max_attacked_rows, b1.max_attacked_rows, b2.max_attacked_rows]
        for value in attacked:
            self.assertIsNotNone(value)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 2)
        self.assertLessEqual(attacked[0], attacked[1])
        self.assertLessEqual(attacked[1], attacked[2])
        certified = [1 - value / 2 for value in attacked]
        self.assertGreaterEqual(certified[0], certified[1])
        self.assertGreaterEqual(certified[1], certified[2])

    def test_report_facing_objective_names_match_qr_semantics(self):
        data = generate_toy_votes(K=5, N=2, L=2, T=3, delta_stab=0.2, delta_val=0.2, target_bias=0.2, seed=13)
        try:
            stab_q1_r1 = solve_structured_stability(
                data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=1, r_cols=1
            )
            stab_q1_rL = solve_structured_stability(
                data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=1, r_cols=2
            )
            stab_qN_r1 = solve_structured_stability(
                data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=2, r_cols=1
            )
            stab_qN_rL = solve_structured_stability(
                data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=2, r_cols=2
            )
            val_q1 = solve_row_col_validity(data.val_votes, data.val_counts, data.target, T=3, influence=data.influence, q_rows=1)
            val_qN = solve_row_col_validity(data.val_votes, data.val_counts, data.target, T=3, influence=data.influence, q_rows=2)
        except Exception as exc:
            if "Gurobi" in str(exc) or "license" in str(exc).lower():
                self.skipTest(f"Gurobi unavailable: {exc}")
            raise

        self.assertEqual(stab_q1_r1.name, "row_col_stability_q1_r1")
        self.assertEqual(stab_q1_rL.name, "row_col_stability_q1_r2")
        self.assertEqual(stab_qN_r1.name, "row_col_stability_q2_r1")
        self.assertEqual(stab_qN_rL.name, "row_col_stability_q2_r2")
        self.assertEqual(val_q1.name, "row_col_validity_q1")
        self.assertEqual(val_qN.name, "row_col_validity_q2")


if __name__ == "__main__":
    unittest.main()
