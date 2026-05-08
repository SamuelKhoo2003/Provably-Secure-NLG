import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from toy_certificate.data import generate_toy_votes
from toy_certificate import experiments
from toy_certificate.experiments import compute_reference_baselines, compute_structured_stability_grid, compute_validity_q_curve


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
            self.assertTrue((base / "validity_scaling_by_L.svg").exists())
            self.assertTrue((base / "stability_structured_by_L.svg").exists())
            self.assertTrue((base / "validity_bias_sweep.svg").exists())

    def test_check_run_script_does_not_run_benchmark_or_plot_csv(self):
        script = Path("scripts/run_toy_check.sh").read_text()

        self.assertIn("compileall toy_certificate tests", script)
        self.assertIn("unittest discover", script)
        self.assertIn("toy_certificate.experiments visualize", script)
        self.assertNotIn("toy_certificate.experiments benchmark", script)
        self.assertNotIn("toy_certificate.experiments plot-csv", script)

    def test_benchmark_wrapper_refreshes_existing_csv_without_generating_data(self):
        script = Path("scripts/run_toy_benchmark.sh").read_text()

        self.assertIn("toy_certificate.experiments visualize", script)
        self.assertIn("toy_certificate.experiments plot-csv", script)
        self.assertIn("run_toy_benchmark_data.sh first", script)
        self.assertNotIn("toy_certificate.experiments benchmark", script)


class GurobiBackedTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
