from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from toy_experiments.csv_io import read_rows_csv
from toy_experiments.experiments import (
    CANONICAL_COLORS,
    MAIN_STABILITY_METRICS,
    SWEEP_STABILITY_METRICS,
    ConfigError,
    build_parser,
    load_experiment_config,
    plot_sweep_csv,
)
from toy_experiments.milp import solve_structured_stability


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "toy_experiments" / "configs"


class StabilityCompetitorConfigTests(unittest.TestCase):
    def test_all_repository_configs_validate_without_mode(self) -> None:
        configs = sorted(CONFIG_DIR.glob("*.yaml"))
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(path=path.name):
                config = load_experiment_config(path)
                self.assertNotIn("stability_competitor_mode", config)

    def test_stale_mode_field_has_clear_deprecation_error(self) -> None:
        stale_config = (CONFIG_DIR / "smoke.yaml").read_text() + "\nstability_competitor_mode: runner_up\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stale.yaml"
            path.write_text(stale_config)
            with self.assertRaisesRegex(
                ConfigError,
                r"deprecated field `stability_competitor_mode`.*always uses all competitors",
            ):
                load_experiment_config(path)

    def test_cli_and_solver_do_not_expose_competitor_mode(self) -> None:
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--stability-competitor-mode", option_strings)
        parameters = inspect.signature(solve_structured_stability).parameters
        self.assertNotIn("competitor_mode", parameters)
        self.assertNotIn("runner_up", parameters)

    def test_plotting_tolerates_old_csv_mode_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old.csv"
            path.write_text(
                "K,N,L,stability_competitor_mode,dpa_stab_row_radius_qN,"
                "row_col_stab_qN_r1,row_col_stab_qN_rL\n"
                "4,1,1,runner_up,1,2,2\n"
            )
            rows = read_rows_csv(path)
            plot_dir = Path(temp_dir) / "plots"
            plot_sweep_csv(str(path), "K", str(plot_dir))
            self.assertTrue((plot_dir / "sweep_K_stability_certificate_vs_K.svg").exists())
        self.assertEqual(rows[0]["stability_competitor_mode"], "runner_up")
        self.assertEqual(rows[0]["row_col_stab_qN_r1"], 2)

    def test_report_facing_plot_labels_do_not_name_runner_up(self) -> None:
        labels = [
            *SWEEP_STABILITY_METRICS,
            *MAIN_STABILITY_METRICS,
            *CANONICAL_COLORS,
        ]
        self.assertFalse(any("runner" in label.lower() for label in labels))


class StabilityCompetitorMilpTests(unittest.TestCase):
    def test_non_runner_up_competitor_can_be_easiest_attack(self) -> None:
        # Token 0 wins 3-2-1. Only the two token-1 shards are attackable.
        # They cannot improve token 1's margin, but poisoning both can move
        # token 2 from one vote to a 3-3 tie with the clean winner.
        votes = np.array([[[0]], [[0]], [[0]], [[1]], [[1]], [[2]]], dtype=np.int64)
        clean_counts = np.array([[[3, 2, 1]]], dtype=np.int64)
        clean_pred = np.array([[0]], dtype=np.int64)
        influence = np.array([[[0]], [[0]], [[0]], [[1]], [[1]], [[0]]], dtype=np.int64)

        result = solve_structured_stability(
            votes,
            clean_counts,
            clean_pred,
            influence,
            q_rows=1,
            r_cols=1,
            gurobi_threads=1,
        )

        self.assertTrue(result.is_optimal)
        self.assertEqual(result.B_star, 2)
        self.assertEqual(result.selected_poisoned_shards, [3, 4])


if __name__ == "__main__":
    unittest.main()
