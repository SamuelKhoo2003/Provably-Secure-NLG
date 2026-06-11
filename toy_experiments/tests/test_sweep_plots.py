from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from toy_experiments.experiments import plot_sweep_csv


class SweepPlotTests(unittest.TestCase):
    def test_additional_sweeps_generate_all_plot_groups(self) -> None:
        sweep_values = {
            "T": [3, 5],
            "delta_stab": [0.1, 0.2],
            "delta_val": [0.1, 0.2],
            "target_bias": [0.2, 0.3],
        }
        metric_values = {
            "dpa_stab_row_radius_qN": 1,
            "row_col_stab_qN_r1": 2,
            "row_col_stab_qN_rL": 3,
            "plain_dpa_val_sequence_qN": 1,
            "tpa_val_sequence_qN": 2,
            "row_col_val_qN": 3,
            "runtime_gurobi_total": 0.1,
        }

        for sweep, values in sweep_values.items():
            with self.subTest(sweep=sweep), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                csv_path = root / "benchmark_results.csv"
                rows = []
                for value in values:
                    row = {
                        "K": 20,
                        "N": 4,
                        "L": 5,
                        "T": 5,
                        "delta_stab": 0.2,
                        "delta_val": 0.2,
                        "target_bias": 0.3,
                        "seed": 0,
                        **metric_values,
                    }
                    row[sweep] = value
                    rows.append(row)
                with csv_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

                with patch("toy_experiments.experiments._save_line_plot") as save_plot:
                    plot_sweep_csv(str(csv_path), sweep=sweep, save_dir=str(root / "plots"))

                self.assertEqual(save_plot.call_count, 3)
                filenames = {call.args[0].name for call in save_plot.call_args_list}
                self.assertEqual(
                    filenames,
                    {
                        f"sweep_{sweep}_stability_certificate_vs_{sweep}.pdf",
                        f"sweep_{sweep}_validity_certificate_vs_{sweep}.pdf",
                        f"sweep_{sweep}_runtime_vs_{sweep}.pdf",
                    },
                )


if __name__ == "__main__":
    unittest.main()
