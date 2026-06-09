from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_certification_curves.py"
)


def load_plot_module():
    spec = importlib.util.spec_from_file_location(
        "plot_certification_curves",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PlotCertificationCurvesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.plotter = load_plot_module()
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"plot dependencies unavailable: {exc}") from exc

    def test_current_method_is_loaded_without_remapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "summary.json").write_text(
                json.dumps({"name": "current", "horizon": 20, "num_prompts": 1})
            )
            (run_dir / "budget_curve_summary.csv").write_text(
                "budget,method,objective_mode,num_prompts,certified_fraction\n"
                "0,aggregate_tpa_final_tool_validity,radius_derived,1,1.0\n"
            )
            frame = self.plotter.load_run(run_dir, None)

        self.assertEqual(
            frame.loc[0, "method"],
            "aggregate_tpa_final_tool_validity",
        )
        self.assertEqual(
            frame.loc[0, "method_label"],
            "Aggregate TPA final-tool validity",
        )

    def test_legacy_method_name_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "budget_curve_summary.csv").write_text(
                "budget,method,objective_mode,num_prompts,certified_fraction\n"
                "0,aggregate_tpa_mcp_validity,radius_derived,1,1.0\n"
            )
            with self.assertRaisesRegex(
                ValueError,
                (
                    "unknown method names.*aggregate_tpa_mcp_validity.*"
                    "Regenerate this run"
                ),
            ):
                self.plotter.load_run(run_dir, None)


if __name__ == "__main__":
    unittest.main()
