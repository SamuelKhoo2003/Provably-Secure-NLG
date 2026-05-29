import subprocess
import tempfile
import unittest
from pathlib import Path

from toy_certificate import experiments


class ConfigDrivenDataTests(unittest.TestCase):
    def test_validity_demo_config_validates(self) -> None:
        config = experiments.load_experiment_config("configs/validity_demo.yaml")
        self.assertEqual(config["generator"], "validity_demo")
        self.assertEqual(config["objective_family"], "validity_only")
        self.assertFalse(config["make_stability_objectives"])

    def test_missing_required_field_errors_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.yaml"
            path.write_text(
                "\n".join(
                    [
                        "generator: toy",
                        "K_values: [4]",
                        "N_values: [2]",
                    ]
                )
            )
            with self.assertRaisesRegex(experiments.ConfigError, "missing required field `L_values`"):
                experiments.load_experiment_config(path)

    def test_wrong_type_errors_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.yaml"
            path.write_text(Path("configs/validity_demo.yaml").read_text().replace("K_values: [20,40,60]", "K_values: 20"))
            with self.assertRaisesRegex(experiments.ConfigError, "field `K_values` .* must be a non-empty list"):
                experiments.load_experiment_config(path)

    def test_yaml_name_does_not_set_cli_preset(self) -> None:
        config = experiments.load_experiment_config("configs/validity_demo.yaml")
        self.assertNotIn("preset", config)

    def test_data_sh_requires_config(self) -> None:
        result = subprocess.run(["./scripts/data.sh"], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR: CONFIG is required.", result.stderr)

    def test_data_sh_config_dry_run_uses_config_only(self) -> None:
        result = subprocess.run(
            ["bash", "-c", "CONFIG=configs/validity_demo.yaml DRY_RUN=1 VERBOSE=1 ./scripts/data.sh"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("Config: configs/validity_demo.yaml", result.stdout)
        self.assertIn("objective_family: validity_only", result.stdout)
        self.assertIn("estimated stability solves: 0", result.stdout)
        self.assertNotIn("--Ks", result.stdout)

    def test_generic_preset_choices_remain_generic(self) -> None:
        parser = experiments.build_parser()
        preset_action = next(action for action in parser._actions if action.dest == "preset")
        self.assertEqual(set(preset_action.choices), {"smoke", "small", "medium", "large"})


if __name__ == "__main__":
    unittest.main()
