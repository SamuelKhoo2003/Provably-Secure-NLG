from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "certify_vote_vectors_runner.py"
)
SPEC = importlib.util.spec_from_file_location("certify_vote_vectors_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class CertificationRunnerTests(unittest.TestCase):
    def test_loader_filters_on_shortest_non_none_prefix(self) -> None:
        vote_vector = ["clean", "clean", "target"]
        common = {
            "vote_vector": vote_vector,
            "vote_counts": dict(Counter(vote_vector)),
            "majority": "clean",
        }
        filtered = {
            **common,
            "token_vote_matrix": [
                [1, 2, 3, None, None],
                [1, 2, None, None, None],
                [1, 2, 3, 4, None],
            ],
        }
        retained = {
            **common,
            "token_vote_matrix": [
                [1, 2, 3, 4, None],
                [1, 2, 3, 5, None],
                [1, 2, 3, 6, None],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "votes.jsonl"
            path.write_text(
                json.dumps(filtered) + "\n" + json.dumps(retained) + "\n"
            )
            rows, report = runner.load_prompt_rows(
                path,
                horizon=3,
                max_prompts=None,
                expected_num_shards=3,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].original_row_index, 1)
        self.assertEqual(rows[0].token_vote_matrix[0], (1, 2, 3))
        self.assertEqual(report.filtered_short_rows, 1)
        self.assertEqual(report.truncated_rows, 1)

    def test_dpa_certified_radius_uses_conservative_tie_convention(self) -> None:
        self.assertEqual(runner.dpa_certified_radius(10, 4), 2)
        self.assertEqual(runner.dpa_certified_radius(5, 5), 0)

    def test_aggregate_tpa_reuses_toy_targeted_partition(self) -> None:
        row = runner.PromptRow(
            original_row_index=0,
            majority_class="clean",
            vote_vector=("clean",) * 6 + ("target",) * 2 + ("other",) * 2,
            token_vote_matrix=tuple((0,) for _ in range(10)),
        )
        # Two changed votes can make target tie clean, so only budget 1 is safe.
        self.assertEqual(runner.compute_aggregate_tpa_radii([row]), [1.0])

    def test_validity_targets_drop_shared_prefix_positions(self) -> None:
        row = runner.PromptRow(
            original_row_index=0,
            majority_class="clean",
            vote_vector=("clean", "clean", "target"),
            token_vote_matrix=((1, 1), (1, 1), (1, 2)),
        )
        grid = np.asarray([[[1, 1, 1], [1, 1, 2]]], dtype=np.int64)
        targets = runner.build_validity_targets([row], grid, 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].active_positions, (1,))

    def test_stability_damage_matches_toy_constraint(self) -> None:
        grid = np.asarray([[[1, 1, 2]]], dtype=np.int64)
        events = runner.build_stability_events(grid, top_competitors=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].margin, 1)
        self.assertEqual(events[0].damage, (2, 2, 0))

    def test_validity_event_checks_every_observed_competitor(self) -> None:
        grid = np.asarray([[[1, 1, 2, 3]]], dtype=np.int64)
        target = runner.ValidityTarget(
            prompt_index=0,
            target_class="target",
            representative_shard_index=2,
            active_positions=(0,),
            target_tokens=(2,),
        )
        events = runner.build_validity_events(grid, [target])
        self.assertEqual(len(events), 1)
        self.assertEqual(
            len(events[0].competitor_margins_and_damage),
            2,
        )

    def test_solver_bound_is_rounded_up_for_certified_lower_bound(self) -> None:
        row = runner.common_milp_row(
            budget=1,
            method="test",
            n=10,
            horizon=1,
            num_shards=3,
            num_events=1,
            solver_status="TIME_LIMIT",
            objective_value=3.0,
            objective_bound=4.2,
            mip_gap=0.4,
            elapsed=1.0,
            top_competitors=1,
            max_targets_per_prompt=1,
        )
        self.assertEqual(row["max_failed_prompts"], 5)
        self.assertEqual(row["certified_prompts_lower_bound"], 5)
        self.assertEqual(row["solver"], "gurobi")


if __name__ == "__main__":
    unittest.main()
