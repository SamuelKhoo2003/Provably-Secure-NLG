from __future__ import annotations

import unittest

import numpy as np

from toy_experiments.baselines import (
    aggregate_plain_dpa_sequence_baselines,
    aggregate_tpa_sequence_baselines,
    compute_reference_baselines,
    compute_stability_baselines,
    compute_validity_baselines,
)
from toy_experiments.data import (
    ToyData,
    compute_counts,
    majority_predictions,
    runner_up_tokens,
)


def make_toy_data() -> ToyData:
    """Build a small deterministic fixture without invoking a solver."""
    stab_votes = np.asarray(
        [
            [[0, 0]],
            [[0, 1]],
            [[1, 1]],
        ],
        dtype=np.int64,
    )
    val_votes = np.asarray(
        [
            [[0, 0]],
            [[0, 1]],
            [[1, 2]],
        ],
        dtype=np.int64,
    )
    stab_counts = compute_counts(stab_votes, T=3)
    val_counts = compute_counts(val_votes, T=3)
    clean_pred = majority_predictions(stab_counts)
    return ToyData(
        stab_votes=stab_votes,
        val_votes=val_votes,
        stab_counts=stab_counts,
        val_counts=val_counts,
        clean_pred=clean_pred,
        runner_up=runner_up_tokens(stab_counts, clean_pred),
        target=np.asarray([[1, 2]], dtype=np.int64),
        base_token=np.asarray([[0, 0]], dtype=np.int64),
        val_base=np.asarray([[0, 0]], dtype=np.int64),
        influence=np.ones_like(stab_votes, dtype=np.int64),
    )


class BaselineSummaryTests(unittest.TestCase):
    def test_full_summary_is_union_of_family_summaries(self) -> None:
        data = make_toy_data()
        expected = {
            **compute_stability_baselines(data),
            **compute_validity_baselines(data),
        }
        self.assertEqual(compute_reference_baselines(data), expected)

    def test_max_token_aggregators_keep_existing_column_names(self) -> None:
        token_radii = np.asarray([[1, 3], [2, 4]], dtype=np.int64)
        self.assertEqual(
            aggregate_plain_dpa_sequence_baselines(token_radii),
            {
                "plain_dpa_val_sequence_q1": 3,
                "plain_dpa_val_sequence_qN": 4,
                "plain_dpa_val_sequence_mean": 3.5,
            },
        )
        self.assertEqual(
            aggregate_tpa_sequence_baselines(token_radii),
            {
                "tpa_val_sequence_q1": 3,
                "tpa_val_sequence_qN": 4,
                "tpa_val_sequence_mean": 3.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
