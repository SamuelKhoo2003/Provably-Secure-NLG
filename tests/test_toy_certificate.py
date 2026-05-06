import unittest

from toy_certificate.data import generate_toy_votes
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
