import unittest

import numpy as np
import pandas as pd

from compare_stitch_methods import (
    aggregate_method_results,
    select_combinations,
    _compute_demo_extent,
    _predict_velocity_field,
)
from src.util.ds_tools import Demonstration, Trajectory


class CompareStitchMethodsTests(unittest.TestCase):
    def test_select_combinations_returns_even_subset(self):
        combos = [(np.array([i, 0.0]), np.array([i + 1.0, 0.0])) for i in range(10)]
        selected = select_combinations(combos, max_combinations=4)
        self.assertEqual(len(selected), 4)
        self.assertTrue(np.allclose(selected[0][0], np.array([0.0, 0.0])))
        self.assertTrue(np.allclose(selected[-1][0], np.array([9.0, 0.0])))

    def test_aggregate_method_results(self):
        df = pd.DataFrame(
            [
                {
                    "dataset_path": "d1",
                    "method": "chain",
                    "success": True,
                    "prediction_rmse": 1.0,
                    "cosine_dissimilarity": 0.2,
                    "dtw_distance_mean": 2.0,
                    "distance_to_attractor_mean": 0.1,
                    "gg_compute_time": 0.3,
                    "ds_compute_time": 0.4,
                    "total_compute_time": 0.7,
                },
                {
                    "dataset_path": "d1",
                    "method": "chain",
                    "success": False,
                    "prediction_rmse": np.nan,
                    "cosine_dissimilarity": np.nan,
                    "dtw_distance_mean": np.nan,
                    "distance_to_attractor_mean": np.nan,
                    "gg_compute_time": np.nan,
                    "ds_compute_time": np.nan,
                    "total_compute_time": np.nan,
                },
            ]
        )
        summary = aggregate_method_results(df)
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["dataset_path"], "d1")
        self.assertEqual(row["method"], "chain")
        self.assertEqual(row["n_cases"], 2)
        self.assertAlmostEqual(row["success_rate"], 0.5)
        self.assertAlmostEqual(row["prediction_rmse_mean"], 1.0)

    def test_compute_demo_extent(self):
        demo1 = Demonstration(
            [Trajectory(x=np.array([[0.0, 1.0], [2.0, 3.0]]), x_dot=np.array([[0.0, 0.0], [0.0, 0.0]]))]
        )
        demo2 = Demonstration(
            [Trajectory(x=np.array([[-1.0, -2.0], [1.0, 4.0]]), x_dot=np.array([[0.0, 0.0], [0.0, 0.0]]))]
        )
        x_min, x_max, y_min, y_max = _compute_demo_extent([demo1, demo2], padding=0.5)
        self.assertAlmostEqual(x_min, -1.5)
        self.assertAlmostEqual(x_max, 2.5)
        self.assertAlmostEqual(y_min, -2.5)
        self.assertAlmostEqual(y_max, 4.5)

    def test_predict_velocity_field_prefers_policy_predictor(self):
        class PolicyDS:
            def predict_velocities(self, points):
                return np.ones_like(points) * 2.0

        pts = np.array([[0.0, 1.0], [2.0, 3.0]])
        vel = _predict_velocity_field(PolicyDS(), pts)
        self.assertTrue(np.allclose(vel, np.ones_like(pts) * 2.0))

    def test_predict_velocity_field_lpv_formula_path(self):
        class DummyDamm:
            def compute_gamma(self, points):
                return np.ones((1, points.shape[0]))

        class LPVDS:
            def __init__(self):
                self.damm = DummyDamm()
                self.A = np.array([-np.eye(2)])
                self.x_att = np.array([0.0, 0.0])

        pts = np.array([[1.0, 0.0], [0.0, 2.0]])
        vel = _predict_velocity_field(LPVDS(), pts)
        self.assertTrue(np.allclose(vel, np.array([[-1.0, 0.0], [0.0, -2.0]])))


if __name__ == "__main__":
    unittest.main()
