import unittest

import numpy as np

from src.stitching.metrics import calculate_ds_metrics, predict_velocities


class _DummyDamm:
    def compute_gamma(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = x.shape[0]
        return np.vstack(
            [
                np.full(n, 0.25, dtype=float),
                np.full(n, 0.75, dtype=float),
            ]
        )


class _DummyLPVDS:
    def __init__(self):
        self.damm = _DummyDamm()
        self.A = np.array(
            [
                [[-1.0, 0.0], [0.0, -1.0]],
                [[-2.0, 0.0], [0.0, -2.0]],
            ],
            dtype=float,
        )
        self.x_att = np.array([0.5, -0.5], dtype=float)


class _CountingPredictDS:
    def __init__(self):
        self.calls = 0

    def predict_velocities(self, x):
        self.calls += 1
        x = np.asarray(x, dtype=float)
        return -x


class MetricsFastExactTests(unittest.TestCase):
    def test_predict_velocities_vectorized_matches_pointwise_formula(self):
        ds = _DummyLPVDS()
        x = np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [-1.0, 0.5],
            ],
            dtype=float,
        )

        pred = predict_velocities(x, ds)

        expected = []
        gamma = ds.damm.compute_gamma(x)
        for i in range(x.shape[0]):
            xi = x[i] - ds.x_att
            vi = np.zeros(2, dtype=float)
            for k in range(ds.A.shape[0]):
                vi += gamma[k, i] * (ds.A[k] @ xi)
            expected.append(vi)
        expected = np.asarray(expected, dtype=float)

        np.testing.assert_allclose(pred, expected, atol=1e-10)

    def test_calculate_ds_metrics_calls_predict_once_for_prediction_terms(self):
        ds = _CountingPredictDS()
        x_ref = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        x_dot_ref = -x_ref.copy()
        trajectories = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)]

        result = calculate_ds_metrics(
            x_ref=x_ref,
            x_dot_ref=x_dot_ref,
            ds=ds,
            sim_trajectories=trajectories,
            initial=np.array([1.0, 1.0], dtype=float),
            attractor=np.array([0.0, 0.0], dtype=float),
        )

        self.assertEqual(ds.calls, 1)
        self.assertAlmostEqual(float(result["prediction_rmse"]), 0.0, places=12)
        self.assertAlmostEqual(float(result["cosine_dissimilarity"]), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
