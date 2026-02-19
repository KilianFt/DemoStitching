import unittest
from types import SimpleNamespace

import numpy as np

from src.lpvds_class import lpvds_class
from src.util.ds_tools import get_gaussian_directions


class _DummyDamm:
    def __init__(self, gamma, gaussian_lists, z):
        self._gamma = np.asarray(gamma, dtype=float)
        self.gaussian_lists = gaussian_lists
        self.z = np.asarray(z, dtype=int)
        self.K = len(gaussian_lists)

    def compute_gamma(self, _x):
        return self._gamma

    def fit(self):
        return self._gamma


class GaussianDirectionTests(unittest.TestCase):
    def test_get_gaussian_directions_uses_argmax_assigned_mean_velocity(self):
        # First 2 points -> comp 0 with +x velocity, last 2 -> comp 1 with +y velocity.
        gamma = np.array(
            [
                [0.9, 0.8, 0.2, 0.1],
                [0.1, 0.2, 0.8, 0.9],
            ]
        )
        lpvds = SimpleNamespace(
            x=np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.2]], dtype=float),
            x_dot=np.array([[2.0, 0.0], [1.5, 0.1], [0.1, 2.0], [0.0, 1.5]], dtype=float),
            x_att=np.array([0.0, 0.0], dtype=float),
            A=np.array([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=float),
            damm=_DummyDamm(
                gamma=gamma,
                gaussian_lists=[{"mu": np.array([1.0, 0.0])}, {"mu": np.array([0.0, 1.0])}],
                z=[1, 1, 0, 0],  # intentionally inconsistent with argmax(gamma)
            ),
        )

        directions = get_gaussian_directions(lpvds)
        self.assertEqual(directions.shape, (2, 2))

        d0_expected = np.array([1.0, 0.0])
        d1_expected = np.array([0.0, 1.0])
        self.assertGreater(np.dot(directions[0], d0_expected), 0.99)
        self.assertGreater(np.dot(directions[1], d1_expected), 0.99)

    def test_get_gaussian_directions_supports_legacy_a_mu_mode(self):
        gamma = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]], dtype=float)
        lpvds = SimpleNamespace(
            x=np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2]], dtype=float),
            x_dot=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float),
            x_att=np.array([0.0, 0.0], dtype=float),
            A=np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], dtype=float),
            damm=_DummyDamm(
                gamma=gamma,
                gaussian_lists=[{"mu": np.array([0.0, 2.0])}, {"mu": np.array([2.0, 0.0])}],
                z=[0, 0, 0],
            ),
        )

        mean_vel_dir = get_gaussian_directions(lpvds, method="mean_velocity")
        legacy_dir = get_gaussian_directions(lpvds, method="a_mu")

        # mean_velocity follows assigned velocities (+x)
        self.assertGreater(np.dot(mean_vel_dir[0], np.array([1.0, 0.0])), 0.99)
        # a_mu follows A @ (mu - x_att), so component 0 points +y
        self.assertGreater(np.dot(legacy_dir[0], np.array([0.0, 1.0])), 0.99)

    def test_get_gaussian_directions_falls_back_when_component_empty(self):
        gamma = np.array(
            [
                [0.9, 0.9, 0.9],
                [0.1, 0.1, 0.1],  # component 1 receives no argmax assignments
            ]
        )
        lpvds = SimpleNamespace(
            x=np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2]], dtype=float),
            x_dot=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float),
            x_att=np.array([0.0, 0.0], dtype=float),
            A=np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=float),
            damm=_DummyDamm(
                gamma=gamma,
                gaussian_lists=[{"mu": np.array([1.0, 0.0])}, {"mu": np.array([0.0, 2.0])}],
                z=[0, 0, 0],
            ),
        )

        directions = get_gaussian_directions(lpvds)
        self.assertGreater(np.dot(directions[0], np.array([1.0, 0.0])), 0.99)
        self.assertGreater(np.dot(directions[1], np.array([0.0, 1.0])), 0.99)

    def test_get_gaussian_directions_rejects_unknown_mode(self):
        lpvds = SimpleNamespace(
            x=np.array([[0.0, 0.0]], dtype=float),
            x_dot=np.array([[1.0, 0.0]], dtype=float),
            x_att=np.array([0.0, 0.0], dtype=float),
            A=np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=float),
            damm=_DummyDamm(gamma=np.array([[1.0]], dtype=float), gaussian_lists=[{"mu": np.array([1.0, 0.0])}], z=[0]),
        )
        with self.assertRaises(ValueError):
            get_gaussian_directions(lpvds, method="not_a_mode")

    def test_cluster_uses_argmax_posterior_assignment(self):
        obj = lpvds_class.__new__(lpvds_class)
        gamma = np.array(
            [
                [0.8, 0.2, 0.7],
                [0.2, 0.8, 0.3],
            ],
            dtype=float,
        )
        obj.damm = _DummyDamm(
            gamma=gamma,
            gaussian_lists=[{"mu": np.array([0.0, 0.0])}, {"mu": np.array([1.0, 1.0])}],
            z=[1, 1, 1],  # deliberately different from argmax(gamma)
        )

        lpvds_class._cluster(obj)

        np.testing.assert_array_equal(obj.assignment_arr, np.array([0, 1, 0], dtype=int))
        self.assertEqual(obj.K, 2)


if __name__ == "__main__":
    unittest.main()
