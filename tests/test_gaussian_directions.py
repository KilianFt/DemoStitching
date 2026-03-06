import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.lpvds_class import lpvds_class
from src.util.ds_tools import get_gaussian_directions


class _DummyDamm:
    def __init__(self, gamma, gaussian_lists, z):
        self._gamma = np.asarray(gamma, dtype=float)
        self.gaussian_lists = gaussian_lists
        self.z = np.asarray(z, dtype=int)
        self.K = len(gaussian_lists)
        self.Mu = np.asarray([np.asarray(g["mu"], dtype=float) for g in gaussian_lists], dtype=float)
        self.Sigma = np.asarray([np.asarray(g.get("sigma", np.eye(self.Mu.shape[1])), dtype=float) for g in gaussian_lists], dtype=float)
        self.Prior = np.asarray([float(g.get("prior", 1.0 / max(self.K, 1))) for g in gaussian_lists], dtype=float)

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

    def test_cluster_compacts_zero_assignment_components(self):
        obj = lpvds_class.__new__(lpvds_class)
        gamma = np.array(
            [
                [0.90, 0.80, 0.10, 0.20],
                [0.05, 0.10, 0.20, 0.30],  # never argmax
                [0.05, 0.10, 0.70, 0.50],
            ],
            dtype=float,
        )
        obj.damm = _DummyDamm(
            gamma=gamma,
            gaussian_lists=[
                {"mu": np.array([0.0, 0.0]), "sigma": np.eye(2), "prior": 0.2},
                {"mu": np.array([1.0, 0.0]), "sigma": np.eye(2), "prior": 0.3},
                {"mu": np.array([2.0, 0.0]), "sigma": np.eye(2), "prior": 0.5},
            ],
            z=[0, 0, 2, 2],
        )

        lpvds_class._cluster(obj)

        self.assertEqual(obj.K, 2)
        self.assertEqual(int(obj.damm.K), 2)
        self.assertEqual(obj.gamma.shape, (2, 4))
        np.testing.assert_allclose(np.sum(obj.gamma, axis=0), np.ones((4,), dtype=float), atol=1e-8)
        np.testing.assert_array_equal(obj.assignment_arr, np.array([0, 0, 1, 1], dtype=int))
        self.assertEqual(obj.damm.Mu.shape[0], 2)
        self.assertEqual(obj.damm.Sigma.shape[0], 2)
        self.assertEqual(len(obj.damm.gaussian_lists), 2)
        np.testing.assert_allclose(np.sum(obj.damm.Prior), 1.0, atol=1e-8)
        np.testing.assert_allclose(
            obj.damm.Prior,
            np.array([0.2, 0.5], dtype=float) / 0.7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.array([g["prior"] for g in obj.damm.gaussian_lists], dtype=float),
            obj.damm.Prior,
            atol=1e-8,
        )

    def test_init_cluster_compacts_zero_assignment_components(self):
        class _InitClusterDamm:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.K = 0
                self.gaussian_lists = []
                self.Mu = np.zeros((0, 2), dtype=float)
                self.Sigma = np.zeros((0, 2, 2), dtype=float)
                self.Prior = np.zeros((0,), dtype=float)

            def compute_gamma(self, _x):
                return np.array(
                    [
                        [0.90, 0.80, 0.10, 0.20],
                        [0.05, 0.10, 0.20, 0.30],  # never argmax
                        [0.05, 0.10, 0.70, 0.50],
                    ],
                    dtype=float,
                )

        obj = lpvds_class.__new__(lpvds_class)
        obj.x = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 0.0],
                [1.1, 0.1],
            ],
            dtype=float,
        )
        obj.x_dot = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        obj.x_dir = obj.x_dot / np.linalg.norm(obj.x_dot, axis=1, keepdims=True)
        obj.nu_0 = 5
        obj.kappa_0 = 1
        obj.psi_dir_0 = 1
        obj.rel_scale = 0.1
        obj.total_scale = 1.0

        gaussian_lists = [
            {"mu": np.array([0.0, 0.0]), "sigma": np.eye(2), "prior": 0.2},
            {"mu": np.array([1.0, 0.0]), "sigma": np.eye(2), "prior": 0.3},
            {"mu": np.array([2.0, 0.0]), "sigma": np.eye(2), "prior": 0.5},
        ]

        with patch("src.lpvds_class.damm_class", _InitClusterDamm):
            lpvds_class.init_cluster(obj, gaussian_lists)

        self.assertEqual(int(obj.K), 2)
        self.assertEqual(int(obj.damm.K), 2)
        self.assertEqual(obj.gamma.shape, (2, 4))
        np.testing.assert_allclose(np.sum(obj.gamma, axis=0), np.ones((4,), dtype=float), atol=1e-8)
        np.testing.assert_array_equal(obj.assignment_arr, np.array([0, 0, 1, 1], dtype=int))
        np.testing.assert_allclose(np.sum(obj.damm.Prior), 1.0, atol=1e-8)
        self.assertEqual(len(obj.damm.gaussian_lists), 2)


if __name__ == "__main__":
    unittest.main()
