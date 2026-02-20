import unittest
from pathlib import Path
import sys
from types import MethodType
from typing import Optional

import networkx as nx
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs import StitchConfig
from src.stitching.chaining import build_chained_ds


_N_SAMPLES_PER_NODE = 80


class _MockDS:
    def __init__(self, mu: np.ndarray, target: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        self.x = np.asarray(mu, dtype=float).reshape(1, -1) + 0.04 * rng.standard_normal((_N_SAMPLES_PER_NODE, 2))
        self.x_dot = np.asarray(target, dtype=float).reshape(1, -1) - self.x
        self.assignment_arr = np.zeros(self.x.shape[0], dtype=int)
        self.A = np.asarray([-np.eye(2)], dtype=float)


class _MockGaussianGraph:
    def __init__(self, states: np.ndarray):
        self.graph = nx.DiGraph()
        self.gaussian_reversal_map = {}
        self._gaussians = {}

        n_nodes = states.shape[0]
        for i, mu in enumerate(states):
            node_id = (i, 0)
            sigma = 0.02 * np.eye(2)
            direction = np.array([1.0, 0.0], dtype=float)
            prior = 1.0 / max(n_nodes, 1)
            self.graph.add_node(
                node_id,
                mean=np.asarray(mu, dtype=float),
                covariance=sigma,
                direction=direction,
                prior=prior,
            )
            self._gaussians[node_id] = (np.asarray(mu, dtype=float), sigma, direction, prior)

    def get_gaussian(self, node_id):
        return self._gaussians[node_id]


class ChainingTransitionPolicyTests(unittest.TestCase):
    @staticmethod
    def _build_chain(
        path_states: np.ndarray,
        method: str,
        attractor: Optional[np.ndarray] = None,
        chain_ds_method: str = "linear",
        precomputed_edge_lookup: Optional[dict] = None,
        chain_overrides: Optional[dict] = None,
    ):
        path_states = np.asarray(path_states, dtype=float)
        if attractor is None:
            attractor = path_states[-1]
        attractor = np.asarray(attractor, dtype=float)
        initial = path_states[0] - np.array([0.25, 0.0], dtype=float)

        ds_set = []
        for i, mu in enumerate(path_states):
            next_idx = min(i + 1, path_states.shape[0] - 1)
            ds_set.append(_MockDS(mu=mu, target=path_states[next_idx], seed=1000 + i))

        gg = _MockGaussianGraph(path_states)
        path_nodes = [(i, 0) for i in range(path_states.shape[0])]

        cfg = StitchConfig()
        cfg.chain.subsystem_edges = 1  # should be ignored: chaining always uses 3-node windows.
        cfg.chain.transition_trigger_method = method
        cfg.chain.ds_method = chain_ds_method
        cfg.chain.enable_recovery = False
        cfg.chain.blend_length_ratio = 0.10
        cfg.chain.use_boundary_ds_initial = False
        cfg.chain.use_boundary_ds_end = False
        if chain_overrides is not None:
            for key, value in chain_overrides.items():
                setattr(cfg.chain, key, value)

        chained = build_chained_ds(
            ds_set,
            gg,
            initial=initial,
            attractor=attractor,
            config=cfg,
            shortest_path_nodes=path_nodes,
            precomputed_edge_lookup=precomputed_edge_lookup,
        )
        return chained, initial, attractor, path_states

    def test_chaining_uses_three_node_windows_with_overlapping_triplets(self):
        path = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ]
        )
        chained, _, _, _ = self._build_chain(path, method="mean_normals")

        self.assertIsNotNone(chained)
        self.assertEqual(chained.subsystem_edges, 2)
        # 3 intermediate segments + 1 attractor segment (last 2 nodes -> attractor).
        self.assertEqual(chained.n_systems, 4)
        np.testing.assert_allclose(chained.node_sources[:, 0], np.array([0.0, 1.0, 2.0, 3.0]), atol=1e-8)
        np.testing.assert_allclose(chained.node_targets[:, 0], np.array([2.0, 3.0, 4.0, 4.0]), atol=1e-8)

        # Intermediate systems use 3-node windows; attractor segment uses 2 nodes.
        expected_per_window = 3 * _N_SAMPLES_PER_NODE
        for fit_points in chained.edge_fit_points[:-1]:
            self.assertEqual(int(fit_points.shape[0]), expected_per_window)
        self.assertEqual(int(chained.edge_fit_points[-1].shape[0]), 2 * _N_SAMPLES_PER_NODE)

    def test_mean_normals_transition_is_centered_on_second_to_last_node(self):
        path = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        chained, _, _, _ = self._build_chain(path, method="mean_normals")

        self.assertIsNotNone(chained)
        expected_anchor = path[1]
        np.testing.assert_allclose(chained.transition_centers[0], expected_anchor, atol=1e-8)

        normal = np.asarray(chained.transition_normals[0], dtype=float)
        normal = normal / np.linalg.norm(normal)
        x_prev_side = expected_anchor - 0.20 * normal
        x_next_side = expected_anchor + 0.20 * normal

        self.assertFalse(chained.trigger_state(0, x_prev_side))
        self.assertTrue(chained.trigger_state(0, x_next_side))

        signed_prev_node = float(np.dot(path[0] - expected_anchor, chained.transition_normals[0]))
        signed_next_node = float(np.dot(path[2] - expected_anchor, chained.transition_normals[0]))
        self.assertLessEqual(signed_prev_node, 0.0)
        self.assertGreaterEqual(signed_next_node, 0.0)

    def test_distance_ratio_transition_uses_last_two_edge_lengths(self):
        path = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ]
        )
        chained, _, _, _ = self._build_chain(path, method="distance_ratio")

        self.assertIsNotNone(chained)
        expected_r = 2.0 / 1.0  # |e1|/|e2|
        self.assertAlmostEqual(float(chained.transition_edge_ratios[0]), expected_r, places=8)

        n1 = path[0]
        n2 = path[2]
        x_no_switch = np.array([1.5, 0.0])
        x_switch = np.array([2.5, 0.0])

        ratio_no_switch = np.linalg.norm(x_no_switch - n1) / np.linalg.norm(x_no_switch - n2)
        ratio_switch = np.linalg.norm(x_switch - n1) / np.linalg.norm(x_switch - n2)
        self.assertLess(ratio_no_switch, expected_r)
        self.assertGreaterEqual(ratio_switch, expected_r)

        self.assertFalse(chained.trigger_state(0, x_no_switch))
        self.assertTrue(chained.trigger_state(0, x_switch))

    def test_both_transition_methods_reach_goal_and_last_subsystem(self):
        path = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.5],
                [3.0, 0.3],
                [4.0, 0.0],
            ]
        )
        attractor = np.array([4.2, 0.1])

        for method in ("mean_normals", "distance_ratio"):
            with self.subTest(method=method):
                chained, initial, _, _ = self._build_chain(path, method=method, attractor=attractor)
                self.assertIsNotNone(chained)

                trajectory, _ = chained.sim(initial[None, :], dt=0.02)
                self.assertLess(np.linalg.norm(trajectory[-1] - attractor), 0.25)
                self.assertIsNotNone(chained.last_sim_indices)
                self.assertEqual(int(np.max(chained.last_sim_indices)), chained.n_systems - 1)

    def test_segmented_chain_matches_linear_interface_and_returns(self):
        path = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.2],
                [3.0, 0.1],
            ]
        )
        chained, initial, attractor, _ = self._build_chain(
            path,
            method="mean_normals",
            chain_ds_method="segmented",
        )
        self.assertIsNotNone(chained)
        self.assertTrue(hasattr(chained, "damm"))
        self.assertEqual(chained.assignment_arr.shape[0], chained.x.shape[0])

        step_out = chained.step_once(initial.copy(), dt=0.02)
        self.assertEqual(len(step_out), 3)
        x_next, velocity, idx = step_out
        self.assertEqual(np.asarray(x_next).shape, initial.shape)
        self.assertEqual(np.asarray(velocity).shape, initial.shape)
        self.assertIsInstance(idx, (int, np.integer))

        trajectory, gamma_history = chained.sim(initial[None, :], dt=0.02)
        self.assertEqual(trajectory.ndim, 2)
        self.assertEqual(gamma_history.ndim, 2)
        self.assertLess(np.linalg.norm(trajectory[-1] - attractor), 0.25)

    def test_segmented_chain_uses_same_configurable_trigger_policy(self):
        path = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ]
        )
        chained, _, _, _ = self._build_chain(
            path,
            method="distance_ratio",
            chain_ds_method="segmented",
        )
        self.assertIsNotNone(chained)
        expected_r = 2.0
        # With use_boundary_ds_initial=False (default) there is no init boundary,
        # so the first transition (index 0) is the first 3-node segment transition.
        self.assertAlmostEqual(float(chained.transition_edge_ratios[0]), expected_r, places=8)
        self.assertFalse(chained.trigger_state(0, np.array([1.5, 0.0])))
        self.assertTrue(chained.trigger_state(0, np.array([2.5, 0.0])))

    def test_transition_times_respect_configured_minimum(self):
        path = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )
        chained, _, _, _ = self._build_chain(
            path,
            method="mean_normals",
            chain_overrides={
                "blend_length_ratio": 0.01,
                "min_transition_time": 0.5,
            },
        )
        self.assertIsNotNone(chained)
        self.assertGreater(chained.transition_times.size, 0)
        self.assertTrue(np.all(chained.transition_times >= 0.5 - 1e-12))

    def test_linear_velocity_is_clipped_by_vmax_without_a_capping(self):
        path = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )
        chained, initial, _, _ = self._build_chain(
            path,
            method="mean_normals",
            chain_overrides={"velocity_max": 0.2},
        )
        self.assertIsNotNone(chained)
        chained.A_seq[0] = -1e8 * np.eye(2)
        x_next, velocity, _ = chained.step_once(initial.copy(), dt=0.02)
        self.assertTrue(np.all(np.isfinite(velocity)))
        self.assertTrue(np.all(np.isfinite(x_next)))
        self.assertLessEqual(float(np.linalg.norm(velocity)), 0.2 + 1e-12)

    def test_non_finite_velocity_raises(self):
        path = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )
        chained, initial, _, _ = self._build_chain(
            path,
            method="mean_normals",
            chain_overrides={"velocity_max": 0.25},
        )
        self.assertIsNotNone(chained)
        chained.A_seq[0][:] = np.nan
        with self.assertRaises(ValueError):
            chained.step_once(initial.copy(), dt=0.02)

    def test_segmented_chain_uses_same_velocity_safeguards(self):
        path = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.2],
                [3.0, 0.1],
            ]
        )
        chained, initial, _, _ = self._build_chain(
            path,
            method="mean_normals",
            chain_ds_method="segmented",
            chain_overrides={"velocity_max": 0.3},
        )
        self.assertIsNotNone(chained)

        def _huge_velocity(_self, x, idx):
            return np.array([1e7, -1e7], dtype=float)

        chained._velocity_for_index = MethodType(_huge_velocity, chained)
        _, velocity, _ = chained.step_once(initial.copy(), dt=0.02)
        self.assertTrue(np.all(np.isfinite(velocity)))
        self.assertLessEqual(float(np.linalg.norm(velocity)), 0.3 + 1e-12)


if __name__ == "__main__":
    unittest.main()
