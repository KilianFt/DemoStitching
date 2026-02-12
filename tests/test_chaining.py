from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from graph_utils import GaussianGraph
from src.stitching.chaining import (
    _edge_fit_samples,
    _edge_fit_data,
    _fit_edge_matrix,
    _fit_linear_system,
    build_chained_ds,
    prepare_chaining_edge_lookup,
)
from src.stitching.graph_paths import shortest_path_nodes
from src.stitching.metrics import calculate_prediction_rmse
from src.util.plot_tools import plot_ds_set_gaussians


class _MockDS:
    def __init__(self, mu: np.ndarray, target: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        self.x = mu + 0.08 * rng.standard_normal((120, 2))
        self.x_dot = target.reshape(1, -1) - self.x
        self.assignment_arr = np.zeros(self.x.shape[0], dtype=int)
        self.A = np.array([-np.eye(2)])


class _ManualDS:
    def __init__(self, x: np.ndarray, x_dot: np.ndarray, A: np.ndarray):
        self.x = np.asarray(x, dtype=float)
        self.x_dot = np.asarray(x_dot, dtype=float)
        self.assignment_arr = np.zeros(self.x.shape[0], dtype=int)
        self.A = np.asarray([A], dtype=float)


class ChainingPolicyTests(unittest.TestCase):
    @staticmethod
    def _make_graph(gaussians):
        gg = GaussianGraph(param_dist=3, param_cos=3)
        gg.add_gaussians(gaussians, reverse_gaussians=False)
        return gg

    def _make_chain(self, enable_recovery=True, edge_data_mode="both_all", initial=None):
        if initial is None:
            initial = np.array([-0.4, 0.0])
        else:
            initial = np.asarray(initial, dtype=float)
        attractor = np.array([3.0, 0.0])

        mus = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        targets = [np.array([1.0, 0.0]), np.array([2.0, 0.0]), attractor]
        ds_set = [_MockDS(mu, target, seed=i + 1) for i, (mu, target) in enumerate(zip(mus, targets))]

        gaussians = {
            (i, 0): {
                "mu": mu,
                "sigma": 0.04 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 1.0 / len(mus),
            }
            for i, mu in enumerate(mus)
        }
        gg = self._make_graph(gaussians)
        path_nodes = shortest_path_nodes(gg, initial_state=initial, target_state=attractor)
        self.assertIsNotNone(path_nodes)

        config = SimpleNamespace(
            chain_trigger_radius=0.10,
            chain_transition_time=0.20,
            chain_recovery_distance=0.30,
            chain_enable_recovery=enable_recovery,
            chain_fit_regularization=1e-4,
            chain_stabilization_margin=1e-3,
            chain_edge_data_mode=edge_data_mode,
            plot_extent=[-1.0, 3.5, -1.0, 1.0],
            # backward-compatible aliases
            chain_switch_threshold=0.10,
            chain_blend_width=0.20,
        )
        chained = build_chained_ds(
            ds_set,
            gg,
            initial=initial,
            attractor=attractor,
            config=config,
            shortest_path_nodes=path_nodes,
        )
        return chained, initial, attractor, config

    def test_chaining_reaches_goal_and_recovers_to_nearest_node(self):
        chained, initial, attractor, _ = self._make_chain()

        self.assertIsNotNone(chained)
        self.assertEqual(chained.K, 3)

        trajectory, _ = chained.sim(initial[None, :], dt=0.05)
        self.assertLess(np.linalg.norm(trajectory[-1] - attractor), 0.08)
        self.assertLess(np.min(np.abs(trajectory[:, 0] - 1.0)), 0.2)
        self.assertLess(np.min(np.abs(trajectory[:, 0] - 2.0)), 0.2)

        disturbed_state = np.array([2.2, 0.6])
        recovered_idx = chained.select_node_index(disturbed_state, current_idx=0)
        self.assertEqual(recovered_idx, chained.n_systems - 1)
        _, velocity, _ = chained.step_once(disturbed_state, dt=0.05, current_idx=0)
        direction_to_target = chained.node_targets[recovered_idx] - disturbed_state
        self.assertGreater(np.dot(velocity, direction_to_target), 0.0)

        rmse = calculate_prediction_rmse(chained.x, chained.x_dot, chained)
        self.assertTrue(np.isfinite(rmse))

    def test_trigger_radius_is_percentage_of_edge_length(self):
        chained, _, _, _ = self._make_chain(enable_recovery=False)
        edge_lengths = np.linalg.norm(np.diff(chained.state_sequence, axis=0), axis=1)
        np.testing.assert_allclose(chained.trigger_radii, 0.10 * edge_lengths)

    def test_timer_trigger_controls_transition_completion(self):
        chained, _, _, _ = self._make_chain(enable_recovery=False)

        chained.reset_runtime(initial_idx=0, start_time=0.0)
        x_probe = np.array([0.98, 0.0])  # close to first DS target (second gaussian center)

        # State trigger should start transition, but timer has not elapsed yet.
        _, _, idx_t0 = chained.step_once(x_probe, dt=0.01, current_time=0.00)
        self.assertTrue(chained.transition_active)
        self.assertEqual(idx_t0, 0)

        _, _, idx_t1 = chained.step_once(x_probe, dt=0.01, current_time=0.10)
        self.assertTrue(chained.transition_active)
        self.assertEqual(idx_t1, 0)

        # After T_i has elapsed, transition timer should switch to next DS.
        _, _, idx_t2 = chained.step_once(x_probe, dt=0.01, current_time=0.21)
        self.assertFalse(chained.transition_active)
        self.assertEqual(idx_t2, 1)

    def test_first_edge_uses_start_gaussian_data(self):
        chained, _, _, _ = self._make_chain(
            enable_recovery=False,
            edge_data_mode="between_orthogonals",
            initial=np.array([0.0, 0.0]),
        )
        self.assertIsNotNone(chained)
        self.assertGreater(chained.edge_fit_points[0].shape[0], 0)

    def test_first_edge_selection_uses_initial_boundary(self):
        initial = np.array([-0.5, 0.0])
        attractor = np.array([2.0, 0.0])
        node0 = np.array([0.0, 0.0])
        node1 = np.array([1.0, 0.0])

        # Source-node samples lie before node0 mean and should still be selected
        # when the first edge starts at the actual initial state.
        ds0 = _ManualDS(
            x=np.array([[-0.40, 0.0], [-0.30, 0.0], [-0.20, 0.0]]),
            x_dot=np.array([[0.15, 0.0], [0.20, 0.0], [0.25, 0.0]]),
            A=-np.eye(2),
        )
        ds1 = _ManualDS(
            x=np.array([[0.90, 0.0], [1.00, 0.0], [1.10, 0.0]]),
            x_dot=np.array([[0.20, 0.0], [0.20, 0.0], [0.20, 0.0]]),
            A=-np.eye(2),
        )
        ds_set = [ds0, ds1]

        gaussians = {
            (0, 0): {
                "mu": node0,
                "sigma": 0.02 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 0.5,
            },
            (1, 0): {
                "mu": node1,
                "sigma": 0.02 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 0.5,
            },
        }
        gg = self._make_graph(gaussians)
        path_nodes = shortest_path_nodes(gg, initial_state=initial, target_state=attractor)
        self.assertIsNotNone(path_nodes)

        config = SimpleNamespace(
            chain_trigger_radius=0.10,
            chain_transition_time=0.20,
            chain_recovery_distance=0.30,
            chain_enable_recovery=False,
            chain_stabilization_margin=1e-3,
            chain_edge_data_mode="between_orthogonals",
            rel_scale=0.11,
            total_scale=0.66,
            nu_0=9,
            kappa_0=0.4,
            psi_dir_0=0.2,
            chain_switch_threshold=0.10,
            chain_blend_width=0.20,
        )
        chained = build_chained_ds(
            ds_set,
            gg,
            initial=initial,
            attractor=attractor,
            config=config,
            shortest_path_nodes=path_nodes,
        )
        self.assertIsNotNone(chained)
        first_fit = np.asarray(chained.edge_fit_points[0], dtype=float)
        self.assertGreater(first_fit.shape[0], 0)
        self.assertTrue(np.any(first_fit[:, 0] < 0.0))

    def test_assignment_arr_is_available_for_plotting(self):
        chained, _, _, config = self._make_chain(enable_recovery=False)
        self.assertEqual(chained.assignment_arr.shape, (chained.x.shape[0],))
        self.assertTrue(np.all(chained.assignment_arr >= 0))
        self.assertTrue(np.all(chained.assignment_arr < chained.K))
        ax = plot_ds_set_gaussians([chained], config, include_trajectory=True)
        self.assertIsNotNone(ax)

    def test_precomputed_edge_lookup_matches_default_build(self):
        chained_default, initial, attractor, config = self._make_chain(enable_recovery=False)

        mus = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        targets = [np.array([1.0, 0.0]), np.array([2.0, 0.0]), attractor]
        ds_set = [_MockDS(mu, target, seed=i + 11) for i, (mu, target) in enumerate(zip(mus, targets))]
        gaussians = {
            (i, 0): {
                "mu": mu,
                "sigma": 0.04 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 1.0 / len(mus),
            }
            for i, mu in enumerate(mus)
        }
        gg = self._make_graph(gaussians)
        path_nodes = shortest_path_nodes(gg, initial_state=initial, target_state=attractor)
        self.assertIsNotNone(path_nodes)
        chain_cfg, edge_lookup = prepare_chaining_edge_lookup(ds_set, gg, config)
        chained_cached = build_chained_ds(
            ds_set,
            gg,
            initial=initial,
            attractor=attractor,
            config=config,
            precomputed_chain_cfg=chain_cfg,
            precomputed_edge_lookup=edge_lookup,
            shortest_path_nodes=path_nodes,
        )

        self.assertIsNotNone(chained_cached)
        self.assertEqual(chained_default.A_seq.shape, chained_cached.A_seq.shape)
        self.assertEqual(chained_default.state_sequence.shape, chained_cached.state_sequence.shape)

    def test_build_chained_ds_uses_graph_shortest_path_when_not_provided(self):
        initial = np.array([-0.4, 0.0])
        attractor = np.array([3.0, 0.0])

        mus = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        targets = [np.array([1.0, 0.0]), np.array([2.0, 0.0]), attractor]
        ds_set = [_MockDS(mu, target, seed=i + 31) for i, (mu, target) in enumerate(zip(mus, targets))]
        gaussians = {
            (i, 0): {
                "mu": mu,
                "sigma": 0.04 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 1.0 / len(mus),
            }
            for i, mu in enumerate(mus)
        }
        gg = self._make_graph(gaussians)
        expected_path = gg.shortest_path(initial, attractor)
        self.assertIsNotNone(expected_path)

        config = SimpleNamespace(
            chain_trigger_radius=0.10,
            chain_transition_time=0.20,
            chain_recovery_distance=0.30,
            chain_enable_recovery=False,
            chain_stabilization_margin=1e-3,
            chain_lmi_tolerance=5e-5,
            chain_edge_data_mode="both_all",
            chain_switch_threshold=0.10,
            chain_blend_width=0.20,
        )
        chained = build_chained_ds(
            ds_set,
            gg,
            initial=initial,
            attractor=attractor,
            config=config,
        )
        self.assertIsNotNone(chained)
        self.assertEqual(chained.path_nodes[:-1], expected_path)

    def test_chain_exposes_edge_fit_points_and_direction_metrics(self):
        chained, _, _, _ = self._make_chain(enable_recovery=False)
        self.assertTrue(hasattr(chained, "edge_fit_points"))
        self.assertTrue(hasattr(chained, "edge_fit_velocities"))
        self.assertTrue(hasattr(chained, "edge_direction_stats"))
        self.assertEqual(len(chained.edge_fit_points), chained.n_systems)
        self.assertEqual(len(chained.edge_fit_velocities), chained.n_systems)
        self.assertEqual(len(chained.edge_direction_stats), chained.n_systems)

        for stats in chained.edge_direction_stats:
            self.assertIn("n_points", stats)
            self.assertIn("frac_forward", stats)
            self.assertIn("min_proj", stats)
            frac = float(stats["frac_forward"])
            if np.isfinite(frac):
                self.assertGreaterEqual(frac, 0.0)
                self.assertLessEqual(frac, 1.0)


class ChainingEdgeDataSelectionTests(unittest.TestCase):
    def test_edge_data_mode_both_all_keeps_all_points(self):
        source_x = np.array([[0.0, 0.0], [0.2, 0.1]])
        source_x_dot = np.array([[1.0, 0.0], [1.0, 0.1]])
        target_x = np.array([[0.8, -0.1], [1.0, 0.0], [1.2, 0.0]])
        target_x_dot = np.array([[0.5, 0.0], [0.2, 0.0], [-0.1, 0.0]])

        fit_x, fit_x_dot = _edge_fit_data(
            source_x=source_x,
            source_x_dot=source_x_dot,
            target_x=target_x,
            target_x_dot=target_x_dot,
            source_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 0.0]),
            mode="both_all",
        )
        self.assertEqual(fit_x.shape[0], source_x.shape[0] + target_x.shape[0])
        self.assertEqual(fit_x_dot.shape[0], source_x_dot.shape[0] + target_x_dot.shape[0])

    def test_edge_data_mode_between_orthogonals_filters_segment_and_direction(self):
        source_x = np.array([[-0.2, 0.0], [0.2, 0.0], [0.4, 0.1], [0.8, 0.0], [1.2, 0.0]])
        source_x_dot = np.array([[1.0, 0.0], [1.0, 0.0], [-0.2, 0.0], [0.4, 0.0], [1.0, 0.0]])
        target_x = np.array([[0.9, 0.0], [1.1, 0.0], [0.6, -0.2]])
        target_x_dot = np.array([[0.2, 0.0], [-0.1, 0.0], [0.3, 0.0]])

        fit_x, fit_x_dot = _edge_fit_data(
            source_x=source_x,
            source_x_dot=source_x_dot,
            target_x=target_x,
            target_x_dot=target_x_dot,
            source_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 0.0]),
            mode="between_orthogonals",
        )

        self.assertEqual(fit_x.shape[0], 4)
        self.assertTrue(np.all(fit_x[:, 0] >= 0.0))
        self.assertTrue(np.all(fit_x[:, 0] <= 1.0))
        self.assertTrue(np.all(fit_x_dot[:, 0] > 0.0))

    def test_edge_data_mode_between_orthogonals_can_be_empty(self):
        source_x = np.array([[0.1, 0.0], [0.4, 0.0]])
        source_x_dot = np.array([[-1.0, 0.0], [-0.5, 0.0]])
        target_x = np.array([[0.6, 0.0], [0.9, 0.0]])
        target_x_dot = np.array([[-0.3, 0.0], [-0.2, 0.0]])

        fit_x, fit_x_dot = _edge_fit_data(
            source_x=source_x,
            source_x_dot=source_x_dot,
            target_x=target_x,
            target_x_dot=target_x_dot,
            source_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 0.0]),
            mode="between_orthogonals",
        )
        self.assertEqual(fit_x.shape[0], 0)
        self.assertEqual(fit_x_dot.shape[0], 0)

    def test_edge_fit_samples_filters_source_only_in_segment_mode(self):
        source_x = np.array([[-0.1, 0.0], [0.2, 0.1], [0.7, -0.2], [1.2, 0.0]])
        source_x_dot = np.array([[1.0, 0.0], [0.3, 0.0], [-0.2, 0.0], [0.2, 0.0]])
        source_data = (source_x, source_x_dot, -np.eye(2), np.eye(2))
        cfg = SimpleNamespace(edge_data_mode="between_orthogonals")
        fit_x, fit_x_dot = _edge_fit_samples(
            source_data=source_data,
            target_data=None,
            source_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 0.0]),
            cfg=cfg,
        )
        self.assertEqual(fit_x.shape[0], 1)
        np.testing.assert_allclose(fit_x[0], np.array([0.2, 0.1]))
        self.assertTrue(np.all(fit_x_dot[:, 0] > 0.0))

    def test_edge_fit_samples_degenerate_segment_keeps_source_data(self):
        source_x = np.array([[0.1, 0.0], [0.3, 0.2], [0.6, -0.1]])
        source_x_dot = np.array([[0.2, 0.0], [0.1, 0.0], [0.3, 0.0]])
        source_data = (source_x, source_x_dot, -np.eye(2), np.eye(2))
        cfg = SimpleNamespace(edge_data_mode="between_orthogonals")
        fit_x, fit_x_dot = _edge_fit_samples(
            source_data=source_data,
            target_data=None,
            source_state=np.array([1.0, 1.0]),
            target_state=np.array([1.0, 1.0]),
            cfg=cfg,
        )
        np.testing.assert_allclose(fit_x, source_x)
        np.testing.assert_allclose(fit_x_dot, source_x_dot)

    def test_constrained_fit_returns_none_if_solver_fails(self):
        x = np.array([[0.0, 0.0], [1.0, 0.5], [1.5, 1.0]])
        x_dot = np.array([[0.4, 0.1], [0.3, 0.2], [0.2, 0.1]])
        target = np.array([2.0, 2.0])

        with patch("cvxpy.problems.problem.Problem.solve", side_effect=Exception("solver error")):
            A = _fit_linear_system(
                x=x,
                x_dot=x_dot,
                target=target,
                stabilization_margin=1e-3,
                lyapunov_P=np.eye(2),
            )
        self.assertIsNone(A)

    def test_constrained_fit_satisfies_lmi(self):
        target = np.array([2.0, -1.0])
        x = np.array([
            [0.0, 0.0],
            [0.5, -0.2],
            [1.2, -0.6],
            [2.5, -1.4],
        ])
        x_dot = -(x - target.reshape(1, -1))
        eps = 1e-3
        A = _fit_linear_system(
            x=x,
            x_dot=x_dot,
            target=target,
            stabilization_margin=eps,
            lyapunov_P=np.eye(2),
        )
        self.assertIsNotNone(A)
        lmi = A.T @ np.eye(2) + np.eye(2) @ A + eps * np.eye(2)
        lambda_max = float(np.max(np.linalg.eigvalsh(0.5 * (lmi + lmi.T))))
        self.assertLessEqual(lambda_max, 5e-5)

    def test_fit_edge_matrix_returns_none_when_constrained_fit_fails(self):
        source_x = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]])
        source_x_dot = np.array([[0.5, 0.0], [0.4, 0.0], [0.3, 0.0]])
        source_data = (source_x, source_x_dot, -np.eye(2), np.eye(2))
        cfg = SimpleNamespace(
            edge_data_mode="both_all",
            fit_regularization=1e-4,
            stabilization_margin=1e-3,
            lmi_tolerance=5e-5,
        )
        with patch("src.stitching.chaining._fit_linear_system", return_value=None):
            A_edge, fit_x, fit_x_dot = _fit_edge_matrix(
                source_data=source_data,
                target_data=None,
                source_state=np.array([0.0, 0.0]),
                target_state=np.array([1.0, 0.0]),
                cfg=cfg,
            )
        self.assertIsNone(A_edge)
        self.assertEqual(fit_x.shape[0], source_x.shape[0])
        self.assertEqual(fit_x_dot.shape[0], source_x_dot.shape[0])

    def test_build_chained_ds_returns_none_if_any_edge_fit_invalid(self):
        initial = np.array([-0.4, 0.0])
        attractor = np.array([3.0, 0.0])
        mus = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        targets = [np.array([1.0, 0.0]), np.array([2.0, 0.0]), attractor]
        ds_set = [_MockDS(mu, target, seed=i + 20) for i, (mu, target) in enumerate(zip(mus, targets))]
        gaussians = {
            (i, 0): {
                "mu": mu,
                "sigma": 0.04 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 1.0 / len(mus),
            }
            for i, mu in enumerate(mus)
        }
        gg = GaussianGraph(param_dist=3, param_cos=3)
        gg.add_gaussians(gaussians, reverse_gaussians=False)
        path_nodes = shortest_path_nodes(gg, initial_state=initial, target_state=attractor)
        self.assertIsNotNone(path_nodes)
        config = SimpleNamespace(
            chain_trigger_radius=0.10,
            chain_transition_time=0.20,
            chain_recovery_distance=0.30,
            chain_enable_recovery=True,
            chain_fit_regularization=1e-4,
            chain_stabilization_margin=1e-3,
            chain_switch_threshold=0.10,
            chain_blend_width=0.20,
        )
        with patch(
            "src.stitching.chaining._fit_edge_matrix",
            return_value=(None, np.zeros((0, 2)), np.zeros((0, 2))),
        ):
            chained = build_chained_ds(
                ds_set,
                gg,
                initial=initial,
                attractor=attractor,
                config=config,
                shortest_path_nodes=path_nodes,
            )
        self.assertIsNone(chained)


if __name__ == "__main__":
    unittest.main()
