from types import SimpleNamespace
import unittest

import numpy as np

from graph_utils import GaussianGraph
from src.stitching.chaining import build_chained_ds
from src.stitching.metrics import calculate_prediction_rmse
from src.util.plot_tools import plot_ds_set_gaussians


class _MockDS:
    def __init__(self, mu: np.ndarray, target: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        self.x = mu + 0.08 * rng.standard_normal((120, 2))
        self.x_dot = target.reshape(1, -1) - self.x
        self.assignment_arr = np.zeros(self.x.shape[0], dtype=int)
        self.A = np.array([-np.eye(2)])


class ChainingPolicyTests(unittest.TestCase):
    def _make_chain(self, enable_recovery=True):
        initial = np.array([-0.4, 0.0])
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
        gg = GaussianGraph(
            gaussians,
            initial=initial,
            attractor=attractor,
            reverse_gaussians=False,
            param_dist=3,
            param_cos=3,
        )
        gg.compute_shortest_path()

        config = SimpleNamespace(
            chain_trigger_radius=0.10,
            chain_transition_time=0.20,
            chain_recovery_distance=0.30,
            chain_enable_recovery=enable_recovery,
            chain_fit_regularization=1e-4,
            chain_fit_blend=0.5,
            chain_stabilization_margin=1e-3,
            plot_extent=[-1.0, 3.5, -1.0, 1.0],
            # backward-compatible aliases
            chain_switch_threshold=0.10,
            chain_blend_width=0.20,
        )
        chained = build_chained_ds(ds_set, gg, initial=initial, attractor=attractor, config=config)
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
        self.assertEqual(recovered_idx, 3)
        _, velocity, _ = chained.step_once(disturbed_state, dt=0.05, current_idx=0)
        direction_to_target = chained.node_targets[recovered_idx] - disturbed_state
        self.assertGreater(np.dot(velocity, direction_to_target), 0.0)

        rmse = calculate_prediction_rmse(chained.x, chained.x_dot, chained)
        self.assertTrue(np.isfinite(rmse))

    def test_timer_trigger_controls_transition_completion(self):
        chained, _, _, _ = self._make_chain(enable_recovery=False)

        chained.reset_runtime(initial_idx=0, start_time=0.0)
        x_probe = np.array([-0.02, 0.0])  # close to mu_2 = first gaussian center

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

    def test_assignment_arr_is_available_for_plotting(self):
        chained, _, _, config = self._make_chain(enable_recovery=False)
        self.assertEqual(chained.assignment_arr.shape, (chained.x.shape[0],))
        self.assertTrue(np.all(chained.assignment_arr >= 0))
        self.assertTrue(np.all(chained.assignment_arr < chained.K))
        ax = plot_ds_set_gaussians([chained], config, include_trajectory=True)
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
