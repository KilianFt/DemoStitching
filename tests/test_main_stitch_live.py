import unittest
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from main_stitch_live import (
    LiveConfig,
    LiveStitchController,
    _collect_live_points_2d,
    _compute_view_extent,
    _predict_velocity_field,
)


class _PredictDS:
    def predict_velocities(self, points):
        return 2.0 * points


class _GammaDS:
    def __init__(self):
        self.A = np.array([-np.eye(2), -2.0 * np.eye(2)])
        self.x_att = np.array([0.0, 0.0])

        class _Damm:
            @staticmethod
            def compute_gamma(points):
                # K x M
                gamma_0 = np.full(points.shape[0], 0.25)
                gamma_1 = np.full(points.shape[0], 0.75)
                return np.vstack([gamma_0, gamma_1])

        self.damm = _Damm()


class _RaisingArtist:
    def remove(self):
        raise NotImplementedError("cannot remove artist")


class _DummyStream:
    def __init__(self):
        self.lines = _RaisingArtist()
        self.arrows = _RaisingArtist()


class LiveStitchVelocityFieldTests(unittest.TestCase):
    def test_live_config_default_figure_size_is_larger(self):
        cfg = LiveConfig()
        self.assertGreaterEqual(cfg.figure_width, 10.0)
        self.assertGreaterEqual(cfg.figure_height, 10.0)

    def test_collect_live_points_includes_demos_gaussians_and_start(self):
        demo_set = [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(x=np.array([[0.0, 0.0], [1.0, 1.0]])),
                    SimpleNamespace(x=np.array([[2.0, 1.5, 7.0]])),
                ]
            )
        ]
        gaussian_map = {
            (0, 0): {"mu": np.array([3.0, 2.0]), "sigma": np.eye(2), "direction": np.array([1.0, 0.0]), "prior": 1.0},
        }
        start = np.array([4.0, 5.0])
        points = _collect_live_points_2d(demo_set, gaussian_map, start_state=start)
        self.assertEqual(points.shape[1], 2)
        self.assertGreaterEqual(points.shape[0], 5)
        self.assertTrue(np.any(np.all(np.isclose(points, np.array([3.0, 2.0])), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(points, np.array([4.0, 5.0])), axis=1)))

    def test_compute_view_extent_is_square_and_has_margin(self):
        points = np.array([[1.0, 1.0], [3.0, 2.0]])
        x_min, x_max, y_min, y_max = _compute_view_extent(points, padding_ratio=0.1, padding_abs=0.2)
        self.assertAlmostEqual(x_max - x_min, y_max - y_min, places=8)
        self.assertLessEqual(x_min, 1.0 - 0.2)
        self.assertGreaterEqual(x_max, 3.0 + 0.2)
        self.assertLessEqual(y_min, 1.0 - 0.2)
        self.assertGreaterEqual(y_max, 2.0 + 0.2)

    def test_compute_view_extent_handles_empty_input(self):
        x_min, x_max, y_min, y_max = _compute_view_extent(np.zeros((0, 2)))
        self.assertEqual((x_min, x_max, y_min, y_max), (-1.0, 1.0, -1.0, 1.0))

    def test_predict_velocity_field_prefers_predict_velocities(self):
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        vel = _predict_velocity_field(_PredictDS(), points)
        np.testing.assert_allclose(vel, 2.0 * points)

    def test_predict_velocity_field_gamma_weighted_fallback(self):
        points = np.array([[2.0, 0.0], [0.0, 2.0]])
        vel = _predict_velocity_field(_GammaDS(), points)
        expected = -(0.25 + 1.5) * points
        np.testing.assert_allclose(vel, expected)

    def test_streamplot_remove_is_tolerant_to_backend_errors(self):
        from main_stitch_live import LiveStitchApp

        app = LiveStitchApp.__new__(LiveStitchApp)
        app.stream = _DummyStream()
        app._remove_streamplot()
        self.assertIsNone(app.stream)

    def test_plan_to_goal_keeps_current_state_and_trajectory(self):
        class _DummyChainDS:
            def __init__(self):
                self.state_sequence = np.array([[1.0, 2.0], [2.0, 3.0]])
                self.reset_calls = []

            def reset_runtime(self, initial_idx=0, start_time=0.0):
                self.reset_calls.append((initial_idx, start_time))

        ctrl = LiveStitchController.__new__(LiveStitchController)
        ctrl.config = SimpleNamespace(ds_method="chain")
        ctrl.current_state = np.array([1.0, 2.0])
        ctrl.start_state = np.array([0.0, 0.0])
        ctrl.goal_state = None
        ctrl.current_ds = None
        ctrl.current_gg = None
        ctrl.current_path_nodes = None
        ctrl.current_chain_idx = None
        ctrl.trajectory = [np.array([0.0, 0.0]), np.array([1.0, 2.0])]
        dummy_ds = _DummyChainDS()

        ctrl._compose_goal_state = lambda goal_xy: np.asarray(goal_xy, dtype=float)
        ctrl._build_chain_ds = lambda initial, attractor: (dummy_ds, object(), [("n0",), ("n1",)])
        ctrl._build_other_ds = lambda initial, attractor: (None, None, None)

        planned = LiveStitchController.plan_to_goal(ctrl, np.array([3.0, 4.0]), keep_trajectory=True)
        self.assertTrue(planned)
        np.testing.assert_allclose(ctrl.current_state, np.array([1.0, 2.0]))
        np.testing.assert_allclose(ctrl.path_anchor_state, np.array([1.0, 2.0]))
        self.assertEqual(len(ctrl.trajectory), 2)
        self.assertEqual(ctrl.current_path_nodes, [("n0",), ("n1",)])
        self.assertEqual(dummy_ds.reset_calls[-1][0], 0)

    def test_path_points_use_fixed_path_anchor_not_current_state(self):
        ctrl = LiveStitchController.__new__(LiveStitchController)
        ctrl.current_gg = SimpleNamespace(
            graph=SimpleNamespace(
                nodes={
                    ("n0",): {"mean": np.array([1.0, 1.0])},
                    ("n1",): {"mean": np.array([2.0, 2.0])},
                }
            )
        )
        ctrl.current_path_nodes = [("n0",), ("n1",)]
        ctrl.path_anchor_state = np.array([0.0, 0.0])
        ctrl.current_state = np.array([9.0, 9.0])
        ctrl.goal_state = np.array([3.0, 3.0])

        points = LiveStitchController.path_points_2d(ctrl)
        np.testing.assert_allclose(points[0], np.array([0.0, 0.0]))
        np.testing.assert_allclose(points[1], np.array([1.0, 1.0]))
        np.testing.assert_allclose(points[2], np.array([2.0, 2.0]))
        np.testing.assert_allclose(points[3], np.array([3.0, 3.0]))

    def test_reset_to_start_only_resets_on_explicit_call(self):
        ctrl = LiveStitchController.__new__(LiveStitchController)
        ctrl.start_state = np.array([0.5, 0.5])
        ctrl.current_state = np.array([3.0, 4.0])
        ctrl.goal_state = np.array([9.0, 1.0])
        ctrl.trajectory = [np.array([1.0, 1.0]), np.array([3.0, 4.0])]
        ctrl.current_ds = object()
        ctrl.current_gg = object()
        ctrl.current_path_nodes = [("a",)]
        ctrl.current_chain_idx = 0

        calls = {}

        def _fake_plan(goal_xy, keep_trajectory=True, initial_override=None):
            calls["goal_xy"] = np.asarray(goal_xy, dtype=float)
            calls["keep_trajectory"] = keep_trajectory
            calls["initial_override"] = np.asarray(initial_override, dtype=float)
            return True

        ctrl.plan_to_goal = _fake_plan
        ok = LiveStitchController.reset_to_start(ctrl)
        self.assertTrue(ok)
        np.testing.assert_allclose(ctrl.current_state, ctrl.start_state)
        self.assertEqual(len(ctrl.trajectory), 1)
        self.assertFalse(calls["keep_trajectory"])
        np.testing.assert_allclose(calls["initial_override"], ctrl.start_state)

    def test_apply_disturbance_updates_state_and_chain_index(self):
        class _DummyChainDS:
            state_sequence = np.array([[0.0, 0.0], [1.5, 2.0], [3.0, 2.0]])

        ctrl = LiveStitchController.__new__(LiveStitchController)
        ctrl.config = SimpleNamespace(ds_method="chain", disturbance_step=0.5)
        ctrl.current_state = np.array([1.0, 2.0])
        ctrl.trajectory = [ctrl.current_state.copy()]
        ctrl.current_ds = _DummyChainDS()
        ctrl.current_chain_idx = 0

        disturbed = LiveStitchController.apply_disturbance(ctrl, np.array([1.0, 0.0]))
        np.testing.assert_allclose(disturbed, np.array([1.5, 2.0]))
        np.testing.assert_allclose(ctrl.current_state, np.array([1.5, 2.0]))
        self.assertEqual(len(ctrl.trajectory), 2)
        self.assertEqual(ctrl.current_chain_idx, 1)

    def test_on_key_press_arrow_applies_disturbance(self):
        from main_stitch_live import LiveStitchApp

        calls = {"direction": None, "redraws": 0}

        class _Ctrl:
            def apply_disturbance(self, direction):
                calls["direction"] = np.asarray(direction, dtype=float)
                return np.array([1.0, 1.0])

        app = LiveStitchApp.__new__(LiveStitchApp)
        app.ctrl = _Ctrl()
        app.disturbance_keys = {
            "left": np.array([-1.0, 0.0]),
            "right": np.array([1.0, 0.0]),
            "up": np.array([0.0, 1.0]),
            "down": np.array([0.0, -1.0]),
        }
        app._redraw_scene = lambda: calls.__setitem__("redraws", calls["redraws"] + 1)

        event = SimpleNamespace(key="left")
        LiveStitchApp._on_key_press(app, event)

        np.testing.assert_allclose(calls["direction"], np.array([-1.0, 0.0]))
        self.assertEqual(calls["redraws"], 1)

    def test_current_chain_direction_stats_returns_active_edge_stats(self):
        ctrl = LiveStitchController.__new__(LiveStitchController)
        ctrl.config = SimpleNamespace(ds_method="chain")
        ctrl.current_chain_idx = 1
        ctrl.current_ds = SimpleNamespace(
            n_systems=3,
            edge_direction_stats=[
                {"n_points": 2, "frac_forward": 1.0, "min_proj": 0.1},
                {"n_points": 3, "frac_forward": 2.0 / 3.0, "min_proj": -0.2},
                {"n_points": 1, "frac_forward": 1.0, "min_proj": 0.01},
            ],
        )

        stats = LiveStitchController.current_chain_direction_stats(ctrl)
        self.assertEqual(stats["n_points"], 3)
        self.assertAlmostEqual(stats["frac_forward"], 2.0 / 3.0)

    def test_draw_chain_fit_points_adds_scatter_and_text(self):
        from main_stitch_live import LiveStitchApp

        app = LiveStitchApp.__new__(LiveStitchApp)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        app.ax = ax
        app.ctrl = SimpleNamespace(
            config=SimpleNamespace(ds_method="chain"),
            current_chain_idx=0,
            current_ds=SimpleNamespace(
                n_systems=2,
                edge_fit_points=[np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[2.0, 2.0]])],
                edge_direction_stats=[
                    {"n_points": 2, "frac_forward": 1.0, "min_proj": 0.5},
                    {"n_points": 1, "frac_forward": 1.0, "min_proj": 0.2},
                ],
            ),
        )
        app.chain_fit_points_artist = None
        app.chain_fit_info_text = None

        LiveStitchApp._draw_chain_fit_points(app)

        self.assertIsNotNone(app.chain_fit_points_artist)
        self.assertIsNotNone(app.chain_fit_info_text)
        self.assertIn("forward", app.chain_fit_info_text.get_text())
        plt.close(fig)

    def test_draw_gaussians_adds_ellipses_and_centers(self):
        from main_stitch_live import LiveStitchApp

        app = LiveStitchApp.__new__(LiveStitchApp)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        app.ax = ax
        app.gaussian_center_artist = None
        app.ctrl = SimpleNamespace(
            current_gg=SimpleNamespace(
                graph=SimpleNamespace(
                    nodes=lambda data=False: [
                        (("g0",), {"mean": np.array([0.0, 0.0]), "covariance": np.array([[0.04, 0.0], [0.0, 0.01]])}),
                        (("g1",), {"mean": np.array([1.0, 1.0]), "covariance": np.array([[0.09, 0.0], [0.0, 0.04]])}),
                        (("tmp",), {"mean": np.array([2.0, 2.0])}),  # non-gaussian node (no covariance)
                    ]
                )
            ),
            gaussian_map={},
        )

        LiveStitchApp._draw_gaussians(app)

        self.assertEqual(len(ax.patches), 2)
        self.assertIsNotNone(app.gaussian_center_artist)
        self.assertEqual(app.gaussian_center_artist.get_offsets().shape[0], 2)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
