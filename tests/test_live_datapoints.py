import unittest
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

_LIVE_MODULE_PATH = Path(__file__).resolve().parents[1] / "live.py"
_REPO_ROOT = str(_LIVE_MODULE_PATH.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIVE_SPEC = importlib.util.spec_from_file_location("live_module_under_test", _LIVE_MODULE_PATH)
_LIVE_MODULE = importlib.util.module_from_spec(_LIVE_SPEC)
assert _LIVE_SPEC.loader is not None
_LIVE_SPEC.loader.exec_module(_LIVE_MODULE)
LiveStitchApp = _LIVE_MODULE.LiveStitchApp
LiveConfig = _LIVE_MODULE.LiveConfig
_collect_live_points_nd = _LIVE_MODULE._collect_live_points_nd
_compute_view_extent_3d = _LIVE_MODULE._compute_view_extent_3d


class LiveUsedDatapointsTests(unittest.TestCase):
    def test_live_config_defaults_include_sim_steps_per_frame(self):
        cfg = LiveConfig()
        self.assertGreaterEqual(int(cfg.sim_steps_per_frame), 1)

    def test_collect_live_points_nd_for_3d(self):
        demo_set = [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(
                        x=np.array(
                            [
                                [0.0, 0.0, 1.0],
                                [1.0, 2.0, 3.0],
                            ]
                        )
                    )
                ]
            )
        ]
        gaussian_map = {
            (0, 0): {
                "mu": np.array([3.0, 4.0, 5.0]),
                "sigma": np.eye(3),
                "direction": np.array([1.0, 0.0, 0.0]),
                "prior": 1.0,
            }
        }
        start = np.array([7.0, 8.0, 9.0])
        points = _collect_live_points_nd(demo_set, gaussian_map, start_state=start, ndim=3)
        self.assertEqual(points.shape[1], 3)
        self.assertGreaterEqual(points.shape[0], 4)
        self.assertTrue(np.any(np.all(np.isclose(points, np.array([3.0, 4.0, 5.0])), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(points, np.array([7.0, 8.0, 9.0])), axis=1)))

    def test_compute_view_extent_3d_is_cubic_and_padded(self):
        pts = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 7.0],
            ]
        )
        x_min, x_max, y_min, y_max, z_min, z_max = _compute_view_extent_3d(
            pts, padding_ratio=0.1, padding_abs=0.2
        )
        self.assertAlmostEqual(x_max - x_min, y_max - y_min, places=8)
        self.assertAlmostEqual(y_max - y_min, z_max - z_min, places=8)
        self.assertLessEqual(x_min, 1.0 - 0.2)
        self.assertGreaterEqual(z_max, 7.0 + 0.2)

    def test_compute_view_extent_3d_is_compacter_than_legacy_padding(self):
        pts = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 7.0],
            ]
        )
        x_min, x_max, y_min, y_max, z_min, z_max = _compute_view_extent_3d(
            pts, padding_ratio=0.1, padding_abs=0.2
        )
        width = x_max - x_min
        legacy_width = 2.0 * (0.5 * 4.0 + max(0.2, 0.1 * 4.0))  # max_span=4
        self.assertLess(width, legacy_width)
        self.assertAlmostEqual(width, y_max - y_min, places=8)
        self.assertAlmostEqual(width, z_max - z_min, places=8)

    def test_draw_ds_field_3d_creates_quiver(self):
        class _DS:
            @staticmethod
            def predict_velocities(points):
                vel = np.zeros_like(points)
                vel[:, 0] = 1.0
                return vel

        app = LiveStitchApp.__new__(LiveStitchApp)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        app.ax = ax
        app.is_3d = True
        app.stream = None
        app.quiver = None
        app.x_min, app.x_max = -1.0, 1.0
        app.y_min, app.y_max = -1.0, 1.0
        app.z_min, app.z_max = -1.0, 1.0
        app.ctrl = SimpleNamespace(
            current_ds=_DS(),
            state_dim=3,
            current_state=np.array([0.0, 0.0, 0.0]),
            config=SimpleNamespace(ds_method="sp_recompute_ds"),
        )

        LiveStitchApp._draw_ds_field(app)

        self.assertIsNotNone(app.quiver)
        plt.close(fig)

    def test_draw_ds_field_2d_chain_draws_region_and_transition_lines(self):
        class _ChainDS:
            n_systems = 3
            node_sources = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=float)
            transition_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
            transition_normals = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
            transition_distances = np.array([1.0, 1.0], dtype=float)

            @staticmethod
            def _velocity_for_index(x, idx):
                x = np.asarray(x, dtype=float).reshape(-1)
                v = np.zeros_like(x)
                v[0] = float(idx + 1)
                return v

        app = LiveStitchApp.__new__(LiveStitchApp)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        app.ax = ax
        app.is_3d = False
        app.stream = None
        app.quiver = None
        app.chain_region_artist = None
        app.chain_transition_line_artists = []
        app.x_min, app.x_max = -1.0, 2.0
        app.y_min, app.y_max = -1.0, 1.0
        app.ctrl = SimpleNamespace(
            current_ds=_ChainDS(),
            state_dim=2,
            current_state=np.array([0.0, 0.0]),
            current_chain_idx=0,
            config=SimpleNamespace(
                ds_method="chain",
                chain=SimpleNamespace(
                    plot_mode="line_regions",
                    plot_grid_resolution=18,
                    plot_show_transition_lines=True,
                    plot_region_alpha=0.3,
                    ds_method="linear",
                ),
            ),
        )

        LiveStitchApp._draw_ds_field(app)

        self.assertIsNotNone(app.stream)
        self.assertIsNotNone(app.chain_region_artist)
        self.assertEqual(app.chain_region_artist.get_array().shape[0], 18)
        self.assertEqual(app.chain_region_artist.get_array().shape[1], 18)
        self.assertEqual(len(app.chain_transition_line_artists), 1)
        self.assertGreater(len(app.chain_transition_line_artists[0].get_segments()), 0)
        plt.close(fig)

    def test_draw_ds_field_2d_chain_active_ds_mode_draws_only_active_field(self):
        class _ChainDS:
            n_systems = 3
            node_sources = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=float)
            transition_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
            transition_normals = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
            transition_distances = np.array([1.0, 1.0], dtype=float)

            @staticmethod
            def _velocity_for_index(x, idx):
                x = np.asarray(x, dtype=float).reshape(-1)
                v = np.zeros_like(x)
                v[0] = float(idx + 1)
                return v

        app = LiveStitchApp.__new__(LiveStitchApp)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        app.ax = ax
        app.is_3d = False
        app.stream = None
        app.quiver = None
        app.chain_region_artist = None
        app.chain_transition_line_artists = []
        app.x_min, app.x_max = -1.0, 2.0
        app.y_min, app.y_max = -1.0, 1.0
        app.ctrl = SimpleNamespace(
            current_ds=_ChainDS(),
            state_dim=2,
            current_state=np.array([0.0, 0.0]),
            current_chain_idx=1,
            config=SimpleNamespace(
                ds_method="chain",
                chain=SimpleNamespace(
                    live_field_mode="active_ds",
                    plot_mode="line_regions",
                    plot_grid_resolution=18,
                    plot_path_bandwidth=0.35,
                    plot_show_transition_lines=True,
                    plot_region_alpha=0.3,
                    ds_method="linear",
                ),
            ),
        )

        LiveStitchApp._draw_ds_field(app)

        self.assertIsNotNone(app.stream)
        self.assertIsNone(app.chain_region_artist)
        self.assertEqual(len(app.chain_transition_line_artists), 0)
        plt.close(fig)

    def test_chain_transition_indicator_shows_progress_and_idle(self):
        app = LiveStitchApp.__new__(LiveStitchApp)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        app.ax = ax
        app.is_3d = False
        app.chain_transition_progress_bg = None
        app.chain_transition_progress_fg = None
        app.chain_transition_progress_text = None
        app.ctrl = SimpleNamespace(
            config=SimpleNamespace(ds_method="chain"),
            current_ds=SimpleNamespace(
                transition_active=True,
                _transition_from_idx=0,
                _transition_t0=1.0,
                _runtime_time=1.5,
                transition_times=np.array([2.0], dtype=float),
            ),
        )

        LiveStitchApp._draw_chain_transition_indicator(app)
        self.assertIsNotNone(app.chain_transition_progress_bg)
        self.assertIsNotNone(app.chain_transition_progress_fg)
        self.assertIsNotNone(app.chain_transition_progress_text)
        self.assertTrue(app.chain_transition_progress_bg.get_visible())
        self.assertGreater(float(app.chain_transition_progress_fg.get_width()), 0.0)
        self.assertIn("transition 0->1", app.chain_transition_progress_text.get_text())

        app.ctrl.current_ds.transition_active = False
        LiveStitchApp._update_chain_transition_indicator(app)
        self.assertFalse(app.chain_transition_progress_bg.get_visible())
        self.assertIn("idle", app.chain_transition_progress_text.get_text().lower())
        plt.close(fig)

    def test_draw_graph_3d_adds_edges(self):
        graph = nx.DiGraph()
        graph.add_node("a", mean=np.array([0.0, 0.0, 0.0]))
        graph.add_node("b", mean=np.array([1.0, 1.0, 1.0]))
        graph.add_edge("a", "b", weight=1.0)

        app = LiveStitchApp.__new__(LiveStitchApp)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        app.ax = ax
        app.is_3d = True
        app.ctrl = SimpleNamespace(
            current_gg=SimpleNamespace(graph=graph),
            chain_base_graph=SimpleNamespace(graph=graph),
        )

        LiveStitchApp._draw_graph(app)

        self.assertGreaterEqual(len(ax.lines), 1)
        plt.close(fig)

    def test_draw_chain_fit_points_uses_ds_x_for_non_chain_method(self):
        app = LiveStitchApp.__new__(LiveStitchApp)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        app.ax = ax
        app.is_3d = False
        app.ctrl = SimpleNamespace(
            config=SimpleNamespace(ds_method="sp_recompute_ds"),
            current_ds=SimpleNamespace(
                x=np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [1.0, 2.0, 3.0],
                        [2.0, 1.0, 5.0],
                    ]
                )
            ),
        )
        app.chain_fit_points_artist = None
        app.chain_fit_info_text = None

        LiveStitchApp._draw_chain_fit_points(app)

        self.assertIsNotNone(app.chain_fit_points_artist)
        self.assertIsNotNone(app.chain_fit_info_text)
        self.assertEqual(app.chain_fit_points_artist.get_offsets().shape[0], 3)
        self.assertIn("DS fit n=3", app.chain_fit_info_text.get_text())
        plt.close(fig)

    def test_on_click_reports_used_datapoints_for_non_chain_method(self):
        app = LiveStitchApp.__new__(LiveStitchApp)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        app.ax = ax
        app.is_3d = False
        app._redraw_scene = lambda: None

        app.ctrl = SimpleNamespace(
            config=SimpleNamespace(ds_method="sp_recompute_ds"),
            current_path_nodes=[("n0",), ("n1",)],
            current_ds=SimpleNamespace(x=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])),
            plan_to_goal=lambda goal: True,
            current_chain_direction_stats=lambda: None,
        )

        event = SimpleNamespace(inaxes=ax, xdata=0.3, ydata=0.6)
        with patch("builtins.print") as mock_print:
            LiveStitchApp._on_click(app, event)

        printed = "\n".join(call.args[0] for call in mock_print.call_args_list if call.args)
        self.assertIn("Used fit datapoints: n=3", printed)
        plt.close(fig)

    def test_animate_advances_multiple_steps_per_frame_without_changing_dt(self):
        class _Line:
            def set_data(self, *_args):
                return None

        class _Point:
            def set_data(self, *_args):
                return None

        calls = {"steps": 0, "indicator": 0, "refresh": 0, "redraw": 0}

        class _Ctrl:
            def __init__(self):
                self.current_ds = object()
                self.current_chain_idx = 0
                self.current_state = np.array([0.0, 0.0], dtype=float)
                self.trajectory = [self.current_state.copy()]
                self.config = SimpleNamespace(
                    ds_method="chain",
                    sim_steps_per_frame=4,
                    chain=SimpleNamespace(live_field_mode="partition"),
                )

            def step_once(self):
                calls["steps"] += 1
                self.current_state = np.array([float(calls["steps"]), 0.0], dtype=float)
                self.trajectory.append(self.current_state.copy())

            @staticmethod
            def current_chain_direction_stats():
                return None

        app = LiveStitchApp.__new__(LiveStitchApp)
        app.ctrl = _Ctrl()
        app.is_3d = False
        app.traj_line = _Line()
        app.point_artist = _Point()
        app._update_chain_transition_indicator = lambda: calls.__setitem__("indicator", calls["indicator"] + 1)
        app._refresh_chain_artists_for_switch = lambda: calls.__setitem__("refresh", calls["refresh"] + 1)
        app._redraw_scene = lambda: calls.__setitem__("redraw", calls["redraw"] + 1)

        LiveStitchApp._animate(app, 0)

        self.assertEqual(calls["steps"], 4)
        self.assertEqual(calls["indicator"], 1)
        self.assertEqual(calls["refresh"], 0)
        self.assertEqual(calls["redraw"], 0)
        np.testing.assert_allclose(app.ctrl.current_state, np.array([4.0, 0.0]))
        self.assertEqual(len(app.ctrl.trajectory), 5)

    def test_animate_chain_switch_in_partition_mode_updates_overlays_without_full_redraw(self):
        class _Line:
            def set_data(self, *_args):
                return None

        class _Point:
            def set_data(self, *_args):
                return None

        calls = {"refresh": 0, "redraw": 0}

        class _Ctrl:
            def __init__(self):
                self.current_ds = object()
                self.current_chain_idx = 0
                self.current_state = np.array([0.0, 0.0], dtype=float)
                self.trajectory = [self.current_state.copy()]
                self.config = SimpleNamespace(
                    ds_method="chain",
                    sim_steps_per_frame=1,
                    chain=SimpleNamespace(live_field_mode="partition"),
                )

            def step_once(self):
                self.current_chain_idx = 1
                self.current_state = np.array([1.0, 0.0], dtype=float)
                self.trajectory.append(self.current_state.copy())

            @staticmethod
            def current_chain_direction_stats():
                return None

        app = LiveStitchApp.__new__(LiveStitchApp)
        app.ctrl = _Ctrl()
        app.is_3d = False
        app.traj_line = _Line()
        app.point_artist = _Point()
        app._update_chain_transition_indicator = lambda: None
        app._refresh_chain_artists_for_switch = lambda: calls.__setitem__("refresh", calls["refresh"] + 1)
        app._redraw_scene = lambda: calls.__setitem__("redraw", calls["redraw"] + 1)

        LiveStitchApp._animate(app, 0)

        self.assertEqual(calls["refresh"], 1)
        self.assertEqual(calls["redraw"], 0)

    def test_animate_chain_switch_in_active_ds_mode_triggers_full_redraw(self):
        class _Line:
            def set_data(self, *_args):
                return None

        class _Point:
            def set_data(self, *_args):
                return None

        calls = {"refresh": 0, "redraw": 0}

        class _Ctrl:
            def __init__(self):
                self.current_ds = object()
                self.current_chain_idx = 0
                self.current_state = np.array([0.0, 0.0], dtype=float)
                self.trajectory = [self.current_state.copy()]
                self.config = SimpleNamespace(
                    ds_method="chain",
                    sim_steps_per_frame=1,
                    chain=SimpleNamespace(live_field_mode="active_ds"),
                )

            def step_once(self):
                self.current_chain_idx = 1
                self.current_state = np.array([1.0, 0.0], dtype=float)
                self.trajectory.append(self.current_state.copy())

            @staticmethod
            def current_chain_direction_stats():
                return None

        app = LiveStitchApp.__new__(LiveStitchApp)
        app.ctrl = _Ctrl()
        app.is_3d = False
        app.traj_line = _Line()
        app.point_artist = _Point()
        app._update_chain_transition_indicator = lambda: None
        app._refresh_chain_artists_for_switch = lambda: calls.__setitem__("refresh", calls["refresh"] + 1)
        app._redraw_scene = lambda: calls.__setitem__("redraw", calls["redraw"] + 1)

        LiveStitchApp._animate(app, 0)

        self.assertEqual(calls["refresh"], 0)
        self.assertEqual(calls["redraw"], 1)


if __name__ == "__main__":
    unittest.main()
