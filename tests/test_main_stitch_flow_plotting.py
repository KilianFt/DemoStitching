import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.stitching.main_stitch_flow import run_single_combination


class _FakeGG:
    gaussian_reversal_map = set()

    def get_gaussian(self, node):
        del node
        return (
            np.array([0.2, 0.3, 0.1], dtype=float),
            np.eye(3, dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
            1.0,
        )


class MainStitchFlowPlottingTests(unittest.TestCase):
    def _config(self, dim: int):
        extent = (-1.0, 2.0, -1.0, 2.0, -1.0, 2.0) if dim >= 3 else (-1.0, 2.0, -1.0, 2.0)
        return SimpleNamespace(
            ds_method="chain",
            save_fig=True,
            combination_timeout_s=30.0,
            plot_extent=extent,
            chain=SimpleNamespace(
                plot_mode="line_regions",
                plot_grid_resolution=20,
                plot_path_bandwidth=0.9,
                plot_show_transition_lines=True,
                plot_region_alpha=0.26,
            ),
        )

    def _demo_set(self, dim: int):
        x = np.array([[0.0, 0.0, 0.0], [0.8, 0.4, 0.2], [1.0, 0.6, 0.3]], dtype=float)
        if dim == 2:
            x = x[:, :2]
        traj = SimpleNamespace(x=x, x_dot=np.ones_like(x))
        demo = SimpleNamespace(trajectories=[traj])
        return [demo]

    def _stitched_ds(self, dim: int):
        x = np.array([[0.0, 0.0, 0.0], [0.7, 0.4, 0.2], [1.1, 0.6, 0.3]], dtype=float)
        x_dot = np.array([[1.0, 0.3, 0.1], [0.7, 0.2, 0.1], [0.3, 0.1, 0.0]], dtype=float)
        if dim == 2:
            x = x[:, :2]
            x_dot = x_dot[:, :2]
        return SimpleNamespace(
            x=x,
            x_dot=x_dot,
            damm=SimpleNamespace(Mu=np.array([[0.0] * dim], dtype=float)),
        )

    def test_run_single_combination_calls_clean_3d_plot_for_3d_results(self):
        plot_clean_3d = Mock()
        plot_composite = Mock()
        stitched = self._stitched_ds(dim=3)
        sim = [stitched.x.copy()]

        result_row, _, _, _ = run_single_combination(
            combination_id=7,
            initial=np.array([0.0, 0.0, 0.0], dtype=float),
            attractor=np.array([1.2, 0.7, 0.4], dtype=float),
            config=self._config(dim=3),
            gg=_FakeGG(),
            norm_demo_set=self._demo_set(dim=3),
            ds_set=[],
            reversed_ds_set=[],
            segment_ds_lookup={},
            demo_set=self._demo_set(dim=3),
            save_fig_indices=None,
            construct_stitched_ds_fn=lambda *args, **kwargs: (stitched, [("a", 0)], {"ds_compute_time": 0.1, "total_compute_time": 0.2}),
            simulate_trajectories_fn=lambda *args, **kwargs: sim,
            calculate_ds_metrics_fn=lambda **kwargs: {"prediction_rmse": 0.1},
            call_with_timeout_fn=lambda timeout_s, label, fn, *args, **kwargs: fn(*args, **kwargs),
            operation_timeout_cls=RuntimeError,
            default_stitching_stats_fn=lambda: {"ds_compute_time": np.nan, "total_compute_time": np.nan},
            nan_ds_metrics_fn=lambda initial, attractor: {"prediction_rmse": np.nan},
            extract_gaussian_node_indices_fn=lambda node_id: (0, 0),
            plot_gg_solution_fn=Mock(),
            plot_ds_set_gaussians_fn=Mock(),
            plot_ds_fn=Mock(),
            plot_composite_fn=plot_composite,
            plot_clean_3d_composite_fn=plot_clean_3d,
        )

        self.assertEqual(result_row["combination_status"], "ok")
        plot_composite.assert_called_once()
        plot_clean_3d.assert_called_once()
        self.assertEqual(plot_clean_3d.call_args.kwargs["save_as"], "7_Composite_3D_Clean")
        self.assertFalse(plot_clean_3d.call_args.kwargs["hide_axis"])

    def test_run_single_combination_skips_clean_3d_plot_for_2d_results(self):
        plot_clean_3d = Mock()
        stitched = self._stitched_ds(dim=2)

        result_row, _, _, _ = run_single_combination(
            combination_id=3,
            initial=np.array([0.0, 0.0], dtype=float),
            attractor=np.array([1.2, 0.7], dtype=float),
            config=self._config(dim=2),
            gg=_FakeGG(),
            norm_demo_set=self._demo_set(dim=2),
            ds_set=[],
            reversed_ds_set=[],
            segment_ds_lookup={},
            demo_set=self._demo_set(dim=2),
            save_fig_indices=None,
            construct_stitched_ds_fn=lambda *args, **kwargs: (stitched, [("a", 0)], {"ds_compute_time": 0.1, "total_compute_time": 0.2}),
            simulate_trajectories_fn=lambda *args, **kwargs: [stitched.x.copy()],
            calculate_ds_metrics_fn=lambda **kwargs: {"prediction_rmse": 0.1},
            call_with_timeout_fn=lambda timeout_s, label, fn, *args, **kwargs: fn(*args, **kwargs),
            operation_timeout_cls=RuntimeError,
            default_stitching_stats_fn=lambda: {"ds_compute_time": np.nan, "total_compute_time": np.nan},
            nan_ds_metrics_fn=lambda initial, attractor: {"prediction_rmse": np.nan},
            extract_gaussian_node_indices_fn=lambda node_id: (0, 0),
            plot_gg_solution_fn=Mock(),
            plot_ds_set_gaussians_fn=Mock(),
            plot_ds_fn=Mock(),
            plot_composite_fn=Mock(),
            plot_clean_3d_composite_fn=plot_clean_3d,
        )

        self.assertEqual(result_row["combination_status"], "ok")
        plot_clean_3d.assert_not_called()


if __name__ == "__main__":
    unittest.main()
