import unittest
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.util.plot_tools import (
    draw_chain_partition_field_2d,
    evaluate_chain_regions,
    plot_ds,
    resolve_chain_plot_mode,
)


class _DummyChainDS:
    def __init__(self):
        self.n_systems = 3
        self.x = np.array(
            [
                [-0.8, -0.1],
                [-0.2, 0.0],
                [0.4, 0.1],
                [1.4, 0.0],
            ],
            dtype=float,
        )
        self.x_dot = np.zeros_like(self.x)
        self.x_att = np.array([2.0, 0.0], dtype=float)
        self.A = np.array([np.eye(2)], dtype=float)
        self.node_sources = np.array(
            [
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )
        self.transition_centers = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )
        self.transition_normals = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )
        self.transition_distances = np.array([1.0, 1.0], dtype=float)

    @staticmethod
    def _velocity_for_index(x, idx):
        x = np.asarray(x, dtype=float).reshape(-1)
        v = np.zeros_like(x)
        v[0] = float(idx + 1)
        return v


class ChainPlotRegionTests(unittest.TestCase):
    def test_resolve_chain_plot_mode_handles_aliases(self):
        self.assertEqual(resolve_chain_plot_mode("line_regions"), "line_regions")
        self.assertEqual(resolve_chain_plot_mode("hard_lines"), "line_regions")
        self.assertEqual(resolve_chain_plot_mode("blend"), "time_blend")
        self.assertEqual(resolve_chain_plot_mode("unknown-value"), "line_regions")

    def test_line_regions_selects_single_active_ds_per_region(self):
        ds = _DummyChainDS()
        points = np.array(
            [
                [-0.4, 0.0],
                [0.2, 0.0],
                [1.3, 0.0],
            ],
            dtype=float,
        )
        velocities, region_idx, weights, _ = evaluate_chain_regions(ds, points, mode="line_regions")

        np.testing.assert_array_equal(region_idx, np.array([0, 1, 2]))
        np.testing.assert_allclose(velocities[:, 0], np.array([1.0, 2.0, 3.0]), atol=1e-8)
        np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(points.shape[0]), atol=1e-8)
        np.testing.assert_allclose(np.max(weights, axis=1), np.ones(points.shape[0]), atol=1e-8)

    def test_time_blend_blends_neighbor_ds_by_transition_length(self):
        ds = _DummyChainDS()
        points = np.array([[0.25, 0.0]], dtype=float)
        velocities, region_idx, weights, _ = evaluate_chain_regions(ds, points, mode="time_blend")

        self.assertEqual(int(region_idx[0]), 1)
        # signed distance to first line = 0.25, transition distance = 1.0
        # => alpha = 0.25; v = 0.75*v0 + 0.25*v1 = 0.75*1 + 0.25*2 = 1.25
        self.assertAlmostEqual(float(weights[0, 0]), 0.75, places=8)
        self.assertAlmostEqual(float(weights[0, 1]), 0.25, places=8)
        self.assertAlmostEqual(float(velocities[0, 0]), 1.25, places=8)

    def test_draw_chain_partition_field_draws_lines_in_line_mode(self):
        ds = _DummyChainDS()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        artists = draw_chain_partition_field_2d(
            ax=ax,
            ds=ds,
            x_min=-1.0,
            x_max=2.0,
            y_min=-1.0,
            y_max=1.0,
            mode="line_regions",
            plot_sample=20,
            show_transition_lines=True,
        )

        self.assertIsNotNone(artists["region_image"])
        self.assertIsNotNone(artists["stream"])
        self.assertEqual(len(artists["transition_lines"]), 2)
        plt.close(fig)

    def test_plot_ds_chain_uses_partition_background(self):
        ds = _DummyChainDS()
        cfg = type(
            "_Cfg",
            (),
            {
                "plot_extent": (-1.0, 2.0, -1.0, 1.0),
                "dataset_path": "dataset/stitching/testing",
                "ds_method": "chain",
                "chain": type(
                    "_ChainCfg",
                    (),
                    {
                        "plot_mode": "line_regions",
                        "plot_grid_resolution": 31,
                        "plot_show_transition_lines": True,
                        "plot_region_alpha": 0.3,
                    },
                )(),
            },
        )()

        ax = plot_ds(
            ds,
            x_test_list=[np.array([[-0.9, 0.0], [1.9, 0.0]], dtype=float)],
            initial=np.array([-1.0, 0.0], dtype=float),
            attractor=np.array([2.0, 0.0], dtype=float),
            config=cfg,
            save_as=None,
            hide_axis=True,
        )
        self.assertGreaterEqual(len(ax.images), 1)
        self.assertEqual(ax.images[0].get_array().shape[0], 31)
        self.assertEqual(ax.images[0].get_array().shape[1], 31)
        self.assertGreaterEqual(len(ax.lines), 3)
        plt.close(ax.figure)


if __name__ == "__main__":
    unittest.main()
