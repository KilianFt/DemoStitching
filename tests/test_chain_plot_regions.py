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

    def test_line_regions_follow_mean_normal_lines_not_node_distance(self):
        ds = _DummyChainDS()
        # Move source nodes so nearest-node ownership would differ, but line partition should stay unchanged.
        ds.node_sources = np.array([[-10.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=float)
        points = np.array([[0.90, 0.0]], dtype=float)
        _, region_idx, weights, _ = evaluate_chain_regions(ds, points, mode="line_regions")

        self.assertEqual(int(region_idx[0]), 1)
        self.assertAlmostEqual(float(weights[0, 1]), 1.0, places=8)

    def test_ambiguous_line_intersection_region_uses_nearest_regime_segment(self):
        ds = _DummyChainDS()
        ds.node_sources = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=float,
        )
        ds.node_targets = np.array(
            [
                [2.0, 0.0],
                [4.0, 0.0],
                [6.0, 0.0],
            ],
            dtype=float,
        )
        # Crossing pattern at x=[3,0.2]: first boundary False, second True -> ambiguous.
        ds.transition_centers = np.array(
            [
                [5.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=float,
        )
        ds.transition_normals = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        points = np.array([[3.0, 0.2]], dtype=float)
        velocities, region_idx, weights, _ = evaluate_chain_regions(ds, points, mode="line_regions")
        # Line-prefix rule alone would pick region 0; fallback should pick nearest regime segment 1.
        self.assertEqual(int(region_idx[0]), 1)
        self.assertAlmostEqual(float(weights[0, 1]), 1.0, places=8)
        self.assertAlmostEqual(float(velocities[0, 0]), 2.0, places=8)

    def test_ambiguous_fallback_stays_local_to_adjacent_regimes(self):
        ds = _DummyChainDS()
        ds.n_systems = 4
        ds.node_sources = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=float,
        )
        ds.node_targets = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=float,
        )
        # Non-monotonic crossing at x=2.1: [False, True, True].
        ds.transition_centers = np.array(
            [
                [5.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=float,
        )
        ds.transition_normals = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )
        points = np.array([[2.1, 0.0]], dtype=float)
        velocities, region_idx, weights, _ = evaluate_chain_regions(ds, points, mode="line_regions")

        # The nearest global regime would be index 2, but fallback must stay local
        # to the boundary pair around idx_line (indices 0/1), so it picks 1.
        self.assertEqual(int(region_idx[0]), 1)
        self.assertAlmostEqual(float(weights[0, 1]), 1.0, places=8)
        self.assertAlmostEqual(float(velocities[0, 0]), 2.0, places=8)

    def test_many_chain_regimes_use_distinct_region_colors(self):
        class _LongChainDS:
            def __init__(self, n_systems):
                self.n_systems = int(n_systems)
                xs = np.arange(self.n_systems, dtype=float)
                self.node_sources = np.column_stack([xs, np.zeros_like(xs)])
                self.node_targets = np.column_stack([xs + 1.0, np.zeros_like(xs)])
                self.transition_centers = np.column_stack([np.arange(self.n_systems - 1, dtype=float), np.zeros((self.n_systems - 1,))])
                self.transition_normals = np.tile(np.array([[1.0, 0.0]], dtype=float), (self.n_systems - 1, 1))

            @staticmethod
            def _velocity_for_index(x, idx):
                x = np.asarray(x, dtype=float).reshape(-1)
                v = np.zeros_like(x)
                v[0] = float(idx + 1)
                return v

        ds = _LongChainDS(25)
        points = np.array([[r - 0.5, 0.0] for r in range(1, ds.n_systems)], dtype=float)
        points = np.vstack([np.array([[-0.5, 0.0]], dtype=float), points])
        _, region_idx, _, rgba = evaluate_chain_regions(ds, points, mode="line_regions")
        self.assertEqual(region_idx.shape[0], ds.n_systems)
        self.assertEqual(len(np.unique(region_idx)), ds.n_systems)
        self.assertEqual(len(np.unique(np.round(rgba[:, :3], 6), axis=0)), ds.n_systems)

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

    def test_time_blend_draws_lines_only_for_non_transition_boundaries(self):
        ds = _DummyChainDS()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        artists = draw_chain_partition_field_2d(
            ax=ax,
            ds=ds,
            x_min=-1.0,
            x_max=2.0,
            y_min=-1.0,
            y_max=1.0,
            mode="time_blend",
            plot_sample=24,
            show_transition_lines=True,
            path_bandwidth=0.25,
        )
        # Both boundaries have nonzero transition distance -> no hard separator lines.
        self.assertEqual(len(artists["transition_lines"]), 0)
        plt.close(fig)

        # Disable transition on one boundary -> draw exactly one line.
        ds = _DummyChainDS()
        ds.transition_distances = np.array([0.0, 1.0], dtype=float)
        ds.transition_times = np.array([0.0, 0.2], dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        artists = draw_chain_partition_field_2d(
            ax=ax,
            ds=ds,
            x_min=-1.0,
            x_max=2.0,
            y_min=-1.0,
            y_max=1.0,
            mode="time_blend",
            plot_sample=24,
            show_transition_lines=True,
            path_bandwidth=0.25,
        )
        self.assertEqual(len(artists["transition_lines"]), 1)
        plt.close(fig)

    def test_time_blend_uses_closest_ds_in_ambiguous_regions(self):
        ds = _DummyChainDS()
        ds.node_sources = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=float,
        )
        ds.node_targets = np.array(
            [
                [2.0, 0.0],
                [4.0, 0.0],
                [6.0, 0.0],
            ],
            dtype=float,
        )
        ds.transition_centers = np.array(
            [
                [5.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=float,
        )
        ds.transition_normals = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        points = np.array([[3.0, 0.2]], dtype=float)
        velocities, region_idx, weights, _ = evaluate_chain_regions(ds, points, mode="time_blend")

        # Ambiguous intersection should remain closest-DS ownership (no blending).
        self.assertEqual(int(region_idx[0]), 1)
        self.assertAlmostEqual(float(weights[0, 1]), 1.0, places=8)
        self.assertAlmostEqual(float(np.sum(weights[0])), 1.0, places=8)
        self.assertAlmostEqual(float(velocities[0, 0]), 2.0, places=8)

    def test_time_blend_reverts_to_closest_ds_far_from_transition_neighborhood(self):
        ds = _DummyChainDS()
        points = np.array([[0.25, 5.0]], dtype=float)
        velocities, region_idx, weights, _ = evaluate_chain_regions(ds, points, mode="time_blend")

        # Same x as blending case, but far from path/transition neighborhood:
        # should use hard assignment only.
        self.assertEqual(int(region_idx[0]), 1)
        self.assertAlmostEqual(float(weights[0, 1]), 1.0, places=8)
        self.assertAlmostEqual(float(weights[0, 0]), 0.0, places=8)
        self.assertAlmostEqual(float(velocities[0, 0]), 2.0, places=8)

    def test_time_blend_keeps_line_assignment_when_no_intersection_ambiguity(self):
        ds = _DummyChainDS()
        # Force line assignment to last DS almost everywhere, independent of proximity.
        ds.transition_centers = np.array([[-100.0, 0.0], [-50.0, 0.0]], dtype=float)
        ds.transition_distances = np.array([0.25, 0.25], dtype=float)
        points = np.array([[-0.8, 0.0]], dtype=float)
        velocities, region_idx, weights, _ = evaluate_chain_regions(ds, points, mode="time_blend")

        # No mean-normal intersection ambiguity: keep line-based assignment.
        self.assertEqual(int(region_idx[0]), 2)
        self.assertAlmostEqual(float(weights[0, 2]), 1.0, places=8)
        self.assertAlmostEqual(float(velocities[0, 0]), 3.0, places=8)

    def test_time_blend_draws_nonadjacent_no_transition_boundaries(self):
        ds = _DummyChainDS()
        # Very narrow middle regime creates sampled nonadjacent contacts (DS0/DS2).
        ds.transition_centers = np.array([[0.0, 0.0], [0.01, 0.0]], dtype=float)
        ds.transition_normals = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
        ds.transition_distances = np.array([0.0, 0.0], dtype=float)
        ds.transition_times = np.array([0.0, 0.0], dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        artists = draw_chain_partition_field_2d(
            ax=ax,
            ds=ds,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            mode="time_blend",
            plot_sample=24,
            show_transition_lines=True,
            path_bandwidth=None,
        )
        # Explicit boundary lines (2) + extra nonadjacent sampled-boundary collection.
        self.assertGreaterEqual(len(artists["transition_lines"]), 3)
        plt.close(fig)

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
            path_bandwidth=0.25,
        )

        self.assertIsNotNone(artists["region_image"])
        self.assertIsNotNone(artists["stream"])
        self.assertEqual(len(artists["transition_lines"]), 2)
        for line in artists["transition_lines"]:
            x_data = np.asarray(line.get_xdata(), dtype=float)
            y_data = np.asarray(line.get_ydata(), dtype=float)
            seg_len = float(np.hypot(x_data[1] - x_data[0], y_data[1] - y_data[0]))
            self.assertAlmostEqual(seg_len, 0.5, places=7)  # 2 * path_bandwidth
        self.assertIn("corridor_mask", artists)
        self.assertGreater(int(np.sum(artists["corridor_mask"])), 0)
        self.assertLess(int(np.sum(artists["corridor_mask"])), artists["corridor_mask"].size)
        rgba = np.asarray(artists["region_image"].get_array(), dtype=float)
        self.assertEqual(rgba.shape[2], 4)
        self.assertTrue(np.all(rgba[:, :, 3][~artists["corridor_mask"]] == 0.0))
        plt.close(fig)

    def test_draw_chain_partition_field_with_none_bandwidth_fills_full_space(self):
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
            path_bandwidth=None,
        )
        mask = np.asarray(artists["corridor_mask"], dtype=bool)
        self.assertTrue(np.all(mask))
        rgba = np.asarray(artists["region_image"].get_array(), dtype=float)
        self.assertTrue(np.all(rgba[:, :, 3] > 0.0))
        plt.close(fig)

    def test_corridor_mask_uses_active_ds_regime_not_any_path_segment(self):
        ds = _DummyChainDS()
        # Force line partition to select last DS almost everywhere.
        ds.transition_centers = np.array([[-100.0, 0.0], [-50.0, 0.0]], dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        x_min, x_max, y_min, y_max = -1.0, 2.0, -1.0, 1.0
        plot_sample = 21
        artists = draw_chain_partition_field_2d(
            ax=ax,
            ds=ds,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            mode="line_regions",
            plot_sample=plot_sample,
            show_transition_lines=True,
            path_bandwidth=0.25,
        )
        mask = np.asarray(artists["corridor_mask"], dtype=bool)
        x_vec = np.linspace(x_min, x_max, plot_sample)
        y_vec = np.linspace(y_min, y_max, plot_sample)
        ix_near_zero = int(np.argmin(np.abs(x_vec - 0.0)))
        ix_near_one = int(np.argmin(np.abs(x_vec - 1.0)))
        iy_zero = int(np.argmin(np.abs(y_vec - 0.0)))
        # Near x=0: close to early path, but far from active DS-2 regime -> should be masked out.
        self.assertFalse(bool(mask[iy_zero, ix_near_zero]))
        # Near x=1: close to active DS-2 regime -> should be visible.
        self.assertTrue(bool(mask[iy_zero, ix_near_one]))
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
                        "plot_path_bandwidth": 0.35,
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
