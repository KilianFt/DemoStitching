import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.util.plot_tools import (
    _sample_cmap_colors,
    plot_demonstration_set,
    plot_ds_set_gaussians,
    plot_gaussian_graph,
    plot_gg_solution,
    plot_clean_3d_composite,
    plot_ds,
    plot_composite,
)


class _DummyDamm:
    def __init__(self):
        self.Mu = np.array([[0.0, 0.0, 0.0]], dtype=float)
        self.Sigma = np.array([np.eye(3)], dtype=float)
        self.Prior = np.array([1.0], dtype=float)
        self.gaussian_lists = [{"mu": self.Mu[0], "sigma": self.Sigma[0], "prior": 1.0}]

    def compute_gamma(self, x):
        x = np.asarray(x, dtype=float)
        return np.ones((1, x.shape[0]), dtype=float)


class _DummyGG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node(("a", 0), mean=np.array([0.0, 0.0, 0.0]), covariance=np.eye(3), direction=np.array([1.0, 0.0, 0.0]), prior=0.5)
        self.graph.add_node(("b", 0), mean=np.array([1.0, 1.0, 1.0]), covariance=np.eye(3), direction=np.array([1.0, 0.0, 0.0]), prior=0.5)
        self.graph.add_edge(("a", 0), ("b", 0), weight=1.0)

    def get_gaussian(self, node):
        data = self.graph.nodes[node]
        return data["mean"], data["covariance"], data["direction"], data["prior"]


class _DummyChainDS:
    def __init__(self):
        self.x = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.2, 0.1],
                [0.8, 0.5, 0.3],
                [1.1, 0.8, 0.4],
            ],
            dtype=float,
        )
        self.x_att = np.array([1.2, 0.9, 0.5], dtype=float)
        self.n_systems = 2
        self.node_sources = np.array([[0.0, 0.0, 0.0], [0.6, 0.35, 0.2]], dtype=float)
        self.node_targets = np.array([[0.6, 0.35, 0.2], [1.2, 0.9, 0.5]], dtype=float)
        self.state_sequence = np.array(
            [[0.0, 0.0, 0.0], [0.6, 0.35, 0.2], [1.2, 0.9, 0.5]],
            dtype=float,
        )
        self.transition_centers = np.array([[0.6, 0.35, 0.2]], dtype=float)
        self.transition_ratio_start_nodes = np.array([[0.45, 0.25, 0.15]], dtype=float)
        self.transition_ratio_nodes = np.array([[0.75, 0.45, 0.28]], dtype=float)
        self.transition_times = np.array([0.2], dtype=float)
        self.transition_distances = np.array([0.15], dtype=float)

    def _velocity_for_index(self, x, idx):
        x = np.asarray(x, dtype=float).reshape(-1)
        gain = 1.0 if int(idx) == 0 else 0.7
        return gain * (self.x_att - x)


class PlotTools3DTests(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            plot_extent=(-1.0, 2.0, -1.0, 2.0, -1.0, 2.0),
            dataset_path="dataset/stitching/testing",
            ds_method="sp_recompute_ds",
            gaussian_direction_method="mean_velocity",
        )

    def _chain_config(self):
        return SimpleNamespace(
            plot_extent=(-1.0, 2.0, -1.0, 2.0, -1.0, 2.0),
            dataset_path="dataset/stitching/testing",
            ds_method="chain",
            gaussian_direction_method="mean_velocity",
            chain=SimpleNamespace(
                plot_mode="line_regions",
                plot_grid_resolution=30,
                plot_path_bandwidth=0.9,
                plot_show_transition_lines=True,
                plot_region_alpha=0.26,
            ),
        )

    def _demo_set(self):
        return [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(
                        x=np.array(
                            [
                                [0.0, 0.0, 0.0],
                                [0.5, 0.2, 0.1],
                                [1.0, 0.8, 0.4],
                            ],
                            dtype=float,
                        ),
                        x_dot=np.array(
                            [
                                [1.0, 0.4, 0.2],
                                [0.8, 0.5, 0.3],
                                [0.4, 0.2, 0.1],
                            ],
                            dtype=float,
                        ),
                    )
                ]
            )
        ]

    def _dummy_ds(self):
        return SimpleNamespace(
            x=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.2, 0.1],
                    [1.0, 0.5, 0.3],
                ],
                dtype=float,
            ),
            x_dot=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.1, 0.1],
                    [0.8, 0.2, 0.1],
                ],
                dtype=float,
            ),
            x_att=np.array([1.2, 0.6, 0.4], dtype=float),
            A=np.array([np.eye(3)], dtype=float),
            damm=_DummyDamm(),
        )

    def test_plot_demonstration_set_uses_3d_axis_for_3d_data(self):
        demo_set = self._demo_set()
        ax = plot_demonstration_set(demo_set, self._config(), save_as=None, hide_axis=True)
        self.assertEqual(ax.name, "3d")
        plt.close(ax.figure)

    def test_plot_ds_set_gaussians_uses_3d_axis_for_3d_data(self):
        ds = self._dummy_ds()
        ax = plot_ds_set_gaussians(
            [ds],
            self._config(),
            initial=np.array([0.0, 0.0, 0.0], dtype=float),
            attractor=np.array([1.2, 0.6, 0.4], dtype=float),
            include_trajectory=True,
            save_as=None,
            hide_axis=True,
        )
        self.assertEqual(ax.name, "3d")
        plt.close(ax.figure)

    def test_plot_graph_and_solution_use_3d_axis_for_3d_graph(self):
        cfg = self._config()
        gg = _DummyGG()
        ax_graph = plot_gaussian_graph(gg, cfg, save_as=None, hide_axis=True)
        self.assertEqual(ax_graph.name, "3d")
        plt.close(ax_graph.figure)

        ax_solution = plot_gg_solution(
            gg,
            solution_nodes=[("a", 0), ("b", 0)],
            initial=np.array([0.0, 0.0, 0.0], dtype=float),
            attractor=np.array([1.4, 1.1, 1.2], dtype=float),
            config=cfg,
            save_as=None,
            hide_axis=True,
        )
        self.assertEqual(ax_solution.name, "3d")
        plt.close(ax_solution.figure)

    def test_plot_ds_uses_3d_axis_for_3d_lpvds(self):
        ds = self._dummy_ds()
        sim = [np.array([[0.0, 0.0, 0.0], [0.7, 0.3, 0.2], [1.1, 0.55, 0.35]], dtype=float)]
        ax = plot_ds(
            ds,
            sim,
            initial=np.array([0.0, 0.0, 0.0], dtype=float),
            attractor=np.array([1.2, 0.6, 0.4], dtype=float),
            config=self._config(),
            save_as=None,
            hide_axis=True,
        )
        self.assertEqual(ax.name, "3d")
        plt.close(ax.figure)

    def test_3d_save_uses_tight_bbox(self):
        demo_set = self._demo_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(
                plot_extent=(-1.0, 2.0, -1.0, 2.0, -1.0, 2.0),
                dataset_path=str(tmpdir),
                ds_method="sp_recompute_ds",
                gaussian_direction_method="mean_velocity",
            )
            with patch("matplotlib.figure.Figure.savefig") as mock_savefig:
                ax = plot_demonstration_set(demo_set, cfg, save_as="tmp_plot", hide_axis=True)
            self.assertGreaterEqual(mock_savefig.call_count, 1)
            _, kwargs = mock_savefig.call_args
            self.assertEqual(kwargs.get("bbox_inches"), "tight")
            self.assertAlmostEqual(float(kwargs.get("pad_inches")), 0.02)
            plt.close(ax.figure)

    def test_2d_save_keeps_default_bbox_behavior(self):
        demo_set = [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(
                        x=np.array([[0.0, 0.0], [1.0, 0.8]], dtype=float),
                        x_dot=np.array([[1.0, 0.8], [1.0, 0.8]], dtype=float),
                    )
                ]
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(
                plot_extent=(-1.0, 2.0, -1.0, 2.0),
                dataset_path=str(tmpdir),
                ds_method="sp_recompute_ds",
                gaussian_direction_method="mean_velocity",
            )
            with patch("matplotlib.figure.Figure.savefig") as mock_savefig:
                ax = plot_demonstration_set(demo_set, cfg, save_as="tmp_plot", hide_axis=True)
            self.assertGreaterEqual(mock_savefig.call_count, 1)
            _, kwargs = mock_savefig.call_args
            self.assertNotIn("bbox_inches", kwargs)
            self.assertNotIn("pad_inches", kwargs)
            plt.close(ax.figure)

    def test_plot_composite_projects_3d_result_into_three_2d_panels(self):
        ds = self._dummy_ds()
        sim = [np.array([[0.0, 0.0, 0.0], [0.7, 0.3, 0.2], [1.1, 0.55, 0.35]], dtype=float)]
        ax = plot_composite(
            _DummyGG(),
            solution_nodes=[("a", 0), ("b", 0)],
            demo_set=self._demo_set(),
            lpvds=ds,
            x_test_list=sim,
            initial=np.array([0.0, 0.0, 0.0], dtype=float),
            attractor=np.array([1.2, 0.6, 0.4], dtype=float),
            config=self._config(),
            save_as=None,
            hide_axis=True,
        )
        figure = ax.figure
        self.assertEqual(len(figure.axes), 3)
        self.assertListEqual([subplot.get_title() for subplot in figure.axes], ["x,y", "x,z", "y,z"])
        self.assertTrue(all(subplot.name == "rectilinear" for subplot in figure.axes))
        plt.close(figure)

    def test_plot_composite_projects_chain_ds_into_three_2d_panels(self):
        sim = [np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.1], [1.0, 0.75, 0.42]], dtype=float)]
        ax = plot_composite(
            _DummyGG(),
            solution_nodes=[("a", 0), ("b", 0)],
            demo_set=self._demo_set(),
            lpvds=_DummyChainDS(),
            x_test_list=sim,
            initial=np.array([0.0, 0.0, 0.0], dtype=float),
            attractor=np.array([1.2, 0.9, 0.5], dtype=float),
            config=self._chain_config(),
            save_as=None,
            hide_axis=True,
        )
        figure = ax.figure
        self.assertEqual(len(figure.axes), 3)
        self.assertTrue(all(subplot.name == "rectilinear" for subplot in figure.axes))
        plt.close(figure)

    def test_plot_clean_3d_composite_returns_axis_and_grid_view(self):
        ds = self._dummy_ds()
        sim = [np.array([[0.0, 0.0, 0.0], [0.7, 0.3, 0.2], [1.1, 0.55, 0.35]], dtype=float)]
        ax = plot_clean_3d_composite(
            _DummyGG(),
            solution_nodes=[("a", 0), ("b", 0)],
            demo_set=self._demo_set(),
            lpvds=ds,
            x_test_list=sim,
            initial=np.array([0.0, 0.0, 0.0], dtype=float),
            attractor=np.array([1.2, 0.6, 0.4], dtype=float),
            config=self._config(),
            save_as=None,
            hide_axis=False,
        )
        self.assertEqual(ax.name, "3d")
        self.assertEqual(ax.get_xlabel(), r'$\xi_1$')
        self.assertEqual(ax.get_ylabel(), r'$\xi_2$')
        self.assertEqual(ax.get_zlabel(), r'$\xi_3$')
        self.assertTrue(ax.xaxis.pane.fill)
        self.assertTrue(ax.yaxis.pane.fill)
        self.assertTrue(ax.zaxis.pane.fill)
        self.assertGreater(len(ax.get_xticklabels()), 0)
        self.assertGreater(len(ax.get_yticklabels()), 0)
        self.assertGreater(len(ax.get_zticklabels()), 0)
        self.assertGreaterEqual(len(ax.lines), 2)
        self.assertGreaterEqual(len(ax.collections), 5)
        plt.close(ax.figure)

    def test_plot_clean_3d_composite_includes_demo_background_for_chain_methods(self):
        sim = [np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.1], [1.0, 0.75, 0.42]], dtype=float)]
        ax = plot_clean_3d_composite(
            _DummyGG(),
            solution_nodes=[("a", 0), ("b", 0)],
            demo_set=self._demo_set(),
            lpvds=_DummyChainDS(),
            x_test_list=sim,
            initial=np.array([0.0, 0.0, 0.0], dtype=float),
            attractor=np.array([1.2, 0.9, 0.5], dtype=float),
            config=self._chain_config(),
            save_as=None,
            hide_axis=True,
        )
        expected_demo_color = _sample_cmap_colors("summer", 1, max_frac=0.7)[0]
        self.assertGreaterEqual(len(ax.lines), 3)
        self.assertTrue(np.allclose(to_rgba(ax.lines[0].get_color()), expected_demo_color))
        plt.close(ax.figure)


if __name__ == "__main__":
    unittest.main()
