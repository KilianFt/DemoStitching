import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.util.plot_tools import (
    plot_demonstration_set,
    plot_ds_set_gaussians,
    plot_gaussian_graph,
    plot_gg_solution,
    plot_ds,
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


class PlotTools3DTests(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            plot_extent=(-1.0, 2.0, -1.0, 2.0, -1.0, 2.0),
            dataset_path="dataset/stitching/testing",
            ds_method="sp_recompute_ds",
            gaussian_direction_method="mean_velocity",
        )

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
        demo_set = [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(
                        x=np.array([[0.0, 0.0, 0.0], [1.0, 0.8, 0.6]], dtype=float),
                        x_dot=np.array([[1.0, 0.8, 0.6], [1.0, 0.8, 0.6]], dtype=float),
                    )
                ]
            )
        ]
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


if __name__ == "__main__":
    unittest.main()
