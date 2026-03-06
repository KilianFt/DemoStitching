import tempfile
import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.util.plot_tools import plot_demonstration_set


class _Traj:
    def __init__(self, x):
        self.x = x


class _Demo:
    def __init__(self, trajectories):
        self.trajectories = trajectories


class PlotToolsFigureCloseTests(unittest.TestCase):
    def _build_demo_set(self):
        t = np.linspace(0.0, 1.0, 25)
        x = np.stack([t, t**2], axis=1)
        return [_Demo([_Traj(x)])]

    def test_plot_demonstration_set_closes_internal_figure_on_save(self):
        plt.close("all")
        demo_set = self._build_demo_set()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(
                plot_extent=(0, 1, 0, 1),
                dataset_path=tmpdir,
                ds_method="unit",
            )

            before = len(plt.get_fignums())
            _ = plot_demonstration_set(demo_set, config, save_as="demo_close_test", hide_axis=True)
            after = len(plt.get_fignums())

            self.assertEqual(before, after)

    def test_plot_demonstration_set_does_not_close_external_axis(self):
        plt.close("all")
        demo_set = self._build_demo_set()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(
                plot_extent=(0, 1, 0, 1),
                dataset_path=tmpdir,
                ds_method="unit",
            )

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            _ = plot_demonstration_set(demo_set, config, ax=ax, save_as="demo_external_ax", hide_axis=True)
            self.assertIn(fig.number, plt.get_fignums())
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
