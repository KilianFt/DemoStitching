import unittest
from types import SimpleNamespace

import numpy as np

from src.util.load_tools import compute_plot_extent_from_demo_set


class LoadToolsPlotExtentTests(unittest.TestCase):
    def test_plot_extent_2d_keeps_legacy_padding(self):
        demo_set = [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(
                        x=np.array([[0.0, 0.0], [2.0, 1.0]], dtype=float),
                        x_dot=np.zeros((2, 2), dtype=float),
                    )
                ]
            )
        ]
        extent = compute_plot_extent_from_demo_set(
            demo_set=demo_set,
            state_dim=2,
            padding_ratio=0.1,
            padding_abs=0.2,
        )
        expected = (-0.2, 2.2, -0.7, 1.7)
        np.testing.assert_allclose(np.asarray(extent, dtype=float), np.asarray(expected, dtype=float), atol=1e-10)

    def test_plot_extent_3d_uses_tighter_padding(self):
        demo_set = [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(
                        x=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float),
                        x_dot=np.zeros((2, 3), dtype=float),
                    )
                ]
            )
        ]
        extent = compute_plot_extent_from_demo_set(
            demo_set=demo_set,
            state_dim=3,
            padding_ratio=0.1,
            padding_abs=0.2,
        )
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        width = x_max - x_min

        # Legacy width (without 3D compaction) for this setup:
        # max_span=3, margin=max(0.2, 0.3)=0.3 => width=2*(1.5+0.3)=3.6
        self.assertLess(width, 3.6)
        self.assertGreater(width, 3.0)
        self.assertAlmostEqual(width, y_max - y_min, places=10)
        self.assertAlmostEqual(width, z_max - z_min, places=10)


if __name__ == "__main__":
    unittest.main()
