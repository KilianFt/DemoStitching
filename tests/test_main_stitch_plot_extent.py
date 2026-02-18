import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from main_stitch import _compute_plot_extent_from_demo_set, _infer_state_dim_from_demo_set


class MainStitchExtentTests(unittest.TestCase):
    def test_infer_state_dim_detects_3d_trajectories(self):
        demo_set = [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(
                        x=np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=float),
                        x_dot=np.zeros((2, 3), dtype=float),
                    )
                ]
            )
        ]
        self.assertEqual(_infer_state_dim_from_demo_set(demo_set), 3)

    def test_compute_plot_extent_returns_2d_bounds_with_padding(self):
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
        extent = _compute_plot_extent_from_demo_set(demo_set, state_dim=2, padding_ratio=0.1, padding_abs=0.2)
        self.assertEqual(len(extent), 4)
        x_min, x_max, y_min, y_max = extent
        self.assertGreater(x_max - x_min, 2.0)
        self.assertGreater(y_max - y_min, 1.0)

    def test_compute_plot_extent_returns_3d_bounds_with_padding(self):
        demo_set = [
            SimpleNamespace(
                trajectories=[
                    SimpleNamespace(
                        x=np.array(
                            [
                                [0.0, 0.0, 0.0],
                                [1.0, 2.0, 3.0],
                            ],
                            dtype=float,
                        ),
                        x_dot=np.zeros((2, 3), dtype=float),
                    )
                ]
            )
        ]
        extent = _compute_plot_extent_from_demo_set(demo_set, state_dim=3, padding_ratio=0.1, padding_abs=0.2)
        self.assertEqual(len(extent), 6)
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        self.assertGreater(x_max - x_min, 1.0)
        self.assertGreater(y_max - y_min, 2.0)
        self.assertGreater(z_max - z_min, 3.0)


if __name__ == "__main__":
    unittest.main()
