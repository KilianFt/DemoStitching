import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.util.load_tools import load_demonstration_set


def _write_traj(file_path: Path, value: float):
    payload = {
        "x": [[value, value], [value + 1.0, value + 1.0]],
        "x_dot": [[1.0, 1.0], [0.0, 0.0]],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


class LoadToolsOrderTests(unittest.TestCase):
    def test_numeric_suffix_sorting_for_demo_and_trajectory_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            demo2 = root / "demonstration_2"
            demo10 = root / "demonstration_10"
            demo2.mkdir()
            demo10.mkdir()

            # Intentionally write out of order to test sorting.
            _write_traj(demo2 / "trajectory_1.json", value=21.0)
            _write_traj(demo2 / "trajectory_0.json", value=20.0)
            _write_traj(demo10 / "trajectory_1.json", value=101.0)
            _write_traj(demo10 / "trajectory_0.json", value=100.0)

            demos = load_demonstration_set(str(root))
            self.assertEqual(len(demos), 2)

            # demonstration_2 should come before demonstration_10
            self.assertTrue(np.allclose(demos[0].trajectories[0].x[0], np.array([20.0, 20.0])))
            self.assertTrue(np.allclose(demos[0].trajectories[1].x[0], np.array([21.0, 21.0])))
            self.assertTrue(np.allclose(demos[1].trajectories[0].x[0], np.array([100.0, 100.0])))
            self.assertTrue(np.allclose(demos[1].trajectories[1].x[0], np.array([101.0, 101.0])))


if __name__ == "__main__":
    unittest.main()

