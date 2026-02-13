import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.util.robottasks_workspace_dataset import build_workspace_composite_dataset


def _mean_endpoint(demo_dir: Path, endpoint_idx: int) -> np.ndarray:
    points = []
    for file_path in sorted(demo_dir.glob("trajectory_*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        points.append(np.asarray(payload["x"][endpoint_idx], dtype=float))
    return np.mean(np.vstack(points), axis=0)


class WorkspaceDatasetBuilderTests(unittest.TestCase):
    def test_builds_connected_workspace_demonstrations(self):
        if not Path("dataset/robottasks/pos_ori").exists():
            self.skipTest("robot task source files are not available in this checkout")

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = build_workspace_composite_dataset(
                output_dir=tmpdir,
                task_data_dir="dataset/robottasks/pos_ori",
                n_trajectories_per_task=2,
                n_points=60,
                include_openbox=True,
                seed=3,
                overwrite=True,
            )

            root = Path(tmpdir)
            demo_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("demonstration_")])
            self.assertEqual(len(demo_dirs), 4)
            self.assertEqual(len(metadata["demonstrations"]), 4)

            with open(root / "workspace_plan.json", "r", encoding="utf-8") as f:
                workspace_plan = json.load(f)
            self.assertEqual(len(workspace_plan["demonstrations"]), 4)

            # New topology constraints:
            # obstacle is the center; pouring starts at obstacle start;
            # pan2stove starts at obstacle end; openbox branches from obstacle midpoint.
            obstacle_start = _mean_endpoint(demo_dirs[0], 0)
            obstacle_end = _mean_endpoint(demo_dirs[0], -1)
            pouring_start = _mean_endpoint(demo_dirs[1], 0)
            pan_start = _mean_endpoint(demo_dirs[2], 0)
            openbox_start = _mean_endpoint(demo_dirs[3], 0)
            obstacle_mid = 0.5 * (obstacle_start + obstacle_end)

            self.assertLess(np.linalg.norm(obstacle_start - pouring_start), 1e-8)
            self.assertLess(np.linalg.norm(obstacle_end - pan_start), 1e-8)
            self.assertLess(np.linalg.norm(obstacle_mid - openbox_start), 1e-8)

            # Structural sanity checks.
            sample_file = demo_dirs[0] / "trajectory_0.json"
            with open(sample_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            x = np.asarray(payload["x"], dtype=float)
            x_dot = np.asarray(payload["x_dot"], dtype=float)
            self.assertEqual(x.shape, (60, 3))
            self.assertEqual(x_dot.shape, (60, 3))
            self.assertTrue(np.all(np.isfinite(x)))
            self.assertTrue(np.all(np.isfinite(x_dot)))


if __name__ == "__main__":
    unittest.main()
