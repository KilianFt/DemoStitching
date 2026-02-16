import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.util.robottasks_workspace_dataset import (
    build_workspace_composite_dataset,
    build_obstacle_to_bottle2shelf_side_dataset,
)


def _mean_endpoint(demo_dir: Path, endpoint_idx: int) -> np.ndarray:
    points = []
    for file_path in sorted(demo_dir.glob("trajectory_*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        points.append(np.asarray(payload["x"][endpoint_idx], dtype=float))
    return np.mean(np.vstack(points), axis=0)


def _mean_trajectory_xy(demo_dir: Path) -> np.ndarray:
    trajectories = []
    for file_path in sorted(demo_dir.glob("trajectory_*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        trajectories.append(np.asarray(payload["x"], dtype=float)[:, :2])
    return np.mean(np.stack(trajectories), axis=0)


def _trajectory_overlap_ratio(
    obstacle_xy: np.ndarray,
    bottle_xy: np.ndarray,
    distance_threshold: float = 2.5,
) -> float:
    obstacle_trimmed = obstacle_xy[: int(0.8 * obstacle_xy.shape[0])]
    bottle_trimmed = bottle_xy[int(0.2 * bottle_xy.shape[0]) :]
    distances = np.linalg.norm(
        obstacle_trimmed[:, None, :] - bottle_trimmed[None, :, :],
        axis=2,
    )
    obstacle_overlap = np.mean(np.min(distances, axis=1) < distance_threshold)
    bottle_overlap = np.mean(np.min(distances, axis=0) < distance_threshold)
    return 0.5 * float(obstacle_overlap + bottle_overlap)


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

    def test_builds_obstacle_to_bottle2shelf_side_dataset(self):
        if not Path("dataset/robottasks/pos_ori").exists():
            self.skipTest("robot task source files are not available in this checkout")

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = build_obstacle_to_bottle2shelf_side_dataset(
                output_dir=tmpdir,
                task_data_dir="dataset/robottasks/pos_ori",
                n_trajectories_per_task=2,
                n_points=80,
                seed=5,
                overwrite=True,
            )

            root = Path(tmpdir)
            demo_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("demonstration_")])
            self.assertEqual(len(demo_dirs), 2)
            self.assertEqual(len(metadata["demonstrations"]), 2)

            obstacle_end = _mean_endpoint(demo_dirs[0], -1)
            bottle_start = _mean_endpoint(demo_dirs[1], 0)
            self.assertLess(np.linalg.norm(obstacle_end - bottle_start), 1e-8)

            self.assertAlmostEqual(float(bottle_start[0]), 37.0, delta=0.25)
            self.assertAlmostEqual(float(bottle_start[1]), -25.0, delta=0.25)

            obstacle_xy = _mean_trajectory_xy(demo_dirs[0])
            bottle_xy = _mean_trajectory_xy(demo_dirs[1])
            overlap_ratio = _trajectory_overlap_ratio(obstacle_xy, bottle_xy)
            self.assertLess(overlap_ratio, 0.15)


if __name__ == "__main__":
    unittest.main()
