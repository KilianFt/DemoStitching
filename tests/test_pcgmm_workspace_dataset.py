import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.util.pcgmm_workspace_dataset import (
    PCGMM_3D_WORKSPACE_PLAN,
    build_pcgmm_3d_workspace_dataset,
)


def _mean_endpoint(demo_dir: Path, endpoint_idx: int) -> np.ndarray:
    points = []
    for file_path in sorted(demo_dir.glob("trajectory_*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        points.append(np.asarray(payload["x"][endpoint_idx], dtype=float))
    return np.mean(np.vstack(points), axis=0)


class PCGMMWorkspaceDatasetBuilderTests(unittest.TestCase):
    def test_builds_connected_pcgmm_workspace(self):
        if not Path("dataset/pc-gmm-data").exists():
            self.skipTest("pc-gmm source files are not available in this checkout")

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = build_pcgmm_3d_workspace_dataset(
                output_dir=tmpdir,
                task_data_dir="dataset/pc-gmm-data",
                n_trajectories_per_task=2,
                n_points=90,
                seed=17,
                overwrite=True,
                visualize=False,
            )

            root = Path(tmpdir)
            demo_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("demonstration_")])
            self.assertEqual(len(demo_dirs), len(PCGMM_3D_WORKSPACE_PLAN))
            self.assertEqual(len(metadata["demonstrations"]), len(PCGMM_3D_WORKSPACE_PLAN))

            with open(root / "pcgmm_workspace_plan.json", "r", encoding="utf-8") as f:
                workspace_plan = json.load(f)
            self.assertEqual(len(workspace_plan["demonstrations"]), len(PCGMM_3D_WORKSPACE_PLAN))

            # Shared central hub:
            # demo0 end == demo1 end == demo2 start == demo3 start
            hub_from_cshape_top = _mean_endpoint(demo_dirs[0], -1)
            hub_from_cshape_bottom = _mean_endpoint(demo_dirs[1], -1)
            hub_to_v1 = _mean_endpoint(demo_dirs[2], 0)
            hub_to_v2 = _mean_endpoint(demo_dirs[3], 0)
            self.assertLess(np.linalg.norm(hub_from_cshape_top - hub_from_cshape_bottom), 1e-8)
            self.assertLess(np.linalg.norm(hub_from_cshape_top - hub_to_v1), 1e-8)
            self.assertLess(np.linalg.norm(hub_from_cshape_top - hub_to_v2), 1e-8)

            # Upper-right overlap node:
            # demo2 end == demo4 start == demo5 start
            upper_end_v1 = _mean_endpoint(demo_dirs[2], -1)
            upper_start_v3 = _mean_endpoint(demo_dirs[4], 0)
            upper_start_cube_pick = _mean_endpoint(demo_dirs[5], 0)
            self.assertLess(np.linalg.norm(upper_end_v1 - upper_start_v3), 1e-8)
            self.assertLess(np.linalg.norm(upper_end_v1 - upper_start_cube_pick), 1e-8)

            # Lower-right overlap node:
            # demo3 end == demo4 end == demo6 start
            lower_end_v2 = _mean_endpoint(demo_dirs[3], -1)
            lower_end_v3 = _mean_endpoint(demo_dirs[4], -1)
            lower_start_pick_box = _mean_endpoint(demo_dirs[6], 0)
            self.assertLess(np.linalg.norm(lower_end_v2 - lower_end_v3), 1e-8)
            self.assertLess(np.linalg.norm(lower_end_v2 - lower_start_pick_box), 1e-8)

            sample_file = demo_dirs[0] / "trajectory_0.json"
            with open(sample_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            x = np.asarray(payload["x"], dtype=float)
            x_dot = np.asarray(payload["x_dot"], dtype=float)
            self.assertEqual(x.shape, (90, 3))
            self.assertEqual(x_dot.shape, (90, 3))
            self.assertTrue(np.all(np.isfinite(x)))
            self.assertTrue(np.all(np.isfinite(x_dot)))

    def test_build_writes_visualization_files(self):
        if not Path("dataset/pc-gmm-data").exists():
            self.skipTest("pc-gmm source files are not available in this checkout")

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = build_pcgmm_3d_workspace_dataset(
                output_dir=tmpdir,
                task_data_dir="dataset/pc-gmm-data",
                n_trajectories_per_task=1,
                n_points=60,
                seed=5,
                overwrite=True,
                visualize=True,
            )
            viz = metadata.get("visualizations", {})
            self.assertIn("individual_tasks_3d", viz)
            self.assertIn("combined_workspace_3d", viz)
            self.assertTrue(Path(viz["individual_tasks_3d"]).exists())
            self.assertTrue(Path(viz["combined_workspace_3d"]).exists())


if __name__ == "__main__":
    unittest.main()
