import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.util.pcgmm_workspace_dataset import (
    PCGMM_3D_SIMPLE_WORKSPACE_PLAN,
    PCGMM_3D_WORKSPACE_PLAN,
    _select_trajectory_indices,
    build_pcgmm_3d_simple_workspace_dataset,
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
    def test_diverse_selection_prefers_separated_trajectories(self):
        t = np.linspace(0.0, 1.0, 120)
        base = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        trajectories = []
        for amp in (-0.35, -0.15, 0.0, 0.15, 0.35):
            traj = base.copy()
            traj[:, 1] = amp * np.sin(np.pi * t)
            trajectories.append(traj)

        rng = np.random.default_rng(7)
        selected = _select_trajectory_indices(
            task_trajectories=trajectories,
            n_select=3,
            n_points=120,
            start_anchor=np.array([0.0, 0.0, 0.0], dtype=float),
            end_anchor=np.array([1.0, 0.0, 0.0], dtype=float),
            rng=rng,
            mode="diverse",
        )

        self.assertEqual(len(selected), 3)
        self.assertIn(0, selected.tolist())
        self.assertIn(4, selected.tolist())

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

    def test_builds_simple_workspace_with_single_shared_overlap_per_demo(self):
        if not Path("dataset/pc-gmm-data").exists():
            self.skipTest("pc-gmm source files are not available in this checkout")

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = build_pcgmm_3d_simple_workspace_dataset(
                output_dir=tmpdir,
                task_data_dir="dataset/pc-gmm-data",
                n_trajectories_per_task=2,
                n_points=80,
                seed=23,
                overwrite=True,
                visualize=False,
            )

            root = Path(tmpdir)
            demo_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("demonstration_")])
            self.assertEqual(len(demo_dirs), len(PCGMM_3D_SIMPLE_WORKSPACE_PLAN))
            self.assertEqual(len(metadata["demonstrations"]), len(PCGMM_3D_SIMPLE_WORKSPACE_PLAN))
            self.assertEqual(metadata.get("selection_mode"), "diverse")

            with open(root / "pcgmm_simple_workspace_plan.json", "r", encoding="utf-8") as f:
                workspace_plan = json.load(f)
            self.assertEqual(len(workspace_plan["demonstrations"]), len(PCGMM_3D_SIMPLE_WORKSPACE_PLAN))

            starts = [_mean_endpoint(demo_dir, 0) for demo_dir in demo_dirs]
            ends = [_mean_endpoint(demo_dir, -1) for demo_dir in demo_dirs]

            # Single shared overlap node across all three demos:
            # demo0 end == demo1 start == demo2 start
            hub = ends[0]
            self.assertLess(np.linalg.norm(hub - starts[1]), 1e-8)
            self.assertLess(np.linalg.norm(hub - starts[2]), 1e-8)

            # Each demo should touch the hub at exactly one endpoint.
            for i in range(3):
                endpoint_dists = (
                    np.linalg.norm(starts[i] - hub),
                    np.linalg.norm(ends[i] - hub),
                )
                near_hub = [d < 1e-8 for d in endpoint_dists]
                self.assertEqual(sum(near_hub), 1, msg=f"demo_{i} has {sum(near_hub)} hub endpoints")

            # Non-hub endpoints remain distinct branches.
            non_hub_points = [starts[0], ends[1], ends[2]]
            self.assertGreater(np.linalg.norm(non_hub_points[0] - non_hub_points[1]), 0.1)
            self.assertGreater(np.linalg.norm(non_hub_points[0] - non_hub_points[2]), 0.1)
            self.assertGreater(np.linalg.norm(non_hub_points[1] - non_hub_points[2]), 0.1)


if __name__ == "__main__":
    unittest.main()
