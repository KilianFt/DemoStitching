import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.util.k3d_dataset import build_k3d_dataset


class K3DDatasetBuilderTests(unittest.TestCase):
    def test_builds_tilted_k3d_dataset_with_expected_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = build_k3d_dataset(
                output_dir=tmpdir,
                n_demo_sets=3,
                n_demos_per_set=4,
                n_points=90,
                seed=5,
                overwrite=True,
            )

            root = Path(tmpdir)
            demo_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("demonstration_")])
            self.assertEqual(len(demo_dirs), 3)
            self.assertEqual(metadata["n_demo_sets"], 3)
            self.assertEqual(metadata["n_demos_per_set"], 4)

            all_points = []
            for demo_dir in demo_dirs:
                files = sorted(demo_dir.glob("trajectory_*.json"))
                self.assertEqual(len(files), 4)
                for file_path in files:
                    with open(file_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    x = np.asarray(payload["x"], dtype=float)
                    x_dot = np.asarray(payload["x_dot"], dtype=float)
                    self.assertEqual(x.shape, (90, 3))
                    self.assertEqual(x_dot.shape, (90, 3))
                    self.assertTrue(np.all(np.isfinite(x)))
                    self.assertTrue(np.all(np.isfinite(x_dot)))

                    # Each trajectory should move in all three axes after tilt.
                    disp = x[-1] - x[0]
                    self.assertGreater(np.abs(disp[0]), 1e-2)
                    self.assertGreater(np.abs(disp[1]), 1e-2)
                    self.assertGreater(np.abs(disp[2]), 1e-2)
                    all_points.append(x)

            # Global cloud should be truly 3D (not close to rank-2 planar).
            stacked = np.vstack(all_points)
            centered = stacked - np.mean(stacked, axis=0, keepdims=True)
            rank = np.linalg.matrix_rank(centered, tol=1e-4)
            self.assertEqual(rank, 3)

            plan_file = root / "k3d_plan.json"
            self.assertTrue(plan_file.exists())
            with open(plan_file, "r", encoding="utf-8") as f:
                plan = json.load(f)
            self.assertEqual(len(plan["demonstrations"]), 3)


if __name__ == "__main__":
    unittest.main()

