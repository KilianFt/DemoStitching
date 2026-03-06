import unittest
from pathlib import Path
import sys
import warnings

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.graph_utils import GaussianGraph


class GraphUtilsShortestPathTreeTests(unittest.TestCase):
    def test_shortest_path_tree_handles_zero_gaussian_eval_without_runtime_warning(self):
        gg = GaussianGraph(param_dist=1.0, param_cos=1.0, bhattacharyya_threshold=0.0)
        gg.add_gaussians(
            {
                (0, 0): {
                    "mu": np.array([0.0, 0.0], dtype=float),
                    "sigma": 1e-6 * np.eye(2, dtype=float),
                    "direction": np.array([1.0, 0.0], dtype=float),
                    "prior": 1.0,
                }
            },
            reverse_gaussians=False,
        )

        target = np.array([1e6, 1e6], dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nodes = gg.shortest_path_tree(target_state=target)

        self.assertIsInstance(nodes, list)
        self.assertIn((0, 0), nodes)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime_warnings), 0)


if __name__ == "__main__":
    unittest.main()
