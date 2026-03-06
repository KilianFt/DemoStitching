import unittest
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from main_stitch import _extract_gaussian_node_indices


class MainStitchNodeIdTests(unittest.TestCase):
    def test_extract_gaussian_node_indices_accepts_standard_and_reversed_nodes(self):
        self.assertEqual(_extract_gaussian_node_indices((2, 7)), (2, 7))
        self.assertEqual(_extract_gaussian_node_indices((2, 7, "reversed")), (2, 7))
        self.assertEqual(_extract_gaussian_node_indices([3, 4, "reversed"]), (3, 4))
        self.assertEqual(
            _extract_gaussian_node_indices(np.array([5, 6, "reversed"], dtype=object)),
            (5, 6),
        )

    def test_extract_gaussian_node_indices_rejects_invalid_ids(self):
        with self.assertRaises(ValueError):
            _extract_gaussian_node_indices((3,))
        with self.assertRaises(TypeError):
            _extract_gaussian_node_indices("not-a-node-id")


if __name__ == "__main__":
    unittest.main()
