import unittest
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs import StitchConfig
from src.stitching.main_stitch_helpers import (
    call_with_timeout,
    extract_chain_segments_from_path_nodes,
    resolve_save_figure_indices,
)


class MainStitchHelperTests(unittest.TestCase):
    def test_call_with_timeout_passthrough_when_disabled(self):
        self.assertEqual(call_with_timeout(0.0, "disabled", lambda x: x + 1, 2), 3)
        self.assertEqual(call_with_timeout(float("nan"), "disabled", lambda: 7), 7)

    def test_extract_chain_segments_matches_triplet_windows(self):
        nodes = [(0, 0), (0, 1), (0, 2), (0, 3)]
        segments = extract_chain_segments_from_path_nodes(nodes)
        self.assertEqual(segments, [((0, 0), (0, 1), (0, 2)), ((0, 1), (0, 2), (0, 3))])

    def test_resolve_save_figure_indices_uses_dataset_defaults(self):
        cfg = StitchConfig()
        cfg.save_fig = True
        cfg.dataset_path = "dataset/stitching/X"
        self.assertEqual(resolve_save_figure_indices(cfg), {24, 29})


if __name__ == "__main__":
    unittest.main()
