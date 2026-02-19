import unittest
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from configs import StitchConfig


class ConfigDataclassTests(unittest.TestCase):
    def test_stitch_config_uses_independent_nested_defaults(self):
        cfg_a = StitchConfig()
        cfg_b = StitchConfig()

        self.assertIsNot(cfg_a.chain, cfg_b.chain)
        self.assertIsNot(cfg_a.damm, cfg_b.damm)

        cfg_a.chain.subsystem_edges = 7
        cfg_a.damm.rel_scale = 0.33

        self.assertNotEqual(cfg_b.chain.subsystem_edges, 7)
        self.assertNotEqual(cfg_b.damm.rel_scale, 0.33)

    def test_stitch_config_exposes_gaussian_direction_method(self):
        cfg = StitchConfig()
        self.assertIn(cfg.gaussian_direction_method, {"mean_velocity", "a_mu"})

    def test_chain_plot_grid_resolution_is_available_and_positive(self):
        cfg = StitchConfig()
        self.assertGreaterEqual(int(cfg.chain.plot_grid_resolution), 8)


if __name__ == "__main__":
    unittest.main()
