import unittest

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


if __name__ == "__main__":
    unittest.main()
