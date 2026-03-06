import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from src.util.plot_tools import _sample_cmap_colors


class PlotToolsColormapTests(unittest.TestCase):
    def test_sample_cmap_colors_handles_continuous_colormap(self):
        colors = _sample_cmap_colors("summer", 4)
        self.assertIsInstance(colors, np.ndarray)
        self.assertEqual(colors.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(colors)))

    def test_sample_cmap_colors_handles_listed_colormap(self):
        colors = _sample_cmap_colors("tab10", 6)
        self.assertIsInstance(colors, np.ndarray)
        self.assertEqual(colors.shape, (6, 4))
        self.assertTrue(np.all(np.isfinite(colors)))

    def test_sample_cmap_colors_respects_fraction_window(self):
        colors = _sample_cmap_colors("summer", 5, min_frac=0.0, max_frac=0.7)
        self.assertEqual(colors.shape, (5, 4))
        np.testing.assert_allclose(colors[0], plt.get_cmap("summer")(0.0), atol=1e-12)
        np.testing.assert_allclose(colors[-1], plt.get_cmap("summer")(0.7), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
