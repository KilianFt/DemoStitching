import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from main_stitch_sweep import SweepConfig, _runner_code, run_sweep


class MainStitchSweepTests(unittest.TestCase):
    def test_runner_code_embeds_requested_overrides(self):
        code = _runner_code(
            dataset_path="dataset/stitching/presentation2",
            ds_method="chain",
            seed=123,
        )
        self.assertIn("self.dataset_path = 'dataset/stitching/presentation2'", code)
        self.assertIn("self.ds_method = 'chain'", code)
        self.assertIn("self.seed = 123", code)
        self.assertIn("main_stitch.main()", code)

    def test_run_sweep_collects_csv_and_figures_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_a"
            method_dir = dataset / "figures" / "chain"
            method_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "prediction_rmse": [1.0, np.nan, 3.0],
                    "cosine_dissimilarity": [0.2, 0.4, 0.6],
                    "dtw_distance_mean": [10.0, 14.0, 16.0],
                    "distance_to_attractor_mean": [0.1, 0.3, np.nan],
                }
            ).to_csv(method_dir / "results_7.csv", index=False)
            (method_dir / "figure_a.png").write_bytes(b"fake-figure")

            calls = []

            def fake_runner(cmd, timeout_s):
                calls.append((cmd, timeout_s))

                class _P:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return _P()

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("chain",),
                seeds=(7,),
                output_dir=str(root / "sweep_out"),
                timeout_s=0.0,
                copy_figures=True,
            )
            df = run_sweep(cfg, runner=fake_runner)

            self.assertEqual(len(calls), 1)
            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(row["status"], "ok")
            self.assertEqual(int(row["n_result_rows"]), 3)
            self.assertAlmostEqual(float(row["prediction_rmse_mean"]), 2.0)
            self.assertAlmostEqual(float(row["cosine_dissimilarity_mean"]), 0.4)
            self.assertTrue(Path(row["log_path"]).exists())
            self.assertTrue(Path(row["results_csv_copy"]).exists())
            self.assertTrue(Path(row["figures_dir_copy"]).exists())
            self.assertTrue((Path(row["figures_dir_copy"]) / "figure_a.png").exists())
            self.assertTrue((root / "sweep_out" / "sweep_results.csv").exists())

    def test_run_sweep_records_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_b"
            dataset.mkdir(parents=True, exist_ok=True)

            def fake_runner(_cmd, _timeout_s):
                class _P:
                    returncode = 9
                    stdout = ""
                    stderr = "failed"

                return _P()

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds",),
                seeds=(11,),
                output_dir=str(root / "sweep_out"),
                timeout_s=0.0,
                copy_figures=True,
            )
            df = run_sweep(cfg, runner=fake_runner)

            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(int(row["return_code"]), 9)
            self.assertEqual(int(row["n_result_rows"]), 0)
            self.assertEqual(row["results_csv_source"], "")
            self.assertEqual(row["results_csv_copy"], "")
            self.assertEqual(row["figures_dir_copy"], "")
            self.assertTrue(Path(row["log_path"]).exists())


if __name__ == "__main__":
    unittest.main()
