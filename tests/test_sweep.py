import tempfile
import unittest
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sweep import SweepConfig, _runner_code, run_sweep


class SweepScriptTests(unittest.TestCase):
    def test_runner_code_includes_chain_overrides(self):
        code = _runner_code(
            dataset_path="dataset/stitching/nodes_1",
            ds_method="chain",
            seed=7,
            n_test_simulations=3,
            chain_ds_method="linear",
            chain_trigger_method="distance_ratio",
            chain_blend_ratio=0.5,
        )
        self.assertIn("self.ds_method = 'chain'", code)
        self.assertIn("self.n_test_simulations = 3", code)
        self.assertNotIn("initialize_iter_strategy", code)
        self.assertIn("main_stitch._compute_segment_DS = _noop_compute_segment_DS", code)
        self.assertIn("self.chain.ds_method = 'linear'", code)
        self.assertIn("self.chain.transition_trigger_method = 'distance_ratio'", code)
        self.assertIn("self.chain.blend_length_ratio = 0.5", code)

    def test_runner_code_does_not_patch_chain_precompute_for_non_chain(self):
        code = _runner_code(
            dataset_path="dataset/stitching/nodes_1",
            ds_method="sp_recompute_ds",
            seed=7,
            n_test_simulations=3,
        )
        self.assertNotIn("main_stitch._compute_segment_DS = _noop_compute_segment_DS", code)

    def test_runner_code_includes_graph_param_overrides(self):
        code = _runner_code(
            dataset_path="dataset/stitching/nodes_1",
            ds_method="sp_recompute_ds",
            seed=9,
            n_test_simulations=3,
            param_dist=2.0,
            param_cos=1.0,
        )
        self.assertIn("self.param_dist = 2.0", code)
        self.assertIn("self.param_cos = 1.0", code)

    def test_chain_trigger_mode_generates_cross_product(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_a"
            method_dir = dataset / "figures" / "chain"
            method_dir.mkdir(parents=True, exist_ok=True)
            def fake_runner(cmd, _timeout_s):
                code = cmd[2]
                match = re.search(r"__SWEEP_RESULTS_PATH__ = '([^']+)'", code)
                self.assertIsNotNone(match)
                results_path = Path(match.group(1))
                results_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"prediction_rmse": [1.0, 3.0]}).to_csv(results_path, index=False)

                class _P:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return _P()

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=(),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="chain_trigger",
                chain_ds_methods=("segmented", "linear"),
                chain_trigger_methods=("mean_normals", "distance_ratio"),
                chain_blend_ratios=(),
                copy_figures=False,
            )
            df = run_sweep(cfg, runner=fake_runner)

            self.assertEqual(len(df), 4)
            self.assertEqual(set(df["ds_method"].tolist()), {"chain"})
            self.assertEqual(set(df["chain_ds_method"].tolist()), {"segmented", "linear"})
            self.assertEqual(
                set(df["chain_transition_trigger_method"].tolist()),
                {"mean_normals", "distance_ratio"},
            )
            self.assertTrue(np.allclose(df["prediction_rmse_mean"].to_numpy(dtype=float), 2.0))
            self.assertTrue(np.all(df["n_test_simulations"].to_numpy(dtype=int) == 3))

    def test_chain_blend_mode_uses_fixed_chain_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_b"
            method_dir = dataset / "figures" / "chain"
            method_dir.mkdir(parents=True, exist_ok=True)
            def fake_runner(cmd, _timeout_s):
                code = cmd[2]
                match = re.search(r"__SWEEP_RESULTS_PATH__ = '([^']+)'", code)
                self.assertIsNotNone(match)
                results_path = Path(match.group(1))
                results_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"prediction_rmse": [2.0]}).to_csv(results_path, index=False)

                class _P:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _P()

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=(),
                seeds=(3,),
                output_dir=str(root / "sweep_out"),
                mode="chain_blend",
                chain_ds_methods=(),
                chain_trigger_methods=(),
                chain_blend_ratios=(0.0, 0.5, 1.0),
                chain_fixed_ds_method="linear",
                chain_fixed_trigger_method="distance_ratio",
                copy_figures=False,
            )
            df = run_sweep(cfg, runner=fake_runner)

            self.assertEqual(len(df), 3)
            self.assertEqual(set(df["chain_ds_method"].tolist()), {"linear"})
            self.assertEqual(set(df["chain_transition_trigger_method"].tolist()), {"distance_ratio"})
            self.assertEqual(
                sorted(df["chain_blend_length_ratio"].to_numpy(dtype=float).tolist()),
                [0.0, 0.5, 1.0],
            )

    def test_graph_params_mode_generates_cross_product(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_graph"
            method_dir = dataset / "figures" / "sp_recompute_ds"
            method_dir.mkdir(parents=True, exist_ok=True)

            def fake_runner(cmd, _timeout_s):
                code = cmd[2]
                match = re.search(r"__SWEEP_RESULTS_PATH__ = '([^']+)'", code)
                self.assertIsNotNone(match)
                results_path = Path(match.group(1))
                results_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"prediction_rmse": [5.0]}).to_csv(results_path, index=False)

                class _P:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return _P()

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="graph_params",
                param_dist_values=(1.0, 2.0),
                param_cos_values=(0.0, 1.0),
                copy_figures=False,
            )
            df = run_sweep(cfg, runner=fake_runner)

            self.assertEqual(len(df), 4)
            self.assertEqual(set(df["ds_method"].tolist()), {"sp_recompute_ds"})
            self.assertEqual(set(df["param_dist"].to_numpy(dtype=float).tolist()), {1.0, 2.0})
            self.assertEqual(set(df["param_cos"].to_numpy(dtype=float).tolist()), {0.0, 1.0})
            self.assertTrue(np.allclose(df["prediction_rmse_mean"].to_numpy(dtype=float), 5.0))

    def test_marks_run_failed_when_no_eval_rows_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_c"
            method_dir = dataset / "figures" / "chain"
            method_dir.mkdir(parents=True, exist_ok=True)

            def fake_runner(cmd, _timeout_s):
                code = cmd[2]
                match = re.search(r"__SWEEP_RESULTS_PATH__ = '([^']+)'", code)
                self.assertIsNotNone(match)
                results_path = Path(match.group(1))
                results_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "combination_id": [np.nan],
                        "ds_method": [np.nan],
                        "prediction_rmse": [np.nan],
                        "cosine_dissimilarity": [np.nan],
                        "dtw_distance_mean": [np.nan],
                    }
                ).to_csv(results_path, index=False)

                class _P:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return _P()

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=(),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="chain_trigger",
                chain_ds_methods=("segmented",),
                chain_trigger_methods=("mean_normals",),
                chain_blend_ratios=(),
                copy_figures=False,
            )
            df = run_sweep(cfg, runner=fake_runner)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "status"], "failed")
            self.assertEqual(df.loc[0, "failure_reason"], "no_evaluation_rows")
            self.assertEqual(int(df.loc[0, "n_eval_rows"]), 0)


if __name__ == "__main__":
    unittest.main()
