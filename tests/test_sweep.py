import tempfile
import unittest
from pathlib import Path
import sys
import contextlib
import io
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep import SweepConfig, _default_run_main, run_sweep


class SweepScriptTests(unittest.TestCase):
    def _ok_runner(self, captures: list[dict]):
        def _runner(stitch_cfg, results_path):
            captures.append(
                {
                    "dataset_path": stitch_cfg.dataset_path,
                    "ds_method": stitch_cfg.ds_method,
                    "seed": stitch_cfg.seed,
                    "n_test_simulations": stitch_cfg.n_test_simulations,
                    "save_fig": stitch_cfg.save_fig,
                    "chain_precompute_segments": getattr(stitch_cfg, "chain_precompute_segments", None),
                    "chain_ds_method": stitch_cfg.chain.ds_method,
                    "chain_trigger_method": stitch_cfg.chain.transition_trigger_method,
                    "chain_blend_ratio": stitch_cfg.chain.blend_length_ratio,
                    "param_dist": stitch_cfg.param_dist,
                    "param_cos": stitch_cfg.param_cos,
                    "rel_scale": stitch_cfg.damm.rel_scale,
                    "results_path": results_path,
                }
            )
            rows = [
                {
                    "ds_compute_time": 0.1,
                    "gg_compute_time": 0.2,
                    "total_compute_time": 0.3,
                },
                {
                    "combination_id": 0,
                    "ds_method": stitch_cfg.ds_method,
                    "prediction_rmse": 2.0,
                    "cosine_dissimilarity": 3.0,
                    "dtw_distance_mean": 4.0,
                    "distance_to_attractor_mean": 5.0,
                    "ds_compute_time": 0.6,
                    "gg_compute_time": 0.7,
                    "total_compute_time": 0.8,
                },
            ]
            pd.DataFrame(rows).to_csv(results_path, index=False)
            return rows

        return _runner

    def test_chain_trigger_mode_generates_cross_product(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_a"
            dataset.mkdir(parents=True, exist_ok=True)
            captures: list[dict] = []

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=(),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="chain_trigger",
                chain_ds_methods=("segmented", "linear"),
                chain_trigger_methods=("mean_normals", "distance_ratio"),
                chain_blend_ratios=(),
            )
            df = run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(len(df), 4)
            self.assertEqual(set(df["ds_method"].tolist()), {"chain"})
            self.assertEqual(set(df["chain_ds_method"].tolist()), {"segmented", "linear"})
            self.assertEqual(
                set(df["chain_transition_trigger_method"].tolist()),
                {"mean_normals", "distance_ratio"},
            )
            self.assertTrue(np.allclose(df["prediction_rmse_mean"].to_numpy(dtype=float), 2.0))
            self.assertTrue(np.all(df["n_test_simulations"].to_numpy(dtype=int) == 3))
            self.assertTrue(np.all(df["status"].to_numpy() == "ok"))

            self.assertEqual(len(captures), 4)
            self.assertTrue(all(c["save_fig"] is False for c in captures))
            self.assertTrue(all(c["chain_precompute_segments"] is False for c in captures))
            self.assertTrue((root / "sweep_out" / "sweep_results.csv").exists())

    def test_chain_blend_mode_uses_fixed_chain_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_b"
            dataset.mkdir(parents=True, exist_ok=True)
            captures: list[dict] = []

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
            )
            df = run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(len(df), 3)
            self.assertEqual(set(df["chain_ds_method"].tolist()), {"linear"})
            self.assertEqual(set(df["chain_transition_trigger_method"].tolist()), {"distance_ratio"})
            self.assertEqual(
                sorted(df["chain_blend_length_ratio"].to_numpy(dtype=float).tolist()),
                [0.0, 0.5, 1.0],
            )

            self.assertEqual(len(captures), 3)
            self.assertEqual(set(c["chain_ds_method"] for c in captures), {"linear"})
            self.assertEqual(set(c["chain_trigger_method"] for c in captures), {"distance_ratio"})

    def test_graph_params_mode_generates_cross_product(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_graph"
            dataset.mkdir(parents=True, exist_ok=True)
            captures: list[dict] = []

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="graph_params",
                param_dist_values=(1.0, 2.0),
                param_cos_values=(0.0, 1.0),
            )
            df = run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(len(df), 4)
            self.assertEqual(set(df["ds_method"].tolist()), {"sp_recompute_ds"})
            self.assertEqual(set(df["param_dist"].to_numpy(dtype=float).tolist()), {1.0, 2.0})
            self.assertEqual(set(df["param_cos"].to_numpy(dtype=float).tolist()), {0.0, 1.0})
            self.assertTrue(np.allclose(df["prediction_rmse_mean"].to_numpy(dtype=float), 2.0))

            expected_pairs = {(1.0, 0.0), (1.0, 1.0), (2.0, 0.0), (2.0, 1.0)}
            captured_pairs = {(float(c["param_dist"]), float(c["param_cos"])) for c in captures}
            self.assertEqual(captured_pairs, expected_pairs)

    def test_rel_scale_mode_generates_cross_product(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_rel"
            dataset.mkdir(parents=True, exist_ok=True)
            captures: list[dict] = []

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="rel_scale",
                rel_scale_values=(0.1, 0.5, 1.0),
            )
            df = run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(len(df), 3)
            self.assertEqual(set(df["ds_method"].tolist()), {"sp_recompute_ds"})
            self.assertEqual(set(df["rel_scale"].to_numpy(dtype=float).tolist()), {0.1, 0.5, 1.0})
            self.assertTrue(np.allclose(df["prediction_rmse_mean"].to_numpy(dtype=float), 2.0))
            self.assertEqual(set(float(c["rel_scale"]) for c in captures), {0.1, 0.5, 1.0})

    def test_marks_run_failed_when_no_eval_rows_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_c"
            dataset.mkdir(parents=True, exist_ok=True)

            def fake_runner(stitch_cfg, results_path):
                del stitch_cfg
                rows = [{"ds_compute_time": 0.1, "gg_compute_time": 0.2}]
                pd.DataFrame(rows).to_csv(results_path, index=False)
                return rows

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=(),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="chain_trigger",
                chain_ds_methods=("segmented",),
                chain_trigger_methods=("mean_normals",),
                chain_blend_ratios=(),
            )
            df = run_sweep(cfg, run_main_fn=fake_runner)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "status"], "failed")
            self.assertEqual(df.loc[0, "failure_reason"], "no_evaluation_rows")
            self.assertEqual(int(df.loc[0, "n_eval_rows"]), 0)

    def test_suppresses_runner_output_but_keeps_progress_counter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_d"
            dataset.mkdir(parents=True, exist_ok=True)

            def noisy_runner(stitch_cfg, results_path):
                del stitch_cfg
                print("NOISY_STDOUT")
                print("NOISY_STDERR", file=sys.stderr)
                rows = [
                    {"ds_compute_time": 0.1, "gg_compute_time": 0.2, "total_compute_time": 0.3},
                    {
                        "combination_id": 0,
                        "ds_method": "sp_recompute_ds",
                        "prediction_rmse": 1.0,
                        "cosine_dissimilarity": 2.0,
                        "dtw_distance_mean": 3.0,
                    },
                ]
                pd.DataFrame(rows).to_csv(results_path, index=False)
                return rows

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
            )
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                df = run_sweep(cfg, run_main_fn=noisy_runner)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "status"], "ok")
            stdout_text = out_buf.getvalue()
            stderr_text = err_buf.getvalue()
            self.assertIn("[1/1] Running", stdout_text)
            self.assertNotIn("NOISY_STDOUT", stdout_text)
            self.assertNotIn("NOISY_STDERR", stdout_text)
            self.assertEqual("", stderr_text)

    def test_timeout_marks_run_failed_and_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_e"
            dataset.mkdir(parents=True, exist_ok=True)

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds", "chain"),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
                timeout_s=1.0,
            )
            with patch("sweep._run_main_with_timeout", return_value=("timeout after 1.0s", True)) as mock_timeout:
                df = run_sweep(cfg, run_main_fn=_default_run_main)

            self.assertEqual(len(df), 2)
            self.assertTrue(np.all(df["status"].to_numpy() == "failed"))
            self.assertTrue(np.all(df["failure_reason"].to_numpy() == "timeout"))
            self.assertTrue(np.all(df["timed_out"].to_numpy(dtype=bool)))
            self.assertEqual(mock_timeout.call_count, 2)

    def test_workers_parallel_branch_runs_all_combinations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_parallel"
            dataset.mkdir(parents=True, exist_ok=True)

            captures: list[tuple[str, int]] = []

            def fast_runner(stitch_cfg, results_path):
                captures.append((stitch_cfg.ds_method, int(stitch_cfg.seed)))
                rows = [
                    {"ds_compute_time": 0.1, "gg_compute_time": 0.2, "total_compute_time": 0.3},
                    {
                        "combination_id": 0,
                        "ds_method": stitch_cfg.ds_method,
                        "prediction_rmse": 1.0,
                        "cosine_dissimilarity": 2.0,
                        "dtw_distance_mean": 3.0,
                    },
                ]
                pd.DataFrame(rows).to_csv(results_path, index=False)
                return rows

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds", "chain"),
                seeds=(1, 2),
                output_dir=str(root / "sweep_out"),
                mode="standard",
                workers=3,
            )
            out_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(io.StringIO()):
                df = run_sweep(cfg, run_main_fn=fast_runner)

            self.assertEqual(len(df), 4)
            self.assertEqual(len(captures), 4)
            self.assertTrue(np.all(df["status"].to_numpy() == "ok"))
            self.assertIn("Using 3 sweep workers", out_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
