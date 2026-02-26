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

from sweep import SweepConfig, _default_run_main, run_sweep, _normalize_triplet_fit_mode
from sweep import load_raw_results


class SweepScriptTests(unittest.TestCase):
    def _ok_runner(self, captures: list[dict]):
        def _runner(stitch_cfg, results_path):
            captures.append(
                {
                    "dataset_path": stitch_cfg.dataset_path,
                    "ds_method": stitch_cfg.ds_method,
                    "seed": stitch_cfg.seed,
                    "n_test_simulations": stitch_cfg.n_test_simulations,
                    "combination_timeout_s": stitch_cfg.combination_timeout_s,
                    "save_fig": stitch_cfg.save_fig,
                    "chain_precompute_segments": getattr(stitch_cfg, "chain_precompute_segments", None),
                    "chain_ds_method": stitch_cfg.chain.ds_method,
                    "chain_trigger_method": stitch_cfg.chain.transition_trigger_method,
                    "chain_triplet_fit_mode": stitch_cfg.chain.triplet_fit_data_mode,
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

    def test_combination_timeout_is_propagated_to_stitch_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_timeout"
            dataset.mkdir(parents=True, exist_ok=True)
            captures: list[dict] = []

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("chain",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
                combination_timeout_s=123.0,
            )
            df = run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(len(df), 1)
            self.assertEqual(len(captures), 1)
            self.assertAlmostEqual(float(captures[0]["combination_timeout_s"]), 123.0)

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

    def test_chain_triplet_fit_mode_generates_expected_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_triplet"
            dataset.mkdir(parents=True, exist_ok=True)
            captures: list[dict] = []

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=(),
                seeds=(5,),
                output_dir=str(root / "sweep_out"),
                mode="chain_triplet_fit",
                chain_fixed_ds_method="segmented",
                chain_fixed_trigger_method="distance_ratio",
                chain_triplet_fit_modes=("all_nodes", "first_two_nodes", "behind_last_edge_plane"),
                chain_triplet_run_method="chain_all",
            )
            df = run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(len(df), 3)
            self.assertEqual(set(df["ds_method"].tolist()), {"chain_all"})
            self.assertEqual(set(df["chain_ds_method"].tolist()), {"segmented"})
            self.assertEqual(set(df["chain_transition_trigger_method"].tolist()), {"distance_ratio"})
            self.assertEqual(
                set(df["chain_triplet_fit_data_mode"].tolist()),
                {"all_nodes", "first_two_nodes", "behind_last_edge_plane"},
            )
            self.assertEqual(len(captures), 3)
            self.assertEqual(
                set(c["chain_triplet_fit_mode"] for c in captures),
                {"all_nodes", "first_two_nodes", "behind_last_edge_plane"},
            )
            self.assertTrue(all(c["chain_precompute_segments"] is True for c in captures))

    def test_chain_triplet_fit_mode_supports_subset_third_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_triplet_subset"
            dataset.mkdir(parents=True, exist_ok=True)
            captures: list[dict] = []

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=(),
                seeds=(7,),
                output_dir=str(root / "sweep_out"),
                mode="chain_triplet_fit",
                chain_fixed_ds_method="segmented",
                chain_fixed_trigger_method="distance_ratio",
                chain_triplet_fit_modes=("subset_third_node",),
                chain_triplet_run_method="chain_all",
            )
            df = run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "chain_triplet_fit_data_mode"], "subset_third_node")
            self.assertEqual(len(captures), 1)
            self.assertEqual(captures[0]["chain_triplet_fit_mode"], "subset_third_node")

    def test_normalize_triplet_fit_mode_accepts_subset_third_aliases(self):
        self.assertEqual(_normalize_triplet_fit_mode("subset_third_node"), "subset_third_node")
        self.assertEqual(_normalize_triplet_fit_mode("subset_third"), "subset_third_node")
        self.assertEqual(_normalize_triplet_fit_mode("third_node_subset"), "subset_third_node")

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
                shared_precompute=False,
            )
            with patch("sweep._run_main_with_timeout", return_value=("timeout after 1.0s", True)) as mock_timeout:
                df = run_sweep(cfg, run_main_fn=_default_run_main)

            self.assertEqual(len(df), 2)
            self.assertTrue(np.all(df["status"].to_numpy() == "failed"))
            self.assertTrue(np.all(df["failure_reason"].to_numpy() == "timeout"))
            self.assertTrue(np.all(df["timed_out"].to_numpy(dtype=bool)))
            self.assertEqual(mock_timeout.call_count, 2)

    def test_marks_run_failed_when_any_eval_combination_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_fail_rows"
            dataset.mkdir(parents=True, exist_ok=True)

            def mixed_status_runner(stitch_cfg, results_path):
                rows = [
                    {"ds_compute_time": 0.1, "gg_compute_time": 0.2, "total_compute_time": 0.3},
                    {
                        "combination_id": 0,
                        "ds_method": stitch_cfg.ds_method,
                        "combination_status": "ok",
                        "prediction_rmse": 1.0,
                        "cosine_dissimilarity": 2.0,
                        "dtw_distance_mean": 3.0,
                    },
                    {
                        "combination_id": 1,
                        "ds_method": stitch_cfg.ds_method,
                        "combination_status": "failed",
                        "combination_failure_reason": "runtime_exception",
                        "prediction_rmse": np.nan,
                        "cosine_dissimilarity": np.nan,
                        "dtw_distance_mean": np.nan,
                    },
                ]
                pd.DataFrame(rows).to_csv(results_path, index=False)
                return rows

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("chain_all",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
            )
            df = run_sweep(cfg, run_main_fn=mixed_status_runner)

            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["failure_reason"], "combination_failures")
            self.assertEqual(int(row["n_eval_rows"]), 2)
            self.assertEqual(int(row["n_eval_ok_rows"]), 1)
            self.assertEqual(int(row["n_eval_failed_rows"]), 1)

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

    def test_chain_timing_summary_uses_solution_and_precompute_parts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_timing"
            dataset.mkdir(parents=True, exist_ok=True)

            def timing_runner(stitch_cfg, results_path):
                rows = [
                    {
                        "ds_compute_time": 1.25,
                        "gg_compute_time": 2.50,
                        "precomputation_time": 3.75,
                    },
                    {
                        "combination_id": 0,
                        "ds_method": stitch_cfg.ds_method,
                        "prediction_rmse": 0.5,
                        "cosine_dissimilarity": 0.4,
                        "dtw_distance_mean": 0.3,
                        "distance_to_attractor_mean": 0.2,
                        "gg_solution_compute_time": 4.5,
                        "ds_compute_time": 5.5,
                        "total_compute_time": 10.0,
                    },
                ]
                pd.DataFrame(rows).to_csv(results_path, index=False)
                return rows

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("chain",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
            )
            df = run_sweep(cfg, run_main_fn=timing_runner)

            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(row["status"], "ok")
            self.assertEqual(int(row["n_eval_rows"]), 1)
            self.assertEqual(int(row["n_precompute_rows"]), 1)
            self.assertAlmostEqual(float(row["gg_solution_compute_time_mean"]), 4.5)
            # Backward-compatible aggregate now falls back to gg_solution_compute_time.
            self.assertAlmostEqual(float(row["gg_compute_time_mean"]), 4.5)
            self.assertAlmostEqual(float(row["ds_compute_time_mean"]), 5.5)
            self.assertAlmostEqual(float(row["total_compute_time_mean"]), 10.0)
            self.assertAlmostEqual(float(row["pre_ds_compute_time_mean"]), 1.25)
            self.assertAlmostEqual(float(row["pre_gg_compute_time_mean"]), 2.5)
            self.assertAlmostEqual(float(row["precomputation_time_mean"]), 3.75)

    def test_load_raw_results_skips_empty_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw_results"
            raw_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "dataset_slug": "dataset__a",
                        "seed": 1,
                        "combination_id": 0,
                        "ds_method": "chain",
                        "prediction_rmse": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "valid.csv", index=False)
            (raw_dir / "empty.csv").write_text("", encoding="utf-8")

            df = load_raw_results(str(root))
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["dataset_slug"], "dataset__a")

    def test_load_raw_results_raises_when_only_empty_csvs_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw_results"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "empty_only.csv").write_text("", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                load_raw_results(str(root))

    def test_includes_pcgmm_simple_dataset_in_sweep_specs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dataset = root / "dataset_keep"
            keep_dataset.mkdir(parents=True, exist_ok=True)
            pcgmm_dataset = "dataset/stitching/pcgmm_3d_workspace_simple"

            captures: list[dict] = []
            cfg = SweepConfig(
                datasets=(pcgmm_dataset, str(keep_dataset)),
                ds_methods=("chain_all",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
            )
            df = run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(len(df), 2)
            self.assertEqual(int(len(captures)), 2)
            self.assertEqual(
                set(df["dataset_path"].tolist()),
                {pcgmm_dataset, str(keep_dataset)},
            )

    def test_shared_precompute_prepared_once_per_signature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_shared"
            dataset.mkdir(parents=True, exist_ok=True)

            seen_artifacts: list[str] = []

            def _fake_run_single(
                cfg, spec, run_main_fn, raw_results_dir, run_index, total_runs, announce_start
            ):
                del cfg, run_main_fn, raw_results_dir, total_runs, announce_start
                seen_artifacts.append(str(spec.get("shared_precompute_artifact_path", "")))
                return {"run_index": run_index, "status": "ok", "dataset_path": spec["dataset_path"]}

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds", "chain"),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
                shared_precompute=True,
            )
            with (
                patch("sweep.get_demonstration_set", return_value=["demo"]),
                patch(
                    "sweep.build_or_load_shared_precompute",
                    return_value={
                        "schema_version": 1,
                        "ds_set": [],
                        "reversed_ds_set": [],
                        "norm_demo_set": [],
                        "gg": None,
                        "ds_compute_time": 0.1,
                        "gg_compute_time": 0.2,
                    },
                ) as precompute_mock,
                patch("sweep._run_single_spec", side_effect=_fake_run_single),
            ):
                run_sweep(cfg, run_main_fn=_default_run_main)

            self.assertEqual(precompute_mock.call_count, 1)
            self.assertEqual(len(seen_artifacts), 2)
            self.assertTrue(all(seen_artifacts))
            self.assertEqual(len(set(seen_artifacts)), 1)

    def test_shared_precompute_skipped_for_custom_runner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_shared_custom"
            dataset.mkdir(parents=True, exist_ok=True)
            captures: list[dict] = []

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
                shared_precompute=True,
            )
            with patch("sweep.build_or_load_shared_precompute") as precompute_mock:
                run_sweep(cfg, run_main_fn=self._ok_runner(captures))

            self.assertEqual(precompute_mock.call_count, 0)
            self.assertEqual(len(captures), 1)

    def test_disable_shared_precompute_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset_shared_disabled"
            dataset.mkdir(parents=True, exist_ok=True)

            def _fake_run_single(
                cfg, spec, run_main_fn, raw_results_dir, run_index, total_runs, announce_start
            ):
                del cfg, run_main_fn, raw_results_dir, total_runs, announce_start
                return {"run_index": run_index, "status": "ok", "dataset_path": spec["dataset_path"]}

            cfg = SweepConfig(
                datasets=(str(dataset),),
                ds_methods=("sp_recompute_ds",),
                seeds=(1,),
                output_dir=str(root / "sweep_out"),
                mode="standard",
                shared_precompute=False,
            )
            with (
                patch("sweep.get_demonstration_set", return_value=["demo"]),
                patch("sweep.build_or_load_shared_precompute") as precompute_mock,
                patch("sweep._run_single_spec", side_effect=_fake_run_single),
            ):
                run_sweep(cfg, run_main_fn=_default_run_main)

            self.assertEqual(precompute_mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()
