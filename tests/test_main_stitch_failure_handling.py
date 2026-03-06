import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs import StitchConfig
from main_stitch import main, _resolve_save_figure_indices, _OperationTimeout


class _FakeSourceDS:
    def __init__(self):
        self.damm = SimpleNamespace(
            Mu=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                ],
                dtype=float,
            ),
            Sigma=np.array([np.eye(2), np.eye(2), np.eye(2)], dtype=float),
            Prior=np.array([1.0, 1.0, 1.0], dtype=float),
        )
        self.x = np.array([[0.0, 0.0]], dtype=float)
        self.x_dot = np.array([[0.0, 0.0]], dtype=float)
        self.assignment_arr = np.array([0], dtype=int)
        self.A = np.array([-np.eye(2), -np.eye(2), -np.eye(2)], dtype=float)


class _FakeStitchedDS:
    def __init__(self):
        self.damm = SimpleNamespace(Mu=np.array([[0.0, 0.0]], dtype=float))
        self.x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
        self.x_dot = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)


class _FakeGaussianGraph:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.gaussians = {}
        self.graph = SimpleNamespace(nodes={})
        self.gaussian_reversal_map = set()

    def add_gaussians(self, gaussians, reverse_gaussians=False):
        del reverse_gaussians
        self.gaussians = dict(gaussians)
        self.graph.nodes = {
            node_id: {"mean": np.asarray(data["mu"], dtype=float), "prior": float(data["prior"])}
            for node_id, data in self.gaussians.items()
        }

    def get_all_simple_paths(self, nr_edges=2):
        del nr_edges
        return [
            [((0, 0), (0, 1)), ((0, 1), (0, 2))],
        ]

    def shortest_path(self, initial_state, target_state):
        del initial_state, target_state
        nodes = sorted(self.graph.nodes.keys())
        if len(nodes) >= 3:
            return nodes[:3]
        return nodes


class MainStitchFailureHandlingTests(unittest.TestCase):
    def _base_config(self, ds_method: str) -> StitchConfig:
        cfg = StitchConfig()
        cfg.dataset_path = "dataset/stitching/2d_large"
        cfg.ds_method = ds_method
        cfg.save_fig = False
        cfg.n_test_simulations = 1
        cfg.chain_precompute_segments = False
        return cfg

    def test_main_stitch_records_failed_combination_and_continues(self):
        source_ds = _FakeSourceDS()
        stitched_ok = _FakeStitchedDS()
        cfg = self._base_config("chain_all")
        cfg.chain_precompute_segments = False

        combo_points = [
            (np.array([0.0, 0.0], dtype=float), np.array([1.0, 1.0], dtype=float)),
            (np.array([0.2, 0.1], dtype=float), np.array([1.2, 1.1], dtype=float)),
        ]

        def _fake_metrics(x_ref, x_dot_ref, ds, sim_trajectories, initial, attractor):
            del x_ref, x_dot_ref, sim_trajectories, initial, attractor
            if ds is None:
                return {
                    "initial_x": 0.0,
                    "initial_y": 0.0,
                    "attractor_x": 0.0,
                    "attractor_y": 0.0,
                    "prediction_rmse": np.nan,
                    "cosine_dissimilarity": np.nan,
                    "dtw_distance_mean": np.nan,
                    "dtw_distance_std": np.nan,
                    "distance_to_attractor_mean": np.nan,
                    "distance_to_attractor_std": np.nan,
                    "trajectory_length_mean": np.nan,
                    "trajectory_length_std": np.nan,
                    "n_simulations": 0,
                }
            return {
                "initial_x": 0.0,
                "initial_y": 0.0,
                "attractor_x": 0.0,
                "attractor_y": 0.0,
                "prediction_rmse": 0.5,
                "cosine_dissimilarity": 0.25,
                "dtw_distance_mean": 1.5,
                "dtw_distance_std": 0.0,
                "distance_to_attractor_mean": 0.2,
                "distance_to_attractor_std": 0.0,
                "trajectory_length_mean": 2.0,
                "trajectory_length_std": 0.0,
                "n_simulations": 1,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with (
                patch("main_stitch.resolve_data_scales", return_value=(1.0, 1.0)),
                patch("main_stitch.get_demonstration_set", return_value=["demo"]),
                patch("main_stitch.infer_state_dim_from_demo_set", return_value=2),
                patch("main_stitch.compute_plot_extent_from_demo_set", return_value=(0.0, 1.0, 0.0, 1.0)),
                patch("main_stitch.apply_lpvds_demowise", return_value=([source_ds], [], ["demo_norm"])),
                patch(
                    "main_stitch.get_gaussian_directions",
                    return_value=[
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                    ],
                ),
                patch("main_stitch.gu.GaussianGraph", _FakeGaussianGraph),
                patch("main_stitch.initialize_iter_strategy", return_value=combo_points),
                patch(
                    "main_stitch.construct_stitched_ds",
                    side_effect=[
                        RuntimeError("solver did not converge"),
                        (stitched_ok, [], {"gg_solution_compute_time": 0.1, "ds_compute_time": 0.2, "total_compute_time": 0.3}),
                    ],
                ),
                patch("main_stitch.simulate_trajectories", return_value=[np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)]),
                patch("main_stitch.calculate_ds_metrics", side_effect=_fake_metrics),
            ):
                all_results = main(config=cfg, results_path=str(csv_path))

            self.assertEqual(len(all_results), 3)
            eval_rows = [row for row in all_results if row.get("combination_id") is not None]
            self.assertEqual(len(eval_rows), 2)
            self.assertEqual(eval_rows[0]["combination_status"], "failed")
            self.assertEqual(eval_rows[0]["combination_failure_reason"], "runtime_exception")
            self.assertIn("solver did not converge", eval_rows[0]["combination_error_message"])
            self.assertEqual(eval_rows[1]["combination_status"], "ok")

            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            eval_df = df[df["combination_id"].notna()].copy()
            self.assertEqual(eval_df["combination_status"].tolist(), ["failed", "ok"])

    def test_chain_all_precompute_temporarily_enables_recompute_gaussians(self):
        source_ds = _FakeSourceDS()
        cfg = self._base_config("chain_all")
        cfg.chain_precompute_segments = True
        cfg.chain.recompute_gaussians = False

        combo_points = [
            (np.array([0.0, 0.0], dtype=float), np.array([1.0, 1.0], dtype=float)),
        ]
        recompute_flags = []

        def _fake_precompute(*args, **kwargs):
            config_obj = kwargs.get("config")
            if config_obj is None and len(args) >= 4:
                config_obj = args[3]
            recompute_flags.append(bool(config_obj.chain.recompute_gaussians))
            return object()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with (
                patch("main_stitch.resolve_data_scales", return_value=(1.0, 1.0)),
                patch("main_stitch.get_demonstration_set", return_value=["demo"]),
                patch("main_stitch.infer_state_dim_from_demo_set", return_value=2),
                patch("main_stitch.compute_plot_extent_from_demo_set", return_value=(0.0, 1.0, 0.0, 1.0)),
                patch("main_stitch.apply_lpvds_demowise", return_value=([source_ds], [], ["demo_norm"])),
                patch(
                    "main_stitch.get_gaussian_directions",
                    return_value=[
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                    ],
                ),
                patch("main_stitch.gu.GaussianGraph", _FakeGaussianGraph),
                patch("main_stitch.initialize_iter_strategy", return_value=combo_points),
                patch("main_stitch._compute_segment_DS", side_effect=_fake_precompute),
                patch(
                    "main_stitch.construct_stitched_ds",
                    return_value=(None, [], {"gg_solution_compute_time": 0.1, "ds_compute_time": 0.2, "total_compute_time": 0.3}),
                ),
            ):
                main(config=cfg, results_path=str(csv_path))

        self.assertGreaterEqual(len(recompute_flags), 1)
        self.assertTrue(all(recompute_flags))
        self.assertFalse(cfg.chain.recompute_gaussians)

    def test_main_stitch_marks_combination_timeout_and_continues(self):
        source_ds = _FakeSourceDS()
        stitched_ok = _FakeStitchedDS()
        cfg = self._base_config("chain_all")
        cfg.chain_precompute_segments = False
        cfg.combination_timeout_s = 600.0

        combo_points = [
            (np.array([0.0, 0.0], dtype=float), np.array([1.0, 1.0], dtype=float)),
            (np.array([0.2, 0.1], dtype=float), np.array([1.2, 1.1], dtype=float)),
        ]
        timeout_calls = {"combination": 0}

        def _fake_call_with_timeout(timeout_s, label, fn, *args, **kwargs):
            del timeout_s
            if str(label).startswith("combination"):
                timeout_calls["combination"] += 1
                if timeout_calls["combination"] == 1:
                    raise _OperationTimeout("combination 0 timeout after 600.000s")
            return fn(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with (
                patch("main_stitch.resolve_data_scales", return_value=(1.0, 1.0)),
                patch("main_stitch.get_demonstration_set", return_value=["demo"]),
                patch("main_stitch.infer_state_dim_from_demo_set", return_value=2),
                patch("main_stitch.compute_plot_extent_from_demo_set", return_value=(0.0, 1.0, 0.0, 1.0)),
                patch("main_stitch.apply_lpvds_demowise", return_value=([source_ds], [], ["demo_norm"])),
                patch(
                    "main_stitch.get_gaussian_directions",
                    return_value=[
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                    ],
                ),
                patch("main_stitch.gu.GaussianGraph", _FakeGaussianGraph),
                patch("main_stitch.initialize_iter_strategy", return_value=combo_points),
                patch("main_stitch._call_with_timeout", side_effect=_fake_call_with_timeout),
                patch(
                    "main_stitch.construct_stitched_ds",
                    return_value=(stitched_ok, [], {"gg_solution_compute_time": 0.1, "ds_compute_time": 0.2, "total_compute_time": 0.3}),
                ),
                patch("main_stitch.simulate_trajectories", return_value=[np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)]),
                patch(
                    "main_stitch.calculate_ds_metrics",
                    return_value={
                        "initial_x": 0.0,
                        "initial_y": 0.0,
                        "attractor_x": 0.0,
                        "attractor_y": 0.0,
                        "prediction_rmse": 0.5,
                        "cosine_dissimilarity": 0.25,
                        "dtw_distance_mean": 1.5,
                        "dtw_distance_std": 0.0,
                        "distance_to_attractor_mean": 0.2,
                        "distance_to_attractor_std": 0.0,
                        "trajectory_length_mean": 2.0,
                        "trajectory_length_std": 0.0,
                        "n_simulations": 1,
                    },
                ),
            ):
                all_results = main(config=cfg, results_path=str(csv_path))

        eval_rows = [row for row in all_results if row.get("combination_id") is not None]
        self.assertEqual(len(eval_rows), 2)
        self.assertEqual(eval_rows[0]["combination_status"], "failed")
        self.assertEqual(eval_rows[0]["combination_failure_reason"], "combination_timeout")
        self.assertIn("timeout after 600.000s", eval_rows[0]["combination_error_message"])
        self.assertEqual(eval_rows[1]["combination_status"], "ok")

    def test_main_stitch_uses_shared_precompute_artifact_when_configured(self):
        source_ds = _FakeSourceDS()
        cfg = self._base_config("chain_all")
        cfg.chain_precompute_segments = False
        cfg.shared_precompute_artifact_path = "/tmp/shared_precompute_dummy.pkl"

        shared_payload = {
            "schema_version": 1,
            "ds_set": [source_ds],
            "reversed_ds_set": [],
            "norm_demo_set": ["demo_norm"],
            "gg": _FakeGaussianGraph(),
            "ds_compute_time": 1.23,
            "gg_compute_time": 2.34,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with (
                patch("main_stitch.resolve_data_scales", return_value=(1.0, 1.0)),
                patch("main_stitch.get_demonstration_set", return_value=["demo"]),
                patch("main_stitch.infer_state_dim_from_demo_set", return_value=2),
                patch("main_stitch.compute_plot_extent_from_demo_set", return_value=(0.0, 1.0, 0.0, 1.0)),
                patch("main_stitch.initialize_iter_strategy", return_value=[]),
                patch("main_stitch.build_or_load_shared_precompute", return_value=shared_payload),
                patch("main_stitch.apply_lpvds_demowise") as apply_mock,
                patch("main_stitch.save_results_dataframe"),
            ):
                all_results = main(config=cfg, results_path=str(csv_path))

        self.assertEqual(len(all_results), 1)
        self.assertEqual(float(all_results[0]["ds_compute_time"]), 1.23)
        self.assertEqual(float(all_results[0]["gg_compute_time"]), 2.34)
        apply_mock.assert_not_called()

    def test_chain_precompute_timeout_is_non_fatal_and_recorded(self):
        source_ds = _FakeSourceDS()
        stitched_ok = _FakeStitchedDS()
        cfg = self._base_config("chain_all")
        cfg.chain_precompute_segments = True
        cfg.combination_timeout_s = 600.0

        combo_points = [
            (np.array([0.0, 0.0], dtype=float), np.array([1.0, 1.0], dtype=float)),
        ]

        def _fake_call_with_timeout(timeout_s, label, fn, *args, **kwargs):
            del timeout_s
            if str(label).startswith("chain precompute segment"):
                raise _OperationTimeout(f"{label} timeout after 600.000s")
            return fn(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with (
                patch("main_stitch.resolve_data_scales", return_value=(1.0, 1.0)),
                patch("main_stitch.get_demonstration_set", return_value=["demo"]),
                patch("main_stitch.infer_state_dim_from_demo_set", return_value=2),
                patch("main_stitch.compute_plot_extent_from_demo_set", return_value=(0.0, 1.0, 0.0, 1.0)),
                patch("main_stitch.apply_lpvds_demowise", return_value=([source_ds], [], ["demo_norm"])),
                patch(
                    "main_stitch.get_gaussian_directions",
                    return_value=[
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                    ],
                ),
                patch("main_stitch.gu.GaussianGraph", _FakeGaussianGraph),
                patch("main_stitch.initialize_iter_strategy", return_value=combo_points),
                patch("main_stitch._call_with_timeout", side_effect=_fake_call_with_timeout),
                patch(
                    "main_stitch.construct_stitched_ds",
                    return_value=(stitched_ok, [], {"gg_solution_compute_time": 0.1, "ds_compute_time": 0.2, "total_compute_time": 0.3}),
                ),
                patch("main_stitch.simulate_trajectories", return_value=[np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)]),
                patch(
                    "main_stitch.calculate_ds_metrics",
                    return_value={
                        "initial_x": 0.0,
                        "initial_y": 0.0,
                        "attractor_x": 0.0,
                        "attractor_y": 0.0,
                        "prediction_rmse": 0.5,
                        "cosine_dissimilarity": 0.25,
                        "dtw_distance_mean": 1.5,
                        "dtw_distance_std": 0.0,
                        "distance_to_attractor_mean": 0.2,
                        "distance_to_attractor_std": 0.0,
                        "trajectory_length_mean": 2.0,
                        "trajectory_length_std": 0.0,
                        "n_simulations": 1,
                    },
                ),
            ):
                all_results = main(config=cfg, results_path=str(csv_path))

        self.assertGreaterEqual(len(all_results), 2)
        pre_row = all_results[0]
        self.assertEqual(int(pre_row["precompute_segment_total"]), 1)
        self.assertEqual(int(pre_row["precompute_segment_ok"]), 0)
        self.assertEqual(int(pre_row["precompute_segment_failed"]), 0)
        self.assertEqual(int(pre_row["precompute_segment_timed_out"]), 1)
        self.assertEqual(int(pre_row["precompute_enumeration_timed_out"]), 0)
        eval_rows = [row for row in all_results if row.get("combination_id") is not None]
        self.assertEqual(len(eval_rows), 1)
        self.assertEqual(eval_rows[0]["combination_status"], "ok")

    def test_chain_precompute_uses_planned_shortest_path_segments(self):
        class _NoAllPathsGaussianGraph(_FakeGaussianGraph):
            def get_all_simple_paths(self, nr_edges=2):
                del nr_edges
                raise AssertionError("get_all_simple_paths should not be called")

        source_ds = _FakeSourceDS()
        cfg = self._base_config("chain_all")
        cfg.chain_precompute_segments = True

        combo_points = [
            (np.array([0.0, 0.0], dtype=float), np.array([1.0, 1.0], dtype=float)),
            (np.array([0.2, 0.1], dtype=float), np.array([1.2, 1.1], dtype=float)),
        ]
        seen_segments = []

        def _fake_precompute(*args, **kwargs):
            segment_nodes = kwargs.get("segment_nodes")
            if segment_nodes is None and len(args) >= 3:
                segment_nodes = args[2]
            seen_segments.append(tuple(segment_nodes))
            return object()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with (
                patch("main_stitch.resolve_data_scales", return_value=(1.0, 1.0)),
                patch("main_stitch.get_demonstration_set", return_value=["demo"]),
                patch("main_stitch.infer_state_dim_from_demo_set", return_value=2),
                patch("main_stitch.compute_plot_extent_from_demo_set", return_value=(0.0, 1.0, 0.0, 1.0)),
                patch("main_stitch.apply_lpvds_demowise", return_value=([source_ds], [], ["demo_norm"])),
                patch(
                    "main_stitch.get_gaussian_directions",
                    return_value=[
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                    ],
                ),
                patch("main_stitch.gu.GaussianGraph", _NoAllPathsGaussianGraph),
                patch("main_stitch.initialize_iter_strategy", return_value=combo_points),
                patch("main_stitch._compute_segment_DS", side_effect=_fake_precompute),
                patch(
                    "main_stitch.construct_stitched_ds",
                    return_value=(None, [], {"gg_solution_compute_time": 0.1, "ds_compute_time": 0.2, "total_compute_time": 0.3}),
                ),
            ):
                main(config=cfg, results_path=str(csv_path))

        # Both combos map to the same shortest-path triplet in this fake graph.
        self.assertEqual(len(seen_segments), 1)
        self.assertEqual(seen_segments[0], ((0, 0), (0, 1), (0, 2)))

    def test_main_stitch_writes_precompute_checkpoint_before_combination_loop(self):
        source_ds = _FakeSourceDS()
        cfg = self._base_config("chain_all")
        cfg.chain_precompute_segments = False

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with (
                patch("main_stitch.resolve_data_scales", return_value=(1.0, 1.0)),
                patch("main_stitch.get_demonstration_set", return_value=["demo"]),
                patch("main_stitch.infer_state_dim_from_demo_set", return_value=2),
                patch("main_stitch.compute_plot_extent_from_demo_set", return_value=(0.0, 1.0, 0.0, 1.0)),
                patch("main_stitch.apply_lpvds_demowise", return_value=([source_ds], [], ["demo_norm"])),
                patch(
                    "main_stitch.get_gaussian_directions",
                    return_value=[
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                    ],
                ),
                patch("main_stitch.gu.GaussianGraph", _FakeGaussianGraph),
                patch(
                    "main_stitch.initialize_iter_strategy",
                    side_effect=RuntimeError("stop_after_precompute"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop_after_precompute"):
                    main(config=cfg, results_path=str(csv_path))

            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            self.assertIn("ds_compute_time", df.columns)
            self.assertIn("gg_compute_time", df.columns)
            self.assertIn("precomputation_time", df.columns)

    def test_resolve_save_figure_indices_uses_dataset_defaults(self):
        cfg_x = self._base_config("chain")
        cfg_x.save_fig = True
        cfg_x.dataset_path = "dataset/stitching/X"
        self.assertEqual(_resolve_save_figure_indices(cfg_x), {24, 29})

        cfg_large = self._base_config("chain")
        cfg_large.save_fig = True
        cfg_large.dataset_path = "dataset/stitching/2d_large"
        self.assertEqual(_resolve_save_figure_indices(cfg_large), {115, 116, 143, 151, 177})

        cfg_skip = self._base_config("chain")
        cfg_skip.save_fig = True
        cfg_skip.dataset_path = "dataset/stitching/pcgmm_3d_workspace_simple"
        self.assertEqual(_resolve_save_figure_indices(cfg_skip), set())


if __name__ == "__main__":
    unittest.main()
