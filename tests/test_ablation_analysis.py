import unittest

import numpy as np
import pandas as pd

from src.util.ablation_analysis import (
    AnalysisConfig,
    add_success_columns,
    best_config_per_method,
    filter_by_ablation_scope,
    filter_eval_rows,
    summarize_by_config,
    summarize_by_method,
)


class AblationAnalysisTests(unittest.TestCase):
    def _base_df(self):
        return pd.DataFrame(
            [
                {
                    "dataset_slug": "dataset__stitching__pcgmm_3d_workspace_simple",
                    "sweep_mode": "pcgmm_damm_grid",
                    "combination_id": 0,
                    "combination_status": "ok",
                    "ds_method": "chain",
                    "distance_to_attractor_mean": 0.005,
                    "dtw_distance_mean": 100.0,
                    "data_position_scale": 5.0,
                    "param_dist": 1.0,
                    "param_cos": 1.0,
                    "bhattacharyya_threshold": 0.01,
                },
                {
                    "dataset_slug": "dataset__stitching__pcgmm_3d_workspace_simple",
                    "sweep_mode": "pcgmm_damm_grid",
                    "combination_id": 1,
                    "combination_status": "ok",
                    "ds_method": "chain",
                    "distance_to_attractor_mean": 0.5,
                    "dtw_distance_mean": 500.0,
                    "data_position_scale": 5.0,
                    "param_dist": 1.0,
                    "param_cos": 1.0,
                    "bhattacharyya_threshold": 0.01,
                },
                {
                    "dataset_slug": "dataset__stitching__pcgmm_3d_workspace_simple",
                    "sweep_mode": "pcgmm_damm_grid",
                    "combination_id": 2,
                    "combination_status": "failed",
                    "ds_method": "chain",
                    "distance_to_attractor_mean": np.nan,
                    "dtw_distance_mean": np.nan,
                    "data_position_scale": 5.0,
                    "param_dist": 1.0,
                    "param_cos": 1.0,
                    "bhattacharyya_threshold": 0.01,
                },
                {
                    "dataset_slug": "dataset__stitching__pcgmm_3d_workspace_simple",
                    "sweep_mode": "pcgmm_damm_grid",
                    "combination_id": 3,
                    "combination_status": "ok",
                    "ds_method": "sp_recompute_ds",
                    "distance_to_attractor_mean": 0.004,
                    "dtw_distance_mean": 50.0,
                    "data_position_scale": 10.0,
                    "param_dist": 2.0,
                    "param_cos": 2.0,
                    "bhattacharyya_threshold": 0.05,
                },
                # Outside allowed data_position_scale; should be filtered.
                {
                    "dataset_slug": "dataset__stitching__pcgmm_3d_workspace_simple",
                    "sweep_mode": "pcgmm_damm_grid",
                    "combination_id": 4,
                    "combination_status": "ok",
                    "ds_method": "sp_recompute_ds",
                    "distance_to_attractor_mean": 0.003,
                    "dtw_distance_mean": 40.0,
                    "data_position_scale": 20.0,
                    "param_dist": 2.0,
                    "param_cos": 2.0,
                    "bhattacharyya_threshold": 0.05,
                },
                # Precompute row; should be excluded by eval filter.
                {
                    "dataset_slug": "dataset__stitching__pcgmm_3d_workspace_simple",
                    "sweep_mode": "pcgmm_damm_grid",
                    "combination_id": np.nan,
                    "combination_status": np.nan,
                    "ds_method": np.nan,
                    "distance_to_attractor_mean": np.nan,
                    "dtw_distance_mean": np.nan,
                    "data_position_scale": 10.0,
                    "param_dist": 2.0,
                    "param_cos": 2.0,
                    "bhattacharyya_threshold": 0.05,
                },
            ]
        )

    def test_filter_scope_and_success_columns(self):
        df = self._base_df()
        cfg = AnalysisConfig(
            dataset_slug="dataset__stitching__pcgmm_3d_workspace_simple",
            sweep_mode="pcgmm_damm_grid",
            goal_tolerance=1e-2,
            allowed_configs={
                "data_position_scale": (5.0, 10.0, 15.0),
                "param_dist": (1.0, 2.0, 3.0),
                "param_cos": (1.0, 2.0, 3.0),
                "bhattacharyya_threshold": (0.01, 0.05, 0.1),
            },
        )
        eval_df = filter_eval_rows(df)
        scoped = filter_by_ablation_scope(eval_df, cfg)
        scoped = add_success_columns(scoped, goal_tolerance=cfg.goal_tolerance)

        self.assertEqual(len(scoped), 4)
        self.assertEqual(int(scoped["ds_build_failed"].sum()), 1)
        self.assertEqual(int(scoped["goal_not_reached"].sum()), 1)
        self.assertEqual(int(scoped["success"].sum()), 2)

    def test_summary_tables(self):
        df = self._base_df()
        cfg = AnalysisConfig(
            allowed_configs={
                "data_position_scale": (5.0, 10.0, 15.0),
                "param_dist": (1.0, 2.0, 3.0),
                "param_cos": (1.0, 2.0, 3.0),
                "bhattacharyya_threshold": (0.01, 0.05, 0.1),
            }
        )
        scoped = add_success_columns(
            filter_by_ablation_scope(filter_eval_rows(df), cfg),
            goal_tolerance=1e-2,
        )

        by_method = summarize_by_method(scoped)
        self.assertTrue({"ds_method", "success_rate", "dtw_success_mean"}.issubset(by_method.columns))
        chain = by_method[by_method["ds_method"] == "chain"].iloc[0]
        self.assertAlmostEqual(float(chain["success_rate"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(chain["dtw_success_mean"]), 100.0, places=6)

        by_config = summarize_by_config(
            scoped,
            config_cols=["data_position_scale", "param_dist", "param_cos", "bhattacharyya_threshold"],
        )
        self.assertGreaterEqual(len(by_config), 2)

        best = best_config_per_method(by_config)
        self.assertEqual(set(best["ds_method"].astype(str)), {"chain", "sp_recompute_ds"})
        self.assertTrue((best["success_rate"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
