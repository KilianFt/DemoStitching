import argparse
from dataclasses import dataclass, asdict
import os
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_stitch import Config as StitchConfig
from src.stitching.ds_stitching import construct_stitched_ds
from src.stitching.metrics import calculate_ds_metrics
from src.util.benchmarking_tools import initialize_iter_strategy
from src.util.ds_tools import apply_lpvds_demowise
from src.util.load_tools import get_demonstration_set


DEFAULT_METHODS = ("sp_recompute_all", "sp_recompute_ds", "sp_recompute_invalid_As", "chain")


@dataclass
class CompareConfig:
    datasets: Tuple[str, ...] = (
        "dataset/stitching/robottasks_workspace_chain",
        "dataset/stitching/nodes_1",
        "dataset/stitching/presentation2",
    )
    methods: Tuple[str, ...] = DEFAULT_METHODS
    max_combinations: int = 4
    n_test_simulations: int = 2
    noise_std: float = 0.03
    seed: int = 42
    output_csv: str = "dataset/stitching/benchmark_method_comparison.csv"
    output_summary_csv: str = "dataset/stitching/benchmark_method_comparison_summary.csv"
    output_figure_dir: str = "dataset/stitching/benchmark_method_figures"
    save_method_figures: bool = True


def select_combinations(
    combinations: Sequence[Tuple[np.ndarray, np.ndarray]],
    max_combinations: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(combinations) <= max_combinations:
        return list(combinations)
    if max_combinations <= 1:
        return [combinations[0]]
    idx = np.linspace(0, len(combinations) - 1, max_combinations, dtype=int)
    idx = np.unique(idx)
    return [combinations[i] for i in idx]


def aggregate_method_results(df: pd.DataFrame) -> pd.DataFrame:
    def safe_nanmean(series):
        arr = np.asarray(series, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.nan
        return float(np.nanmean(arr))

    grouped = []
    for (dataset_path, method), part in df.groupby(["dataset_path", "method"], dropna=False):
        grouped.append(
            {
                "dataset_path": dataset_path,
                "method": method,
                "n_cases": int(len(part)),
                "success_rate": float(np.mean(part["success"].astype(float))),
                "prediction_rmse_mean": safe_nanmean(part["prediction_rmse"]),
                "cosine_dissimilarity_mean": safe_nanmean(part["cosine_dissimilarity"]),
                "dtw_distance_mean": safe_nanmean(part["dtw_distance_mean"]),
                "distance_to_attractor_mean": safe_nanmean(part["distance_to_attractor_mean"]),
                "gg_compute_time_mean": safe_nanmean(part["gg compute time"]),
                "ds_compute_time_mean": safe_nanmean(part["ds compute time"]),
                "total_compute_time_mean": safe_nanmean(part["total compute time"]),
            }
        )
    return pd.DataFrame(grouped)


def _simulate_trajectories(ds, initial: np.ndarray, cfg: CompareConfig):
    if ds is None:
        return None
    rng = np.random.default_rng(cfg.seed)
    x_inits = [
        initial + rng.normal(0.0, cfg.noise_std, initial.shape[0])
        for _ in range(cfg.n_test_simulations)
    ]
    trajectories = []
    for x_0 in x_inits:
        trajectories.append(ds.sim(x_0[None, :], dt=0.01)[0])
    return trajectories


def _method_config(method: str, seed: int) -> StitchConfig:
    cfg = StitchConfig()
    cfg.ds_method = method
    cfg.seed = seed
    cfg.save_fig = False
    cfg.n_test_simulations = 1
    cfg.noise_std = 0.0
    return cfg


def _compute_demo_extent(demo_set, padding: float = 0.5):
    all_points = np.vstack([traj.x[:, :2] for demo in demo_set for traj in demo.trajectories])
    x_min, y_min = np.min(all_points, axis=0) - padding
    x_max, y_max = np.max(all_points, axis=0) + padding
    return x_min, x_max, y_min, y_max


def _plot_method_overlay(
    dataset_name: str,
    method: str,
    combo_id: int,
    demo_set,
    ds,
    trajectory,
    save_dir: str,
):
    if ds is None or trajectory is None:
        return None

    os.makedirs(save_dir, exist_ok=True)
    x_min, x_max, y_min, y_max = _compute_demo_extent(demo_set)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
    _plot_projected_ds_background(ax, ds, x_min, x_max, y_min, y_max)

    # demonstrations
    demo_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(demo_set)))
    for i, demo in enumerate(demo_set):
        for traj in demo.trajectories:
            ax.plot(traj.x[:, 0], traj.x[:, 1], color=demo_colors[i], linewidth=1.2, alpha=0.35)

    # executed path
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="magenta", linewidth=3.0, label="executed path")
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color="gold", edgecolor="black", s=70, zorder=4, label="start")
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color="red", edgecolor="black", s=70, zorder=4, label="end")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(f"{dataset_name} | {method} | combo {combo_id}")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"{dataset_name}_{method}_combo_{combo_id}.png")
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    return out_path


def _predict_velocity_field(ds, points: np.ndarray) -> np.ndarray:
    # Preferred path for chain and any custom policies.
    if hasattr(ds, "predict_velocities"):
        return ds.predict_velocities(points)

    gamma = ds.damm.compute_gamma(points)  # K x M
    vel = np.zeros_like(points)
    for k in range(ds.A.shape[0]):
        vel += gamma[k][:, None] * ((ds.A[k] @ (points - ds.x_att).T).T)
    return vel


def _plot_projected_ds_background(ax, ds, x_min: float, x_max: float, y_min: float, y_max: float):
    plot_sample = 55
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(x_min, x_max, plot_sample),
        np.linspace(y_min, y_max, plot_sample),
    )
    xy = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    dim = ds.x.shape[1]
    points = np.zeros((xy.shape[0], dim))
    points[:, :2] = xy
    if dim > 2:
        anchor = np.asarray(ds.x_att).reshape(-1)
        for d in range(2, dim):
            points[:, d] = anchor[d]

    vel = _predict_velocity_field(ds, points)
    u = vel[:, 0].reshape(plot_sample, plot_sample)
    v = vel[:, 1].reshape(plot_sample, plot_sample)
    ax.streamplot(x_mesh, y_mesh, u, v, density=2.6, color="black", arrowsize=1.0, arrowstyle="->")


def run_comparison(cfg: CompareConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(cfg.seed)
    rows = []

    for dataset_path in cfg.datasets:
        print(f"\n=== Dataset: {dataset_path} ===")
        dataset_name = os.path.basename(dataset_path.rstrip("/"))
        demo_set = get_demonstration_set(dataset_path)
        fit_cfg = _method_config(method="sp_recompute_ds", seed=cfg.seed)
        ds_set, reversed_ds_set, norm_demo_set = apply_lpvds_demowise(demo_set, fit_cfg)

        combo_cfg = _method_config(method="sp_recompute_ds", seed=cfg.seed)
        all_combinations = initialize_iter_strategy(combo_cfg, demo_set)
        combinations = select_combinations(all_combinations, cfg.max_combinations)
        print(f"Using {len(combinations)} / {len(all_combinations)} start-goal combinations")

        for method in cfg.methods:
            print(f"  -> Method: {method}")
            method_cfg = _method_config(method=method, seed=cfg.seed)
            for combo_id, (initial, attractor) in enumerate(combinations):
                try:
                    stitched_ds, gg, _, ds_stats = construct_stitched_ds(
                        method_cfg, norm_demo_set, ds_set, reversed_ds_set, initial, attractor
                    )
                except Exception as e:
                    stitched_ds = None
                    ds_stats = {
                        "gg compute time": np.nan,
                        "ds compute time": np.nan,
                        "total compute time": np.nan,
                    }
                    print(f"    combo {combo_id}: FAILED during construction: {e}")

                if stitched_ds is None:
                    ds_metrics = {
                        "initial_x": initial[0],
                        "initial_y": initial[1],
                        "attractor_x": attractor[0],
                        "attractor_y": attractor[1],
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
                else:
                    sim_trajectories = _simulate_trajectories(stitched_ds, initial, cfg)
                    ds_metrics = calculate_ds_metrics(
                        x_ref=stitched_ds.x,
                        x_dot_ref=stitched_ds.x_dot,
                        ds=stitched_ds,
                        sim_trajectories=sim_trajectories,
                        initial=initial,
                        attractor=attractor,
                    )
                    figure_path = None
                    if cfg.save_method_figures and len(sim_trajectories) > 0:
                        figure_path = _plot_method_overlay(
                            dataset_name=dataset_name,
                            method=method,
                            combo_id=combo_id,
                            demo_set=demo_set,
                            ds=stitched_ds,
                            trajectory=sim_trajectories[0],
                            save_dir=os.path.join(cfg.output_figure_dir, dataset_name),
                        )
                if stitched_ds is None:
                    figure_path = None

                row = {
                    "dataset_path": dataset_path,
                    "method": method,
                    "combo_id": combo_id,
                    "success": stitched_ds is not None,
                    "figure_path": figure_path,
                }
                row.update(ds_stats)
                row.update(ds_metrics)
                rows.append(row)

    df = pd.DataFrame(rows)
    summary = aggregate_method_results(df)

    os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    summary.to_csv(cfg.output_summary_csv, index=False)
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Compare stitching DS methods across datasets.")
    parser.add_argument("--max-combinations", type=int, default=4)
    parser.add_argument("--n-test-simulations", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default="dataset/stitching/benchmark_method_comparison.csv")
    parser.add_argument(
        "--output-summary-csv",
        type=str,
        default="dataset/stitching/benchmark_method_comparison_summary.csv",
    )
    parser.add_argument(
        "--output-figure-dir",
        type=str,
        default="dataset/stitching/benchmark_method_figures",
    )
    parser.add_argument(
        "--no-save-figures",
        action="store_true",
        help="Disable per-method overlay figure generation.",
    )
    args = parser.parse_args()

    cfg = CompareConfig(
        max_combinations=args.max_combinations,
        n_test_simulations=args.n_test_simulations,
        noise_std=args.noise_std,
        seed=args.seed,
        output_csv=args.output_csv,
        output_summary_csv=args.output_summary_csv,
        output_figure_dir=args.output_figure_dir,
        save_method_figures=not args.no_save_figures,
    )
    print("Running method comparison with config:")
    print(asdict(cfg))

    _, summary = run_comparison(cfg)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n=== Summary ===")
    print(summary.sort_values(["dataset_path", "method"]).to_string(index=False))
    print(f"\nSaved detailed rows: {os.path.abspath(cfg.output_csv)}")
    print(f"Saved summary rows: {os.path.abspath(cfg.output_summary_csv)}")


if __name__ == "__main__":
    main()
