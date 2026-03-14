from __future__ import annotations

import os

import numpy as np

from configs import StitchConfig
import src.graph_utils as gu
from src.stitching.chaining import _compute_segment_DS, _resolve_triplet_fit_data_mode
from src.stitching.ds_stitching import construct_stitched_ds
from src.stitching.main_stitch_flow import run_chain_segment_precompute, run_single_combination
from src.stitching.main_stitch_helpers import (
    OperationTimeout as _OperationTimeout,
    call_with_timeout as _call_with_timeout,
    checkpoint_results_csv as _checkpoint_results_csv,
    default_stitching_stats as _default_stitching_stats,
    extract_chain_segments_from_path_nodes as _extract_chain_segments_from_path_nodes,
    extract_gaussian_node_indices as _extract_gaussian_node_indices,
    nan_ds_metrics as _nan_ds_metrics,
    resolve_save_figure_indices as _resolve_save_figure_indices,
    simulate_trajectories,
)
from src.stitching.metrics import calculate_ds_metrics, save_results_dataframe
from src.stitching.shared_precompute import build_or_load_shared_precompute
from src.util.benchmarking_tools import initialize_iter_strategy
from src.util.ds_tools import apply_lpvds_demowise, get_gaussian_directions
from src.util.load_tools import (
    compute_plot_extent_from_demo_set,
    get_demonstration_set,
    infer_state_dim_from_demo_set,
    resolve_data_scales,
)
from src.util.plot_tools import (
    plot_composite,
    plot_clean_3d_composite,
    plot_demonstration_set,
    plot_ds,
    plot_ds_set_gaussians,
    plot_gaussian_graph,
    plot_gg_solution,
)

# Backward-compatible wrappers used by tests and notebook code.
def _infer_state_dim_from_demo_set(demo_set):
    return infer_state_dim_from_demo_set(demo_set)


def _compute_plot_extent_from_demo_set(demo_set, **kwargs):
    return compute_plot_extent_from_demo_set(demo_set, **kwargs)


def main(config: StitchConfig | None = None, results_path: str | None = None):
    if config is None:
        config = StitchConfig()

    np.random.seed(config.seed)
    if config.save_folder_override:
        save_folder = config.save_folder_override
    else:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
    final_results_path = results_path or (save_folder + f"results_{config.seed}.csv")
    os.makedirs(os.path.dirname(final_results_path), exist_ok=True)

    # Create/truncate early so sweeps can observe that the run started.
    with open(final_results_path, "w", encoding="utf-8"):
        pass

    save_fig_indices = _resolve_save_figure_indices(config)
    if config.save_fig and save_fig_indices is not None:
        print(f"Saving figures only for indices: {sorted(save_fig_indices)}")

    # ============= Load/create a set of demonstrations =============
    data_position_scale, data_velocity_scale = resolve_data_scales(config)
    demo_set = get_demonstration_set(
        config.dataset_path,
        position_scale=data_position_scale,
        velocity_scale=data_velocity_scale,
    )
    state_dim = infer_state_dim_from_demo_set(demo_set)
    config.plot_extent = compute_plot_extent_from_demo_set(demo_set, state_dim=state_dim)
    if config.save_fig and save_fig_indices is None:
        plot_demonstration_set(demo_set, config, save_as="Demonstrations_Raw", hide_axis=True)

    # ============= Shared LPV-DS + Gaussian Graph precompute =============
    shared_precompute = build_or_load_shared_precompute(
        config=config,
        demo_set=demo_set,
        apply_lpvds_demowise_fn=apply_lpvds_demowise,
        get_gaussian_directions_fn=get_gaussian_directions,
        gaussian_graph_cls=gu.GaussianGraph,
    )
    ds_set = shared_precompute["ds_set"]
    reversed_ds_set = shared_precompute["reversed_ds_set"]
    norm_demo_set = shared_precompute["norm_demo_set"]
    gg = shared_precompute["gg"]
    ds_compute_time = float(shared_precompute["ds_compute_time"])
    gg_compute_time = float(shared_precompute["gg_compute_time"])
    if config.save_fig and save_fig_indices is None:
        plot_demonstration_set(norm_demo_set, config, save_as="Demonstrations_Norm", hide_axis=True)
        plot_ds_set_gaussians(
            ds_set,
            config,
            include_trajectory=True,
            save_as="Demonstrations_Gaussians",
            hide_axis=True,
        )

    # ============= Construct/Load Gaussian Graph (already in shared precompute) =============
    if config.save_fig and save_fig_indices is None:
        plot_gaussian_graph(gg, config, save_as="Gaussian_Graph", hide_axis=True)

    pre_computation_results = {
        "ds_compute_time": ds_compute_time,
        "gg_compute_time": gg_compute_time,
        "precomputation_time": 0.0,
        "precompute_segment_total": 0,
        "precompute_segment_ok": 0,
        "precompute_segment_failed": 0,
        "precompute_segment_timed_out": 0,
        "precompute_enumeration_timed_out": 0,
    }

    # Checkpoint before initializing combinations so interrupted runs retain setup timings.
    _checkpoint_results_csv([pre_computation_results], final_results_path)

    # ============= Initialize combination iterator =============
    init_attr_combinations = initialize_iter_strategy(config, demo_set)

    # ============= Pre-compute segment DSs (for chaining only) =============
    segment_ds_lookup = run_chain_segment_precompute(
        config=config,
        gg=gg,
        ds_set=ds_set,
        init_attr_combinations=init_attr_combinations,
        pre_computation_results=pre_computation_results,
        final_results_path=final_results_path,
        checkpoint_results_csv_fn=_checkpoint_results_csv,
        call_with_timeout_fn=_call_with_timeout,
        operation_timeout_cls=_OperationTimeout,
        compute_segment_ds_fn=_compute_segment_DS,
        resolve_triplet_fit_data_mode_fn=_resolve_triplet_fit_data_mode,
        extract_chain_segments_from_path_nodes_fn=_extract_chain_segments_from_path_nodes,
    )

    # ============= Test various initial/attractor combinations =============
    all_results = [pre_computation_results]
    _checkpoint_results_csv(all_results, final_results_path)

    for i, (initial, attractor) in enumerate(init_attr_combinations):
        print(
            f"Processing combination {i + 1} of {len(init_attr_combinations)} "
            "#######################################"
        )

        result_row, stitched_ds, stitching_stats, ds_metrics = run_single_combination(
            combination_id=i,
            initial=initial,
            attractor=attractor,
            config=config,
            gg=gg,
            norm_demo_set=norm_demo_set,
            ds_set=ds_set,
            reversed_ds_set=reversed_ds_set,
            segment_ds_lookup=segment_ds_lookup,
            demo_set=demo_set,
            save_fig_indices=save_fig_indices,
            construct_stitched_ds_fn=construct_stitched_ds,
            simulate_trajectories_fn=simulate_trajectories,
            calculate_ds_metrics_fn=calculate_ds_metrics,
            call_with_timeout_fn=_call_with_timeout,
            operation_timeout_cls=_OperationTimeout,
            default_stitching_stats_fn=_default_stitching_stats,
            nan_ds_metrics_fn=_nan_ds_metrics,
            extract_gaussian_node_indices_fn=_extract_gaussian_node_indices,
            plot_gg_solution_fn=plot_gg_solution,
            plot_ds_set_gaussians_fn=plot_ds_set_gaussians,
            plot_ds_fn=plot_ds,
            plot_composite_fn=plot_composite,
            plot_clean_3d_composite_fn=plot_clean_3d_composite,
        )

        all_results.append(result_row)
        _checkpoint_results_csv(all_results, final_results_path)

        if result_row.get("combination_status") == "ok" and stitched_ds is not None:
            print(f"Successful Stitched DS construction: {stitching_stats['total_compute_time']:.2f} s")
            gg_time = stitching_stats.get("gg_solution_compute_time")
            if gg_time is not None:
                print(f"  Gaussian Graph: {gg_time:.2f} s, ")
            else:
                print("  Gaussian Graph: N/A")
            print(f"  Stitched DS: {stitching_stats['ds_compute_time']:.2f} s")
            print("Metrics:")
            print(f"  RMSE: {ds_metrics['prediction_rmse']:.4f}")
            print(f"  Cosine Dissimilarity: {ds_metrics['cosine_dissimilarity']:.4f}")
            print(
                "  DTW Distance: "
                f"{ds_metrics['dtw_distance_mean']:.4f} ± {ds_metrics['dtw_distance_std']:.4f}"
            )
        else:
            print("Stitched DS construction failed.")

    if all_results:
        save_results_dataframe(all_results, final_results_path)
    else:
        print("No results to save.")

    return all_results


if __name__ == "__main__":
    main()
