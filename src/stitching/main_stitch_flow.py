from __future__ import annotations

import time

import numpy as np

from .metrics import demo_set_spread


def _update_precompute_results(
    pre_computation_results: dict,
    *,
    precompute_segment_total: int,
    precompute_segment_ok: int,
    precompute_segment_failed: int,
    precompute_segment_timed_out: int,
    precompute_enumeration_timed_out: int,
    precomputation_time: float,
):
    pre_computation_results["precomputation_time"] = float(precomputation_time)
    pre_computation_results["precompute_segment_total"] = int(precompute_segment_total)
    pre_computation_results["precompute_segment_ok"] = int(precompute_segment_ok)
    pre_computation_results["precompute_segment_failed"] = int(precompute_segment_failed)
    pre_computation_results["precompute_segment_timed_out"] = int(precompute_segment_timed_out)
    pre_computation_results["precompute_enumeration_timed_out"] = int(precompute_enumeration_timed_out)


def run_chain_segment_precompute(
    *,
    config,
    gg,
    ds_set,
    init_attr_combinations,
    pre_computation_results: dict,
    final_results_path: str,
    checkpoint_results_csv_fn,
    call_with_timeout_fn,
    operation_timeout_cls,
    compute_segment_ds_fn,
    resolve_triplet_fit_data_mode_fn,
    extract_chain_segments_from_path_nodes_fn,
):
    """Precompute chain segment DS objects for planned shortest-path segments."""
    t0 = time.time()
    segment_ds_lookup = {}
    precompute_segment_total = int(pre_computation_results.get("precompute_segment_total", 0))
    precompute_segment_ok = int(pre_computation_results.get("precompute_segment_ok", 0))
    precompute_segment_failed = int(pre_computation_results.get("precompute_segment_failed", 0))
    precompute_segment_timed_out = int(pre_computation_results.get("precompute_segment_timed_out", 0))
    precompute_enumeration_timed_out = int(pre_computation_results.get("precompute_enumeration_timed_out", 0))

    precompute_chain_segments = bool(getattr(config, "chain_precompute_segments", True))
    if config.ds_method in {"chain", "chain_all"} and precompute_chain_segments:
        triplet_fit_mode = resolve_triplet_fit_data_mode_fn(config.chain)
        prev_recompute = bool(config.chain.recompute_gaussians)
        if config.ds_method == "chain_all":
            config.chain.recompute_gaussians = True
        try:
            planned_segments = set()
            for combo_idx, (initial, attractor) in enumerate(init_attr_combinations):
                try:
                    path_nodes = call_with_timeout_fn(
                        config.combination_timeout_s,
                        f"chain precompute shortest path combo {combo_idx}",
                        gg.shortest_path,
                        initial,
                        attractor,
                    )
                except operation_timeout_cls as exc:
                    precompute_enumeration_timed_out += 1
                    print(f"Warning: {exc}. Skipping precompute path for combo {combo_idx}.")
                    continue
                except Exception as exc:
                    print(f"Warning: shortest-path extraction failed for combo {combo_idx}: {exc}")
                    continue

                if path_nodes is None or len(path_nodes) == 0:
                    continue
                for segment_nodes in extract_chain_segments_from_path_nodes_fn(path_nodes):
                    planned_segments.add(tuple(segment_nodes))

            all_segments_to_precompute = list(planned_segments)
            precompute_segment_total = int(len(all_segments_to_precompute))
            for seg_idx, segment_nodes in enumerate(all_segments_to_precompute, start=1):
                if len(segment_nodes) == 0:
                    continue

                try:
                    segment_ds = call_with_timeout_fn(
                        config.combination_timeout_s,
                        f"chain precompute segment {segment_nodes}",
                        compute_segment_ds_fn,
                        ds_set,
                        gg,
                        segment_nodes,
                        config,
                    )
                    segment_ds_lookup[(segment_nodes, triplet_fit_mode)] = segment_ds
                    if triplet_fit_mode == "all_nodes":
                        segment_ds_lookup[segment_nodes] = segment_ds
                    precompute_segment_ok += 1
                except operation_timeout_cls as exc:
                    precompute_segment_timed_out += 1
                    print(f"Warning: failed to precompute segment {segment_nodes}: {exc}")
                except Exception as exc:
                    precompute_segment_failed += 1
                    print(f"Warning: failed to precompute segment {segment_nodes}: {exc}")

                if seg_idx % 25 == 0 or seg_idx == precompute_segment_total:
                    _update_precompute_results(
                        pre_computation_results,
                        precompute_segment_total=precompute_segment_total,
                        precompute_segment_ok=precompute_segment_ok,
                        precompute_segment_failed=precompute_segment_failed,
                        precompute_segment_timed_out=precompute_segment_timed_out,
                        precompute_enumeration_timed_out=precompute_enumeration_timed_out,
                        precomputation_time=time.time() - t0,
                    )
                    checkpoint_results_csv_fn([pre_computation_results], final_results_path)
        finally:
            config.chain.recompute_gaussians = prev_recompute

    _update_precompute_results(
        pre_computation_results,
        precompute_segment_total=precompute_segment_total,
        precompute_segment_ok=precompute_segment_ok,
        precompute_segment_failed=precompute_segment_failed,
        precompute_segment_timed_out=precompute_segment_timed_out,
        precompute_enumeration_timed_out=precompute_enumeration_timed_out,
        precomputation_time=time.time() - t0,
    )
    return segment_ds_lookup


def run_single_combination(
    *,
    combination_id: int,
    initial,
    attractor,
    config,
    gg,
    norm_demo_set,
    ds_set,
    reversed_ds_set,
    segment_ds_lookup,
    demo_set,
    save_fig_indices,
    construct_stitched_ds_fn,
    simulate_trajectories_fn,
    calculate_ds_metrics_fn,
    call_with_timeout_fn,
    operation_timeout_cls,
    default_stitching_stats_fn,
    nan_ds_metrics_fn,
    extract_gaussian_node_indices_fn,
    plot_gg_solution_fn,
    plot_ds_set_gaussians_fn,
    plot_ds_fn,
    plot_composite_fn,
):
    """Run one initial/attractor evaluation, including optional plotting."""
    combination_status = "failed"
    combination_failure_reason = ""
    combination_error_message = ""
    stitching_stats = default_stitching_stats_fn()
    ds_metrics = nan_ds_metrics_fn(initial, attractor)
    stitched_ds = None

    def _process_combination():
        local_combination_status = "failed"
        local_combination_failure_reason = ""
        local_combination_error_message = ""
        local_stitching_stats = default_stitching_stats_fn()
        local_ds_metrics = nan_ds_metrics_fn(initial, attractor)
        local_stitched_ds = None
        local_gg_solution_nodes = []
        local_simulated_trajectories = None

        print("Constructing Gaussian Graph and Stitched DS...")
        stitch_result = construct_stitched_ds_fn(
            config,
            gg,
            norm_demo_set,
            ds_set,
            reversed_ds_set,
            initial,
            attractor,
            segment_ds_lookup=segment_ds_lookup,
        )
        if isinstance(stitch_result, tuple) and len(stitch_result) == 4:
            local_stitched_ds, _gg_obj, local_gg_solution_nodes, stitching_stats_raw = stitch_result
            del _gg_obj
        elif isinstance(stitch_result, tuple) and len(stitch_result) == 3:
            local_stitched_ds, local_gg_solution_nodes, stitching_stats_raw = stitch_result
        else:
            raise RuntimeError("construct_stitched_ds returned unexpected result shape")
        if isinstance(stitching_stats_raw, dict):
            local_stitching_stats.update(stitching_stats_raw)

        if (
            local_stitched_ds is None
            or not hasattr(local_stitched_ds, "damm")
            or local_stitched_ds.damm is None
            or not hasattr(local_stitched_ds.damm, "Mu")
        ):
            print("Warning: skipping Stitched DS object with incomplete DAMM clustering")
            local_stitched_ds = None
            local_combination_failure_reason = "ds_construction_failed"
            raw_error = str(local_stitching_stats.get("construction_error", "")).strip()
            local_combination_error_message = raw_error
        else:
            print("Simulating trajectories...")
            local_simulated_trajectories = simulate_trajectories_fn(local_stitched_ds, initial, config)

            print("Calculating DS metrics...")
            mean_mindist, std_mindist = demo_set_spread(norm_demo_set)

            if config.ds_method.startswith("spt"):
                sp_nodes = gg.shortest_path(initial, attractor)
                sp_x_parts, sp_xd_parts = [], []
                for node_id in sp_nodes:
                    ds_idx, gaussian_idx = extract_gaussian_node_indices_fn(node_id)
                    mask = ds_set[ds_idx].assignment_arr == gaussian_idx
                    ax = ds_set[ds_idx].x[mask]
                    axd = ds_set[ds_idx].x_dot[mask]
                    if node_id in gg.gaussian_reversal_map:
                        axd = -axd
                    sp_x_parts.append(ax)
                    sp_xd_parts.append(axd)
                x_ref = np.vstack(sp_x_parts)
                x_dot_ref = np.vstack(sp_xd_parts)
            else:
                x_ref = local_stitched_ds.x
                x_dot_ref = local_stitched_ds.x_dot

            local_ds_metrics = calculate_ds_metrics_fn(
                x_ref=x_ref,
                x_dot_ref=x_dot_ref,
                ds=local_stitched_ds,
                sim_trajectories=local_simulated_trajectories,
                initial=initial,
                attractor=attractor,
                mean_mindist=mean_mindist,
                std_mindist=std_mindist
            )
            local_combination_status = "ok"

            if config.save_fig:
                should_save_combo_figs = (save_fig_indices is None) or (combination_id in save_fig_indices)
            else:
                should_save_combo_figs = False
            if should_save_combo_figs:
                plot_gg_solution_fn(
                    gg,
                    local_gg_solution_nodes,
                    initial,
                    attractor,
                    config,
                    save_as=f"{combination_id}_Gaussian_Graph_Solution",
                    hide_axis=True,
                )
                plot_ds_set_gaussians_fn(
                    [local_stitched_ds],
                    config,
                    initial=initial,
                    attractor=attractor,
                    include_trajectory=True,
                    save_as=f"{combination_id}_Stitched_DS_Gaussians",
                    hide_axis=True,
                )
                plot_ds_fn(
                    local_stitched_ds,
                    local_simulated_trajectories,
                    initial,
                    attractor,
                    config,
                    save_as=f"{combination_id}_Stitched_DS_Simulation",
                    hide_axis=True,
                )
                plot_composite_fn(
                    gg,
                    local_gg_solution_nodes,
                    demo_set,
                    local_stitched_ds,
                    local_simulated_trajectories,
                    initial,
                    attractor,
                    config,
                    save_as=f"{combination_id}_Composite",
                    hide_axis=True,
                )
        return (
            local_combination_status,
            local_combination_failure_reason,
            local_combination_error_message,
            local_stitching_stats,
            local_ds_metrics,
            local_stitched_ds,
        )

    try:
        (
            combination_status,
            combination_failure_reason,
            combination_error_message,
            stitching_stats,
            ds_metrics,
            stitched_ds,
        ) = call_with_timeout_fn(
            config.combination_timeout_s,
            f"combination {combination_id}",
            _process_combination,
        )
    except operation_timeout_cls as exc:
        combination_status = "failed"
        combination_failure_reason = "combination_timeout"
        combination_error_message = str(exc)
        print(f"Warning: combination {combination_id} timed out: {combination_error_message}")
    except Exception as exc:
        combination_status = "failed"
        combination_failure_reason = "runtime_exception"
        combination_error_message = f"{type(exc).__name__}: {exc}"
        print(f"Warning: combination {combination_id} failed with error: {combination_error_message}")

    result_row = {
        "combination_id": combination_id,
        "ds_method": config.ds_method,
        "combination_status": combination_status,
        "combination_failure_reason": combination_failure_reason,
        "combination_error_message": combination_error_message,
    } | stitching_stats | ds_metrics

    return result_row, stitched_ds, stitching_stats, ds_metrics
