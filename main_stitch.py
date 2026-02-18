import numpy as np

from src.stitching.metrics import save_results_dataframe, calculate_ds_metrics
from src.util.load_tools import get_demonstration_set, resolve_data_scales
from src.util.benchmarking_tools import initialize_iter_strategy
from src.stitching.ds_stitching import construct_stitched_ds
from src.util.ds_tools import apply_lpvds_demowise
from src.util.plot_tools import plot_demonstration_set, plot_ds_set_gaussians, plot_gaussian_graph, plot_gg_solution, plot_ds
from configs import StitchConfig
import time
import src.graph_utils as gu
from src.util.ds_tools import get_gaussian_directions


def simulate_trajectories(ds, initial, config):
    """Simulates multiple trajectories from noisy initial conditions.

    Args:
        ds: Dynamical system to simulate, or None.
        initial: Base initial point for trajectory generation.
        config: Configuration object with noise_std and n_test_simulations.

    Returns:
        list: Simulated trajectories from noisy initial points, or None if ds is None.
    """
    if ds is None:
        return None

    # --- simulate
    # TODO figure out initial/final position from data
    x_inits = [initial + np.random.normal(0, config.noise_std, initial.shape[0]) for _ in
               range(config.n_test_simulations)]
    simulated_trajectories = []
    for x_0 in x_inits:
        simulated_trajectories.append(ds.sim(x_0[None, :], dt=0.01)[0])

    return simulated_trajectories

def main():
    config = StitchConfig()
    np.random.seed(config.seed)
    pre_computation_results = dict()

    save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"

    # ============= Load/create a set of demonstrations =============
    data_position_scale, data_velocity_scale = resolve_data_scales(config)
    demo_set = get_demonstration_set(
        config.dataset_path,
        position_scale=data_position_scale,
        velocity_scale=data_velocity_scale,
    )
    plot_demonstration_set(demo_set, config, save_as='Demonstrations_Raw', hide_axis=True)

    #  ============= Fit a DS to each demonstration =============
    t0 = time.time()
    ds_set, reversed_ds_set, norm_demo_set = apply_lpvds_demowise(demo_set, config.damm)
    pre_computation_results['ds_compute_time'] = time.time() - t0
    plot_demonstration_set(norm_demo_set, config, save_as='Demonstrations_Norm', hide_axis=True)
    plot_ds_set_gaussians(ds_set, config, include_trajectory=True, save_as='Demonstrations_Gaussians', hide_axis=True)

    #  ============= Construct Gaussian Graph =============
    t0 = time.time()
    gaussians = {(i, j): {'mu': mu, 'sigma': sigma, 'direction': direction, 'prior': prior}
                 for i, ds in enumerate(ds_set)
                 for j, (mu, sigma, direction, prior) in
                 enumerate(zip(ds.damm.Mu, ds.damm.Sigma, get_gaussian_directions(ds), ds.damm.Prior))}
    gg = gu.GaussianGraph(param_dist=config.param_dist,
                          param_cos=config.param_cos,
                          bhattacharyya_threshold=config.bhattacharyya_threshold)
    gg.add_gaussians(gaussians, reverse_gaussians=config.reverse_gaussians)
    pre_computation_results['gg_compute_time'] = time.time() - t0
    plot_gaussian_graph(gg, config, save_as='Gaussian_Graph', hide_axis=True)

    #  ============= Pre-compute segment DSs (for chaining only) =============
    t0 = time.time()
    segment_ds_lookup = dict()
    if config.ds_method == "chain":
        # TODO precompute
        pass
    pre_computation_results['precomputation_time'] = time.time() - t0

    # ============= Test various initial/attractor combinations =============
    init_attr_combinations = initialize_iter_strategy(config, demo_set)
    all_results = [pre_computation_results]
    for i, (initial, attractor) in enumerate(init_attr_combinations):
        print(f"Processing combination {i+1} of {len(init_attr_combinations)} #######################################")

        # Construct Gaussian Graph and Stitched DS
        print('Constructing Gaussian Graph and Stitched DS...')
        stitched_ds, gg_solution_nodes, stitching_stats = construct_stitched_ds(
            config, gg, norm_demo_set, ds_set, reversed_ds_set, initial, attractor
        )

        if stitched_ds is None or not hasattr(stitched_ds, 'damm') or stitched_ds.damm is None or not hasattr(stitched_ds.damm, 'Mu'):
            print(f"Warning: Skipping Stitched DS object with incomplete DAMM clustering")
            stitched_ds = None

        if stitched_ds is None:
            ds_metrics = calculate_ds_metrics(
                x_ref=None,
                x_dot_ref=None,
                ds=stitched_ds,
                sim_trajectories=None,
                initial=initial,
                attractor=attractor
            )
        else:

            # Simulate trajectories
            print('Simulating trajectories...')
            simulated_trajectories = simulate_trajectories(stitched_ds, initial, config)

            # Calculate DS metrics
            print('Calculating DS metrics...')
            ds_metrics = calculate_ds_metrics(
                x_ref=stitched_ds.x,
                x_dot_ref=stitched_ds.x_dot,
                ds=stitched_ds,
                sim_trajectories=simulated_trajectories,
                initial=initial,
                attractor=attractor
            )

            # Plot
            if config.save_fig:
                plot_gg_solution(gg, gg_solution_nodes, initial, attractor, config, save_as=f'{i}_Gaussian_Graph_Solution', hide_axis=True)
                plot_ds_set_gaussians([stitched_ds], config, initial=initial, attractor=attractor, include_trajectory=True, save_as=f'{i}_Stitched_DS_Gaussians', hide_axis=True)
                plot_ds(stitched_ds, simulated_trajectories, initial, attractor, config, save_as=f'{i}_Stitched_DS_Simulation', hide_axis=True)

        # Compile and append results
        results = {'combination_id': i, 'ds_method': config.ds_method,} | stitching_stats | ds_metrics
        all_results.append(results)

        # Print
        if stitched_ds is not None:
            print(f'Successful Stitched DS construction: {stitching_stats["total compute time"]:.2f} s')
            print(f'  Gaussian Graph: {stitching_stats["gg solution compute time"]:.2f} s, ')
            print(f'  Stitched DS: {stitching_stats["ds compute time"]:.2f} s')
            print(f'Metrics:')
            print(f'  RMSE: {ds_metrics["prediction_rmse"]:.4f}')
            print(f'  Cosine Dissimilarity: {ds_metrics["cosine_dissimilarity"]:.4f}')
            print(f'  DTW Distance: {ds_metrics["dtw_distance_mean"]:.4f} ± {ds_metrics["dtw_distance_std"]:.4f}')
        else:
            print("Stitched DS construction failed.")

    # Save Results to CSV
    if all_results:
        results_path = save_folder + f"results_{config.seed}.csv"
        save_results_dataframe(all_results, results_path)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
    
