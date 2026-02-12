import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.util import plot_tools
from src.stitching.metrics import save_results_dataframe, calculate_ds_metrics
from src.util.load_tools import get_demonstration_set, resolve_data_scales
from src.util.benchmarking_tools import initialize_iter_strategy
from src.stitching.ds_stitching import construct_stitched_ds
from src.util.ds_tools import apply_lpvds_demowise
from src.util.plot_tools import plot_demonstration_set, plot_ds_set_gaussians, plot_gaussian_graph

# TODO
# - all_paths_all

# ds_method options:
# - ["recompute_all"] Recompute using shortest path
# - ["recompute_ds"] Recompute only DS
# - ["reuse"] Reuse A's from step 1 and only recompute them if they are invalid wrt P
# - ["all_paths_all"] Fit DS to each node and use all paths
# - ["all_paths_ds"] Fit DS to each node and use all paths for DS
# - ["all_paths_reuse"] Fit DS to each node and use all paths for DS and reuse A's from step 1
# - ["chain"] Fit one linear DS per path node and switch/blend between them online

@dataclass
class Config:
    dataset_path: str = "./dataset/stitching/presentation2"
    force_preprocess: bool = True
    initial: Optional[np.ndarray] = None#np.array([4,15])
    attractor: Optional[np.ndarray] = None#np.array([14,2])
    ds_method: str = "chain"
    reverse_gaussians: bool = True
    param_dist: int = 3
    param_cos: int = 3
    n_demos: int = 5 # number of demonstrations to generate
    noise_std: float = 0.05 # standard deviation of noise added to demonstrations
    plot_extent = (0, 15, 0, 15) # (x_min, x_max, y_min, y_max)
    n_test_simulations: int = 2 # number of test simulations for metrics
    save_fig: bool = True
    seed: int = 42 # 42, 100, 3215, 21

    # DAMM settings
    rel_scale: float = 0.1 # 0.7
    total_scale: float = 1.0 # 1.5
    nu_0: int = 2*6 # 5
    kappa_0: float = 1.0
    psi_dir_0: float = 0.1 # 1.0
    data_position_scale: float = 1.0
    data_velocity_scale: Optional[float] = None

    # Chaining settings
    chain_switch_threshold: float = 0.12
    chain_blend_width: float = 0.18
    chain_trigger_radius: float = 0.10
    chain_transition_time: float = 5.0
    chain_start_node_candidates: int = 1
    chain_goal_node_candidates: int = 1
    chain_recovery_distance: float = 0.35
    chain_enable_recovery: bool = False
    chain_stabilization_margin: float = 1e-3
    chain_lmi_tolerance: float = 5e-5
    chain_edge_data_mode: str = "between_orthogonals"  # ["both_all", "between_orthogonals"]

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
    config = Config()
    np.random.seed(config.seed)

    save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"

    # Load/create a set of demonstrations
    data_position_scale, data_velocity_scale = resolve_data_scales(config)
    demo_set = get_demonstration_set(
        config.dataset_path,
        position_scale=data_position_scale,
        velocity_scale=data_velocity_scale,
    )
    plot_demonstration_set(demo_set, config, file_name='Demonstrations_Raw')

    # Fit a DS to each demonstration
    ds_set, reversed_ds_set, norm_demo_set = apply_lpvds_demowise(demo_set, config)
    plot_demonstration_set(norm_demo_set, config, file_name='Demonstrations_Norm')
    plot_ds_set_gaussians(ds_set, config, include_points=True, file_name='Demonstrations_Gaussians')

    # Determine iteration strategy based on config
    combinations = initialize_iter_strategy(config, demo_set)

    all_results = []
    for i, (initial, attractor) in enumerate(combinations):
        print(f"Processing combination {i+1} of {len(combinations)} #######################################")

        # Construct Gaussian Graph and Stitched DS
        print('Constructing Gaussian Graph and Stitched DS...')
        stitched_ds, gg, ds_stats = construct_stitched_ds(config, norm_demo_set, ds_set, reversed_ds_set, initial, attractor)
        
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
            plot_ds_set_gaussians([stitched_ds], config, include_points=True, file_name=f'stitched_gaussians_{i}')

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
            if i == 0 and config.save_fig:
                plot_gaussian_graph(gg, config, save_as='Gaussian_Graph')
            if config.save_fig:
                plot_tools.plot_gaussians_with_ds(gg, stitched_ds, simulated_trajectories, save_folder, i, config)
                plot_gaussian_graph(gg, config, save_as=f'gg_path_{i}')

        # Compile and append results
        results = {'combination_id': i, 'ds_method': config.ds_method,} | ds_stats | ds_metrics
        all_results.append(results)

        # Print
        if stitched_ds is not None:
            print(f'Successful Stitched DS construction: {ds_stats["total compute time"]:.2f} s')
            print(f'  Gaussian Graph: {ds_stats["gg compute time"]:.2f} s, ')
            print(f'  Stitched DS: {ds_stats["ds compute time"]:.2f} s')
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
    
