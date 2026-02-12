import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.util import plot_tools
from src.stitching.metrics import save_results_dataframe, calculate_ds_metrics
from src.util.load_tools import get_demonstration_set
from src.util.benchmarking_tools import initialize_iter_strategy
from src.stitching.ds_stitching import construct_stitched_ds
from src.util.ds_tools import apply_lpvds_demowise
from src.util.plot_tools import plot_demonstration_set, plot_ds_set_gaussians, plot_gaussian_graph, plot_gg_solution, plot_ds

# TODO
# - all_paths_all

# ds_method options:
# - ["sp_recompute_all"]            Uses shortest path, extracts raw traj. points, recomputes Gaussians and DS.
# - ["sp_recompute_ds"]             Uses shortest path, keeps Gaussians but recomputes DS.
# - ["sp_recompute_invalid_As"]     Uses shortest path, selects a P near the attractor, recomputes any incompatible As.
# - ["sp_recompute_P"]              Uses shortest path, keeps Gaussians and As, tries to find a P.
# - ["spt_recompute_all"]           Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["spt_recompute_ds"]            Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["spt_recompute_invalid_As"]    Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["chain"]                       Fit one linear DS per path node and switch/blend between them online.

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
    nu_0: int = 5 # 5
    kappa_0: int = 0.1 # 1.0
    psi_dir_0: int = 0.1 # 1.0

    # Chaining settings
    chain_switch_threshold: float = 0.12
    chain_blend_width: float = 0.18
    chain_trigger_radius: float = 0.12
    chain_trigger_radius_scale: float = 0.0
    chain_trigger_radius_min: float = 0.05
    chain_trigger_radius_max: float = 0.35
    chain_transition_time: float = 0.18
    chain_recovery_distance: float = 0.35
    chain_enable_recovery: bool = True
    chain_fit_regularization: float = 1e-4
    chain_fit_blend: float = 0.0 # this determines if the "original" A matrix should be blended
    chain_stabilization_margin: float = 1e-3

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
    demo_set = get_demonstration_set(config.dataset_path)
    plot_demonstration_set(demo_set, config, save_as='Demonstrations_Raw')

    # Fit a DS to each demonstration
    ds_set, reversed_ds_set, norm_demo_set = apply_lpvds_demowise(demo_set, config)
    plot_demonstration_set(norm_demo_set, config, save_as='Demonstrations_Norm')
    plot_ds_set_gaussians(ds_set, config, include_trajectory=True, save_as='Demonstrations_Gaussians')

    # Determine iteration strategy based on config
    init_attr_combinations = initialize_iter_strategy(config, demo_set)

    all_results = []
    for i, (initial, attractor) in enumerate(init_attr_combinations):
        print(f"Processing combination {i+1} of {len(init_attr_combinations)} #######################################")

        # Construct Gaussian Graph and Stitched DS
        print('Constructing Gaussian Graph and Stitched DS...')
        stitched_ds, gg, gg_solution_nodes, ds_stats = construct_stitched_ds(config, norm_demo_set, ds_set, reversed_ds_set, initial, attractor)

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
            if i == 0 and config.save_fig:
                plot_gaussian_graph(gg, config, save_as='Gaussian_Graph')
            if config.save_fig:
                plot_gg_solution(gg, gg_solution_nodes, config, save_as=f'{i}_Gaussian_Graph_Solution')
                plot_ds_set_gaussians([stitched_ds], config, include_trajectory=True, save_as=f'{i}_Stitched_DS_Gaussians')
                plot_ds(stitched_ds, simulated_trajectories, config, save_as=f'{i}_Stitched_DS_Simulation')

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
    
