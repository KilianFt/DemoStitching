import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.util import plot_tools
from src.stitching import build_ds
from src.stitching.metrics import save_results_dataframe, calculate_ds_metrics
from src.util.load_tools import get_demonstration_set
from src.util.benchmarking_tools import initialize_iter_strategy
from src.stitching.ds_stitching import construct_stitched_ds
from src.util.ds_tools import apply_lpvds_demowise
from src.util.plot_tools import plot_demonstration_set, plot_ds_set_gaussians, plot_gaussian_graph

# TODO
# - implement recalculating P?

# ds_method options:
# - ["recompute_all"] Recompute using shortest path
# - ["recompute_ds"] Recompute only DS
# - ["reuse"] Reuse A's from step 1 and only recompute them if they are invalid wrt P
# - ["chain"] Fit DS to each node and switch between attractors to reach goal

@dataclass
class Config:
    dataset_path: str = "./dataset/stitching/testing"
    force_preprocess: bool = True
    initial: Optional[np.ndarray] = None #np.array([4,15])
    attractor: Optional[np.ndarray] = None #np.array([14,2])
    ds_method: str = "all_paths_ds" # ["recompute_all", "recompute_ds", "reuse", "all_paths_all", "all_paths_ds", "chain"]
    reverse_gaussians: bool = True
    param_dist: int = 3
    param_cos: int = 3
    n_demos: int = 5 # number of demonstrations to generate
    noise_std: float = 0.05 # standard deviation of noise added to demonstrations
    plot_extent = (-2, 20, -2, 20) # (x_min, x_max, y_min, y_max)
    n_test_simulations: int = 2 # number of test simulations for metrics
    save_fig: bool = True

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
        simulated_trajectories.append(ds.sim(x_0[None, :], dt=0.01))

    return simulated_trajectories

def main():

    config = Config()
    save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"

    # Load/create a set of demonstrations
    demo_set = get_demonstration_set(config.dataset_path)
    plot_demonstration_set(demo_set, config, file_name='Demonstrations_Raw')

    # Fit a DS to each demonstration
    ds_set, norm_demo_set = apply_lpvds_demowise(demo_set)
    plot_demonstration_set(norm_demo_set, config, file_name='Demonstrations_Norm')
    plot_ds_set_gaussians(ds_set, config, include_points=True, file_name='Demonstrations_Gaussians')

    # Determine iteration strategy based on config
    combinations = initialize_iter_strategy(config, demo_set)

    all_results = []
    for i, (initial, attractor) in enumerate(combinations):
        print(f"Processing combination {i+1} of {len(combinations)} #######################################")

        # Construct Gaussian Graph and Stitched DS
        print('Constructing Gaussian Graph and Stitched DS...')
        stitched_ds, gg, ds_stats = construct_stitched_ds(config, norm_demo_set, ds_set, initial, attractor)
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

        # Compile and append results
        results = {'combination_id': i, 'ds_method': config.ds_method,} | ds_stats | ds_metrics
        all_results.append(results)

        # Plot
        if i == 0 and config.save_fig:
            plot_gaussian_graph(gg, config, bare=True, save_as='Gaussian_Graph')
        if config.save_fig:
            plot_tools.plot_gaussians_with_ds(gg, stitched_ds, simulated_trajectories, save_folder, i, config)
            plot_gaussian_graph(gg, config, save_as=f'gg_path_{i}')

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
        results_path = save_folder + "results.csv"
        save_results_dataframe(all_results, results_path)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
    