import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.util import plot_tools
import graph_utils as gu
from src.stitching import build_ds
from src.stitching.metrics import calculate_all_metrics, save_results_dataframe
from src.util.load_tools import get_ds_set
from src.util.stitching import initialize_iter_strategy, get_nan_results

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
    force_preprocess: bool = False
    initial: Optional[np.ndarray] = None #np.array([4,15])
    attractor: Optional[np.ndarray] = None #np.array([14,2])
    ds_method: str = "recompute_all" # ["recompute_all", "recompute_ds", "reuse", "chain"]
    reverse_gaussians: bool = True
    param_dist: int = 3
    param_cos: int = 3
    n_demos: int = 5 # number of demonstrations to generate
    noise_std: float = 0.05 # standard deviation of noise added to demonstrations
    x_min: float = -2
    x_max: float = 20
    y_min: float = -2
    y_max: float = 20
    n_test_simulations: int = 2 # number of test simulations for metrics
    save_fig: bool = True

def build_stitched_ds(ds_set, initial, attractor, config):

    # build the gaussian graph
    t0 = time.time()
    gg = gu.GaussianGraph(ds_set["centers"],
                          ds_set["sigmas"],
                          ds_set["directions"],
                          attractor=attractor,
                          initial=initial,
                          reverse_gaussians=config.reverse_gaussians,
                          param_dist=config.param_dist,
                          param_cos=config.param_cos)
    gg.compute_shortest_path()
    gg_time = time.time() - t0

    # build the DS
    try:
        t0 = time.time()
        ds = build_ds(gg, ds_set, attractor, config.ds_method, config.reverse_gaussians)
        ds_time = time.time() - t0
    except:
        ds = None
        ds_time = None

    stats = {'gg time': gg_time, 'ds time': ds_time, 'total time': gg_time + (ds_time if ds_time is not None else 0)}

    return gg, ds, stats

def simulate_trajectories(ds, initial, config):

    # --- simulate
    # TODO figure out initial/final position from data
    x_inits = [initial + np.random.normal(0, config.noise_std, initial.shape[0]) for _ in
               range(config.n_test_simulations)]
    simulation_positions = []
    for x_0 in x_inits:
        simulation_positions.append(ds.sim(x_0[None, :], dt=0.01))

    return simulation_positions

def main():

    config = Config()
    save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
    # os.makedirs(save_folder, exist_ok=True)

    # load set of DSs
    ds_set = get_ds_set(config)

    # determine iteration strategy based on config
    combinations = initialize_iter_strategy(config, ds_set["x_initial_sets"], ds_set["x_attrator_sets"])
    n_iters = len(combinations)
    
    all_results = []
    for i, (initial, attractor) in enumerate(combinations):

        # Start timing for this iteration
        iteration_start_time = time.time()
        print("Processing combination {} of {}".format(i+1, n_iters))

        # Build gaussian graph and stitched DS
        gg, stitched_ds, stats = build_stitched_ds(ds_set, initial, attractor, config)

        # If building the Stitched DS failed, return NaN metrics
        if stitched_ds is None:
            print("Failed to build DS")
            iteration_time = time.time() - iteration_start_time
            nan_result = get_nan_results(i, config.ds_method, initial, attractor)
            nan_result['compute_time'] = iteration_time
            all_results.append(nan_result)
            print(f"  Failed iteration completed in {iteration_time:.3f}s")
            continue

        # Building the Stitched DS was successful
        print(f'  Gaussian Graph and Stitched DS built in {stats["total time"]:.2f}s')

        # Simulate some trajectories
        simulation_positions = simulate_trajectories(stitched_ds, initial, config)

        # Compute metrics
        try:
            metrics_result = calculate_all_metrics(
                x_ref=stitched_ds.x,
                x_dot_ref=stitched_ds.x_dot,
                lpvds=stitched_ds,
                x_test_list=simulation_positions,
                initial=initial,
                attractor=attractor,
                ds_method=config.ds_method,
                combination_id=i
            )
            iteration_time = time.time() - iteration_start_time
            metrics_result['compute_time'] = iteration_time

            all_results.append(metrics_result)

            print(f"  Metrics calculated: RMSE={metrics_result['prediction_rmse']:.4f}, "
                  f"Cosine={metrics_result['cosine_dissimilarity']:.4f}, "
                  f"DTW={metrics_result['dtw_distance_mean']:.4f}±{metrics_result['dtw_distance_std']:.4f}, "
                  f"Time={iteration_time:.3f}s")
        except Exception as e:
            print(f"  Failed to calculate metrics: {e}")

        # Plot
        if i == 0 and config.save_fig:
            plot_tools.save_initial_plots(gg, ds_set, save_folder, config)
        if config.save_fig:
            plot_tools.plot_gaussians_with_ds(gg, stitched_ds, simulation_positions, save_folder, i, config)

    # Save Results to CSV
    if all_results:
        results_path = save_folder + "results.csv"
        save_results_dataframe(all_results, results_path)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
    