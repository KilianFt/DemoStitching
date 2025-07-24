import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.util import plot_tools
import graph_utils as gu
from src.stitching import build_ds
from src.stitching.metrics import calculate_all_metrics, save_results_dataframe
from src.util.load_tools import load_data_from_file
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
    force_preprocess: bool = True
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



def main():
    config = Config()

    # load data
    data = load_data_from_file(config.dataset_path, config.n_demos, config.noise_std, config.force_preprocess)

    # determine iteration strategy based on config
    combinations, save_folder = initialize_iter_strategy(config, data["x_initial_sets"], data["x_attrator_sets"])
    n_iters = len(combinations)
    
    all_results = []
    for i, (initial, attractor) in enumerate(combinations):
        print("Processing combination {} of {}".format(i+1, n_iters))

        # build graph
        gg = gu.GaussianGraph(data["centers"],
                              data["sigmas"],
                              data["directions"],
                              attractor=attractor,
                              initial=initial,
                              reverse_gaussians=config.reverse_gaussians,
                              param_dist=config.param_dist,
                              param_cos=config.param_cos)
        
        if i == 0 and config.save_fig:
            plot_tools.save_initial_plots(gg, data, save_folder)

        gg.compute_shortest_path()

        # build ds
        try:
            lpvds = build_ds(gg, data, attractor, config.ds_method, config.reverse_gaussians)
        except:
            print("Failed to build DS")
            nan_result = get_nan_results(i, config.ds_method, initial, attractor)
            all_results.append(nan_result)
            continue

        # simulate
        # TODO figure out initial/final position from data
        x_inits = [initial+np.random.normal(0, config.noise_std, initial.shape[0]) for _ in range(config.n_test_simulations)]
        x_test_list = []
        for x_0 in x_inits:
            x_test_list.append(lpvds.sim(x_0[None,:], dt=0.01))

        # Calculate metrics
        try:
            metrics_result = calculate_all_metrics(
                x_ref=lpvds.x,
                x_dot_ref=lpvds.x_dot,
                lpvds=lpvds,
                x_test_list=x_test_list,
                initial=initial,
                attractor=attractor,
                ds_method=config.ds_method,
                combination_id=i
            )
            all_results.append(metrics_result)
            
            print(f"  Metrics calculated: RMSE={metrics_result['prediction_rmse']:.4f}, "
                  f"Cosine={metrics_result['cosine_dissimilarity']:.4f}, "
                  f"DTW={metrics_result['dtw_distance_mean']:.4f}±{metrics_result['dtw_distance_std']:.4f}")
        except Exception as e:
            print(f"  Failed to calculate metrics: {e}")

        # plot
        if config.save_fig:
            plot_tools.plot_gaussians_with_ds(gg, lpvds, x_test_list, save_folder, i, config)

    # Save results dataframe
    if all_results:
        results_path = save_folder + "results.csv"
        save_results_dataframe(all_results, results_path)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
    