import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from dataclasses import dataclass
from typing import Optional

from src.util import plot_tools
import graph_utils as gu
from src.stitching import build_ds
from src.stitching.utils import initial_goal_picker
from src.util.load_tools import load_data_from_file
from src.stitching.PlottingLib import plot_gaussians


# TODO
# - implement recalculating P?
# - calculate metrics for all methods
# - method where all gaussians point towards goal


# ds_method options:
# - ["recompute_all"] Recompute using shortest path
# - ["recompute_ds"] Recompute only DS
# - ["reuse"] Reuse A's from step 1 and only recompute them if they are invalid wrt P
# - ["chain"] Fit DS to each node and switch between attractors to reach goal

@dataclass
class Config:
    input_opt: Optional[int] = 2
    data_file: Optional[str] = "test_1" # this only matters for option 3
    initial: Optional[np.ndarray] = np.array([4,15]) # np.array([3,15])
    attractor: Optional[np.ndarray] = np.array([14,2]) # np.array([17,2])
    ds_method: str = "recompute_ds" # ["recompute_all", "recompute_ds", "reuse", "chain"]
    reverse_gaussians: bool = True
    param_dist: int = 3
    param_cos: int = 3
    x_min: float = -2
    x_max: float = 20
    y_min: float = -2
    y_max: float = 20
    save_fig: bool = True


def main():
    config = Config()

    input_message = '''
    Please choose a data input option:
    1. X - trajectory sets
    2. Intersecting trajectories
    3. Load drawn trajectories (from trajectory drawer)
    Enter the corresponding option number: '''
    if config.input_opt is None:
        config.input_opt = int(input(input_message))


    # 1) load data
    data, data_hash = load_data_from_file(config.input_opt, config.data_file)

    # pick start and goal
    # initial, attractor = initial_goal_picker(data)

    all_points = data["x_inits"] + data["x_atts"]
    n_iters = len(list(permutations(all_points, 2)))
    save_folder = f"./figures/stitching/{data_hash}/{config.ds_method}-2/"
    os.makedirs(save_folder, exist_ok=True)
    x_min, x_max, y_min, y_max = config.x_min, config.x_max, config.y_min, config.y_max

    # main loop to iter over all permutations of initial and attractor positions
    print("Number of combinations: {}".format(n_iters))
    for i, (initial, attractor) in enumerate([(config.initial, config.attractor)]):
    # for i, (initial, attractor) in enumerate(permutations(all_points, 2)):
        print("Processing combination {} of {}".format(i+1, n_iters))
        # build graph
        gg = gu.GaussianGraph(data["centers"], data["sigmas"], data["directions"],
                            attractor=attractor, initial=initial,
                            reverse_gaussians=config.reverse_gaussians, param_dist=config.param_dist, param_cos=config.param_cos)
        if i == 0:
            # initial plots that only need to be computed once
            fig, axs = plt.subplots(1, 1, figsize=(8,6), sharex=True, sharey=True)
            gg.plot(ax=axs)
            plt.savefig(save_folder + "graph.png")
            plt.close()
            fig, axs = plt.subplots(1, 1, figsize=(8,6), sharex=True, sharey=True)

            plot_gaussians(data["centers"], data["sigmas"], data["directions"], ax=axs)
            # plot_tools.plot_gaussians(data["centers"], data["sigmas"], data["directions"], ax=axs)
            plt.savefig(save_folder + "gaussians.png")
            plt.close()

        gg.compute_shortest_path()

        # build DS
        try:
            lpvds = build_ds(gg, data, attractor, config.ds_method, config.reverse_gaussians)
        except:
            print("Failed to build DS")
            continue

        # simulate
        x_inits = [initial+np.random.normal(0, 0.5, initial.shape[0]) for _ in range(2)]
        x_test_list = []
        for x_0 in x_inits:
            x_test_list.append(lpvds.sim(x_0[None,:], dt=0.01))

        # plot
        if config.save_fig:
            fig, axs = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True)

            #gg.plot_shortest_path_gaussians(ax=axs[0])
            mus, sigmas, directions = gg.get_gaussians(gg.shortest_path[1:-1])
            plot_gaussians(mus, sigmas, directions, ax=axs[0])
            plot_tools.plot_ds_2d(lpvds.x, x_test_list, lpvds, ax=axs[1], x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

            axs[1].set_xlim(x_min, x_max)
            axs[1].set_ylim(y_min, y_max)
            axs[0].set_xlim(x_min, x_max)
            axs[0].set_ylim(y_min, y_max)
            plt.tight_layout()
            plt.savefig(save_folder + "ds_{}.png".format(i))
            plt.close()


    # if hasattr(lpvds.damm, "Mu"):
    #     centers = lpvds.damm.Mu
    #     assignment_arr = lpvds.assignment_arr
    #     mean_xdot = np.zeros((lpvds.damm.K, lpvds.x.shape[1]))
    #     for k in range(lpvds.damm.K):
    #         mean_xdot[k] = np.mean(lpvds.x_dot[assignment_arr==k], axis=0)
    #     plot_tools.plot_gaussians(centers, lpvds.damm.Sigma, mean_xdot, ax=axs[0,1])


if __name__ == "__main__":
    main()
    