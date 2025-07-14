import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.util import plot_tools
import graph_utils as gu
import src.stitching as st


def load_gaussians_from_file(input_opt):
    filename = "./dataset/stitching/nodes_{}.pkl".format(input_opt)
    if os.path.exists(filename):
        print("Using cached nodes")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Calculating nodes")
        data = st.calculate_gaussians(input_opt)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    return data


def main():
    input_message = '''
    Please choose a data input option:
    1. X - trajectory sets
    2. Intersecting trajectories
    Enter the corresponding option number: '''
    input_opt  = input(input_message)

    # --------- config ---------
    # problem setup
    initial = np.array([4,15]) # np.array([3,15])
    attractor = np.array([14,2]) # np.array([17,2])
    # initial = np.array([0,0])
    # attractor = np.array([14,2])

    # LPVDS parameters
    # gmm_variabt = "PC-GMM" TODO
    ds_method = "recompute" # options: ["recompute", "reuse", "switch"]
    reverse_gaussians = True # if True, duplicate gaussians and reverse directions
    rebuild_lpvds = False # if True, recompute clustering from scratch instead of using existing gaussians
    
    # graph parameters
    param_dist = 3 # parameter for distance
    param_cos = 2 # parameter for directionality



    # --------- main ---------
    # 1) get GMM centers and directionality
    data = load_gaussians_from_file(input_opt)
    # plot_tools.plot_gaussians(data["centers"], data["sigmas"], data["directions"])


    # 2) build graph
    gg = gu.GaussianGraph(data["centers"], data["sigmas"], data["directions"],
                          attractor=attractor, initial=initial,
                          reverse_gaussians=reverse_gaussians, param_dist=param_dist, param_cos=param_cos)
    gg.compute_shortest_path()
    gg.plot()
    # gg.plot_shortest_path_gaussians()


    # 3) build DS
    # Options:
    # - ["recompute"] Recompute using shortest path
    # - ["reuse"] Calculate A's in step 1 and reuse them. Check GAS in this case.
    # - ["switch"] Fit DS to each node and switch between attractors to reach goal
    lpvds = st.build_ds(gg, data, attractor, ds_method, reverse_gaussians, rebuild_lpvds=rebuild_lpvds)


    # 4) simulate
    x_inits = [initial+np.random.normal(0, 0.5, initial.shape[0]) for _ in range(3)]
    x_test_list = []
    for x_0 in x_inits:
        x_test_list.append(lpvds.sim(x_0[None,:], dt=0.01))

    plot_tools.plot_ds_2d(lpvds.x, x_test_list, lpvds)
    plt.show()


if __name__ == "__main__":
    main()
    