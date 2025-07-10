import os
import json

import numpy as np
import matplotlib.pyplot as plt
from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class


def calculate_nodes(input_opt, plot_results=False):
    # data list with each entry containing x, x_dot, x_att, x_init
    data = load_tools.load_data_stitch(int(input_opt))

    if plot_results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    node_centers = []
    node_directions = []
    for i in range(len(data)):
        x, x_dot, x_att, x_init = data[i]
        lpvds = lpvds_class(x, x_dot, x_att)
        lpvds._cluster()

        # get directionality
        centers = lpvds.damm.Mu
        assignment_arr = lpvds.assignment_arr
        # get mean xdot per cluster
        # NOTE could be modified to have weighted average
        mean_xdot = np.zeros((lpvds.damm.K, x.shape[1]))
        for k in range(lpvds.damm.K):
            mean_xdot[k] = np.mean(x_dot[assignment_arr==k], axis=0)

        for k in range(lpvds.damm.K):
            sigma = lpvds.damm.Sigma[k, :, :]
            mu = lpvds.damm.Mu[k, :]
            if plot_results:
                plot_tools.plot_2d_gaussian(mu, sigma, ax = ax)
        
        # plot mean xdot per cluster
        if plot_results:
            for k in range(lpvds.damm.K):
                ax.arrow(centers[k, 0],
                        centers[k, 1],
                        mean_xdot[k, 0],
                        mean_xdot[k, 1],
                        head_width=0.5,
                        head_length=0.5,
                        fc='r',
                        ec='r',
                        zorder=10
                        )

        node_centers.extend(centers.tolist())
        node_directions.extend(mean_xdot.tolist())

        # plot original data if wanted
        # ax.scatter(x[:, 0], x[:, 1], color='k', s=5, label='original data')

    if plot_results:
        plt.tight_layout()
        plt.show()  

    return node_centers, node_directions


def maybe_calculate_nodes(input_opt):
    filename = "./dataset/stitching/nodes_{}.json".format(input_opt)
    if os.path.exists(filename):
        print("Using cached nodes")
        with open(filename, 'r') as f:
            node_centers, node_directions = json.load(f)
    else:
        print("Calculating nodes")
        node_centers, node_directions = calculate_nodes(input_opt)
        with open(filename, 'w') as f:
            json.dump([node_centers, node_directions], f)
    return node_centers, node_directions

def main():
    input_message = '''
    Please choose a data input option:
    1. X - trajectory sets
    2. Intersecting trajectories
    Enter the corresponding option number: '''
    input_opt  = input(input_message)

    # 1) get GMM centers and directionality
    node_centers, node_directions = maybe_calculate_nodes(input_opt)

    # 2) build graph
    print(node_centers)
    print(node_directions)


if __name__ == "__main__":
    main()
    