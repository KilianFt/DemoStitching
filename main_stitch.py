import os
import json

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class

import graph_utils as gu


def calculate_gaussians(input_opt):
    # data list with each entry containing x, x_dot, x_att, x_init
    data = load_tools.load_data_stitch(int(input_opt))
    node_centers = []
    node_sigmas = []
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

        node_centers.extend(centers.tolist())
        node_directions.extend(mean_xdot.tolist())
        node_sigmas.extend(lpvds.damm.Sigma.tolist())

    return np.array(node_centers), np.array(node_directions), np.array(node_sigmas)


def plot_gaussians(gaussian_mu, gaussian_sigma, gaussian_direction):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for k in range(len(gaussian_mu)):
        mu = gaussian_mu[k]
        sigma = gaussian_sigma[k]
        plot_tools.plot_2d_gaussian(mu, sigma, ax = ax)    

        ax.arrow(gaussian_mu[k, 0],
                 gaussian_mu[k, 1],
                 gaussian_direction[k, 0],
                 gaussian_direction[k, 1],
                 head_width=0.5,
                 head_length=0.5,
                 fc='r',
                 ec='r',
                 zorder=10
                 )

    plt.tight_layout()
    plt.show()  


def load_gaussians_from_file(input_opt):
    filename = "./dataset/stitching/nodes_{}.json".format(input_opt)
    if os.path.exists(filename):
        print("Using cached nodes")
        with open(filename, 'r') as f:
            node_centers, node_directions, node_sigmas = json.load(f)
            node_centers = np.array(node_centers)
            node_directions = np.array(node_directions)
            node_sigmas = np.array(node_sigmas)
    else:
        print("Calculating nodes")
        node_centers, node_directions, node_sigmas = calculate_gaussians(input_opt)
        with open(filename, 'w') as f:
            json.dump([node_centers.tolist(), node_directions.tolist(), node_sigmas.tolist()], f)
    return node_centers, node_sigmas, node_directions


def main():
    input_message = '''
    Please choose a data input option:
    1. X - trajectory sets
    2. Intersecting trajectories
    Enter the corresponding option number: '''
    input_opt  = input(input_message)

    # 1) get GMM centers and directionality
    gaussian_mu, gaussian_sigma, gaussian_direction = load_gaussians_from_file(input_opt)
    plot_gaussians(gaussian_mu, gaussian_sigma, gaussian_direction)

    # 2) build graph
    initial = np.array([3,15])
    attractor = np.array([17,2])
    gg = gu.GaussianGraph(gaussian_mu, gaussian_sigma, gaussian_direction,
                          attractor=attractor, initial=initial,
                          reverse_gaussians=True, param_dist=3, param_cos=2)
    gg.compute_shortest_path()
    print(gg.shortest_path)

    gg.plot()


if __name__ == "__main__":
    main()
    