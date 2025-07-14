from src.lpvds_class import lpvds_class
from scipy.stats import multivariate_normal
import numpy as np


def build_ds(gg, data, attractor, ds_method, reverse_gaussians, rebuild_lpvds=False):
    if ds_method == "recompute":
        path_len = len(gg.shortest_path)
        gaussian_list = []

        x_att = attractor[None,:]
        all_x = np.vstack(data["xs"])
        all_x_dot = np.vstack(data["x_dots"])
        all_assignment_arr = np.hstack(data["assignment_arrs"])

        filtered_xs = []
        filtered_x_dots = []

        for node_id in gg.shortest_path[1:-1]:
            mu, sigma, direction = gg.get_gaussian(node_id)

            gaussian_list.append({   
                "prior" : 1 / path_len,
                "mu"    : mu,
                "sigma" : sigma,
                "rv"    : multivariate_normal(mu, sigma, allow_singular=True)
            })

            if reverse_gaussians and node_id >= gg.N:
                assign_id = node_id - gg.N
                x_dot_direction = -1
            else:
                assign_id = node_id
                x_dot_direction = 1

            node_xs = all_x[all_assignment_arr==assign_id]
            node_x_dots = x_dot_direction * all_x_dot[all_assignment_arr==assign_id]

            assert np.allclose(direction, node_x_dots.mean(axis=0)), f"Direction mismatch for node {node_id}"

            filtered_xs.append(node_xs)
            filtered_x_dots.append(node_x_dots)

        filtered_xs = np.vstack(filtered_xs)
        filtered_x_dots = np.vstack(filtered_x_dots)

        # compute DS
        lpvds = lpvds_class(filtered_xs, filtered_x_dots, x_att)
        if rebuild_lpvds:
            lpvds.begin()
        else:
            lpvds.init_cluster(gaussian_list)
            lpvds._optimize()
    else:
        raise NotImplementedError(f"Invalid ds_method: {ds_method}")

    return lpvds