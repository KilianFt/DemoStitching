from scipy.stats import multivariate_normal
import numpy as np

from src.lpvds_class import lpvds_class
from src.dsopt.dsopt_class import dsopt_class
from src.util.stitching import is_negative_definite
from src.stitching.optimization import compute_valid_A


def build_ds(gg, data, attractor, ds_method, reverse_gaussians):
    if ds_method == "recompute_all":
        lpvds = recompute_ds(gg, data, attractor, reverse_gaussians, rebuild_lpvds=True)
    elif ds_method == "recompute_ds":
        lpvds = recompute_ds(gg, data, attractor, reverse_gaussians, rebuild_lpvds=False)
    elif ds_method == "reuse":
        lpvds = reuse_ds(gg, data, attractor, reverse_gaussians)
    elif ds_method == "chain":
        raise NotImplementedError(f"Invalid ds_method: {ds_method}")
    else:
        raise NotImplementedError(f"Invalid ds_method: {ds_method}")

    return lpvds


def reuse_ds(gg, data, attractor, reverse_gaussians):
    # get A's that are in shortest path
    path_len = len(gg.shortest_path)
    x_att = attractor[None,:]
    all_x = data["x_sets"] # [D, M, N]
    all_x_dot = data["x_dot_sets"] # [D, M, N]
    all_assignment_arr = data["assignment_arrs"] # [D, M]

    # select P
    # TODO is this fine? why not negative P?
    P = None
    min_dist = np.inf
    for i, potential_x_att in enumerate(data["x_attrator_sets"]):
        dist = np.linalg.norm(potential_x_att - attractor)
        if dist < min_dist:
            min_dist = dist
            P = data["Ps"][i]

    As = []
    gaussian_list = []
    filtered_xs = []
    filtered_x_dots = []

    for node_id in gg.shortest_path[1:-1]:
        mu, sigma, direction = gg.get_gaussian(node_id)
        if reverse_gaussians and node_id >= gg.N:
            assign_id = node_id - gg.N
            x_dot_direction = -1
        else:
            assign_id = node_id
            x_dot_direction = 1

        node_A = x_dot_direction * data["As"][assign_id]
        node_x = all_x[all_assignment_arr==assign_id]
        node_x_dot = x_dot_direction * all_x_dot[all_assignment_arr==assign_id]

        # check if they are valid with respect to P
        valid_A = is_negative_definite(node_A + np.transpose(node_A))
        valid_wrt_p = is_negative_definite(np.transpose(node_A) @ P + P @ node_A)
        if not valid_A or not valid_wrt_p:
            updated_A = compute_valid_A(node_A, P, node_x, x_att, node_x_dot)
            node_A = updated_A

            valid_A = is_negative_definite(updated_A + np.transpose(updated_A))
            valid_wrt_p = is_negative_definite(np.transpose(updated_A) @ P + P @ updated_A)
            assert valid_A and valid_wrt_p, "Updated A is not valid"

        gaussian_list.append({   
            "prior" : 1 / path_len,
            "mu"    : mu,
            "sigma" : sigma,
            "rv"    : multivariate_normal(mu, sigma, allow_singular=True)
        })
        As.append(node_A)

        filtered_xs.append(node_x)
        filtered_x_dots.append(node_x_dot)

    filtered_xs = np.vstack(filtered_xs)
    filtered_x_dots = np.vstack(filtered_x_dots)
    lpvds = lpvds_class(filtered_xs, filtered_x_dots, x_att)
    lpvds.init_cluster(gaussian_list)
    lpvds.A = np.array(As)
    lpvds.ds_opt = dsopt_class(lpvds.x, lpvds.x_dot, lpvds.x_att, lpvds.gamma, lpvds.assignment_arr)
    lpvds.ds_opt.P = P

    return lpvds


def recompute_ds(gg, data, attractor, reverse_gaussians, rebuild_lpvds):
    path_len = len(gg.shortest_path)
    gaussian_list = []

    x_att = attractor[None,:]
    all_x = data["x_sets"]
    all_x_dot = data["x_dot_sets"]
    all_assignment_arr = data["assignment_arrs"]

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

    return lpvds