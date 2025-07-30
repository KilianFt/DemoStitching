from scipy.stats import multivariate_normal
import numpy as np
import time
from src.lpvds_class import lpvds_class
from src.dsopt.dsopt_class import dsopt_class
from src.util.benchmarking_tools import is_negative_definite
from src.stitching.optimization import compute_valid_A
from src.util.ds_tools import get_guassian_directions
import graph_utils as gu

def construct_stitched_ds(config, demo_set, ds_set, initial, attractor):
    """Constructs a stitched dynamical system based on configuration method.

    Args:
        config: Configuration object with ds_method attribute.
        demo_set: List of Demonstrations (normalized).
        ds_set: List of DSs, one for each demonstration in demo_set.
        initial: Initial point for the system.
        attractor: Target attractor point.

    Returns:
        tuple: (stitched_ds, gaussian_graph, timing_stats)

    Raises:
        NotImplementedError: For unsupported or invalid ds_method values.
    """
    if config.ds_method == 'recompute_all':
        return NEW_recompute_ds(ds_set, demo_set, initial, attractor, config, recompute_gaussians=True)
    elif config.ds_method == 'recompute_ds':
        return NEW_recompute_ds(ds_set, initial, attractor, config, recompute_gaussians=False)
    elif config.ds_method == 'reuse':
        raise NotImplementedError(f"Reuse method is not implemented yet.")
    elif config.ds_method == 'chain':
        raise NotImplementedError(f"Chain method is not implemented yet.")
    else:
        raise NotImplementedError(f"Invalid ds_method: {config.ds_method}")

def NEW_recompute_ds(ds_set, demo_set, initial, attractor, config, recompute_gaussians):
    """Constructs a stitched dynamical system by recomputing components along shortest path.

    Args:
        ds_set: Dataset with trajectory data and gaussian assignments.
        demo_set: List of Demonstrations (normalized).
        initial: Initial point for path planning.
        attractor: Target attractor point.
        config: Configuration object with DS parameters.
        recompute_gaussians: If True, recomputes gaussians and linear systems;
            if False, only recomputes linear systems.

    Returns:
        tuple: (stitched_ds, gaussian_graph, timing_stats)
    """
    # Initialize stats dictionary
    stats = dict()

    # ############## GAUSSIAN GRAPH ##############
    # Construct Gaussian Graph and compute the shortest path
    t0 = time.time()

    gaussians = {(i,j): {'mu': mu, 'sigma': sigma, 'direction': direction, 'prior': prior}
                 for i, ds in enumerate(ds_set)
                 for j, (mu, sigma, direction, prior) in enumerate(zip(ds.damm.Mu, ds.damm.Sigma, get_guassian_directions(ds), ds.damm.Prior))}
    gg = gu.GaussianGraph(gaussians,
                          attractor=attractor,
                          initial=initial,
                          reverse_gaussians=config.reverse_gaussians,
                          param_dist=config.param_dist,
                          param_cos=config.param_cos)
    gg.compute_shortest_path()

    stats['gg compute time'] = time.time() - t0

    # ############## DS ##############
    t0 = time.time()

    # Collect the gaussians along the shortest path
    priors = [gg.graph.nodes[node_id]['prior'] for node_id in gg.shortest_path[1:-1]]
    priors = [prior / sum(priors) for prior in priors]
    gaussians = []
    for i, node_id in enumerate(gg.shortest_path[1:-1]):
        mu, sigma, direction, prior = gg.get_gaussian(node_id)
        gaussians.append({
            'prior': priors[i],  # use normalized prior
            'mu': mu,
            'sigma': sigma,
            'rv': multivariate_normal(mu, sigma, allow_singular=True)
        })

    # collect the trajectory points that are assigned to the gaussians along the shortest path
    filtered_x = []
    filtered_x_dot = []
    for node_id in gg.shortest_path[1:-1]:

        ds_idx = node_id[0]
        gaussian_idx = node_id[1]

        assigned_x = ds_set[ds_idx].x[ds_set[ds_idx].assignment_arr == gaussian_idx]
        assigned_x_dot = ds_set[ds_idx].x_dot[ds_set[ds_idx].assignment_arr == gaussian_idx]

        # reverse velocity if gaussian is reversed
        assigned_x_dot = -assigned_x_dot if node_id in gg.gaussian_reversal_map else assigned_x_dot

        filtered_x.append(assigned_x)
        filtered_x_dot.append(assigned_x_dot)

    filtered_x = np.vstack(filtered_x)
    filtered_x_dot = np.vstack(filtered_x_dot)

    # compute DS
    x_att = attractor[None,:]
    try:
        stitched_ds = lpvds_class(filtered_x, filtered_x_dot, x_att)
        if recompute_gaussians:     # compute new gaussians and linear systems (As)
            stitched_ds.begin()
        else:                       # compute only linear systems (As)
            stitched_ds.init_cluster(gaussians)
            stitched_ds._optimize()

    except Exception as e:
        print(f'Failed to construct Stitched DS: {e}')
        stitched_ds = None

    stats['ds compute time'] = time.time() - t0
    stats['total compute time'] = time.time() - t0

    return stitched_ds, gg, stats

def NEW_reuse_ds(gg, ds_set, attractor, reverse_gaussians):

    # Collect useful data
    path_len = len(gg.shortest_path)
    x_att = attractor[None,:]
    all_x = ds_set["x_sets"] # [D, M, N]
    all_x_dot = ds_set["x_dot_sets"] # [D, M, N]
    all_assignment_arr = ds_set["assignment_arrs"] # [D, M]

    # Select P from demo with closest attractor to current attractor
    # TODO is this fine? why not negative P?
    P = None
    min_dist = np.inf
    for i, potential_x_att in enumerate(ds_set["x_attrator_sets"]):
        dist = np.linalg.norm(potential_x_att - attractor)
        if dist < min_dist:
            min_dist = dist
            P = ds_set["Ps"][i]


    As = []
    gaussian_list = []
    filtered_xs = []
    filtered_x_dots = []
    for node_id in gg.shortest_path[1:-1]:

        # Get Gaussian, determine if reverse gaussians
        mu, sigma, direction = gg.get_gaussian(node_id)
        if reverse_gaussians and node_id >= gg.n_gaussians:
            assign_id = node_id - gg.n_gaussians
            x_dot_direction = -1
        else:
            assign_id = node_id
            x_dot_direction = 1

        # Collect A, x, x_dot (reversed if needed)
        node_A = x_dot_direction * ds_set["As"][assign_id]
        node_x = all_x[all_assignment_arr==assign_id]
        node_x_dot = x_dot_direction * all_x_dot[all_assignment_arr==assign_id]

        # check if A is valid with respect to P
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

def build_ds(gg, ds_set, attractor, ds_method, reverse_gaussians):
    if ds_method == "recompute_all":
        lpvds = recompute_ds(gg, ds_set, attractor, reverse_gaussians, rebuild_lpvds=True)
    elif ds_method == "recompute_ds":
        lpvds = recompute_ds(gg, ds_set, attractor, reverse_gaussians, rebuild_lpvds=False)
    elif ds_method == "reuse":
        lpvds = reuse_ds(gg, ds_set, attractor, reverse_gaussians)
    elif ds_method == "chain":
        raise NotImplementedError(f"Chain method is not implemented yet.")
    else:
        raise NotImplementedError(f"Invalid ds_method: {ds_method}")

    return lpvds

def reuse_ds(gg, ds_set, attractor, reverse_gaussians):

    # Collect useful data
    path_len = len(gg.shortest_path)
    x_att = attractor[None,:]
    all_x = ds_set["x_sets"] # [D, M, N]
    all_x_dot = ds_set["x_dot_sets"] # [D, M, N]
    all_assignment_arr = ds_set["assignment_arrs"] # [D, M]

    # Select P from demo with closest attractor to current attractor
    # TODO is this fine? why not negative P?
    P = None
    min_dist = np.inf
    for i, potential_x_att in enumerate(ds_set["x_attrator_sets"]):
        dist = np.linalg.norm(potential_x_att - attractor)
        if dist < min_dist:
            min_dist = dist
            P = ds_set["Ps"][i]


    As = []
    gaussian_list = []
    filtered_xs = []
    filtered_x_dots = []
    for node_id in gg.shortest_path[1:-1]:

        # Get Gaussian, determine if reverse gaussians
        mu, sigma, direction = gg.get_gaussian(node_id)
        if reverse_gaussians and node_id >= gg.n_gaussians:
            assign_id = node_id - gg.n_gaussians
            x_dot_direction = -1
        else:
            assign_id = node_id
            x_dot_direction = 1

        # Collect A, x, x_dot (reversed if needed)
        node_A = x_dot_direction * ds_set["As"][assign_id]
        node_x = all_x[all_assignment_arr==assign_id]
        node_x_dot = x_dot_direction * all_x_dot[all_assignment_arr==assign_id]

        # check if A is valid with respect to P
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

def recompute_ds(gg, ds_set, attractor, reverse_gaussians, rebuild_lpvds):

    path_len = len(gg.shortest_path)
    gaussian_list = []

    x_att = attractor[None,:]
    all_x = ds_set["x_sets"]
    all_x_dot = ds_set["x_dot_sets"]
    all_assignment_arr = ds_set["assignment_arrs"]

    # Extract demonstration points and gaussians for only the points in the shortest path
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

        if reverse_gaussians and node_id >= gg.n_gaussians:
            assign_id = node_id - gg.n_gaussians
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
    if rebuild_lpvds:  # compute new guassians and linear systems (As)
        lpvds.begin()
    else:              # compute only linear systems (As)
        lpvds.init_cluster(gaussian_list)
        lpvds._optimize()

    return lpvds
