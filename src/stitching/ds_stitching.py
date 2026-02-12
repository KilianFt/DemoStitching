from scipy.stats import multivariate_normal
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.lpvds_class import lpvds_class
from src.dsopt.dsopt_class import dsopt_class
from src.util.benchmarking_tools import is_negative_definite
from src.stitching.optimization import compute_valid_A, find_lyapunov_function
from src.stitching.chaining import build_chained_ds
from src.util.ds_tools import get_gaussian_directions
import graph_utils as gu


def construct_stitched_ds(config, norm_demo_set, ds_set, reversed_ds_set, initial, attractor):
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
    if config.ds_method == 'sp_recompute_all':
        return recompute_ds(ds_set, initial, attractor, config, recompute_gaussians=True)
    elif config.ds_method == 'sp_recompute_ds':
        return recompute_ds(ds_set, initial, attractor, config, recompute_gaussians=False)
    elif config.ds_method == 'sp_recompute_invalid_As':
        return reuse_ds(ds_set, reversed_ds_set, initial, attractor, config)
    elif config.ds_method == 'sp_recompute_P':
        return reuse_A(ds_set, initial, attractor, config)
    elif config.ds_method == 'spt_recompute_all':
        return all_paths(ds_set, attractor, config, recompute_gaussians=True)
    elif config.ds_method == 'spt_recompute_ds':
        return all_paths(ds_set, attractor, config, recompute_gaussians=False)
    elif config.ds_method == 'spt_recompute_invalid_As':
        return all_paths_reuse(ds_set, reversed_ds_set, initial, attractor, config)
    elif config.ds_method == 'chain':
        return chain_ds(ds_set, initial, attractor, config)
    else:
        raise NotImplementedError(f"Invalid ds_method: {config.ds_method}")

def chain_ds(ds_set, initial, attractor, config):
    """Builds a chained DS that tracks a shortest Gaussian-graph path by switching targets."""
    stats = dict()

    # ############## GAUSSIAN GRAPH ##############
    t0 = time.time()
    gaussians = {
        (i, j): {'mu': mu, 'sigma': sigma, 'direction': direction, 'prior': prior}
        for i, ds in enumerate(ds_set)
        for j, (mu, sigma, direction, prior) in enumerate(
            zip(ds.damm.Mu, ds.damm.Sigma, get_gaussian_directions(ds), ds.damm.Prior)
        )
    }
    gg = gu.GaussianGraph(param_dist=config.param_dist, param_cos=config.param_cos)
    gg.add_gaussians(gaussians, reverse_gaussians=config.reverse_gaussians)
    gg_solution_nodes = gg.shortest_path(initial, attractor)
    stats['gg compute time'] = time.time() - t0

    # ############## DS ##############
    t_ds = time.time()
    try:
        stitched_ds = build_chained_ds(ds_set, gg, gg_solution_nodes, initial=initial, attractor=attractor, config=config)
    except Exception as e:
        print(f'Failed to construct Chained DS: {e}')
        stitched_ds = None

    stats['ds compute time'] = time.time() - t_ds
    stats['total compute time'] = time.time() - t0
    return stitched_ds, gg, gg_solution_nodes, stats

def recompute_ds(ds_set, initial, attractor, config, recompute_gaussians):
    """Builds a stitched dynamical system by following the shortest path through a Gaussian graph.

    Args:
        ds_set: List of DS objects, each with Gaussian mixture parameters and trajectory data.
        initial: Initial point for the system.
        attractor: Target attractor point.
        config: Configuration object specifying parameters for graph construction and DS computation.
        recompute_gaussians: If True, re-estimates both Gaussians and dynamics; if False, only dynamics.

    Returns:
        tuple: (stitched_ds, gg, stats) where
            - stitched_ds: Learned stitched DS object, or None on failure.
            - gg: Constructed GaussianGraph object along the path.
            - stats: Dictionary with timing information for major steps.
    """
    # Initialize stats dictionary
    stats = dict()

    # ############## GAUSSIAN GRAPH ##############
    t0 = time.time()
    gaussians = {(i,j): {'mu': mu, 'sigma': sigma, 'direction': direction, 'prior': prior}
                 for i, ds in enumerate(ds_set)
                 for j, (mu, sigma, direction, prior) in enumerate(zip(ds.damm.Mu, ds.damm.Sigma, get_gaussian_directions(ds), ds.damm.Prior))}
    gg = gu.GaussianGraph(param_dist=config.param_dist, param_cos=config.param_cos)
    gg.add_gaussians(gaussians, reverse_gaussians=config.reverse_gaussians)
    gg_solution_nodes = gg.shortest_path(initial, attractor)
    stats['gg compute time'] = time.time() - t0

    # ############## DS ##############
    t_ds = time.time()

    # Collect the gaussians along the shortest path
    priors = [gg.graph.nodes[node_id]['prior'] for node_id in gg_solution_nodes]
    priors = [prior / sum(priors) for prior in priors]
    gaussians = []
    for i, node_id in enumerate(gg_solution_nodes):
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
    for node_id in gg_solution_nodes:

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
    x_att = attractor
    try:
        stitched_ds = lpvds_class(filtered_x, filtered_x_dot, x_att,
                                  rel_scale=getattr(config, 'rel_scale', 0.7),
                                  total_scale=getattr(config, 'total_scale', 1.5),
                                  nu_0=getattr(config, 'nu_0', 5),
                                  kappa_0=getattr(config, 'kappa_0', 1),
                                  psi_dir_0=getattr(config, 'psi_dir_0', 1))
        if recompute_gaussians:     # compute new gaussians and linear systems (As)
            result = stitched_ds.begin()
            if not result:
                print('Failed to construct Stitched DS: DAMM clustering failed')
                stitched_ds = None
        else:                       # compute only linear systems (As)
            stitched_ds.init_cluster(gaussians)
            stitched_ds._optimize()

    except Exception as e:
        print(f'Failed to construct Stitched DS: {e}')
        stitched_ds = None

    stats['ds compute time'] = time.time() - t_ds
    stats['total compute time'] = time.time() - t0

    return stitched_ds, gg, gg_solution_nodes, stats

def _process_node(args):
    """Helper function to process a single node in parallel.
    
    Args:
        args: Tuple containing (node_id, ds_set, gg, P, attractor)
        
    Returns:
        Tuple containing (node_A, assigned_x, assigned_x_dot)
    """
    node_id, ds_set, gg, P, attractor = args
    
    ds_idx = node_id[0]
    gaussian_idx = node_id[1]

    assigned_x = ds_set[ds_idx].x[ds_set[ds_idx].assignment_arr == gaussian_idx]
    assigned_x_dot = ds_set[ds_idx].x_dot[ds_set[ds_idx].assignment_arr == gaussian_idx]

    # reverse velocity if gaussian is reversed
    assigned_x_dot = -assigned_x_dot if node_id in gg.gaussian_reversal_map else assigned_x_dot
    node_A = ds_set[ds_idx].A[gaussian_idx] * (-1 if node_id in gg.gaussian_reversal_map else 1)

    # check if A is valid with respect to P
    valid_A = is_negative_definite(node_A + np.transpose(node_A))
    valid_wrt_p = is_negative_definite(np.transpose(node_A) @ P + P @ node_A)
    if not valid_A or not valid_wrt_p:
        updated_A = compute_valid_A(node_A, P, assigned_x, attractor, assigned_x_dot)
        node_A = updated_A

        valid_A = is_negative_definite(updated_A + np.transpose(updated_A))
        valid_wrt_p = is_negative_definite(np.transpose(updated_A) @ P + P @ updated_A)
        assert valid_A and valid_wrt_p, "Updated A is not valid"

    return node_A, assigned_x, assigned_x_dot

def reuse_ds(ds_set, reversed_ds_set, initial, attractor, config):
    # Initialize stats dictionary
    stats = dict()

    # ############## GAUSSIAN GRAPH ##############
    t0 = time.time()
    gaussians = {(i, j): {'mu': mu, 'sigma': sigma, 'direction': direction, 'prior': prior}
                 for i, ds in enumerate(ds_set)
                 for j, (mu, sigma, direction, prior) in
                 enumerate(zip(ds.damm.Mu, ds.damm.Sigma, get_gaussian_directions(ds), ds.damm.Prior))}
    gg = gu.GaussianGraph(param_dist=config.param_dist, param_cos=config.param_cos)
    gg.add_gaussians(gaussians, reverse_gaussians=config.reverse_gaussians)
    gg_solution_nodes = gg.shortest_path(initial, attractor)
    stats['gg compute time'] = time.time() - t0

    # ############## DS ##############
    t_ds = time.time()

    # Select P from demo with closest attractor to current attractor
    P = None
    min_dist = np.inf
    for i, ds_class in enumerate(ds_set + reversed_ds_set):
        dist = np.linalg.norm(ds_class.x_att - attractor)
        if dist < min_dist:
            min_dist = dist
            P = ds_class.ds_opt.P

    if min_dist > 1:
        print("WARN: reuse did not find good attractor")

    # Collect the gaussians along the shortest path
    priors = [gg.graph.nodes[node_id]['prior'] for node_id in gg_solution_nodes]
    priors = [prior / sum(priors) for prior in priors]
    gaussians = []
    for i, node_id in enumerate(gg_solution_nodes):
        mu, sigma, direction, prior = gg.get_gaussian(node_id)
        gaussians.append({
            'prior': priors[i],  # use normalized prior
            'mu': mu,
            'sigma': sigma,
            'rv': multivariate_normal(mu, sigma, allow_singular=True)
        })

    # collect the trajectory points that are assigned to the gaussians along the shortest path (optimized)
    nodes_to_process = gg_solution_nodes
    
    # Prepare arguments for processing
    process_args = [(node_id, ds_set, gg, P, attractor) for node_id in nodes_to_process]
    
    # Process nodes - optimized to avoid hanging
    As = []
    filtered_x = []
    filtered_x_dot = []
    
    # Check if parallel processing is worth it (overhead vs benefit)
    if len(process_args) <= 2:
        # For small number of nodes, sequential processing is faster
        results = [_process_node(args) for args in process_args]
    else:
        # Use executor.map() which is more efficient and avoids hanging
        with ProcessPoolExecutor(max_workers=min(len(process_args), 4)) as executor:
            results = list(executor.map(_process_node, process_args))
    
    # Extract results in the correct order
    for node_A, assigned_x, assigned_x_dot in results:
        As.append(node_A)
        filtered_x.append(assigned_x)
        filtered_x_dot.append(assigned_x_dot)

    filtered_x = np.vstack(filtered_x)
    filtered_x_dot = np.vstack(filtered_x_dot)
    As = np.array(As)

    lpvds = lpvds_class(filtered_x, filtered_x_dot, attractor,
                        rel_scale=getattr(config, 'rel_scale', 0.7),
                        total_scale=getattr(config, 'total_scale', 1.5),
                        nu_0=getattr(config, 'nu_0', 5),
                        kappa_0=getattr(config, 'kappa_0', 1),
                        psi_dir_0=getattr(config, 'psi_dir_0', 1))
    lpvds.init_cluster(gaussians)
    lpvds.A = np.array(As)
    lpvds.ds_opt = dsopt_class(lpvds.x, lpvds.x_dot, lpvds.x_att, lpvds.gamma, lpvds.assignment_arr)
    lpvds.ds_opt.P = P

    stats['ds compute time'] = time.time() - t_ds
    stats['total compute time'] = time.time() - t0

    return lpvds, gg, gg_solution_nodes, stats

def all_paths(ds_set, attractor, config, recompute_gaussians):
    """Constructs a stitched dynamical system by aggregating all node-wise shortest paths in a Gaussian graph.

    Args:
        ds_set: List of DS objects, each with Gaussian mixture parameters and trajectory data.
        attractor: Target attractor point for the system.
        config: Configuration object specifying parameters for graph construction and DS computation.
        recompute_gaussians: If True, re-estimates both Gaussians and dynamics; if False, only linear dynamics.

    Returns:
        tuple: (stitched_ds, gg, stats) where
            - stitched_ds: Learned stitched DS object, or None on failure.
            - gg: Constructed GaussianGraph object (with node-wise shortest paths).
            - stats: Dictionary with timing information for major steps.
    """

    # Initialize stats dictionary
    stats = dict()

    # ############## GAUSSIAN GRAPH ##############
    t0 = time.time()
    gaussians = {(i, j): {'mu': mu, 'sigma': sigma, 'direction': direction, 'prior': prior}
                 for i, ds in enumerate(ds_set)
                 for j, (mu, sigma, direction, prior) in
                 enumerate(zip(ds.damm.Mu, ds.damm.Sigma, get_gaussian_directions(ds), ds.damm.Prior))}
    gg = gu.GaussianGraph(param_dist=config.param_dist, param_cos=config.param_cos)
    gg.add_gaussians(gaussians, reverse_gaussians=config.reverse_gaussians)
    gg_solution_nodes = gg.shortest_path_tree(target_state=attractor)
    stats['gg compute time'] = time.time() - t0

    # ############## DS ##############
    t0 = time.time()

    # Collect the shortest-path-tree gaussians and normalize their priors
    priors = [gg.graph.nodes[node_id]['prior'] for node_id in gg_solution_nodes]
    priors = [prior / sum(priors) for prior in priors]
    gaussians = []
    for i, node_id in enumerate(gg_solution_nodes):
        mu, sigma, direction, prior = gg.get_gaussian(node_id)
        gaussians.append({
            'prior': priors[i],  # use normalized prior
            'mu': mu,
            'sigma': sigma,
            'rv': multivariate_normal(mu, sigma, allow_singular=True)
        })

    # Collect the trajectory points that are assigned to each gaussian
    filtered_x = []
    filtered_x_dot = []
    for node_id in gg_solution_nodes:

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
    x_att = attractor
    try:
        stitched_ds = lpvds_class(filtered_x, filtered_x_dot, x_att,
                                  rel_scale=getattr(config, 'rel_scale', 0.7),
                                  total_scale=getattr(config, 'total_scale', 1.5),
                                  nu_0=getattr(config, 'nu_0', 5),
                                  kappa_0=getattr(config, 'kappa_0', 1),
                                  psi_dir_0=getattr(config, 'psi_dir_0', 1))
        if recompute_gaussians:  # compute new gaussians and linear systems (As)
            result = stitched_ds.begin()
            if result is None:
                print('Failed to construct Stitched DS: DAMM clustering failed')
                stitched_ds = None
        else:  # compute only linear systems (As)
            stitched_ds.init_cluster(gaussians)
            try:
                stitched_ds._optimize()
            except Exception as opt_e:
                print(f'Failed to optimize Stitched DS: {opt_e}')
                stitched_ds = None

    except Exception as e:
        print(f'Failed to construct Stitched DS: {e}')
        stitched_ds = None

    stats['ds compute time'] = time.time() - t0
    stats['total compute time'] = time.time() - t0

    return stitched_ds, gg, gg_solution_nodes, stats


def all_paths_reuse(ds_set, reversed_ds_set, initial, attractor, config):
    # Initialize stats dictionary
    stats = dict()

    # ############## GAUSSIAN GRAPH ##############
    t0 = time.time()
    gaussians = {(i, j): {'mu': mu, 'sigma': sigma, 'direction': direction, 'prior': prior}
                 for i, ds in enumerate(ds_set)
                 for j, (mu, sigma, direction, prior) in
                 enumerate(zip(ds.damm.Mu, ds.damm.Sigma, get_gaussian_directions(ds), ds.damm.Prior))}
    gg = gu.GaussianGraph(param_dist=config.param_dist, param_cos=config.param_cos)
    gg.add_gaussians(gaussians, reverse_gaussians=config.reverse_gaussians)
    gg_solution_nodes = gg.shortest_path_tree(target_state=attractor)
    stats['gg compute time'] = time.time() - t0

    # ############## DS ##############
    t_ds = time.time()

    # Select P from demo with closest attractor to current attractor
    P = None
    min_dist = np.inf
    is_attractor = False
    for i, ds_class in enumerate(ds_set + reversed_ds_set):
        dist = np.linalg.norm(ds_class.x_att - attractor)
        if dist < min_dist:
            min_dist = dist
            is_attractor = True
            P = ds_class.ds_opt.P

    # Collect the gaussians along the shortest path
    priors = [gg.graph.nodes[node_id]['prior'] for node_id in gg_solution_nodes]
    priors = [prior / sum(priors) for prior in priors]
    gaussians = []
    for i, node_id in enumerate(gg_solution_nodes):
        mu, sigma, direction, prior = gg.get_gaussian(node_id)
        gaussians.append({
            'prior': priors[i],  # use normalized prior
            'mu': mu,
            'sigma': sigma,
            'rv': multivariate_normal(mu, sigma, allow_singular=True)
        })

    # collect the trajectory points that are assigned to the gaussians along the shortest path (parallel)
    nodes_to_process = gg_solution_nodes
    
    # Prepare arguments for parallel processing
    process_args = [(node_id, ds_set, gg, P, attractor) for node_id in nodes_to_process]
    
    # Process nodes in parallel
    As = []
    filtered_x = []
    filtered_x_dot = []
    
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        future_to_index = {executor.submit(_process_node, args): i for i, args in enumerate(process_args)}
        
        # Collect results in order
        results = [None] * len(nodes_to_process)
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    
    # Extract results in the correct order
    for node_A, assigned_x, assigned_x_dot in results:
        As.append(node_A)
        filtered_x.append(assigned_x)
        filtered_x_dot.append(assigned_x_dot)

    filtered_x = np.vstack(filtered_x)
    filtered_x_dot = np.vstack(filtered_x_dot)
    As = np.array(As)

    lpvds = lpvds_class(filtered_x, filtered_x_dot, attractor,
                        rel_scale=getattr(config, 'rel_scale', 0.7),
                        total_scale=getattr(config, 'total_scale', 1.5),
                        nu_0=getattr(config, 'nu_0', 5),
                        kappa_0=getattr(config, 'kappa_0', 1),
                        psi_dir_0=getattr(config, 'psi_dir_0', 1))
    lpvds.init_cluster(gaussians)
    lpvds.A = np.array(As)
    lpvds.ds_opt = dsopt_class(lpvds.x, lpvds.x_dot, lpvds.x_att, lpvds.gamma, lpvds.assignment_arr)
    lpvds.ds_opt.P = P

    stats['ds compute time'] = time.time() - t_ds
    stats['total compute time'] = time.time() - t0

    return lpvds, gg, gg_solution_nodes, stats

def reuse_A(ds_set, initial, attractor, config):

    # Initialize stats dictionary
    stats = dict()

    # ############## GAUSSIAN GRAPH ##############
    t0 = time.time()
    gaussians = {(i, j): {'mu': mu, 'sigma': sigma, 'direction': direction, 'prior': prior}
                 for i, ds in enumerate(ds_set)
                 for j, (mu, sigma, direction, prior) in
                 enumerate(zip(ds.damm.Mu, ds.damm.Sigma, get_gaussian_directions(ds), ds.damm.Prior))}
    gg = gu.GaussianGraph(param_dist=config.param_dist, param_cos=config.param_cos)
    gg.add_gaussians(gaussians, reverse_gaussians=config.reverse_gaussians)
    gg_solution_nodes = gg.shortest_path_tree(target_state=attractor)
    stats['gg compute time'] = time.time() - t0

    # ############## DS ##############
    t0 = time.time()

    # Collect the gaussians that eventually lead to the target
    priors = [gg.graph.nodes[node_id]['prior'] for node_id in gg_solution_nodes]
    priors = [prior / sum(priors) for prior in priors]
    gaussians = []
    for i, node_id in enumerate(gg_solution_nodes):
        mu, sigma, direction, prior = gg.get_gaussian(node_id)
        gaussians.append({
            'prior': priors[i],  # use normalized prior
            'mu': mu,
            'sigma': sigma,
            'rv': multivariate_normal(mu, sigma, allow_singular=True)
        })

    # Collect the trajectory points that are assigned to each gaussian
    filtered_x = []
    filtered_x_dot = []
    As = []
    for node_id in gg_solution_nodes:

        ds_idx = node_id[0]
        gaussian_idx = node_id[1]

        assigned_x = ds_set[ds_idx].x[ds_set[ds_idx].assignment_arr == gaussian_idx]
        assigned_x_dot = ds_set[ds_idx].x_dot[ds_set[ds_idx].assignment_arr == gaussian_idx]

        # reverse velocity if gaussian is reversed
        assigned_x_dot = -assigned_x_dot if node_id in gg.gaussian_reversal_map else assigned_x_dot
        
        node_A = ds_set[ds_idx].A[gaussian_idx] * (-1 if node_id in gg.gaussian_reversal_map else 1)

        filtered_x.append(assigned_x)
        filtered_x_dot.append(assigned_x_dot)
        As.append(node_A)

    filtered_x = np.vstack(filtered_x)
    filtered_x_dot = np.vstack(filtered_x_dot)
    As = np.array(As)

    # try to find valid P
    # TODO make use of formulas found in homework
    P = find_lyapunov_function(As, attractor.shape[0])
    
    if P is None:
        print("Warning: Could not find a valid Lyapunov function P. Using identity matrix.")
        P = -np.eye(attractor.shape[0])  # Fallback to negative identity
    
    # Verify P satisfies conditions for all A matrices
    all_valid = True
    for i, node_A in enumerate(As):
        # check if A is valid with respect to P
        valid_wrt_p = is_negative_definite(np.transpose(node_A) @ P + P @ node_A)
        p_nd = is_negative_definite(P)
        p_symetric = np.allclose(P, np.transpose(P))
        
        if not (valid_wrt_p and p_nd and p_symetric):
            print(f"Warning: A matrix {i} does not satisfy Lyapunov conditions with found P")
            all_valid = False
    
    if all_valid:
        print(f"Successfully found Lyapunov function P for all {len(As)} A matrices")

    lpvds = lpvds_class(filtered_x, filtered_x_dot, attractor,
                        rel_scale=getattr(config, 'rel_scale', 0.7),
                        total_scale=getattr(config, 'total_scale', 1.5),
                        nu_0=getattr(config, 'nu_0', 5),
                        kappa_0=getattr(config, 'kappa_0', 1),
                        psi_dir_0=getattr(config, 'psi_dir_0', 1))
    lpvds.init_cluster(gaussians)
    lpvds.A = np.array(As)
    lpvds.ds_opt = dsopt_class(lpvds.x, lpvds.x_dot, lpvds.x_att, lpvds.gamma, lpvds.assignment_arr)
    lpvds.ds_opt.P = P

    stats['ds compute time'] = time.time() - t0
    stats['total compute time'] = time.time() - t0

    return lpvds, gg, gg_solution_nodes, stats