import numpy as np
from scipy.io import loadmat

from src.lpvds_class import lpvds_class

def compute_weighted_average(x, x_dot, centers, sigmas, assignment_arr):
    mean_xdot = np.zeros((centers.shape[0], x.shape[1]))
    for k in range(centers.shape[0]):
        # Get points assigned to cluster k
        assigned_mask = assignment_arr == k
        if np.sum(assigned_mask) == 0:
            continue
        
        assigned_x = x[assigned_mask]
        assigned_xdot = x_dot[assigned_mask]
        
        # Compute Gaussian weights for assigned points
        center = centers[k]
        sigma = sigmas[k]
        
        # Calculate multivariate Gaussian weights
        diff = assigned_x - center
        # Handle potential singular covariance matrix
        try:
            sigma_inv = np.linalg.inv(sigma)
            weights = np.exp(-0.5 * np.sum(diff @ sigma_inv * diff, axis=1))
        except np.linalg.LinAlgError:
            # Fallback to identity if sigma is singular
            weights = np.exp(-0.5 * np.sum(diff**2, axis=1))
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Compute weighted average
        mean_xdot[k] = np.sum(assigned_xdot * weights[:, np.newaxis], axis=0)

    return mean_xdot

def compute_average(x, x_dot, centers, sigmas, assignment_arr):
    mean_xdot = np.zeros((centers.shape[0], x.shape[1]))
    for k in range(centers.shape[0]):
        mean_xdot[k] = np.mean(x_dot[assignment_arr==k], axis=0)
    return mean_xdot


def lpvds_per_demo(data):
    """
    data list with each entry containing x, x_dot, x_att, x_init
    Runs LPVDS on each demo set and returns the aggregated results
    """
    n_demo_sets = len(data) # L
    node_centers = []
    node_sigmas = []
    node_directions = []
    xs = []
    x_dots = []
    x_atts = []
    x_inits = []
    assignment_arrs = []
    n_centers = 0
    As = []
    Ps = []
    
    for i in range(n_demo_sets):
        x, x_dot, x_att, x_init = data[i]
        lpvds = lpvds_class(x, x_dot, x_att)
        lpvds.begin()

        # get directionality
        centers = lpvds.damm.Mu
        assignment_arr = lpvds.assignment_arr
        sigmas = lpvds.damm.Sigma
        # Compute weighted average using Gaussian weights
        # mean_xdot = compute_weighted_average(x, x_dot, centers, sigmas, assignment_arr)
        mean_xdot = compute_average(x, x_dot, centers, sigmas, assignment_arr)

        node_centers.extend(centers.tolist())
        node_directions.extend(mean_xdot.tolist())
        node_sigmas.extend(sigmas.tolist())
        xs.append(x)
        x_dots.append(x_dot)
        x_atts.append(np.squeeze(x_att))
        x_inits.append(np.squeeze(x_init).mean(axis=0))
        As.append(lpvds.A)
        Ps.append(lpvds.ds_opt.P)

        # ensures that the assignment array is unique
        assignment_arrs.append(assignment_arr+n_centers)
        n_centers += centers.shape[0]

    # aggregate results
    # N: dimension of the data (states)
    # K: number of demo sets
    # L: number of demonstrations per demo set
    # M: number of observations per trajectory
    # D: total number of nodes (across all demo sets)

    # print(x_inits)

    return {
        "centers": np.array(node_centers), # [D, N] center of nodes
        "directions": np.array(node_directions), # [D, N] direction of nodes
        "sigmas": np.array(node_sigmas), # [D, N, N] covariance of nodes
        "x_sets": np.vstack(xs), # [D, M, N] states
        "x_dot_sets": np.vstack(x_dots), # [D, M, N] velocities
        "x_attrator_sets": np.array(x_atts), # [K, N] attractor positions
        "x_initial_sets": np.array(x_inits), # [K, N] initial positions (mean per set)
        # "all_x_initial": np.vstack(x_inits), # [D, N] all initial positions
        "assignment_arrs": np.hstack(assignment_arrs), # [D, M] node assignment arrays
        "As": np.vstack(As), # [D, N, N] A matrices
        "Ps": np.array(Ps), # [K, N, N] P matrices
    }


def _pre_process(x, x_dot):
    """ 
    Roll out nested lists into a single list of M entries

    Parameters:
    -------
        x:     an L-length list of [M, N] NumPy array: L number of trajectories, each containing M observations of N dimension,
    
        x_dot: an L-length list of [M, N] NumPy array: L number of trajectories, each containing M observations velocities of N dimension

    Note:
    -----
        M can vary and need not be same between trajectories
    """

    L = len(x)
    x_init = []
    x_shifted = []

    x_att  = [x[l][-1, :]  for l in range(L)]  
    x_att_mean  =  np.mean(np.array(x_att), axis=0, keepdims=True)
    for l in range(L):
        x_init.append(x[l][0].reshape(1, -1))

        x_diff = x_att_mean - x_att[l]
        x_shifted.append(x_diff.reshape(1, -1) + x[l])

    for l in range(L):
        if l == 0:
            x_rollout = x_shifted[l]
            x_dot_rollout = x_dot[l]
        else:
            x_rollout = np.vstack((x_rollout, x_shifted[l]))
            x_dot_rollout = np.vstack((x_dot_rollout, x_dot[l]))

    return  x_rollout, x_dot_rollout, x_att_mean, x_init



def _process_bag(path):
    """ Process .mat files that is converted from .bag files """

    data_ = loadmat(r"{}".format(path))
    data_ = data_['data_ee_pose']
    L = data_.shape[1]

    x     = []
    x_dot = [] 

    sample_step = 5
    vel_thresh  = 1e-3 
    
    for l in range(L):
        data_l = data_[0, l]['pose'][0,0]
        pos_traj  = data_l[:3, ::sample_step]
        quat_traj = data_l[3:7, ::sample_step]
        time_traj = data_l[-1, ::sample_step].reshape(1,-1)

        raw_diff_pos = np.diff(pos_traj)
        vel_mag = np.linalg.norm(raw_diff_pos, axis=0).flatten()
        first_non_zero_index = np.argmax(vel_mag > vel_thresh)
        last_non_zero_index = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)

        if first_non_zero_index >= last_non_zero_index:
            raise Exception("Sorry, vel are all zero")

        pos_traj  = pos_traj[:, first_non_zero_index:last_non_zero_index]
        quat_traj = quat_traj[:, first_non_zero_index:last_non_zero_index]
        time_traj = time_traj[:, first_non_zero_index:last_non_zero_index]
        vel_traj = np.diff(pos_traj) / np.diff(time_traj)
        
        x.append(pos_traj[:, 0:-1].T)
        x_dot.append(vel_traj.T)

    return x, x_dot
