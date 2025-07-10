import os, sys
import numpy as np
import pyLasaDataset as lasa
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R


def load_data(input_opt):
    """
    Return:
    -------
        x:     a [M, N] NumPy array: M observations of N dimension
    
        x_dot: a [M, N] NumPy array: M observations velocities of N dimension

        x_att: a [1, N] NumPy array of attractor

        x_init: an L-length list of [1, N] NumPy array: L number of trajectories, each containing an initial point of N dimension
    """

    if input_opt == 1:
        print("\nYou selected PC-GMM benchmark data.\n")
        pcgmm_list = ["3D_sink", "3D_viapoint_1", "3D-cube-pick", "3D_viapoint_2", "2D_Lshape",  "2D_incremental_1", "2D_multi-behavior", "2D_messy-snake"]

        message = """Available Models: \n"""
        for i in range(len(pcgmm_list)):
            message += "{:2}) {: <18} ".format(i+1, pcgmm_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '

        data_opt = int(input(message))
        if data_opt == 0:
            sys.exit()
        elif data_opt<0 or data_opt>len(pcgmm_list):
            print("Invalid data option")
            sys.exit()

        data_name  = str(pcgmm_list[data_opt-1]) + ".mat"
        input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "pc-gmm-data", data_name)

        data_ = loadmat(r"{}".format(input_path))
        data_ = np.array(data_["data"])

        N     = int(data_[0, 0].shape[0]/2)
        if N == 2:
            L = data_.shape[1]
            x     = [data_[0, l][:N, :].T  for l in range(L)]
            x_dot = [data_[0, l][N:, :].T  for l in range(L)]
        elif N == 3:
            L = data_.shape[0]
            L_sub = np.random.choice(range(L), 6, replace=False)

            x     = [data_[l, 0][:N, :].T  for l in range(L)]
            x_dot = [data_[l, 0][N:, :].T  for l in range(L)]


    elif input_opt == 2:
        print("\nYou selected LASA benchmark dataset.\n")

        # suppress print message from lasa package
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        sys.stdout = original_stdout

        lasa_list = ["Angle", "BendedLine", "CShape", "DoubleBendedLine", "GShape", "heee", "JShape", "JShape_2", "Khamesh", "Leaf_1",
        "Leaf_2", "Line", "LShape", "NShape", "PShape", "RShape", "Saeghe", "Sharpc", "Sine", "Snake",
        "Spoon", "Sshape", "Trapezoid", "Worm", "WShape", "Zshape", "Multi_Models_1", "Multi_Models_2", "Multi_Models_3", "Multi_Models_4"]

        message = """Available Models: \n"""
        for i in range(len(lasa_list)):
            message += "{:2}) {: <18} ".format(i+1, lasa_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '
        
        data_opt = int(input(message))

        if data_opt == 0:
            sys.exit()
        elif data_opt<0 or data_opt > len(lasa_list):
            print("Invalid data option")
            sys.exit()

        data = getattr(lasa.DataSet, lasa_list[data_opt-1])
        demos = data.demos 
        sub_sample = 1
        L = len(demos)

        x     = [demos[l].pos[:, ::sub_sample].T for l in range(L)]
        x_dot = [demos[l].vel[:, ::sub_sample].T for l in range(L)]


    elif input_opt == 3:
        print("\nYou selected Damm demo dataset.\n")

        damm_list = ["bridge", "Nshape", "orientation"]
        
        message = """Available Models: \n"""
        for i in range(len(damm_list)):
            message += "{:2}) {: <18} ".format(i+1, damm_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '
        
        data_opt = int(input(message))
    
        folder_name = str(damm_list[data_opt-1])
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "damm-demo-data", folder_name, "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 4:
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "obstacle", "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 'demo':
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 'increm':
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "increm", "all.mat")
        x, x_dot    = _process_bag(input_path)


    # elif input_opt == 5:
    #     # make x shape data where each trajectory is diagonal, but with more samples along the same paths
    #     n_samples = 101  # number of points per trajectory
    #     t = np.linspace(0, 1, n_samples)

    #     # Trajectory 1: from (2, 2) to (0, 0) (main diagonal)
    #     traj_target = np.vstack((2 - 2 * t, 2 - 2 * t)).T
    #     # Trajectory 2: from (0, 2) to (2, 0) (anti-diagonal)
    #     traj_other = np.vstack((2 * t, 2 - 2 * t)).T

    #     x_sets = [[traj_target], [traj_other]]
    #     # Compute velocities as finite differences between successive points
    #     # x_dot = [np.gradient(traj1, axis=0), np.gradient(traj2, axis=0)]
    #     x_dot_target = [np.repeat(np.array([[-0.5, -0.5]]), n_samples, axis=0)]
    #     x_dot_other = [np.repeat(np.array([[0.5, -0.5]]), n_samples, axis=0)]
    #     x_dot_sets = [x_dot_target, x_dot_other]
    #     return _pre_process_stitch(x_sets, x_dot_sets)

    return _pre_process(x, x_dot)


def load_data_stitch(input_opt):
    """
    Return:
    -------
        x:     a [M, N] NumPy array: M observations of N dimension
    
        x_dot: a [M, N] NumPy array: M observations velocities of N dimension

        x_att: a [1, N] NumPy array of attractor

        x_init: an L-length list of [1, N] NumPy array: L number of trajectories, each containing an initial point of N dimension
    """

    if input_opt == 1:
        n_samples = 101  # number of points per trajectory
        n_demos = 5
        t = np.linspace(0, 1, n_samples)

        # Trajectory 1: from (10, 10) to (0, 0) (main diagonal)
        base_traj_target = np.vstack((10 - 10 * t, 10 - 10 * t)).T
        trajs_target = []
        for _ in range(n_demos):
            noisy_traj = base_traj_target + np.random.normal(0, 0.05, base_traj_target.shape)
            trajs_target.append(noisy_traj)

        # Trajectory 2: from (0, 10) to (10, 0) (anti-diagonal)
        base_traj_other = np.vstack((10 * t, 10 - 10 * t)).T
        trajs_other = []
        for _ in range(n_demos):
            noisy_traj = base_traj_other + np.random.normal(0, 0.05, base_traj_other.shape)
            trajs_other.append(noisy_traj)

        x_sets = [trajs_target, trajs_other]

        # Calculate velocities
        x_dot_sets = []
        for trajs in x_sets:
            x_dot_trajs = [np.gradient(traj, axis=0) for traj in trajs]
            x_dot_sets.append(x_dot_trajs)

    elif input_opt == 2:
        n_points = 40
        n_demos = 5

        # Trajectory 1
        trajs1 = []
        base_x1_1 = np.linspace(0, 15, n_points) + np.sin(np.linspace(0, 15, n_points))
        base_x1_2 = np.linspace(0, 15, n_points) + np.cos(0.2*np.linspace(0, 15, n_points))
        for _ in range(n_demos):
            x1_1 = base_x1_1 + np.random.normal(0, 0.2, n_points)
            x1_2 = base_x1_2 + np.random.normal(0, 0.2, n_points)
            trajs1.append(np.vstack((x1_1, x1_2)).T)

        # Trajectory 2
        trajs2 = []
        base_x2_1 = np.linspace(10, 14, n_points)
        base_x2_2 = np.linspace(8, 2, n_points) + np.sin(0.4*np.linspace(8, 2, n_points))
        for _ in range(n_demos):
            x2_1 = base_x2_1 + np.random.normal(0, 0.2, n_points)
            x2_2 = base_x2_2 + np.random.normal(0, 0.2, n_points)
            trajs2.append(np.vstack((x2_1, x2_2)).T)

        # Trajectory 3
        trajs3 = []
        base_x3_1 = np.linspace(4, 5, n_points)
        base_x3_2 = np.linspace(6.5, 13, n_points) + np.sin(0.6*np.linspace(6, 13, n_points))
        for _ in range(n_demos):
            x3_1 = base_x3_1 + np.random.normal(0, 0.2, n_points)
            x3_2 = base_x3_2 + np.random.normal(0, 0.2, n_points)
            trajs3.append(np.vstack((x3_1, x3_2)).T)

        x_sets = [trajs1, trajs2, trajs3]

        # Calculate velocities
        x_dot_sets = []
        for trajs in x_sets:
            x_dot_trajs = [np.gradient(traj, axis=0) for traj in trajs]
            x_dot_sets.append(x_dot_trajs)

    # return _pre_process_stitch(x_sets, x_dot_sets)
    return [_pre_process(x, x_dot) for x, x_dot in zip(x_sets, x_dot_sets)]


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


def _pre_process_stitch(x_sets, x_dot_sets):
    """Apply the same normalization as `_pre_process`, but for multiple sets.

    The first element of `x_sets` is the *target* set that reaches the goal.
    We compute the mean attractor (goal) from the last sample of every
    trajectory in this first set.  For **every** trajectory in **every** set,
    we apply an individual shift so that that trajectory's endpoint coincides
    with this mean attractor:

        shift = x_att_mean - traj[-1]

    This keeps relative shapes intact while aligning all endpoints to the
    shared goal.  Velocities remain unchanged.

    Parameters
    ----------
    x_sets : list[list[np.ndarray]]
        Nested list of position trajectories.
    x_dot_sets : list[list[np.ndarray]]
        Nested list of corresponding velocity trajectories.

    Returns
    -------
    x_rollout : np.ndarray
        Stacked, shifted position samples.
    x_dot_rollout : np.ndarray
        Stacked velocity samples (unshifted).
    x_att_mean : np.ndarray
        The mean attractor used for alignment (shape 1×N).
    x_init : list[np.ndarray]
        Initial point of every original trajectory (unshifted).
    """

    # ----------------------------
    # 1. Compute mean attractor
    # ----------------------------
    target_set = x_sets[0]
    if len(target_set) == 0:
        raise ValueError("x_sets[0] must contain at least one trajectory")

    x_att = [traj[-1] for traj in target_set]
    x_att_mean = np.mean(np.array(x_att), axis=0, keepdims=True)  # [1, N]

    # ----------------------------
    # 2. Shift & roll out
    # ----------------------------
    x_rollout = None
    x_dot_rollout = None
    x_init = []

    for x_set, xdot_set in zip(x_sets, x_dot_sets):
        if len(x_set) != len(xdot_set):
            raise ValueError("Mismatch between x_sets and x_dot_sets lengths")

        for traj, vel in zip(x_set, xdot_set):
            # store initial (unshifted)
            x_init.append(traj[0:1])

            # individual shift bringing endpoint to mean attractor
            traj_shifted = traj + x_att_mean #- traj[-1])

            if x_rollout is None:
                x_rollout = traj_shifted
                x_dot_rollout = vel
            else:
                x_rollout = np.vstack((x_rollout, traj_shifted))
                x_dot_rollout = np.vstack((x_dot_rollout, vel))

    return x_rollout, x_dot_rollout, x_att_mean, x_init


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