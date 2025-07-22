import os, sys
import pickle
import numpy as np
import pyLasaDataset as lasa
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from src.stitching.trajectory_drawer import TrajectoryDrawer, plot_trajectories

from src.stitching.preprocessing import calculate_demo_lpvds

def load_data_from_file(input_opt, data_file=None):

    if input_opt < 3:
        data_file = f"nodes_{input_opt}"
    elif data_file is not None:
        data_file = data_file
    else:
        raise ValueError("Invalid input option")

    filename = "./dataset/stitching/{}.pkl".format(data_file)
    if os.path.exists(filename):
        print("Using cached nodes")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Calculating nodes")
        data, used_data_file = calculate_demo_lpvds(input_opt, data_file)
        if used_data_file != data_file:
            print(f"Using data file: {used_data_file}")
            filename = "./dataset/stitching/{}.pkl".format(used_data_file)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    return data, data_file


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


    return _pre_process(x, x_dot)


def load_data_stitch(input_opt, data_file=None):
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
        data_hash = "nodes_1"

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

        data_hash = "nodes_2"

    elif input_opt == 3:
        x_sets, x_dot_sets, used_file = load_drawn_trajectories(data_file)
        data_hash = used_file

    # Flatten demo sets for processing
    plot_trajs = []
    for demo_set in x_sets:
        plot_trajs.extend(demo_set)
    
    # Plot the loaded/generated trajectories
    plot_trajectories(plot_trajs, "Loaded Trajectories")

    # return _pre_process_stitch(x_sets, x_dot_sets)
    return [_pre_process(x, x_dot) for x, x_dot in zip(x_sets, x_dot_sets)], data_hash


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


def load_drawn_trajectories(data_file=None):
    """Load trajectories from trajectory drawer files."""
    print("\n=== Load Drawn Trajectories ===")
    print("Options:")
    print("1. Load existing trajectory file")
    print("2. Draw new trajectories interactively")
    
    data_path = "./dataset/stitching/"

    if data_file is not None:
        file_path = data_path + data_file + "_traj.pkl"
        if not os.path.exists(file_path):
            print(f"File {file_path} not found!")
            choice = "2"
        else:
            choice = "1"
    else:
        choice = input("Choose option (1 or 2): ").strip()
    
    drawer = TrajectoryDrawer()
    if choice == "1":
        # Load existing file
        if data_file is not None:
            filename = data_file
        else:
            filename = input("Enter trajectory filename (without .pkl extension): ").strip()
        if not filename:
            print("No filename provided. Using default: sample_trajectories.pkl")
            filename = "sample_trajectories"
            
        file_path = data_path + filename + "_traj.pkl"

        if not os.path.exists(file_path):
            print(f"File {file_path} not found!")
            return None
            
        trajectories, velocities = drawer.load_trajectories(file_path)
        if not trajectories:
            return None
        
        
    elif choice == "2":
        # Draw new trajectories
        print("Starting interactive drawing session...")
        drawer.start_interactive_drawing()
        
        if not drawer.trajectories:
            print("No trajectories were drawn.")
            return None
            
        trajectories = drawer.trajectories
        
        # Ask to save
        save_choice = input("Save drawn trajectories? (y/n): ").strip().lower()
        if save_choice == 'y':
            if data_file is not None:
                filename = data_file
            else:
                filename = input("Enter filename (without .pkl extension): ").strip()
            if filename:
                drawer.save_trajectories(data_path + f"{filename}_traj.pkl")
        else:
            return None

        trajectories, velocities = drawer.load_trajectories(data_path + f"{filename}_traj.pkl")
                
    else:
        print("Invalid choice.")
        return None


    # Ask if user wants to generate more demonstrations
    # generate_demos = input("Generate multiple demonstrations with noise? (y/n): ").strip().lower()
    # if generate_demos == 'y':
    n_demos = 3 #int(input("Number of demonstrations per trajectory (default 5): ") or 5)
    noise_std = 0.05#float(input("Noise standard deviation (default 0.2): ") or 0.2)
        
    noisy_trajectories, noisy_velocities = drawer.generate_multiple_demos(trajectories, velocities, n_demos, noise_std)
    # drawer.plot_trajectory_set(noisy_trajectories, "Generated Demonstrations")
        
    return noisy_trajectories, noisy_velocities, filename