import os, sys
import pickle
import numpy as np
from typing import List
import pyLasaDataset as lasa
from scipy.io import loadmat

from src.stitching.trajectory_drawer import TrajectoryDrawer, plot_trajectories
from src.util.preprocessing import lpvds_per_demo, _pre_process, _process_bag


def load_data_from_file(data_file, n_demos=5, noise_std=0.05, force_preprocess=False):
    """
    Load trajectory data from file with caching support.
    
    Loads processed trajectory data from a cached pickle file if it exists,
    otherwise calculates the data from raw trajectories and caches the result.
    
    Parameters:
    -----------
    data_file : str
        Name of the data file (without extension)
    n_demos : int, optional
        Number of demonstrations to generate per base trajectory (default: 5)
    noise_std : float, optional
        Standard deviation of noise to add to demonstrations (default: 0.05)
        
    Returns:
    --------
    dict
        Processed trajectory data containing centers, directions, sigmas, etc.
        
    Raises:
    -------
    ValueError
        If data_file is None
    """
    if data_file is None:
        raise ValueError("No data file provided")

    # load data from file if exists
    filename = "./dataset/stitching/{}.pkl".format(data_file)
    if os.path.exists(filename) and not force_preprocess:
        print("Using cached nodes")
        with open(filename, 'rb') as f:
            processed_data = pickle.load(f)
    else:
        # calculate nodes if not yet calculated
        print("Calculating nodes")
        raw_data = load_data_stitch(data_file=data_file, n_demos=n_demos, noise_std=noise_std)

        processed_data = lpvds_per_demo(raw_data)
        with open(filename, 'wb') as f:
            pickle.dump(processed_data, f)

    return processed_data


def load_nodes_1():
    """
    Generate two intersecting diagonal trajectories for testing.
    
    Creates two base trajectories:
    1. Main diagonal: from (10, 10) to (0, 0)
    2. Anti-diagonal: from (0, 10) to (10, 0)
    
    Returns:
    --------
    tuple[List[np.ndarray], List[np.ndarray]]
        x_sets: List of trajectory position arrays, each [M, 2]
        x_dot_sets: List of trajectory velocity arrays, each [M, 2]
    """
    n_samples = 101  # number of points per trajectory
    t = np.linspace(0, 1, n_samples)

    # Trajectory 1: from (10, 10) to (0, 0) (main diagonal)
    base_traj_target = np.vstack((10 - 10 * t, 10 - 10 * t)).T

    # Trajectory 2: from (0, 10) to (10, 0) (anti-diagonal)
    base_traj_other = np.vstack((10 * t, 10 - 10 * t)).T

    x_sets = [base_traj_target, base_traj_other]

    # Calculate velocities
    x_dot_sets = [np.gradient(traj, axis=0) for traj in x_sets]

    return x_sets, x_dot_sets


def load_nodes_2():
    """
    Generate three curved trajectories with sinusoidal components.
    
    Creates three base trajectories with different mathematical patterns:
    1. Trajectory with sine and cosine modulation
    2. Linear trajectory with sine modulation  
    3. Nearly vertical trajectory with sine modulation
    
    Returns:
    --------
    tuple[List[np.ndarray], List[np.ndarray]]
        raw_x_sets: List of trajectory position arrays, each [M, 2]
        raw_x_dot_sets: List of trajectory velocity arrays, each [M, 2]
    """
    n_points = 40

    # Trajectory 1
    raw_x_sets = []
    raw_x_dot_sets = []
    base_x1_1 = np.linspace(0, 15, n_points) + np.sin(np.linspace(0, 15, n_points))
    base_x1_2 = np.linspace(0, 15, n_points) + np.cos(0.2*np.linspace(0, 15, n_points))
    raw_x_sets.append(np.vstack((base_x1_1, base_x1_2)).T)
    raw_x_dot_sets.append(np.gradient(raw_x_sets[-1], axis=0))

    # Trajectory 2
    base_x2_1 = np.linspace(10, 14, n_points)
    base_x2_2 = np.linspace(8, 2, n_points) + np.sin(0.4*np.linspace(8, 2, n_points))
    raw_x_sets.append(np.vstack((base_x2_1, base_x2_2)).T)
    raw_x_dot_sets.append(np.gradient(raw_x_sets[-1], axis=0))

    # Trajectory 3
    base_x3_1 = np.linspace(4, 5, n_points)
    base_x3_2 = np.linspace(6.5, 13, n_points) + np.sin(0.6*np.linspace(6, 13, n_points))
    raw_x_sets.append(np.vstack((base_x3_1, base_x3_2)).T)
    raw_x_dot_sets.append(np.gradient(raw_x_sets[-1], axis=0))

    return raw_x_sets, raw_x_dot_sets


def generate_multiple_demos(base_trajectories: List[np.ndarray], base_velocities: List[np.ndarray], 
                            n_demos: int = 5, noise_std: float = 0.2) -> List[List[np.ndarray]]:
    """
    Generate multiple noisy demonstrations from base trajectories.
    
    Takes base trajectories and generates multiple variations by adding Gaussian noise.
    Velocities are recalculated from the noisy position trajectories using gradient.
    
    Parameters:
    -----------
    base_trajectories : List[np.ndarray]
        List of base trajectory position arrays, each [M, N]
    base_velocities : List[np.ndarray]
        List of base trajectory velocity arrays, each [M, N]
    n_demos : int, optional
        Number of demonstrations to generate per base trajectory (default: 5)
    noise_std : float, optional
        Standard deviation of Gaussian noise to add (default: 0.2)
        
    Returns:
    --------
    tuple[List[List[np.ndarray]], List[List[np.ndarray]]]
        demo_sets: List of demo sets, each containing n_demos position trajectories
        demo_vel_sets: List of demo sets, each containing n_demos velocity trajectories
    """
    demo_sets = []
    demo_vel_sets = []
    
    for base_traj, base_vel in zip(base_trajectories, base_velocities):
        demos = []
        demo_vels = []
        for _ in range(n_demos):
            noise = np.random.normal(0, noise_std, base_traj.shape)
            demo = base_traj + noise
            demos.append(demo)
            
            # noise_vel = np.random.normal(0, noise_std, base_vel.shape)
            noise_vel = np.gradient(demo, axis=0)
            demo_vels.append(noise_vel)#base_vel + noise_vel)
        demo_sets.append(demos)
        demo_vel_sets.append(demo_vels)
        
    print(f"Generated {n_demos} demonstrations for each of {len(base_trajectories)} base trajectories")
    return demo_sets, demo_vel_sets


def load_data_stitch(data_file, n_demos=5, noise_std=0.05):
    """
    Load and process trajectory data for stitching applications.
    
    Loads base trajectories from various sources (predefined nodes or drawn trajectories),
    generates multiple noisy demonstrations, and preprocesses them for stitching.
    
    Parameters:
    -----------
    data_file : str
        Data source identifier. Options:
        - "nodes_1": Two intersecting diagonal trajectories
        - "nodes_2": Three curved trajectories with sinusoidal components
        - Other: Load from trajectory drawer files
    n_demos : int, optional
        Number of demonstrations to generate per base trajectory (default: 5)
    noise_std : float, optional
        Standard deviation of noise to add to demonstrations (default: 0.05)
        
    Returns:
    --------
    List[tuple]
        A list of tuples, where each tuple contains:
            x:     a [M, N] NumPy array: M observations of N dimension
            x_dot: a [M, N] NumPy array: M observations velocities of N dimension
            x_att: a [1, N] NumPy array of attractor
            x_init: an L-length list of [1, N] NumPy array: L number of trajectories, each containing an initial point of N dimension
    """

    if data_file == "nodes_1":
        raw_x_sets, raw_x_dot_sets = load_nodes_1()
    elif data_file == "nodes_2":
        raw_x_sets, raw_x_dot_sets = load_nodes_2()
    else:
        # x_sets, x_dot_sets = load_drawn_trajectories(data_file)
        raw_x_sets, raw_x_dot_sets = load_drawn_trajectories(data_file)

    x_sets, x_dot_sets = generate_multiple_demos(raw_x_sets, raw_x_dot_sets, n_demos, noise_std)

    # Plot the loaded/generated trajectories
    plot_trajs = []
    for demo_set in x_sets:
        plot_trajs.extend(demo_set)
    plot_trajectories(plot_trajs, "Loaded Trajectories")

    # return _pre_process_stitch(x_sets, x_dot_sets)
    return [_pre_process(x, x_dot) for x, x_dot in zip(x_sets, x_dot_sets)]


def load_drawn_trajectories(data_file):
    """
    Load trajectories from trajectory drawer files or create new ones interactively.
    
    Attempts to load existing trajectory files from the dataset/stitching/ directory.
    If the file doesn't exist, provides an option to draw new trajectories interactively
    using the TrajectoryDrawer interface.
    
    Parameters:
    -----------
    data_file : str
        Base name of the trajectory file (without extension)
        Will look for {data_file}_traj.pkl in dataset/stitching/
        
    Returns:
    --------
    tuple[List[np.ndarray], List[np.ndarray]] or None
        trajectories: List of trajectory position arrays, each [M, 2]
        velocities: List of trajectory velocity arrays, each [M, 2]
        Returns None if no trajectories are loaded or drawn
    """
    print("\n=== Load Drawn Trajectories ===")
    print("Options:")
    print("1. Load existing trajectory file")
    print("2. Draw new trajectories interactively")
    
    data_path = "./dataset/stitching/"
    file_path = data_path + data_file + "_traj.pkl"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        choice = "2"
    else:
        choice = "1"
    
    drawer = TrajectoryDrawer()
    if choice == "1":
        # Load existing file
        file_path = data_path + data_file + "_traj.pkl"

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
            drawer.save_trajectories(data_path + f"{data_file}_traj.pkl")
        else:
            return None

        trajectories, velocities = drawer.load_trajectories(data_path + f"{data_file}_traj.pkl")
                
    else:
        print("Invalid choice.")
        return None

    return trajectories, velocities


def load_data(input_opt):
    """
    Load trajectory data from various benchmark datasets.
    
    Provides an interactive interface to load data from different sources including
    PC-GMM benchmark data, LASA benchmark dataset, and custom demo datasets.
    
    Parameters:
    -----------
    input_opt : int or str
        Data source option:
        - 1: PC-GMM benchmark data (3D/2D trajectories)
        - 2: LASA benchmark dataset (various trajectory shapes)
        - 3: Damm demo dataset (bridge, Nshape, orientation)
        - 4: Demo dataset with obstacles
        - 'demo': General demo dataset
        - 'increm': Incremental demo dataset
        
    Returns:
    --------
    tuple
        Preprocessed trajectory data containing:
        - x: [M, N] NumPy array: M observations of N dimension
        - x_dot: [M, N] NumPy array: M observations velocities of N dimension
        - x_att: [1, N] NumPy array of attractor
        - x_init: L-length list of [1, N] NumPy arrays: initial points for L trajectories
        
    Notes:
    ------
    This function provides interactive prompts for dataset selection and may call
    sys.exit() if invalid options are selected or user chooses to exit.
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