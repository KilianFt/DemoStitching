import os, sys
import json
import pickle
import numpy as np
import pyLasaDataset as lasa
from scipy.io import loadmat
from collections import namedtuple
from src.util.plot_tools import plot_trajectories, plot_demonstration_set
from src.util.preprocessing import lpvds_per_demo, _pre_process, _process_bag
from src.util.generate_data import generate_data

Demoset = namedtuple('Demoset', ['x', 'x_dot'])

def get_ds_set(config):
    """Loads or computes a DS set from demonstration data with automatic caching.

    Args:
        config: Configuration object with dataset_path and force_preprocess attributes.

    Returns:
        DS set: Computed dynamical system set from demonstration trajectories.

    Raises:
        ValueError: If config.dataset_path is None.
    """
    demoset_path = config.dataset_path
    if demoset_path is None:
        raise ValueError("A demonstration set path must be provided.")

    # Load demonstration set's DS set if it has already been computed exists
    ds_set_filename = "{}/preprocessed.pkl".format(demoset_path)
    if os.path.exists(ds_set_filename) and not config.force_preprocess:

        print(f'Using cached DS set at \"{ds_set_filename}\"')
        with open(ds_set_filename, 'rb') as f:
            ds_set = pickle.load(f)
        return ds_set

    if not config.force_preprocess:
        print(f'Could not find an existing DS set at \"{demoset_path}\".')

    # No DS set has been computed (or config forces recompute). Locate existing demos.
    demosets = load_demoset(demoset_path)
    if demosets is None:
        print(f'No existing demonstrations found in \"{demoset_path}\". Drawing new demonstrations.')
        generate_data(demoset_path)
        demosets = load_demoset(demoset_path)
    else:
        print(f'Found existing demonstrations in \"{demoset_path}\".')

    # Compute the DS set from the loaded demonstration set
    print(f'Computing DS set from demonstrations in \"{demoset_path}\".')
    plot_demonstration_set(demosets, config)
    preprocessed_demoset = [_pre_process(x, x_dot) for x, x_dot in zip(demosets.x, demosets.x_dot)]
    ds_set = lpvds_per_demo(preprocessed_demoset)

    # Save the DS set to a file for future use
    print(f'Saving DS set to \"{demoset_path}\".')
    with open(ds_set_filename, 'wb') as f:
        pickle.dump(ds_set, f)

    return ds_set

def load_demoset(demoset_path):
    """Loads a demoset from demonstration folders of trajectory JSON files.

    Args:
        demoset_path (str): Path to directory containing demonstration set folders.

    Returns:
        Demoset or None: Namedtuple with x (positions) and x_dot (velocities) lists,
            or None if path doesn't exist or contains no dataset folders.
    """
    if not os.path.exists(demoset_path):
        return None

    # Collect all demonstration folders
    demonstration_folders = []
    for folder_name in os.listdir(demoset_path):
       if 'dataset' in folder_name:
           demonstration_folders.append(folder_name)

    # Return if the folder does not contain any demonstrations
    if not demonstration_folders:
        return None

    # Collect all trajectories from every demonstration folder
    demoset_x = []
    demoset_x_dot = []
    for demo_folder in demonstration_folders:
        demo_path = os.path.join(demoset_path, demo_folder)

        demonstration_x = []
        demonstration_x_dot = []

        for trajectory_file in os.listdir(demo_path):
            trajectory_path = os.path.join(demo_path, trajectory_file)

            trajectory = json.load(open(trajectory_path))
            demonstration_x.append(np.array(trajectory["x"]))
            demonstration_x_dot.append(np.array(trajectory["x_dot"]))

        demoset_x.append(demonstration_x)
        demoset_x_dot.append(demonstration_x_dot)

    return Demoset(demoset_x, demoset_x_dot)

def load_data_from_file(demoset_path, config):
    """
    Load trajectory data from file with caching support.
    
    Loads processed trajectory data from a cached pickle file if it exists,
    otherwise calculates the data from raw trajectories and caches the result.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file (without extension)
    config : Config
        Configuration object containing parameters for data loading
        
    Returns:
    --------
    dict
        Processed trajectory data containing centers, directions, sigmas, etc.
        
    Raises:
    -------
    ValueError
        If data_file is None
    """

    if demoset_path is None:
        raise ValueError("A demonstration set path must be provided.")

    # Load demonstration set's DS set if it has already been computed exists
    ds_set_filename = "{}/preprocessed.pkl".format(demoset_path)
    if os.path.exists(ds_set_filename) and not config.force_preprocess:

        print(f'Using cached DS set at \"{ds_set_filename}\"')
        with open(ds_set_filename, 'rb') as f:
            processed_data = pickle.load(f)

    # No DS set has been computed (or forced recompute). Locate the demos (or draw them if not found)
    else:
        print(f'Could not find an existing DS \"{demoset_path}\", locating demos.')
        os.makedirs(demoset_path, exist_ok=True)
        raw_data = load_data_stitch(dataset_path=demoset_path, config=config)

        processed_data = lpvds_per_demo(raw_data)
        with open(ds_set_filename, 'wb') as f:
            pickle.dump(processed_data, f)

    return processed_data

def load_data_stitch(dataset_path, config):
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
    config : Config
        Configuration object containing parameters for data loading
        
    Returns:
    --------
    List[tuple]
        A list of tuples, where each tuple contains:
            x:     a [M, N] NumPy array: M observations of N dimension
            x_dot: a [M, N] NumPy array: M observations velocities of N dimension
            x_att: a [1, N] NumPy array of attractor
            x_init: an L-length list of [1, N] NumPy array: L number of trajectories, each containing an initial point of N dimension
    """

    # generate data if not exists, else load it
        
    dataset_exists = any("dataset" in task_dataset_path for task_dataset_path in os.listdir(dataset_path))
    if not dataset_exists:
        print("Generating data")
        generate_data(dataset_path)

    # if dataset_path == "nodes_1":
    #     raw_x_sets, raw_x_dot_sets = load_nodes_1()
    #     x_sets, x_dot_sets = generate_multiple_demos(raw_x_sets, raw_x_dot_sets, n_demos, noise_std)
    # elif dataset_path == "nodes_2":
    #     raw_x_sets, raw_x_dot_sets = load_nodes_2()
    #     x_sets, x_dot_sets = generate_multiple_demos(raw_x_sets, raw_x_dot_sets, n_demos, noise_std)
    # else:
    #     # x_sets, x_dot_sets = load_drawn_trajectories(data_file)
    #     raw_x_sets, raw_x_dot_sets = load_drawn_trajectories(dataset_path)
    #     x_sets, x_dot_sets = generate_multiple_demos(raw_x_sets, raw_x_dot_sets, n_demos, noise_std)

    # get all folder with "dataset" in dataset_path
    x_sets = []
    x_dot_sets = []
    for task_dataset_path in os.listdir(dataset_path):
        if "dataset" in task_dataset_path:
            task_x = []
            task_x_dot = []
            task_dataset_path = os.path.join(dataset_path, task_dataset_path)
            for demo_path in os.listdir(task_dataset_path):
                data = json.load(open(os.path.join(task_dataset_path, demo_path)))
                demo_x, demo_x_dot = data["x"], data["x_dot"]
                task_x.append(np.array(demo_x))
                task_x_dot.append(np.array(demo_x_dot))
            x_sets.append(task_x)
            x_dot_sets.append(task_x_dot)

    # Plot the loaded/generated trajectories
    plot_trajs = []
    for demo_set in x_sets:
        plot_trajs.extend(demo_set)
    save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
    plot_trajectories(plot_trajs, "Loaded Trajectories", save_folder, config=config)

    # return _pre_process_stitch(x_sets, x_dot_sets)
    return [_pre_process(x, x_dot) for x, x_dot in zip(x_sets, x_dot_sets)]

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