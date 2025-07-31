import os, sys
import json
import pickle
import numpy as np
import pyLasaDataset as lasa
from scipy.io import loadmat
from src.util.ds_tools import lpvds_per_demo, _pre_process, _process_bag, Demonstration, Demoset, Trajectory
from src.util.generate_data import generate_data

def get_demonstration_set(demoset_path):
    """Loads demonstrations, or prompts user to generate new ones if none exist.

    Args:
        demoset_path: Path to directory containing demonstration folders.

    Returns:
        list: List (set of demonstrations) of lists (demonstrations) of Trajectory objects.
    """
    demoset = load_demonstration_set(demoset_path)
    if demoset is None:
        print(f'No demonstrations found in \"{demoset_path}\". Drawing new demonstrations.')
        generate_data(demoset_path)
        demoset = load_demonstration_set(demoset_path)

    return demoset

def load_demonstration_set(demoset_path):
    """Loads demonstration trajectories from a folder of trajectory JSON files.

    Args:
        demoset_path: Path to folder containing demonstration subfolders.

    Returns:
        list: List of Demonstration objects with concatenated trajectory data
            and individual Trajectory objects, or None if path doesn't exist
            or no demonstrations found.
    """

    if not os.path.exists(demoset_path):
        print(f'Path \"{demoset_path}\" does not exist.')
        return None

    # Collect all demonstration folders
    demonstration_folders = []
    for folder_name in os.listdir(demoset_path):
       if 'demonstration' in folder_name:
           demonstration_folders.append(folder_name)

    # Return if the folder does not contain any demonstrations
    if not demonstration_folders:
        print(f'No demonstration folders found in \"{demoset_path}\".')
        return None

    # Collect all trajectories from every demonstration folder and save them as Demonstration objects
    demonstrations = []
    for demo_folder in demonstration_folders:
        demo_path = os.path.join(demoset_path, demo_folder)

        trajectories = []
        for trajectory_file in os.listdir(demo_path):
            trajectory_path = os.path.join(demo_path, trajectory_file)
            trajectory = json.load(open(trajectory_path))
            trajectory = Trajectory(np.array(trajectory['x']), np.array(trajectory['x_dot']))
            trajectories.append(trajectory)

        demonstrations.append(
            Demonstration(trajectories)
        )

    return demonstrations
