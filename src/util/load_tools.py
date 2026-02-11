import os
import json
import numpy as np
from src.util.ds_tools import Demonstration, Trajectory
from src.util.generate_data import generate_data


def _numeric_suffix_sort_key(name):
    stem = os.path.splitext(name)[0]
    suffix = stem.split('_')[-1]
    if suffix.isdigit():
        return (stem[: -len(suffix)], int(suffix), name)
    return (stem, float('inf'), name)

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
    for folder_name in sorted(os.listdir(demoset_path), key=_numeric_suffix_sort_key):
       if 'demonstration' in folder_name or 'dataset' in folder_name:
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
        for trajectory_file in sorted(os.listdir(demo_path), key=_numeric_suffix_sort_key):
            trajectory_path = os.path.join(demo_path, trajectory_file)
            with open(trajectory_path, "r", encoding="utf-8") as f:
                trajectory = json.load(f)
            trajectory = Trajectory(np.array(trajectory['x']), np.array(trajectory['x_dot']))
            trajectories.append(trajectory)

        demonstrations.append(
            Demonstration(trajectories)
        )

    return demonstrations
