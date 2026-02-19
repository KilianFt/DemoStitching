import os
import json
import numpy as np
from src.util.ds_tools import Demonstration, Trajectory
from src.util.generate_data import generate_data


def infer_state_dim_from_demo_set(demo_set):
    for demo in demo_set:
        for traj in demo.trajectories:
            x = np.asarray(traj.x, dtype=float)
            if x.ndim == 2 and x.shape[0] > 0 and x.shape[1] > 0:
                return int(x.shape[1])
    return 2


def compute_plot_extent_from_demo_set(
    demo_set,
    state_dim: int,
    padding_ratio: float = 0.08,
    padding_abs: float = 0.5,
):
    dim = 3 if int(state_dim) >= 3 else 2
    points = []
    for demo in demo_set:
        for traj in demo.trajectories:
            x = np.asarray(traj.x, dtype=float)
            if x.ndim == 2 and x.shape[0] > 0 and x.shape[1] >= dim:
                points.append(x[:, :dim])

    if len(points) == 0:
        if dim == 3:
            return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        return (-1.0, 1.0, -1.0, 1.0)

    pts = np.vstack(points)
    finite_mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite_mask]
    if pts.shape[0] == 0:
        if dim == 3:
            return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        return (-1.0, 1.0, -1.0, 1.0)

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    max_span = float(np.max(spans))
    margin = max(float(padding_abs), float(padding_ratio) * max_span)
    half_extent = 0.5 * max_span + margin
    center = 0.5 * (mins + maxs)
    lows = center - half_extent
    highs = center + half_extent

    if dim == 3:
        return (
            float(lows[0]), float(highs[0]),
            float(lows[1]), float(highs[1]),
            float(lows[2]), float(highs[2]),
        )
    return (
        float(lows[0]), float(highs[0]),
        float(lows[1]), float(highs[1]),
    )

def _numeric_suffix_sort_key(name):
    stem = os.path.splitext(name)[0]
    suffix = stem.split('_')[-1]
    if suffix.isdigit():
        return (stem[: -len(suffix)], int(suffix), name)
    return (stem, float('inf'), name)

def _resolve_data_scales(position_scale=1.0, velocity_scale=None):
    if position_scale is None:
        position_scale = 1.0
    if velocity_scale is None:
        velocity_scale = position_scale
    position_scale = float(position_scale)
    velocity_scale = float(velocity_scale)
    if not np.isfinite(position_scale) or position_scale <= 0.0:
        position_scale = 1.0
    if not np.isfinite(velocity_scale) or velocity_scale <= 0.0:
        velocity_scale = position_scale
    return position_scale, velocity_scale


def resolve_data_scales(config=None):
    if config is None:
        return 1.0, 1.0
    position_scale = getattr(config, "data_position_scale", getattr(config, "damm_position_scale", 1.0))
    velocity_scale = getattr(config, "data_velocity_scale", getattr(config, "damm_velocity_scale", None))
    return _resolve_data_scales(position_scale=position_scale, velocity_scale=velocity_scale)


def get_demonstration_set(demoset_path, position_scale=1.0, velocity_scale=None):
    """Loads demonstrations, or prompts user to generate new ones if none exist.

    Args:
        demoset_path: Path to directory containing demonstration folders.
        position_scale: Scalar multiplier applied to loaded positions.
        velocity_scale: Scalar multiplier applied to loaded velocities.

    Returns:
        list: List (set of demonstrations) of lists (demonstrations) of Trajectory objects.
    """
    demoset = load_demonstration_set(
        demoset_path,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
    )
    if demoset is None:
        print(f'No demonstrations found in \"{demoset_path}\". Drawing new demonstrations.')
        generate_data(demoset_path)
        demoset = load_demonstration_set(
            demoset_path,
            position_scale=position_scale,
            velocity_scale=velocity_scale,
        )

    return demoset

def load_demonstration_set(demoset_path, position_scale=1.0, velocity_scale=None):
    """Loads demonstration trajectories from a folder of trajectory JSON files.

    Args:
        demoset_path: Path to folder containing demonstration subfolders.
        position_scale: Scalar multiplier applied to loaded positions.
        velocity_scale: Scalar multiplier applied to loaded velocities.

    Returns:
        list: List of Demonstration objects with concatenated trajectory data
            and individual Trajectory objects, or None if path doesn't exist
            or no demonstrations found.
    """
    position_scale, velocity_scale = _resolve_data_scales(position_scale, velocity_scale)

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
            trajectory = Trajectory(
                np.asarray(trajectory['x'], dtype=float) * position_scale,
                np.asarray(trajectory['x_dot'], dtype=float) * velocity_scale,
            )
            trajectories.append(trajectory)

        demonstrations.append(
            Demonstration(trajectories)
        )

    return demonstrations
