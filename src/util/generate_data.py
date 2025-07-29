import os
import numpy as np
import json
from typing import List

from src.stitching.trajectory_drawer import EnhancedTrajectoryDrawer


def generate_data(dataset_path):


    x_sets, x_dot_sets = draw_trajectories()

    # save each trajectory to a file
    for i, (task_x, task_x_dot) in enumerate(zip(x_sets, x_dot_sets)):
        task_dataset_path = os.path.join(dataset_path, "demonstration_{}".format(i))
        os.makedirs(task_dataset_path, exist_ok=True)
        for j, (demo_x, demo_x_dot) in enumerate(zip(task_x, task_x_dot)):
            # save to json
            with open(os.path.join(task_dataset_path, "trajectory_{}.json".format(j)), "w") as f:
                json.dump({"x": demo_x.tolist(), "x_dot": demo_x_dot.tolist()}, f)    


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
    n_demos = 5
    noise_std = 0.05
    x_sets, x_dot_sets = generate_multiple_demos(x_sets, x_dot_sets, n_demos, noise_std)
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
    n_demos = 5
    noise_std = 0.05
    x_sets, x_dot_sets = generate_multiple_demos(raw_x_sets, raw_x_dot_sets, n_demos, noise_std)
    return x_sets, x_dot_sets


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


def draw_trajectories():
    """
    Draw multiple trajectory sets interactively.
    
    Provides an enhanced interactive drawing interface that allows users to:
    1. Draw multiple trajectories within a set
    2. Start new trajectory sets
    3. Organize trajectories into sets for different tasks/behaviors
    
    Returns:
    --------
    tuple[List[List[np.ndarray]], List[List[np.ndarray]]] or None
        x_sets: List of trajectory sets, each containing multiple trajectories
        x_dot_sets: List of velocity sets, each containing velocities for trajectories
        Returns None if no trajectories are drawn
    """
    
    drawer = EnhancedTrajectoryDrawer()
    
    drawer.start_interactive_drawing()
    
    if not drawer.trajectory_sets:
        print("No trajectory sets were drawn.")
        return None
    
    # Convert trajectory sets to the required format and calculate velocities
    x_sets = []
    x_dot_sets = []
    
    for traj_set in drawer.trajectory_sets:
        if len(traj_set) > 0:  # Only include non-empty sets
            x_sets.append(traj_set)
            # Calculate velocities for each trajectory in the set
            x_dot_set = []
            for traj in traj_set:
                x_dot = np.gradient(traj, axis=0)
                x_dot_set.append(x_dot)
            x_dot_sets.append(x_dot_set)
    
    print(f"Generated {len(x_sets)} trajectory sets with {[len(s) for s in x_sets]} trajectories each")
    return x_sets, x_dot_sets
