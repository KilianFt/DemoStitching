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
