import numpy as np
# from scipy.io import loadmat
from src.lpvds_class import lpvds_class
from collections import namedtuple

Demoset = namedtuple('Demoset', ['x', 'x_dot'])
Trajectory = namedtuple('Trajectory', ['x', 'x_dot'])

class Demonstration:
    """Class representing a set of trajectories in a demonstration."""

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.x = []
        self.x_dot = []
        for trajectory in trajectories:
            self.x.extend(trajectory.x)
            self.x_dot.extend(trajectory.x_dot)

        self.x = np.array(self.x)
        self.x_dot = np.array(self.x_dot)

    def __str__(self):
        return f'Demonstration with {len(self.trajectories)} trajectories, total points: {self.x.shape[0]}'

def apply_lpvds_demowise(demo_set, config=None):
    """Fits an LPV-DS to each demonstration in the set after aligning to a common attractor.

    Args:
        demo_set: List of Demonstration objects with trajectory data.
        config: Configuration object with DAMM settings.

    Returns:
        tuple: (ds_set, norm_demo_set) where ds_set contains LPV-DS parameter
            objects for each demonstration, and norm_demo_set is the corresponding
            set with trajectories shifted to a common attractor.
    """
    # Normalize each demonstration to a common attractor
    norm_demo_set, attractors, initial_points = normalize_demo_set(demo_set)

    rel_scale = getattr(config, 'rel_scale', 0.7)
    total_scale = getattr(config, 'total_scale', 1.5)
    nu_0 = getattr(config, 'nu_0', 5)
    kappa_0 = getattr(config, 'kappa_0', 1)
    psi_dir_0 = getattr(config, 'psi_dir_0', 1)

    # Apply LPV-DS to each normalized demonstration
    ds_set = []
    reversed_ds_set = []
    for demo, attractor, initial_point in zip(norm_demo_set, attractors, initial_points):

        # Apply LPV-DS normal direction
        for i in range(10):
            try:
                lpvds = lpvds_class(demo.x, demo.x_dot, x_att=attractor, 
                                    rel_scale=rel_scale, total_scale=total_scale,
                                    nu_0=nu_0, kappa_0=kappa_0, psi_dir_0=psi_dir_0)
                result = lpvds.begin()
                if not result:
                    print('Failed to fit LPV-DS, retrying...')
                else:
                    ds_set.append(lpvds)
                    break
            except Exception as e:
                print(f"Failed to fit LPV-DS, retrying... Error: {e}")


        # Apply LPV-DS reverse direction
        for i in range(10):
            try:
                lpvds = lpvds_class(demo.x, -demo.x_dot, x_att=initial_point,
                                    rel_scale=rel_scale, total_scale=total_scale,
                                    nu_0=nu_0, kappa_0=kappa_0, psi_dir_0=psi_dir_0)
                result = lpvds.begin()
                if not result:
                    print('Failed to fit LPV-DS, retrying...')
                else:
                    reversed_ds_set.append(lpvds)
                    break
            except Exception as e:
                print(f"Failed to fit LPV-DS, retrying... Error: {e}")

    return ds_set, reversed_ds_set, norm_demo_set

def normalize_demo_set(demo_set):
    """Shifts demonstration trajectories so their endpoints coincide at a common mean attractor.

    Args:
        demo_set: List of Demonstration objects, each with trajectory data.

    Returns:
        tuple: (normalized_demonstrations, attractors), where each normalized
            Demonstration has trajectories ending at the shared attractor.
    """
    normalized_demonstrations = []
    attractors = []
    initial_points = []
    for demo in demo_set:

        # Get the demonstration's attractor (mean of end points)
        end_points = [traj.x[-1] for traj in demo.trajectories]
        attractor = np.mean(end_points, axis=0)
        attractors.append(attractor)

        # Get the demonstration's initial point (mean of start points)
        demo_inits = [traj.x[0] for traj in demo.trajectories]
        initial_point = np.mean(demo_inits, axis=0)
        initial_points.append(initial_point)

        # Normalize each trajectory to end at the attractor position
        normalized_trajectories = []
        for trajectory in demo.trajectories:

            normalized_x = trajectory.x - trajectory.x[-1] + attractor
            normalized_trajectories.append(
                Trajectory(x=normalized_x, x_dot=trajectory.x_dot)
            )

        # Compile the normalized demonstration
        normalized_demo = Demonstration(normalized_trajectories)
        normalized_demonstrations.append(normalized_demo)

    return normalized_demonstrations, attractors, initial_points

def get_gaussian_directions(lpvds, method: str = "mean_velocity"):
    """Computes normalized direction vectors for each Gaussian in an LPV-DS.

    Args:
        lpvds: LPV-DS object with damm.gaussian_lists (with 'mu') and A matrices.

    Returns:
        np.ndarray: Array of normalized direction vectors for all Gaussians.
    """
    def _normalize_or_none(v):
        v = np.asarray(v, dtype=float).reshape(-1)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n <= 1e-12:
            return None
        return v / n

    method = str(method).strip().lower()
    if method not in {"mean_velocity", "a_mu"}:
        raise ValueError(f"Unknown gaussian direction method: {method}")

    x = np.asarray(lpvds.x, dtype=float)
    x_dot = np.asarray(lpvds.x_dot, dtype=float)
    x_att = np.asarray(lpvds.x_att, dtype=float).reshape(-1)
    assignment_arr = None
    if method == "mean_velocity":
        gamma = np.asarray(lpvds.damm.compute_gamma(x), dtype=float)
        assignment_arr = np.argmax(gamma, axis=0)

    directions = []
    for i, gaussian in enumerate(lpvds.damm.gaussian_lists):
        d = None

        if method == "mean_velocity":
            # Primary definition: mean velocity from points assigned by highest posterior.
            mask = assignment_arr == i
            if np.any(mask):
                d = _normalize_or_none(np.mean(x_dot[mask], axis=0))

        # Fallback for empty/degenerate mean velocity or legacy mode.
        if d is None:
            try:
                d = _normalize_or_none(lpvds.A[i] @ (np.asarray(gaussian["mu"], dtype=float) - x_att))
            except Exception:
                d = None

        if d is None:
            d = np.zeros(x.shape[1], dtype=float)
            d[0] = 1.0
        directions.append(d)

    directions = np.asarray(directions, dtype=float)

    return directions
