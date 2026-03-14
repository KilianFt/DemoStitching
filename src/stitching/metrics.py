import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cdist
from fastdtw import fastdtw


def predict_velocities(x_positions, lpvds):
    """
    Predict velocities for given positions using the LPVDS model.
    
    Args:
        x_positions: Input positions (M x n)
        lpvds: Trained LPVDS model
        
    Returns:
        x_dot_pred: Predicted velocities (M x n)
    """
    # Allow policy-specific predictors (e.g., chained DS with state-dependent switching).
    if hasattr(lpvds, "predict_velocities"):
        x_dot_pred = lpvds.predict_velocities(x_positions)
        return np.asarray(x_dot_pred)

    x_positions = np.atleast_2d(np.asarray(x_positions, dtype=float))
    gamma = np.asarray(lpvds.damm.compute_gamma(x_positions), dtype=float)
    x_shift = x_positions - np.asarray(lpvds.x_att, dtype=float).reshape(1, -1)

    x_dot_pred = np.zeros_like(x_shift)
    for k in range(lpvds.A.shape[0]):
        x_dot_pred += gamma[k, :].reshape(-1, 1) * (x_shift @ np.asarray(lpvds.A[k], dtype=float).T)

    return x_dot_pred


def _prediction_rmse_from_pred(x_dot_ref, x_dot_pred):
    x_dot_ref = np.asarray(x_dot_ref, dtype=float)
    x_dot_pred = np.asarray(x_dot_pred, dtype=float)
    squared_errors = np.sum((x_dot_ref - x_dot_pred) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_errors)))


def _cosine_dissimilarity_from_pred(x_dot_ref, x_dot_pred):
    x_dot_ref = np.asarray(x_dot_ref, dtype=float)
    x_dot_pred = np.asarray(x_dot_pred, dtype=float)

    ref_norms = np.linalg.norm(x_dot_ref, axis=1)
    pred_norms = np.linalg.norm(x_dot_pred, axis=1)
    denom = ref_norms * pred_norms

    dots = np.sum(x_dot_ref * x_dot_pred, axis=1)
    cosine_sim = np.zeros_like(dots)
    valid = denom > 0.0
    cosine_sim[valid] = dots[valid] / denom[valid]
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    cosine_dissimilarity = np.ones_like(cosine_sim)
    cosine_dissimilarity[valid] = 1.0 - cosine_sim[valid]
    return float(np.mean(cosine_dissimilarity))


def calculate_prediction_rmse(x_ref, x_dot_ref, lpvds):
    """
    Calculate Prediction RMSE = 1/M * sum(||x_dot_ref - f(x_ref)||^2)
    
    Args:
        x_ref: Reference positions (M x n)
        x_dot_ref: Reference velocities (M x n)
        lpvds: Trained LPVDS model
        
    Returns:
        float: RMSE value
    """
    # Predict velocities using the model
    x_dot_pred = predict_velocities(x_ref, lpvds)
    
    return _prediction_rmse_from_pred(x_dot_ref=x_dot_ref, x_dot_pred=x_dot_pred)


def calculate_cosine_similarity(x_ref, x_dot_ref, lpvds):
    """
    Calculate Prediction cosine similarity = 1/M * sum(1 - (f(x_ref)^T * x_dot_ref) / (||f(x_ref)|| * ||x_dot_ref||))
    
    Args:
        x_ref: Reference positions (M x n)
        x_dot_ref: Reference velocities (M x n)
        lpvds: Trained LPVDS model
        
    Returns:
        float: Average cosine dissimilarity (1 - cosine similarity)
    """
    # Predict velocities using the model
    x_dot_pred = predict_velocities(x_ref, lpvds)
    
    return _cosine_dissimilarity_from_pred(x_dot_ref=x_dot_ref, x_dot_pred=x_dot_pred)


def calculate_dtw_distance(traj_ref, traj_sim):
    """
    Calculate Dynamic Time Warping Distance between reference and simulated trajectories.
    
    Args:
        traj_ref: Reference trajectory (M x n)
        traj_sim: Simulated trajectory (N x n)
        
    Returns:
        float: DTW distance
    """
    # Use fastdtw for efficient computation
    distance, _ = fastdtw(traj_ref, traj_sim, dist=euclidean)
    return distance

def demo_set_spread(demo_set):
    """ Calculates the average distance for each point to another point from another trajectory"""
    trajectories = {}
    for i, demo in enumerate(demo_set):
        demo_trajectories = getattr(demo, "trajectories", None)
        if demo_trajectories is None:
            continue
        for j, traj in enumerate(demo_trajectories):
            x = np.asarray(getattr(traj, "x", []), dtype=float)
            if x.ndim != 2 or x.shape[0] < 2:
                continue
            trajectories[(i, j)] = traj

    if len(trajectories) <= 1:
        return 0.0, 0.0

    # skip last point since it is the attractor (same for all trajectories) in a demo
    all_pos = np.concatenate([traj.x[:-1] for traj in trajectories.values()], axis=0)
    traj_idx = np.concatenate([i*np.ones(len(traj.x)-1) for i, traj in enumerate(trajectories.values())])

    all_distances = np.linalg.norm(all_pos[:, None, :] - all_pos[None, :, :], axis=-1)
    min_distances = np.zeros(all_pos.shape[0])
    for i in range(len(trajectories)):

        traj_i_indices = np.where(traj_idx == i)[0]
        traj_i_distances = all_distances[traj_i_indices][:, traj_idx != i]
        min_distances[traj_i_indices] = np.min(traj_i_distances, axis=1)

    mean_mindist, std_mindist = np.mean(min_distances), np.std(min_distances)

    return mean_mindist, std_mindist

def calculate_demo_spread(x_ref, x_test, mean_mindist, std_mindist):
    """
    Evaluates how far the test trajectory points are from the reference trajectory.
    Returns a score between 0 and 1, where 1 means it's within the typical demo spread.
    Works for arbitrary dimensions (2D, 3D, etc.).
    """
    if len(x_test) == 0 or len(x_ref) == 0:
        return 0.0

    # Compute pairwise distances between all points in x_test and x_ref
    # x_test: (N, D), x_ref: (M, D) -> distances: (N, M)
    distances = cdist(x_test, x_ref)
    
    # For each test point, find the distance to the closest reference point
    min_dists = np.min(distances, axis=1)

    # Evaluate min_dist relative to the normal distribution of min dists among the demo points
    scores = np.ones_like(min_dists)
    mask = min_dists >= mean_mindist
    scores[mask] = np.exp(-0.5 * ((min_dists[mask] - mean_mindist) / std_mindist) ** 2)

    return float(np.mean(scores))


def calculate_ds_metrics(x_ref, x_dot_ref, ds, sim_trajectories, initial, attractor, mean_mindist=None, std_mindist=None):
    """Calculates performance metrics for a dynamical system.

    Args:
        x_ref: Reference trajectory positions.
        x_dot_ref: Reference trajectory velocities.
        ds: Dynamical system to evaluate, or None.
        sim_trajectories: List of simulated trajectories.
        initial: Initial point coordinates.
        attractor: Attractor point coordinates.
        mean_mindist: Mean of min distance for demo spread calculation.
        std_mindist: Std of min distance for demo spread calculation.

    Returns:
        dict: Performance metrics including RMSE, DTW distances, and trajectory stats.
    """
    # If DS is None, return NaN metrics
    if ds is None:
        return {
            'initial_x': initial[0],
            'initial_y': initial[1],
            'attractor_x': attractor[0],
            'attractor_y': attractor[1],
            'prediction_rmse': np.nan,
            'cosine_dissimilarity': np.nan,
            'dtw_distance_mean': np.nan,
            'dtw_distance_std': np.nan,
            'demo_spread_mean': np.nan,
            'demo_spread_std': np.nan,
            'distance_to_attractor_mean': np.nan,
            'distance_to_attractor_std': np.nan,
            'trajectory_length_mean': np.nan,
            'trajectory_length_std': np.nan,
            'n_simulations': 0
        }

    # Calculate prediction metrics on reference data (single forward pass).
    x_dot_pred = predict_velocities(x_ref, ds)
    prediction_rmse = _prediction_rmse_from_pred(x_dot_ref=x_dot_ref, x_dot_pred=x_dot_pred)
    cosine_similarity = _cosine_dissimilarity_from_pred(x_dot_ref=x_dot_ref, x_dot_pred=x_dot_pred)

    # Calculate DTW distances for all simulated trajectories
    dtw_distances = []
    final_distances_to_attractor = []
    trajectory_lengths = []
    final_positions = []
    demo_spreads = []
    for trajectory in sim_trajectories:

        if np.any(np.isnan(trajectory)):
            print(f"Warning: Skipping trajectory with NaNs")
            dtw_distances.append(np.nan)
            final_distances_to_attractor.append(np.nan)
            trajectory_lengths.append(len(trajectory))
            continue

        # Calculate DTW distance to reference trajectory
        dtw_distance = calculate_dtw_distance(x_ref, trajectory)
        dtw_distances.append(dtw_distance)

        # calculate in data percentage
        if mean_mindist is not None and std_mindist is not None:
            demo_spread = calculate_demo_spread(x_ref, trajectory, mean_mindist, std_mindist)
        else:
            demo_spread = np.nan

        # Calculate final position metrics
        final_pos = trajectory[-1]
        final_positions.append(final_pos)
        final_distances_to_attractor.append(np.linalg.norm(final_pos - attractor))
        trajectory_lengths.append(len(trajectory))
        demo_spreads.append(demo_spread)

    # Aggregate metrics
    result = {
        'initial_x': initial[0],
        'initial_y': initial[1],
        'attractor_x': attractor[0],
        'attractor_y': attractor[1],
        'prediction_rmse': prediction_rmse,
        'cosine_dissimilarity': cosine_similarity,
        'dtw_distance_mean': np.mean(dtw_distances),
        'dtw_distance_std': np.std(dtw_distances),
        'demo_spread_mean': np.mean(demo_spreads),
        'demo_spread_std': np.std(demo_spreads),
        'distance_to_attractor_mean': np.mean(final_distances_to_attractor),
        'distance_to_attractor_std': np.std(final_distances_to_attractor),
        'trajectory_length_mean': np.mean(trajectory_lengths),
        'trajectory_length_std': np.std(trajectory_lengths),
        'n_simulations': len(sim_trajectories)
    }

    return result


def save_results_dataframe(all_results, save_path):
    """
    Save all results to a CSV file.
    
    Args:
        all_results: List of all result dictionaries
        save_path: Path to save the CSV file
    """
    if not all_results:
        print("No results to save.")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    
    # Print summary statistics
    print(f"\nResults saved to: {save_path}")
    print(f"Total simulations: {len(df)}")
    print(f"Average Prediction RMSE: {df['prediction_rmse'].mean():.4f} ± {df['prediction_rmse'].std():.4f}")
    print(f"Average Cosine Dissimilarity: {df['cosine_dissimilarity'].mean():.4f} ± {df['cosine_dissimilarity'].std():.4f}")
    print(f"Average DTW Distance: {df['dtw_distance_mean'].mean():.4f} ± {df['dtw_distance_mean'].std():.4f}")
    print(f"Average Distance to Attractor: {df['distance_to_attractor_mean'].mean():.4f} ± {df['distance_to_attractor_mean'].std():.4f}")
