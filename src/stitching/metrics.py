import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
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
    x_dot_pred = []
    
    for i in range(len(x_positions)):
        x = x_positions[i:i+1].T  # Shape (n, 1) as expected by _step
        
        # Use the same logic as in _step method
        x_dot = np.zeros((x.shape[0], 1))
        gamma = lpvds.damm.logProb(x)
        
        for k in range(lpvds.K):
            x_dot += gamma[k, 0] * lpvds.A[k] @ (x - lpvds.x_att.T)
        
        x_dot_pred.append(x_dot.flatten())
    
    return np.array(x_dot_pred)


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
    
    # Calculate squared errors
    squared_errors = np.sum((x_dot_ref - x_dot_pred) ** 2, axis=1)
    
    # Return RMSE
    return np.sqrt(np.mean(squared_errors))


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
    
    cosine_dissimilarities = []
    
    for i in range(len(x_ref)):
        pred_vec = x_dot_pred[i]
        ref_vec = x_dot_ref[i]
        
        # Calculate norms
        pred_norm = np.linalg.norm(pred_vec)
        ref_norm = np.linalg.norm(ref_vec)
        
        # Avoid division by zero
        if pred_norm == 0 or ref_norm == 0:
            cosine_dissimilarity = 1.0  # Maximum dissimilarity
        else:
            # Calculate cosine similarity
            cosine_sim = np.dot(pred_vec, ref_vec) / (pred_norm * ref_norm)
            # Clamp to [-1, 1] to handle numerical errors
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
            cosine_dissimilarity = 1.0 - cosine_sim
        
        cosine_dissimilarities.append(cosine_dissimilarity)
    
    return np.mean(cosine_dissimilarities)


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


def calculate_all_metrics(x_ref, x_dot_ref, lpvds, x_test_list, initial, attractor, 
                         ds_method, combination_id):
    """
    Calculate all metrics for a given configuration.
    
    Args:
        x_ref: Reference positions from training data
        x_dot_ref: Reference velocities from training data
        lpvds: Trained LPVDS model
        x_test_list: List of simulated trajectories
        initial: Initial position
        attractor: Goal/attractor position
        ds_method: DS method used
        combination_id: Combination identifier
        
    Returns:
        dict: Single dictionary containing aggregated metrics for this combination
    """
    # Calculate prediction metrics on reference data (once per combination)
    prediction_rmse = calculate_prediction_rmse(x_ref, x_dot_ref, lpvds)
    cosine_similarity = calculate_cosine_similarity(x_ref, x_dot_ref, lpvds)
    
    # Calculate DTW distances for all simulated trajectories
    dtw_distances = []
    final_distances_to_attractor = []
    trajectory_lengths = []
    final_positions = []
    
    for x_test in x_test_list:
        # Calculate DTW distance to reference trajectory
        dtw_distance = calculate_dtw_distance(x_ref, x_test)
        dtw_distances.append(dtw_distance)
        
        # Calculate final position metrics
        final_pos = x_test[-1]
        final_positions.append(final_pos)
        final_distances_to_attractor.append(np.linalg.norm(final_pos - attractor))
        trajectory_lengths.append(len(x_test))
    
    # Aggregate DTW and trajectory metrics
    result = {
        'combination_id': combination_id,
        'ds_method': ds_method,
        'initial_x': initial[0],
        'initial_y': initial[1],
        'attractor_x': attractor[0],
        'attractor_y': attractor[1],
        'prediction_rmse': prediction_rmse,
        'cosine_dissimilarity': cosine_similarity,
        'dtw_distance_mean': np.mean(dtw_distances),
        'dtw_distance_std': np.std(dtw_distances),
        'distance_to_attractor_mean': np.mean(final_distances_to_attractor),
        'distance_to_attractor_std': np.std(final_distances_to_attractor),
        'trajectory_length_mean': np.mean(trajectory_lengths),
        'trajectory_length_std': np.std(trajectory_lengths),
        'n_simulations': len(x_test_list)
    }
    
    return result

def calculate_ds_metrics(x_ref, x_dot_ref, ds, sim_trajectories, initial, attractor):
    """Calculates performance metrics for a dynamical system.

    Args:
        x_ref: Reference trajectory positions.
        x_dot_ref: Reference trajectory velocities.
        ds: Dynamical system to evaluate, or None.
        sim_trajectories: List of simulated trajectories.
        initial: Initial point coordinates.
        attractor: Attractor point coordinates.

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
            'distance_to_attractor_mean': np.nan,
            'distance_to_attractor_std': np.nan,
            'trajectory_length_mean': np.nan,
            'trajectory_length_std': np.nan,
            'n_simulations': 0
        }

    # Calculate prediction metrics on reference data (once per combination)
    prediction_rmse = calculate_prediction_rmse(x_ref, x_dot_ref, ds)
    cosine_similarity = calculate_cosine_similarity(x_ref, x_dot_ref, ds)

    # Calculate DTW distances for all simulated trajectories
    dtw_distances = []
    final_distances_to_attractor = []
    trajectory_lengths = []
    final_positions = []
    for trajectory in sim_trajectories:

        # Calculate DTW distance to reference trajectory
        dtw_distance = calculate_dtw_distance(x_ref, trajectory)
        dtw_distances.append(dtw_distance)

        # Calculate final position metrics
        final_pos = trajectory[-1]
        final_positions.append(final_pos)
        final_distances_to_attractor.append(np.linalg.norm(final_pos - attractor))
        trajectory_lengths.append(len(trajectory))

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
