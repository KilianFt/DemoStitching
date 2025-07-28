from itertools import permutations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button


def initialize_iter_strategy(config, x_initial_sets, x_attrator_sets):
    if config.initial is not None and config.attractor is not None:
        # Use specified initial and attractor positions
        combinations = [(config.initial, config.attractor)]
        print("Using specified initial and attractor positions")
    else:
        # Use all permutations of available points
        all_points = np.vstack((x_initial_sets, x_attrator_sets))
        combinations = list(permutations(all_points, 2))
        print("Using all permutations of initial and attractor positions")


    return combinations

def get_nan_results(i, ds_method, initial, attractor):
    return {
                'combination_id': i,
                'ds_method': ds_method,
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

def is_negative_definite(matrix):
    """
    Checks if a square matrix is negative definite.

    A matrix 'A' is negative definite if the quadratic form x^T * A * x is
    strictly negative for every non-zero real vector x.

    This condition is satisfied if and only if the symmetric part of the matrix,
    (A + A^T) / 2, has all strictly negative eigenvalues.

    Args:
        matrix (array-like): The matrix to check. It can be a list of lists or a NumPy array.

    Returns:
        bool: True if the matrix is negative definite, False otherwise.
    """
    try:
        # Convert input to a NumPy array to ensure it has the necessary methods.
        A = np.array(matrix, dtype=float)
    except (ValueError, TypeError):
        # Return False if the input cannot be converted to a numeric NumPy array.
        print("Error: Input could not be converted to a numeric matrix.")
        return False

    # A matrix must be square to be considered for definiteness.
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print(f"Error: Input is not a square matrix. Shape is {A.shape}.")
        return False

    # For the quadratic form x^T*A*x, only the symmetric part of A matters.
    # The definiteness of A is determined by the eigenvalues of its symmetric part.
    symmetric_part = (A + A.T) / 2
    
    try:
        # For a real symmetric matrix, all eigenvalues are guaranteed to be real.
        eigenvalues = np.linalg.eigvals(symmetric_part)
    except np.linalg.LinAlgError:
        # The eigenvalue computation may fail for ill-conditioned matrices.
        print("Error: Eigenvalue computation did not converge.")
        return False

    # The matrix is negative definite if and only if all its eigenvalues are negative.
    return np.all(eigenvalues < 0)

def initial_goal_picker(data):
    """
    Interactive UI for selecting initial and goal positions from Gaussian data.
    
    Args:
        data (dict): Dictionary containing 'centers', 'sigmas', 'directions' arrays
        
    Returns:
        tuple: (initial_position, goal_position) as numpy arrays
    """
    centers = data['centers']
    sigmas = data.get('sigmas', None)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Select Initial (Green) and Goal (Red) Positions\nDrag the colored points to desired locations')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    # Plot all Gaussians as circles
    gaussian_circles = []
    for i, center in enumerate(centers):
        if sigmas is not None and len(sigmas) > i:
            # Use sigma to determine circle size (assuming 2D)
            if sigmas[i].ndim == 2:
                # Take average of diagonal elements as radius
                radius = np.sqrt(np.mean(np.diag(sigmas[i])))# * 2
            else:
                radius = np.mean(sigmas[i])# * 2
        else:
            radius = 0.1  # Default radius
        
        circle = Circle(center[:2], radius, fill=False, color='blue', alpha=0.5, linewidth=1)
        ax.add_patch(circle)
        gaussian_circles.append(circle)

    ax.scatter(centers[:, 0], centers[:, 1], s=50, marker='x', color='blue', alpha=0.5)

    # Set axis limits based on data
    x_min, x_max = np.min(centers[:, 0]) - 1, np.max(centers[:, 0]) + 1
    y_min, y_max = np.min(centers[:, 1]) - 1, np.max(centers[:, 1]) + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Initialize positions (center of data range)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Create draggable points
    initial_point = ax.plot(center_x - 0.5, center_y, 'go', markersize=15, label='Initial', picker=True)[0]
    goal_point = ax.plot(center_x + 0.5, center_y, 'ro', markersize=15, label='Goal', picker=True)[0]
    
    ax.legend()
    
    # State variables
    selected_point = None
    positions = {
        'initial': np.array([center_x - 0.5, center_y]),
        'goal': np.array([center_x + 0.5, center_y])
    }
    
    def on_pick(event):
        nonlocal selected_point
        if event.artist == initial_point:
            selected_point = 'initial'
        elif event.artist == goal_point:
            selected_point = 'goal'
    
    def on_motion(event):
        nonlocal selected_point
        if selected_point is not None and event.inaxes == ax:
            # Update position
            positions[selected_point] = np.array([event.xdata, event.ydata])
            
            # Update visual
            if selected_point == 'initial':
                initial_point.set_data([event.xdata], [event.ydata])
            elif selected_point == 'goal':
                goal_point.set_data([event.xdata], [event.ydata])
            
            fig.canvas.draw_idle()
    
    def on_release(event):
        nonlocal selected_point
        selected_point = None
    
    # Connect events
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    
    # Add confirm button
    ax_button = plt.axes([0.81, 0.01, 0.15, 0.05])
    button = Button(ax_button, 'Confirm Selection')
    
    confirmed = {'value': False}
    
    def on_confirm(event):
        confirmed['value'] = True
        plt.close(fig)
    
    button.on_clicked(on_confirm)
    
    # Show plot and wait for user interaction
    plt.tight_layout()
    plt.show()
    
    # Return selected positions
    return positions['initial'], positions['goal']


# --- Example Usage ---
if __name__ == '__main__':
    # Example 1: A negative definite matrix
    neg_def_matrix = [[-3, 1], [1, -3]]
    print(f"Is the matrix {neg_def_matrix} negative definite? ", is_negative_definite(neg_def_matrix))
    # Expected output: True

    # Example 2: A positive definite matrix (should be False)
    pos_def_matrix = [[2, -1], [-1, 2]]
    print(f"Is the matrix {pos_def_matrix} negative definite? ", is_negative_definite(pos_def_matrix))
    # Expected output: False

    # Example 3: A non-symmetric but negative definite matrix
    # Its symmetric part is [[-2, 1.5], [1.5, -4]], which is negative definite.
    non_sym_matrix = [[-2, 2], [1, -4]]
    print(f"Is the matrix {non_sym_matrix} negative definite? ", is_negative_definite(non_sym_matrix))
    # Expected output: True

    # Example 4: A singular matrix (one eigenvalue is zero, so not strictly negative)
    singular_matrix = [[-1, 1], [1, -1]]
    print(f"Is the matrix {singular_matrix} negative definite? ", is_negative_definite(singular_matrix))
    # Expected output: False
    
    # Example 5: A non-square matrix
    non_square_matrix = [[1, 2, 3], [4, 5, 6]]
    print(f"Is the matrix {non_square_matrix} negative definite? ", is_negative_definite(non_square_matrix))
    # Expected output: False
