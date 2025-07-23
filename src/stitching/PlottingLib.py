import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gaussians(mus, sigmas, directions=None, resolution=100, extent=None,
                   scaling='sqrt', ax=None):
    """Plot heatmap of summed 2D Gaussian evaluations on a grid with 2-sigma ellipses and direction arrows.

    Args:
        mus: Array of Gaussian means (N x 2)
        sigmas: Array of covariance matrices (N x 2 x 2)
        directions: Array of direction vectors (N x 2) - optional for arrows
        resolution: Grid resolution, int or tuple (width, height). Default 100.
        extent: Optional ((xmin, xmax), (ymin, ymax)) for plot bounds. If None, inferred from means.
        scaling: Scaling method for visualization. Options: 'linear', 'log', 'exponential', 'sqrt'
        ax: Optional matplotlib axis

    Returns:
        matplotlib axis object with heatmap
    """
    # Ensure inputs are numpy arrays
    mus = np.array(mus)
    sigmas = np.array(sigmas)

    # Handle single gaussian case
    if mus.ndim == 1:
        mus = mus.reshape(1, -1)
        sigmas = sigmas.reshape(1, 2, 2)

    # Determine plot extent
    if extent is None:
        buffer = 3 * np.sqrt(np.max([np.trace(sigma) for sigma in sigmas]))
        xmin, ymin = mus.min(axis=0) - buffer
        xmax, ymax = mus.max(axis=0) + buffer
    else:
        (xmin, xmax), (ymin, ymax) = extent

    # Handle resolution parameter
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    # Create coordinate grid
    x = np.linspace(xmin, xmax, resolution[0])
    y = np.linspace(ymin, ymax, resolution[1])
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # Evaluate and sum all Gaussians at each grid point
    total_values = np.zeros(len(grid_points))

    for mu, sigma in zip(mus, sigmas):
        diff = grid_points - mu
        inv_sigma = np.linalg.inv(sigma)
        mahalanobis_sq = np.sum(diff @ inv_sigma * diff, axis=1)
        norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
        gaussian_values = norm_const * np.exp(-0.5 * mahalanobis_sq)
        total_values += gaussian_values

    # Reshape back to grid
    heatmap = total_values.reshape(resolution[1], resolution[0])

    # Apply scaling transformation
    if scaling == 'exponential':
        scaled_heatmap = np.exp(heatmap) - 1
    elif scaling == 'log':
        epsilon = np.max(heatmap) * 1e-10
        scaled_heatmap = np.log(heatmap + epsilon)
    elif scaling == 'sqrt':
        scaled_heatmap = np.sqrt(heatmap)
    else:  # linear
        scaled_heatmap = heatmap

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(scaled_heatmap, extent=[xmin, xmax, ymin, ymax],
                   origin='lower', cmap='viridis', aspect='equal')

    # Add 2-sigma ellipses for each Gaussian
    for mu, sigma in zip(mus, sigmas):
        # Eigendecomposition for ellipse orientation and size
        eigenvals, eigenvecs = np.linalg.eigh(sigma)
        order = eigenvals.argsort()[::-1]  # Sort in descending order
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # Calculate ellipse parameters for 2-sigma boundary
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 4 * np.sqrt(eigenvals[0])   # 2-sigma width (2 * 2 * sqrt)
        height = 4 * np.sqrt(eigenvals[1])  # 2-sigma height (2 * 2 * sqrt)

        # Draw 2-sigma ellipse
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle,
                          edgecolor='white', facecolor='none', linewidth=1, linestyle='-')
        ax.add_patch(ellipse)

    # Add direction arrows
    if directions is not None:
        directions = np.array(directions)
        if directions.ndim == 1:
            directions = directions.reshape(1, -1)

        directions_norm = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10)

        # FIXED: Use ymin instead of ymax, and scale appropriately
        arrow_scale = min(xmax - xmin, ymax - ymin) * 0.05

        for mu, dir_vec in zip(mus, directions_norm):
            ax.arrow(mu[0], mu[1],
                     dir_vec[0] * arrow_scale, dir_vec[1] * arrow_scale,
                     head_width=arrow_scale*0.2, head_length=arrow_scale*0.15,
                     fc='white', ec='black', linewidth=2, alpha=0.8)

    return ax



