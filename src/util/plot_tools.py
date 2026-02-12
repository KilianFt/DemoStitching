import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from src.util.ds_tools import get_gaussian_directions
import random
import os

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": 20
})

def plot_demonstration_set(demo_set, config, ax=None, save_as=None):
    """Plots grouped demonstration trajectories with start and end points, optionally saving the figure.

    Args:
        demo_set: List of Demonstration objects, each containing Trajectory objects to plot.
        config: Configuration object specifying plot extent, dataset path, and ds_method.
        ax: Optional matplotlib axis. If None, a new figure and axis are created.
        save_as: Optional filename (without extension) to save the plot as PNG.

    Returns:
        matplotlib.axes.Axes: The axis containing the plotted demonstration set.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Generate colors from colormap - one color per demonstration
    colors = plt.cm.get_cmap('tab10', len(demo_set)).colors

    # Plot each demonstration with its assigned color
    for i, demo in enumerate(demo_set):
        ax = primitive_plot_demo(ax, demo, color=colors[i])

    # Plot settings
    if hasattr(config, 'plot_extent'):
        ax.set_xlim(config.plot_extent[0], config.plot_extent[1])
        ax.set_ylim(config.plot_extent[2], config.plot_extent[3])
    ax.set_aspect('equal')
    plt.tight_layout()

    # Save the figure if file_name is provided
    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder + save_as + '.pdf')

    # Close the figure if we created it
    if ax is None:
        plt.close()

    return ax

def plot_ds_set_gaussians(ds_set, config, include_trajectory=False, ax=None, save_as=None):
    """Plots means, covariances, and directions for all Gaussians in the DS set, with optional data point arrows.

    Args:
        ds_set: List of DS objects, each containing fitted Gaussian parameters and data assignments.
        config: Configuration object providing plotting and dataset settings.
        include_trajectory: If True, overlays trajectory points and velocity arrows assigned to Gaussians.
        ax: Optional matplotlib axis. If None, creates a new figure and axis.
        save_as: Optional filename (without extension) to save the plot as PNG.

    Returns:
        matplotlib.axes.Axes: The axis with the plotted Gaussians and any optional data points.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the gaussians
    colors = plt.cm.get_cmap('tab10', len(ds_set)).colors
    for i, ds in enumerate(ds_set):
        mus = ds.damm.Mu
        sigmas = ds.damm.Sigma
        directions = get_gaussian_directions(ds)

        for mu, sigma, direction in zip(mus, sigmas, directions):
            ax = primitive_plot_gaussian(ax, mu, sigma, color=colors[i], direction=direction, sigma_bound=2)

    # Add trajectory points if requested
    if include_trajectory:
        for i, ds in enumerate(ds_set):
            ax = primitive_plot_trajectory_points(ax, ds.x, ds.x_dot, color=colors[i])

    # set limits
    if hasattr(config, 'plot_extent'):
        ax.set_xlim(config.plot_extent[0], config.plot_extent[1])
        ax.set_ylim(config.plot_extent[2], config.plot_extent[3])
    ax.set_aspect('equal')
    plt.tight_layout()

    # save the figure
    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder + save_as + '.pdf')

    return ax

def plot_gg_solution(gg, solution_nodes, config, ax=None, save_as=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax = gg.plot(ax=ax, nodes=solution_nodes)

    # plot gaussians
    for node in solution_nodes:
        mu, sigma, direction, _ = gg.get_gaussian(node)
        ax = primitive_plot_gaussian(ax, mu, sigma, color='orange', direction=direction)

    # set limits
    if hasattr(config, 'plot_extent'):
        ax.set_xlim(config.plot_extent[0], config.plot_extent[1])
        ax.set_ylim(config.plot_extent[2], config.plot_extent[3])
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder + save_as + '.pdf')

    return ax

def plot_ds(lpvds, x_test_list, config, ax=None, save_as=None):
    """Plots the DS vector field and simulated trajectories in 2D.

    Args:
        lpvds: The DS object containing the fitted model.
        x_test_list: List of simulated trajectory arrays to plot.
        config: Configuration object with plotting extent and dataset settings.
        ax: Optional matplotlib axis. If None, creates a new figure and axis.
        save_as: Optional filename (without extension) to save the plot as PNG.

    Returns:
        matplotlib.axes.Axes: The axis with the plotted DS vector field and trajectories.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    plot_ds_2d(lpvds.x, x_test_list, lpvds, ax=ax,
               x_min=config.plot_extent[0], x_max=config.plot_extent[1],
               y_min=config.plot_extent[2], y_max=config.plot_extent[3])

    # set limits
    if hasattr(config, 'plot_extent'):
        ax.set_xlim(config.plot_extent[0], config.plot_extent[1])
        ax.set_ylim(config.plot_extent[2], config.plot_extent[3])
    ax.set_aspect('equal')
    plt.tight_layout()

    # save the figure
    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder + save_as + '.pdf')

    return ax

def plot_gaussian_graph(gg, config, ax=None, save_as=None):
    """Plots a GaussianGraph.

    Args:
        gg: a GaussianGraph
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax = gg.plot(ax=ax)

    # Apply config settings if provided
    if hasattr(config, 'plot_extent'):
        ax.set_xlim(config.plot_extent[0], config.plot_extent[1])
        ax.set_ylim(config.plot_extent[2], config.plot_extent[3])
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder + save_as + '.pdf')
        plt.close()

    return ax


# These will possibly be removed at a later stage (avoid using if an alternative above is available)
def plot_gmm(x_train, label, damm, ax = None):
    """ passing damm object to plot the ellipsoids of clustering results"""
    N = x_train.shape[1]

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
    color_mapping = np.take(colors, label)


    if ax is None:
        fig = plt.figure(figsize=(12, 10))
    if N == 2:
        if ax is None:
            ax = fig.add_subplot()
        ax.scatter(x_train[:, 0], x_train[:, 1], color=color_mapping[:], alpha=0.4, label="Demonstration")

    elif N == 3:
        if ax is None:
            ax = fig.add_subplot(projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], 'o', color=color_mapping[:], s=3, alpha=0.4, label="Demonstration")

        K = damm.K
        for k in range(K):
            _, s, rotation = np.linalg.svd(damm.Sigma[k, :, :])  # find the rotation matrix and radii of the axes
            radii = np.sqrt(s) * 1.5                        # set the scale factor yourself
            u = np.linspace(0.0, 2.0 * np.pi, 60)
            v = np.linspace(0.0, np.pi, 60)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))   # calculate cartesian coordinates for the ellipsoid surface
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + damm.Mu[k, :]
            ax.plot_surface(x, y, z, rstride=3, cstride=3, color=colors[k], linewidth=0.1, alpha=0.3, shade=True) 


        ax.set_xlabel(r'$\xi_1$', fontsize=38, labelpad=20)
        ax.set_ylabel(r'$\xi_2$', fontsize=38, labelpad=20)
        ax.set_zlabel(r'$\xi_3$', fontsize=38, labelpad=20)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis='z', which='major', pad=15)

def plot_incremental_ds(new_data, prev_data, att, x_test_list):

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(prev_data[::4, 0], prev_data[::4, 1], prev_data[::4, 2], color='r', s=10,  label='original data')
    ax.scatter(new_data[::4, 0], new_data[::4, 1], new_data[::4, 2], color = 'magenta', s=10,  label='new data')

    ax.scatter(att[0, 0], att[0, 1], att[0, 2], marker=(8, 2, 0), s=150, c='k', label='Target')

    new_label = mlines.Line2D([], [], color='red',
                        linewidth=3, label='Old Demo')
    old_label = mlines.Line2D([], [], color='magenta',
                        linewidth=3, label='New Demo')
    ax.legend(handles=[new_label, old_label])

    L = len(x_test_list)
    for l in range(L):
        x_test = x_test_list[l]
        if l != L - 1:
            ax.plot3D(x_test[:, 0], x_test[:, 1], x_test[:, 2], 'k', linewidth=3.5)
        else:
            ax.plot3D(x_test[:, 0], x_test[:, 1], x_test[:, 2], 'k', linewidth=3.5, label='Reproduction')

    ax.axis('auto')
    ax.set_xlabel(r'$\xi_1(m)$')
    ax.set_ylabel(r'$\xi_2(m)$')
    ax.set_zlabel(r'$\xi_3(m)$')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


# Plotting primitives: accepts ax, adds to it, and returns it
def primitive_plot_demo(ax, demo, color=None):
    """Plots a single demonstration's trajectories with start and end points, optionally using a specified color.
    """
    # Params
    alpha = 1
    linewidth = 1
    marker_size = 8

    # Select random color if not provided
    if color is None:
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    # Plot all trajectories in this demonstration with the same color
    for traj in demo.trajectories:
        ax.plot(traj.x[:, 0], traj.x[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        # Start point (green)
        ax.plot(traj.x[0, 0], traj.x[0, 1], 'go', markersize=marker_size)
        # End point (red)
        ax.plot(traj.x[-1, 0], traj.x[-1, 1], 'ro', markersize=marker_size)

    return ax

def primitive_plot_gaussian(ax, mu, sigma, color=None, sigma_bound=2, resolution=200, direction=None):

    # Params
    sigma_bound_color = 'black'
    sigma_bound_linewidth = 0.25
    direction_arrow_color = 'white'
    direction_arrow_size = 0.1
    sigma_extent = 3  # extent of the grid in terms of std deviations

    # Select random color if not provided
    if color is None:
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    # Create coordinate grid
    x_min = mu[0] - sigma_extent * np.sqrt(sigma[0, 0])
    x_max = mu[0] + sigma_extent * np.sqrt(sigma[0, 0])
    x = np.linspace(x_min, x_max, resolution)
    y_min = mu[1] - sigma_extent * np.sqrt(sigma[1, 1])
    y_max = mu[1] + sigma_extent * np.sqrt(sigma[1, 1])
    y = np.linspace(y_min, y_max, resolution)

    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # Evaluate each grid point wrt the Gaussian
    diff = grid_points - mu
    inv_sigma = np.linalg.inv(sigma)
    mahalanobis_sq = np.sum(diff @ inv_sigma * diff, axis=1)
    norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
    gaussian_values = norm_const * np.exp(-0.5 * mahalanobis_sq)
    gaussian_values_normalized = gaussian_values / np.max(gaussian_values)  # Normalize for visualization
    gaussian_evaluations = gaussian_values_normalized

    # create heatmap
    heatmap = gaussian_evaluations.reshape(resolution, resolution)
    cmap = LinearSegmentedColormap.from_list('custom_heatmap', [(1, 1, 1, 0), color])
    ax.imshow(heatmap, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, aspect='equal')

    # Add 2-sigma ellipses for each Gaussian
    if sigma_bound is not None:

        # Eigendecomposition for ellipse orientation and size
        eigenvals, eigenvecs = np.linalg.eigh(sigma)
        order = eigenvals.argsort()[::-1]  # Sort in descending order
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # Calculate ellipse parameters for x-sigma boundary
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = sigma_bound ** 2 * np.sqrt(eigenvals[0])  # 2-sigma width (2 * 2 * sqrt)
        height = sigma_bound ** 2 * np.sqrt(eigenvals[1])  # 2-sigma height (2 * 2 * sqrt)

        # Draw 2-sigma ellipse
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle,
                          edgecolor=sigma_bound_color, facecolor='none',
                          linewidth=sigma_bound_linewidth, linestyle='-')
        ax.add_patch(ellipse)

    # Add direction arrows
    if direction is not None:
        direction_norm = direction / (np.linalg.norm(direction, keepdims=True) + 1e-10)
        arrow_scale = min(np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1])) * direction_arrow_size

        ax.arrow(mu[0], mu[1], direction_norm[0] * arrow_scale, direction_norm[1] * arrow_scale,
                 width=arrow_scale,
                 fc=direction_arrow_color, ec=direction_arrow_color)

    return ax

def primitive_plot_trajectory_points(ax, x, x_dot, color='blue', alpha=0.5):
    """Plots arrows at each point in x with direction given by x_dot."""

    # params
    size = 0.015

    for i in range(len(x)):
        ax.arrow(x[i, 0], x[i, 1], x_dot[i, 0] * size, x_dot[i, 1] * size,
                 width=size, fc=color, ec=color, alpha=alpha)

    return ax

def plot_ds_2d(x_train, x_test_list, lpvds, title=None, ax=None, x_min=None, x_max=None, y_min=None, y_max=None):
    """ passing lpvds object to plot the streamline of DS (only in 2D)"""
    A = lpvds.A
    att = lpvds.x_att

    if ax is None:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot()

    ax.scatter(x_train[:, 0], x_train[:, 1], color='k', s=5, label='original data')
    for idx, x_test in enumerate(x_test_list):
        ax.plot(x_test[:, 0], x_test[:, 1], color='r', linewidth=2)

    if x_min is None or x_max is None or y_min is None or y_max is None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

    plot_sample = 50
    x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, plot_sample), np.linspace(y_min, y_max, plot_sample))
    X = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

    if hasattr(lpvds, "vector_field"):
        dx = lpvds.vector_field(X.T).T
    else:
        gamma = lpvds.damm.compute_gamma(X.T)
        for k in np.arange(len(A)):
            if k == 0:
                dx = gamma[k].reshape(1, -1) * (
                            A[k] @ (X - att.reshape(1, -1).T))  # gamma[k].reshape(1, -1): [1, num] dim x num
            else:
                dx += gamma[k].reshape(1, -1) * (A[k] @ (X - att.reshape(1, -1).T))

    u = dx[0, :].reshape((plot_sample, plot_sample))
    v = dx[1, :].reshape((plot_sample, plot_sample))

    ax.streamplot(x_mesh, y_mesh, u, v, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")
    ax.scatter(att[0], att[1], color='g', s=100, alpha=0.7)
    ax.set_aspect('equal')

    if title is not None:
        ax.set_title(title)

def plot_ds_3d(x_train, x_test_list):
    N = x_train.shape[1]

    fig = plt.figure(figsize=(12, 10))
    if N == 2:
        ax = fig.add_subplot()
        ax.scatter(x_train[:, 0], x_train[:, 1], color='k', s=1, alpha=0.4, label="Demonstration")
        for idx, x_test in enumerate(x_test_list):
            ax.plot(x_test[:, 0], x_test[:, 1], color='b')

    elif N == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], 'o', color='k', s=3, alpha=0.4, label="Demonstration")

        for idx, x_test in enumerate(x_test_list):
            ax.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], color='b')
        ax.set_xlabel(r'$\xi_1$', fontsize=38, labelpad=20)
        ax.set_ylabel(r'$\xi_2$', fontsize=38, labelpad=20)
        ax.set_zlabel(r'$\xi_3$', fontsize=38, labelpad=20)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis='z', which='major', pad=15)
