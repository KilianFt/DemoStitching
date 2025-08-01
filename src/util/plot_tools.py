from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import multivariate_normal
from src.util.ds_tools import get_gaussian_directions
import random
import os
import networkx as nx

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": 20
})

def plot_gaussians_with_ds(gg, lpvds, x_test_list, save_folder, i, config):
    fig, axs = plt.subplots(1, 1, figsize=(8,8), sharex=True, sharey=True)
    if gg.shortest_path is not None:
        mus, sigmas, directions, _ = gg.get_gaussian(gg.shortest_path[1:-1])
    elif gg.node_wise_shortest_path is not None:
        mus, sigmas, directions, _ = gg.get_gaussian(gg.node_wise_shortest_path)
    else:
        raise Exception("No shortest path computed before plotting.")
    plot_gaussians(config, mus, sigmas, directions, ax=axs, resolution=1000)
    # gg.plot_shortest_path_gaussians(ax=axs[0])
    axs.set_xlim(config.plot_extent[0], config.plot_extent[1])
    axs.set_ylim(config.plot_extent[2], config.plot_extent[3])
    axs.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_folder + "stitched_pre_gaussians_{}.png".format(i), dpi=300)
    plt.close()

    fig, axs = plt.subplots(1, 1, figsize=(8,8), sharex=True, sharey=True)
    plot_ds_2d(lpvds.x, x_test_list, lpvds, ax=axs, x_min=config.plot_extent[0], x_max=config.plot_extent[1], y_min=config.plot_extent[2], y_max=config.plot_extent[3])
    axs.set_xlim(config.plot_extent[0], config.plot_extent[1])
    axs.set_ylim(config.plot_extent[2], config.plot_extent[3])
    axs.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_folder + "stitched_ds_{}.png".format(i), dpi=300)
    plt.close()

    # Plot updates gaussians from lpvds if they were updated
    """
    if hasattr(lpvds.damm, "Mu"):
        fig, axs = plt.subplots(1, 1, figsize=(8,8), sharex=True, sharey=True)
        centers = lpvds.damm.Mu
        assignment_arr = lpvds.assignment_arr
        mean_xdot = np.zeros((lpvds.damm.K, lpvds.x.shape[1]))
        for k in range(lpvds.damm.K):
            mean_xdot[k] = np.mean(lpvds.x_dot[assignment_arr==k], axis=0)
        plot_gaussians(config, centers, lpvds.damm.Sigma, mean_xdot, ax=axs, resolution=1000)
        axs.set_xlim(config.plot_extent[0], config.plot_extent[1])
        axs.set_ylim(config.plot_extent[2], config.plot_extent[3])
        axs.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(save_folder + "stitched_updated_gaussians_{}.png".format(i), dpi=300)
        plt.close()
    """

def plot_gaussians(config, mus, sigmas, directions=None, resolution=800, ax=None):
    """Plots a heatmap of summed 2D Gaussians, their 2-sigma ellipses, and optional direction arrows.

    Args:
        config: Configuration object with plotting extent (plot_extent).
        mus: Array-like of shape (N, 2), mean vectors for each Gaussian.
        sigmas: Array-like of shape (N, 2, 2), covariance matrices for each Gaussian.
        directions: Optional array-like of (N, 2), direction vectors for each Gaussian.
        resolution: Int or tuple, grid resolution for the density heatmap (default 400).
        ax: Optional matplotlib axis. If None, a new figure and axis are created.

    Returns:
        matplotlib.axes.Axes: The axis containing the plotted Gaussians and ellipses.
    """

    # create color pallet
    colors = ['#ffffff', '#00a7c4']
    cmap = LinearSegmentedColormap.from_list('custom_heatmap', colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # initialize lists for means and covariances
    N = len(mus)
    mus = np.array(mus)
    sigmas = np.array(sigmas)

    # Handle single gaussian case
    if mus.ndim == 1:
        mus = mus.reshape(1, -1)
        sigmas = sigmas.reshape(1, 2, 2)

    # Handle resolution parameter
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    # Create coordinate grid
    if hasattr(config, 'plot_extent') is None:
        buffer = 3 * np.sqrt(np.max([np.trace(sigma) for sigma in sigmas]))
        xmin, ymin = mus.min(axis=0) - buffer
        xmax, ymax = mus.max(axis=0) + buffer
        extent = (xmin, xmax, ymin, ymax)
    else:
        extent = config.plot_extent
    x = np.linspace(extent[0], extent[1], resolution[0])
    y = np.linspace(extent[2], extent[3], resolution[1])
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # Evaluate each grid point wrt each Gaussian
    gaussian_evaluations = np.zeros((N, len(grid_points)))
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        diff = grid_points - mu
        inv_sigma = np.linalg.inv(sigma)
        mahalanobis_sq = np.sum(diff @ inv_sigma * diff, axis=1)
        norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
        gaussian_values = norm_const * np.exp(-0.5 * mahalanobis_sq)
        gaussian_values_normalized = gaussian_values / np.max(gaussian_values)  # Normalize for visualization
        gaussian_evaluations[i] = gaussian_values_normalized

    # aggregate evaluations
    total_values = np.max(gaussian_evaluations, axis=0)

    # create heatmap
    heatmap = total_values.reshape(resolution[1], resolution[0])
    im = ax.imshow(heatmap, extent=extent, origin='lower', cmap=cmap, aspect='equal')

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
                          edgecolor='black', facecolor='none', linewidth=0.25, linestyle='-')
        ax.add_patch(ellipse)

    # Add direction arrows
    if directions is not None:
        directions = np.array(directions)
        if directions.ndim == 1:
            directions = directions.reshape(1, -1)

        directions_norm = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10)
        arrow_scale = min(extent[1] - extent[0], extent[3] - extent[2]) * 0.01

        for mu, dir_vec in zip(mus, directions_norm):
            ax.arrow(mu[0], mu[1],
                     dir_vec[0] * arrow_scale, dir_vec[1] * arrow_scale,
                     head_width=arrow_scale*0.2, head_length=arrow_scale*0.15,
                     fc='white', ec='white', linewidth=2, alpha=0.8)

    # set tight layout
    ax.set_aspect('equal')
    plt.tight_layout()
    return ax

def plot_ds_set_gaussians(ds_set, config, include_points=False, ax=None, file_name=None):
    """Plots means, covariances, and directions for all Gaussians in the DS set, with optional data point arrows.

    Args:
        ds_set: List of DS objects, each containing fitted Gaussian parameters and data assignments.
        config: Configuration object providing plotting and dataset settings.
        include_points: If True, overlays trajectory points and velocity arrows assigned to Gaussians.
        ax: Optional matplotlib axis. If None, creates a new figure and axis.
        file_name: Optional filename (without extension) to save the plot as PNG.

    Returns:
        matplotlib.axes.Axes: The axis with the plotted Gaussians and any optional data points.
    """

    # Collect the gaussians from all DS'
    mu = []
    sigma = []
    directions = []
    for ds in ds_set:
        mu.append(ds.damm.Mu)
        sigma.append(ds.damm.Sigma)
        directions.append(get_gaussian_directions(ds))
    mu = np.vstack(mu)
    sigma = np.vstack(sigma)
    directions = np.vstack(directions)

    # Plot the Gaussians
    ax = plot_gaussians(config, mu, sigma, directions, ax=ax)

    # Add trajectory points if requested
    if include_points:

        # get color map (one color for each gaussian from each ds)
        num_gaussians = mu.shape[0]
        colors = plt.cm.hsv(np.linspace(0, 1, num_gaussians))
        random.shuffle(colors)

        # plot points assigned to each gaussian with the corresponding color (k)
        k = 0
        for i in range(len(ds_set)):
            for j in range(ds_set[i].K):

                # get the points assigned to the j-th gaussian of the i-th ds
                assigned_x = ds_set[i].x[ds_set[i].assignment_arr == j]
                assigned_x_dot = ds_set[i].x_dot[ds_set[i].assignment_arr == j]
                for p in range(len(assigned_x)):
                    ax.arrow(assigned_x[p, 0], assigned_x[p, 1], assigned_x_dot[p, 0] * 0.1, assigned_x_dot[p, 1] * 0.1,
                             head_width=0.05, head_length=0.05, fc=colors[k], ec=colors[k], alpha=0.5)
                k += 1

    # save the figure
    if file_name is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder + file_name + '.png', dpi=800)

    return ax

def plot_demonstration_set(demo_set, config, ax=None, file_name=None):
    """Plots grouped demonstration trajectories with start and end points, optionally saving the figure.

    Args:
        demo_set: List of Demonstration objects, each containing Trajectory objects to plot.
        config: Configuration object specifying plot extent, dataset path, and ds_method.
        ax: Optional matplotlib axis. If None, a new figure and axis are created.
        file_name: Optional filename (without extension) to save the plot as PNG.

    Returns:
        matplotlib.axes.Axes: The axis containing the plotted demonstration set.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Number of demonstrations
    n_demos = len(demo_set)

    # Generate colors from colormap - one color per demonstration
    colors = plt.cm.viridis(np.linspace(0, 1, n_demos+2))
    colors = colors[1:-1]

    # Plot each demonstration with its assigned color
    for i, demo in enumerate(demo_set):
        demo_color = colors[i]

        # Plot all trajectories in this demonstration with the same color
        for traj in demo.trajectories:
            ax.plot(traj.x[:, 0], traj.x[:, 1], color=demo_color, linewidth=1, alpha=1)
            # Start point (green)
            ax.plot(traj.x[0, 0], traj.x[0, 1], 'go', markersize=8)
            # End point (red)
            ax.plot(traj.x[-1, 0], traj.x[-1, 1], 'ro', markersize=8)

    # Plot settings
    if hasattr(config, 'plot_extent'):
        ax.set_xlim(config.plot_extent[0], config.plot_extent[1])
        ax.set_ylim(config.plot_extent[2], config.plot_extent[3])
    ax.set_aspect('equal')
    plt.tight_layout()

    # Save the figure if file_name is provided
    if file_name is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder + file_name + '.png', dpi=600)

    # Close the figure if we created it
    if ax is None:
        plt.close()

    return ax

def plot_gaussian_graph(gg, config, bare=False, ax=None, save_as=None):
    """Plots a GaussianGraph.

    Args:
        gg: a GaussianGraph
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    nxgraph = gg.graph

    # Get positions for nodes
    pos = {node: nxgraph.nodes[node]['mean'] for node in gg.gaussian_ids}
    if gg.attractor_id:
        pos[gg.attractor_id] = nxgraph.nodes[gg.attractor_id]['pos']
    if gg.initial_id:
        pos[gg.initial_id] = nxgraph.nodes[gg.initial_id]['pos']

    # Create colormap for nodes
    colormap = []
    for node in nxgraph.nodes:
        if node is gg.attractor_id and not bare:
            colormap.append('red')
        elif node is gg.initial_id and not bare:
            colormap.append('green')
        else:
            colormap.append('teal')

    # Draw nodes manually using scatter plot
    node_positions = np.array([pos[node] for node in nxgraph.nodes])
    ax.scatter(node_positions[:, 0], node_positions[:, 1],
               c=colormap, s=100, zorder=3)

    # Extract edge weights and normalize for alpha values
    edges = nxgraph.edges(data=True)
    weights = [edata['weight'] for _, _, edata in edges]

    # Normalize weights for alpha values (higher weight = lower alpha)
    norm_param = 10
    normalize_weights = np.array(weights) / min(weights)
    normalize_weights = normalize_weights - 1
    normalize_weights = normalize_weights * norm_param / np.median(normalize_weights)
    normalize_weights = normalize_weights + 1
    alphas = [np.exp(1 - w) for w in normalize_weights]

    # Draw edges manually
    node_buffer_start = 0.3
    node_buffer_end = 0.3
    line_width = 0.5
    arrow_head_size = 8

    for (u, v, edata), alpha in zip(edges, alphas):
        start_pos = pos[u]
        end_pos = pos[v]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = np.sqrt(dx ** 2 + dy ** 2)

        if length > 0:
            # Normalize direction vector
            dx_norm = dx / length
            dy_norm = dy / length

            # Calculate adjusted start and end points based on buffer distances
            adjusted_start_x = start_pos[0] + dx_norm * node_buffer_start
            adjusted_start_y = start_pos[1] + dy_norm * node_buffer_start
            adjusted_end_x = end_pos[0] - dx_norm * node_buffer_end
            adjusted_end_y = end_pos[1] - dy_norm * node_buffer_end

            # Draw the line between adjusted points
            ax.plot([adjusted_start_x, adjusted_end_x], [adjusted_start_y, adjusted_end_y],
                    'k-', alpha=alpha, zorder=1, linewidth=line_width)

            # Draw arrow head at the end of the adjusted line
            # Position arrow slightly before the adjusted end point
            arrow_tail_x = adjusted_end_x - dx_norm * 0.05
            arrow_tail_y = adjusted_end_y - dy_norm * 0.05

            ax.annotate('', xy=(adjusted_end_x, adjusted_end_y),
                        xytext=(arrow_tail_x, arrow_tail_y),
                        arrowprops=dict(arrowstyle='->', color='black',
                                        alpha=alpha, lw=line_width,
                                        mutation_scale=arrow_head_size),
                        zorder=2)

    # Draw shortest path manually
    if gg.shortest_path is not None and not bare:
        path_edges = [(gg.shortest_path[i], gg.shortest_path[i + 1])
                      for i in range(len(gg.shortest_path) - 1)]

        for u, v in path_edges:
            start_pos = pos[u]
            end_pos = pos[v]

            # Draw thick path line
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                    'magenta', alpha=0.25, linewidth=10, zorder=0)

            # Draw large arrow for path
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx ** 2 + dy ** 2)

            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
                arrow_start_x = end_pos[0] - dx_norm * 0.15
                arrow_start_y = end_pos[1] - dy_norm * 0.15

                ax.annotate('', xy=(end_pos[0], end_pos[1]),
                            xytext=(arrow_start_x, arrow_start_y),
                            arrowprops=dict(arrowstyle='->', color='magenta',
                                            alpha=0.25, lw=10), zorder=0)

    # Apply config settings if provided
    if hasattr(config, 'plot_extent'):
        ax.set_xlim(config.plot_extent[0], config.plot_extent[1])
        ax.set_ylim(config.plot_extent[2], config.plot_extent[3])
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder + save_as + '.png', dpi=300)
        plt.close()

    return ax

def plot_trajectory_points(x, x_dot, ax):

    # plot an arrow at each x point with direction x_dot
    for i in range(len(x)):
        ax.arrow(x[i, 0], x[i, 1], x_dot[i, 0] * 0.1, x_dot[i, 1] * 0.1,
                 head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.5)


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

def plot_ds_2d(x_train, x_test_list, lpvds, title=None, ax=None, x_min=None, x_max=None, y_min=None, y_max=None):
    """ passing lpvds object to plot the streamline of DS (only in 2D)"""
    A = lpvds.A
    att = lpvds.x_att

    if ax is None:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot()

    ax.scatter(x_train[:, 0], x_train[:, 1], color='k', s=5, label='original data')
    for idx, x_test in enumerate(x_test_list):
        ax.plot(x_test[:, 0], x_test[:, 1], color= 'r', linewidth=2)

    if x_min is None or x_max is None or y_min is None or y_max is None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

    plot_sample = 50
    x_mesh,y_mesh = np.meshgrid(np.linspace(x_min,x_max,plot_sample),np.linspace(y_min,y_max,plot_sample))
    X = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
    gamma = lpvds.damm.logProb(X.T)
    for k in np.arange(len(A)):
        if k == 0:
            dx = gamma[k].reshape(1, -1) * (A[k] @ (X - att.reshape(1,-1).T))  # gamma[k].reshape(1, -1): [1, num] dim x num
        else:
            dx +=  gamma[k].reshape(1, -1) * (A[k] @ (X - att.reshape(1,-1).T))
    u = dx[0,:].reshape((plot_sample,plot_sample))
    v = dx[1,:].reshape((plot_sample,plot_sample))

    ax.streamplot(x_mesh,y_mesh,u,v, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")
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
            ax.plot(x_test[:, 0], x_test[:, 1], color= 'b')
        
    elif N == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], 'o', color='k', s=3, alpha=0.4, label="Demonstration")

        for idx, x_test in enumerate(x_test_list):
            ax.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], color= 'b')
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
