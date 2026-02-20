import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.collections import LineCollection
from typing import Optional
from src.util.ds_tools import get_gaussian_directions
import random
import os

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": 20
})

def _infer_demo_dim(demo_set):
    for demo in demo_set:
        for traj in getattr(demo, "trajectories", []):
            x = np.asarray(getattr(traj, "x", []), dtype=float)
            if x.ndim == 2 and x.shape[1] > 0:
                return int(x.shape[1])
    return 2

def _infer_ds_dim(ds_set):
    if ds_set is None or len(ds_set) == 0:
        return 2
    x = np.asarray(getattr(ds_set[0], "x", []), dtype=float)
    if x.ndim == 2 and x.shape[1] > 0:
        return int(x.shape[1])
    return 2

def _infer_graph_dim(gg):
    for _, node_data in gg.graph.nodes(data=True):
        mu = node_data.get("mean", None)
        if mu is None:
            continue
        mu = np.asarray(mu, dtype=float).reshape(-1)
        if mu.shape[0] > 0:
            return int(mu.shape[0])
    return 2

def _create_axis(dim, ax=None, figsize=(8, 8)):
    if ax is not None:
        return ax
    fig = plt.figure(figsize=figsize)
    if dim >= 3:
        return fig.add_subplot(projection='3d')
    return fig.add_subplot()

def _apply_plot_extent(ax, config, dim):
    if not hasattr(config, "plot_extent"):
        return
    extent = getattr(config, "plot_extent")
    if extent is None:
        return
    if dim >= 3:
        if len(extent) >= 4:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        if len(extent) >= 6:
            ax.set_zlim(extent[4], extent[5])
    else:
        if len(extent) >= 4:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])


def _save_axis_figure(ax, save_path: str, dim: int):
    save_kwargs = {}
    if int(dim) >= 3:
        # 3D exports otherwise keep large canvas whitespace independent of axis limits.
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = 0.02
    ax.figure.savefig(save_path, **save_kwargs)

def _plot_graph_3d(ax, gg, highlight_nodes=None):
    nodes = []
    for node_id, node_data in gg.graph.nodes(data=True):
        mu = node_data.get("mean", None)
        if mu is None:
            continue
        mu = np.asarray(mu, dtype=float).reshape(-1)
        if mu.shape[0] < 3:
            continue
        nodes.append((node_id, mu))

    if len(nodes) == 0:
        return ax

    node_lookup = {node_id: mu for node_id, mu in nodes}
    all_pts = np.vstack([mu for _, mu in nodes])
    ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], s=14, color='teal', alpha=0.8)

    for u, v in gg.graph.edges:
        if u not in node_lookup or v not in node_lookup:
            continue
        pu = node_lookup[u]
        pv = node_lookup[v]
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], [pu[2], pv[2]], color='black', alpha=0.25, linewidth=0.7)

    if highlight_nodes is not None:
        pts = []
        for node in highlight_nodes:
            if node in node_lookup:
                pts.append(node_lookup[node])
        if len(pts) > 0:
            pts = np.vstack(pts)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=26, color='orange', alpha=0.95)

    ax.set_xlabel(r'$\xi_1$')
    ax.set_ylabel(r'$\xi_2$')
    ax.set_zlabel(r'$\xi_3$')
    return ax

def plot_demonstration_set(demo_set, config, ax=None, save_as=None, hide_axis=False):
    """Plots grouped demonstration trajectories with start and end points, optionally saving the figure.

    Args:
        demo_set: List of Demonstration objects, each containing Trajectory objects to plot.
        config: Configuration object specifying plot extent, dataset path, and ds_method.
        ax: Optional matplotlib axis. If None, a new figure and axis are created.
        save_as: Optional filename (without extension) to save the plot as PNG.

    Returns:
        matplotlib.axes.Axes: The axis containing the plotted demonstration set.
    """
    dim = _infer_demo_dim(demo_set)
    external_ax = ax is not None
    ax = _create_axis(dim, ax=ax, figsize=(8, 8))

    # Generate colors from colormap - one color per demonstration
    colors = plt.cm.get_cmap('tab10', len(demo_set)).colors

    # Plot each demonstration with its assigned color
    for i, demo in enumerate(demo_set):
        ax = primitive_plot_demo(ax, demo, color=colors[i])

    # Plot settings
    _apply_plot_extent(ax, config, dim)
    if hide_axis:
        ax.axis('off')
    if dim < 3:
        ax.set_aspect('equal')
    plt.tight_layout()

    # Save the figure if file_name is provided
    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        _save_axis_figure(ax, save_folder + save_as + '.pdf', dim)
        if not external_ax:
            plt.close(ax.figure)

    return ax

def plot_ds_set_gaussians(ds_set, config, initial=None, attractor=None, include_trajectory=False, ax=None, save_as=None, hide_axis=False):
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
    dim = _infer_ds_dim(ds_set)
    external_ax = ax is not None
    ax = _create_axis(dim, ax=ax, figsize=(8, 8))

    colors = plt.cm.get_cmap('tab10', len(ds_set)).colors

    # Add trajectory points if requested
    if include_trajectory:
        for i, ds in enumerate(ds_set):
            ax = primitive_plot_trajectory_points(ax, ds.x, ds.x_dot, color=colors[i])

    # Plot the gaussians
    for i, ds in enumerate(ds_set):
        mus = ds.damm.Mu
        sigmas = ds.damm.Sigma
        directions = get_gaussian_directions(
            ds,
            method=config.gaussian_direction_method,
        )

        for mu, sigma, direction in zip(mus, sigmas, directions):
            ax = primitive_plot_gaussian(ax, mu, sigma, color=colors[i], direction=direction, sigma_bound=2)

    # plot initial and attractor
    if initial is not None:
        ax = primitive_plot_point(ax, initial, color='red')
    if attractor is not None:
        ax = primitive_plot_point(ax, attractor, color='green')

    # set limits
    _apply_plot_extent(ax, config, dim)
    if hide_axis:
        ax.axis('off')
    if dim < 3:
        ax.set_aspect('equal')
    plt.tight_layout()

    # save the figure
    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        _save_axis_figure(ax, save_folder + save_as + '.pdf', dim)
        if not external_ax:
            plt.close(ax.figure)

    return ax

def plot_gg_solution(gg, solution_nodes, initial, attractor, config, ax=None, save_as=None, hide_axis=False):
    dim = _infer_graph_dim(gg)
    external_ax = ax is not None
    ax = _create_axis(dim, ax=ax, figsize=(8, 8))

    if dim >= 3:
        ax = _plot_graph_3d(ax, gg, highlight_nodes=solution_nodes)
    else:
        ax = gg.plot(ax=ax)
        for node in solution_nodes:
            mu, sigma, direction, _ = gg.get_gaussian(node)
            ax = primitive_plot_gaussian(ax, mu, sigma, color='orange', direction=direction)

    # plot initial and attractor
    ax = primitive_plot_point(ax, initial, color='red')
    ax = primitive_plot_point(ax, attractor, color='green')

    # set limits
    _apply_plot_extent(ax, config, dim)
    if hide_axis:
        ax.axis('off')
    if dim < 3:
        ax.set_aspect('equal')
    plt.tight_layout()

    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        _save_axis_figure(ax, save_folder + save_as + '.pdf', dim)
        if not external_ax:
            plt.close(ax.figure)

    return ax

def plot_ds(lpvds, x_test_list, initial, attractor, config, ax=None, save_as=None, hide_axis=False):
    """Plots the DS vector field/trajectories in 2D, and trajectories in 3D.

    Args:
        lpvds: The DS object containing the fitted model.
        x_test_list: List of simulated trajectory arrays to plot.
        config: Configuration object with plotting extent and dataset settings.
        ax: Optional matplotlib axis. If None, creates a new figure and axis.
        save_as: Optional filename (without extension) to save the plot as PNG.

    Returns:
        matplotlib.axes.Axes: The axis with the plotted DS vector field and trajectories.
    """
    dim = int(np.asarray(lpvds.x, dtype=float).shape[1])
    external_ax = ax is not None
    ax = _create_axis(dim, ax=ax, figsize=(8, 8))
    x_test_list = [] if x_test_list is None else x_test_list
    if dim >= 3:
        ax = plot_ds_3d(lpvds.x, x_test_list, ax=ax, att=attractor)
    else:
        chain_cfg = getattr(config, "chain", None)
        plot_ds_2d(
            lpvds.x,
            x_test_list,
            lpvds,
            ax=ax,
            x_min=config.plot_extent[0],
            x_max=config.plot_extent[1],
            y_min=config.plot_extent[2],
            y_max=config.plot_extent[3],
            chain_plot_mode=getattr(chain_cfg, "plot_mode", "line_regions"),
            chain_plot_resolution=int(max(8, getattr(chain_cfg, "plot_grid_resolution", 60))),
            chain_plot_path_bandwidth=getattr(chain_cfg, "plot_path_bandwidth", 0.9),
            show_chain_transition_lines=bool(getattr(chain_cfg, "plot_show_transition_lines", True)),
            chain_region_alpha=float(getattr(chain_cfg, "plot_region_alpha", 0.26)),
        )

    # plot initial and attractor
    ax = primitive_plot_point(ax, initial, color='red')
    ax = primitive_plot_point(ax, attractor, color='green')

    # set limits
    _apply_plot_extent(ax, config, dim)
    if hide_axis:
        ax.axis('off')
    if dim < 3:
        ax.set_aspect('equal')
    plt.tight_layout()

    # save the figure
    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        _save_axis_figure(ax, save_folder + save_as + '.pdf', dim)
        if not external_ax:
            plt.close(ax.figure)

    return ax

def plot_gaussian_graph(gg, config, ax=None, save_as=None, hide_axis=False):
    """Plots a GaussianGraph.

    Args:
        gg: a GaussianGraph
    """
    dim = _infer_graph_dim(gg)
    external_ax = ax is not None
    ax = _create_axis(dim, ax=ax, figsize=(8, 8))
    if dim >= 3:
        ax = _plot_graph_3d(ax, gg)
    else:
        ax = gg.plot(ax=ax)

    # Apply config settings if provided
    _apply_plot_extent(ax, config, dim)
    if hide_axis:
        ax.axis('off')
    if dim < 3:
        ax.set_aspect('equal')
    plt.tight_layout()

    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        _save_axis_figure(ax, save_folder + save_as + '.pdf', dim)
        if not external_ax:
            plt.close(ax.figure)

    return ax

def plot_composite(gg, solution_nodes, demo_set, lpvds, x_test_list, initial, attractor, config, ax=None, save_as=None, hide_axis=False):

    dim = _infer_graph_dim(gg)
    external_ax = ax is not None
    ax = _create_axis(dim, ax=ax, figsize=(12, 12))

    # plot raw demonstrations

    colors = plt.cm.get_cmap('tab10', len(demo_set)).colors
    for i, demo in enumerate(demo_set):
        ax = primitive_plot_demo(ax, demo, linewidth=6, alpha=0.5, marker_size=6, color=colors[i])

    # Plot DS
    dim = int(np.asarray(lpvds.x, dtype=float).shape[1])
    ax = _create_axis(dim, ax=ax, figsize=(8, 8))
    x_test_list = [] if x_test_list is None else x_test_list
    if dim >= 3:
        ax = plot_ds_3d(lpvds.x, x_test_list, ax=ax, att=attractor)
    else:
        chain_cfg = getattr(config, "chain", None)
        plot_ds_2d(
            lpvds.x,
            x_test_list,
            lpvds,
            ax=ax,
            x_min=config.plot_extent[0],
            x_max=config.plot_extent[1],
            y_min=config.plot_extent[2],
            y_max=config.plot_extent[3],
            arrowsize=2,
            include_raw_data=False,
            linewidth=7,
            marker_size=200,
            stream_density=1,
            stream_color='black',
            stream_width=0.5,
            chain_plot_mode=getattr(chain_cfg, "plot_mode", "line_regions"),
            chain_plot_resolution=int(max(8, getattr(chain_cfg, "plot_grid_resolution", 60))),
            chain_plot_path_bandwidth=getattr(chain_cfg, "plot_path_bandwidth", 0.9),
            show_chain_transition_lines=bool(getattr(chain_cfg, "plot_show_transition_lines", True)),
            chain_region_alpha=float(getattr(chain_cfg, "plot_region_alpha", 0.26)),
        )

    # plot solution gaussians
    if dim <= 2:
        for node in solution_nodes:
            mu, sigma, direction, _ = gg.get_gaussian(node)
            ax = primitive_plot_gaussian(ax, mu, sigma, color='orange', direction=None, sigma_bound=None)

    # plot initial and attractor
    ax = primitive_plot_point(ax, initial, color='red')
    ax = primitive_plot_point(ax, attractor, color='green')

    # set limits
    _apply_plot_extent(ax, config, dim)
    if hide_axis:
        ax.axis('off')
    if dim < 3:
        ax.set_aspect('equal')
    plt.tight_layout()

    # save the figure
    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        _save_axis_figure(ax, save_folder + save_as + '.pdf', dim)
        if not external_ax:
            plt.close(ax.figure)

    return ax

def plot_gaussian_graph(gg, config, ax=None, save_as=None, hide_axis=False):
    """Plots a GaussianGraph.

    Args:
        gg: a GaussianGraph
    """
    dim = _infer_graph_dim(gg)
    external_ax = ax is not None
    ax = _create_axis(dim, ax=ax, figsize=(8, 8))
    if dim >= 3:
        ax = _plot_graph_3d(ax, gg)
    else:
        ax = gg.plot(ax=ax)

    # Apply config settings if provided
    _apply_plot_extent(ax, config, dim)
    if hide_axis:
        ax.axis('off')
    if dim < 3:
        ax.set_aspect('equal')
    plt.tight_layout()

    if save_as is not None:
        save_folder = f"{config.dataset_path}/figures/{config.ds_method}/"
        os.makedirs(save_folder, exist_ok=True)
        _save_axis_figure(ax, save_folder + save_as + '.pdf', dim)
        if not external_ax:
            plt.close(ax.figure)

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
def primitive_plot_demo(ax, demo, color=None, linewidth=1, alpha=1.0, marker_size=8):
    """Plots a single demonstration's trajectories with start and end points, optionally using a specified color.
    """

    # Select random color if not provided
    if color is None:
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    # Plot all trajectories in this demonstration with the same color
    for traj in demo.trajectories:
        x = np.asarray(traj.x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
            continue
        if x.shape[1] >= 3:
            ax.plot(x[:, 0], x[:, 1], x[:, 2], color=color, linewidth=linewidth, alpha=alpha)
            ax.scatter(x[0, 0], x[0, 1], x[0, 2], color='red', s=marker_size * 6)
            ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], color='green', s=marker_size * 6)
        else:
            ax.plot(x[:, 0], x[:, 1], color=color, linewidth=linewidth, alpha=alpha)
            # Start point (red)
            ax.plot(x[0, 0], x[0, 1], 'ro', markersize=marker_size)
            # End point (green)
            ax.plot(x[-1, 0], x[-1, 1], 'go', markersize=marker_size)

    return ax

def primitive_plot_gaussian(ax, mu, sigma, color=None, sigma_bound=2, sigma_bound_linewidth=0.25, resolution=200, direction=None):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float)

    # Params
    sigma_bound_color = 'black'
    direction_arrow_color = 'white'
    direction_arrow_size = 0.1
    sigma_extent = 3  # extent of the grid in terms of std deviations

    # Select random color if not provided
    if color is None:
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    if mu.shape[0] >= 3:
        ax.scatter(mu[0], mu[1], mu[2], color=color, s=28, alpha=0.9)
        if direction is not None:
            direction = np.asarray(direction, dtype=float).reshape(-1)
            direction_norm = direction / (np.linalg.norm(direction) + 1e-10)
            if sigma.ndim == 2 and sigma.shape[0] >= 3 and sigma.shape[1] >= 3:
                scale = float(np.sqrt(np.max(np.diag(sigma[:3, :3]))))
            else:
                scale = 0.1
            scale = max(scale, 1e-3)
            ax.quiver(
                mu[0],
                mu[1],
                mu[2],
                direction_norm[0],
                direction_norm[1],
                direction_norm[2],
                length=scale,
                normalize=True,
                color='black',
                linewidth=0.9,
            )
        return ax

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
    x = np.asarray(x, dtype=float)
    x_dot = np.asarray(x_dot, dtype=float)
    if x.ndim != 2 or x_dot.ndim != 2 or x.shape[0] == 0:
        return ax
    if x.shape[1] >= 3 and x_dot.shape[1] >= 3:
        size = 0.04
        for i in range(len(x)):
            ax.quiver(
                x[i, 0],
                x[i, 1],
                x[i, 2],
                x_dot[i, 0],
                x_dot[i, 1],
                x_dot[i, 2],
                length=size,
                normalize=False,
                color=color,
                alpha=alpha,
                linewidth=0.4,
            )
        return ax

    # params
    size = 0.015

    for i in range(len(x)):
        ax.arrow(x[i, 0], x[i, 1], x_dot[i, 0] * size, x_dot[i, 1] * size,
                 width=size, fc=color, ec=color, alpha=alpha)

    return ax

def primitive_plot_point(ax, point, color='red', marker='o', size=100, label=None):
    """Plots a single point with specified color, marker, and size."""
    point = np.asarray(point, dtype=float).reshape(-1)
    if point.shape[0] >= 3:
        ax.scatter(point[0], point[1], point[2], color=color, marker=marker, s=size, label=label)
    else:
        ax.scatter(point[0], point[1], color=color, marker=marker, s=size, label=label)
    return ax

def resolve_chain_plot_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    aliases = {
        "line_regions": "line_regions",
        "hard_lines": "line_regions",
        "hard": "line_regions",
        "time_blend": "time_blend",
        "blend": "time_blend",
        "blended": "time_blend",
    }
    if mode not in aliases:
        return "line_regions"
    return aliases[mode]


def _is_chain_ds_for_region_plot(ds) -> bool:
    return (
        hasattr(ds, "n_systems")
        and hasattr(ds, "transition_centers")
        and hasattr(ds, "transition_normals")
        and hasattr(ds, "_velocity_for_index")
    )


def _chain_nominal_index_from_lines(ds, x: np.ndarray) -> int:
    idx_line, _ = _chain_line_state(ds, x)
    return int(idx_line)


def _chain_nominal_index_from_lines_with_regime_fallback(ds, x: np.ndarray) -> int:
    x = np.asarray(x, dtype=float).reshape(-1)
    n_systems = int(max(1, getattr(ds, "n_systems", 1)))
    n_trans = min(
        n_systems - 1,
        len(np.asarray(getattr(ds, "transition_centers", []))),
        len(np.asarray(getattr(ds, "transition_normals", []))),
    )
    if n_trans <= 0:
        return 0

    idx_line, ambiguous = _chain_line_state(ds, x)
    if not ambiguous:
        return idx_line

    if x.shape[0] < 2:
        return idx_line
    xy = x[:2].reshape(1, 2)
    # Keep fallback local to the adjacent regimes around the line-prefix
    # boundary. This avoids non-local color islands that contradict the
    # transition-line interpretation.
    left_idx = int(np.clip(idx_line, 0, n_systems - 1))
    right_idx = int(np.clip(left_idx + 1, 0, n_systems - 1))
    if right_idx == left_idx:
        return left_idx

    dist = np.full((n_systems,), np.inf, dtype=float)
    for idx in (left_idx, right_idx):
        seg = _chain_regime_segment_2d(ds, idx)
        if seg is None:
            continue
        a, b = seg
        dist[idx] = _distance_points_to_segment_2d(xy, a, b)[0]
    finite = np.isfinite(dist)
    if np.any(finite):
        return int(np.argmin(dist))
    return idx_line


def _chain_line_state(ds, x: np.ndarray):
    x = np.asarray(x, dtype=float).reshape(-1)
    n_systems = int(max(1, getattr(ds, "n_systems", 1)))
    n_trans = min(
        n_systems - 1,
        len(np.asarray(getattr(ds, "transition_centers", []))),
        len(np.asarray(getattr(ds, "transition_normals", []))),
    )
    if n_trans <= 0:
        return 0, False

    crossed = []
    for k in range(n_trans):
        center = np.asarray(ds.transition_centers[k], dtype=float).reshape(-1)
        normal = np.asarray(ds.transition_normals[k], dtype=float).reshape(-1)
        dim = min(center.shape[0], normal.shape[0], x.shape[0])
        if dim <= 0:
            crossed.append(False)
            continue
        signed = float(np.dot(x[:dim] - center[:dim], normal[:dim]))
        crossed.append(bool(signed >= 0.0))

    idx_line = 0
    for c in crossed:
        if c:
            idx_line += 1
        else:
            break
    idx_line = int(np.clip(idx_line, 0, n_systems - 1))

    seen_false = False
    ambiguous = False
    for c in crossed:
        if not c:
            seen_false = True
        elif seen_false:
            ambiguous = True
            break
    return idx_line, ambiguous


def _chain_velocity_for_idx(ds, x: np.ndarray, idx: int) -> np.ndarray:
    idx = int(np.clip(idx, 0, int(max(1, ds.n_systems)) - 1))
    x = np.asarray(x, dtype=float).reshape(-1)
    velocity = np.asarray(ds._velocity_for_index(x, idx), dtype=float).reshape(-1)
    if velocity.shape[0] != x.shape[0]:
        raise ValueError("Chain DS velocity dimension mismatch.")
    return velocity


def _chain_transition_progress(ds, boundary_idx: int, x: np.ndarray) -> float:
    centers = np.asarray(getattr(ds, "transition_centers", []), dtype=float)
    normals = np.asarray(getattr(ds, "transition_normals", []), dtype=float)
    if boundary_idx < 0 or boundary_idx >= len(centers) or boundary_idx >= len(normals):
        return 1.0

    center = np.asarray(centers[boundary_idx], dtype=float).reshape(-1)
    normal = np.asarray(normals[boundary_idx], dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    dim = min(center.shape[0], normal.shape[0], x.shape[0])
    if dim <= 0:
        return 1.0

    signed = float(np.dot(x[:dim] - center[:dim], normal[:dim]))
    if signed <= 0.0:
        return 0.0

    distances = np.asarray(getattr(ds, "transition_distances", []), dtype=float).reshape(-1)
    if boundary_idx < len(distances):
        distance = float(distances[boundary_idx])
        if not np.isfinite(distance) or distance <= 1e-12:
            return 1.0
        transition_length = distance
    else:
        transition_length = 1.0

    return float(np.clip(signed / transition_length, 0.0, 1.0))


def _boundary_has_transition_zone(ds, boundary_idx: int) -> bool:
    distances = np.asarray(getattr(ds, "transition_distances", []), dtype=float).reshape(-1)
    if boundary_idx < len(distances):
        d = float(distances[boundary_idx])
        if np.isfinite(d) and d > 1e-12:
            return True
    times = np.asarray(getattr(ds, "transition_times", []), dtype=float).reshape(-1)
    if boundary_idx < len(times):
        t = float(times[boundary_idx])
        if np.isfinite(t) and t > 1e-12:
            return True
    return False


def _is_local_to_transition_neighborhood(ds, boundary_idx: int, x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] < 2:
        return True
    distances = np.asarray(getattr(ds, "transition_distances", []), dtype=float).reshape(-1)
    if boundary_idx >= len(distances):
        return True
    transition_length = float(distances[boundary_idx])
    if not np.isfinite(transition_length) or transition_length <= 1e-12:
        return True

    xy = x[:2].reshape(1, 2)
    seg_left = _chain_regime_segment_2d(ds, boundary_idx)
    seg_right = _chain_regime_segment_2d(ds, boundary_idx + 1)
    dists = []
    if seg_left is not None:
        dists.append(float(_distance_points_to_segment_2d(xy, seg_left[0], seg_left[1])[0]))
    if seg_right is not None:
        dists.append(float(_distance_points_to_segment_2d(xy, seg_right[0], seg_right[1])[0]))
    if len(dists) == 0:
        return True
    neighborhood_scale = 1.5
    return float(min(dists)) <= neighborhood_scale * transition_length


def _closest_regime_info_2d(ds, x: np.ndarray):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] < 2:
        return None, np.inf, np.array([], dtype=float)
    xy = x[:2].reshape(1, 2)
    n_systems = int(max(1, getattr(ds, "n_systems", 1)))
    dist = np.full((n_systems,), np.inf, dtype=float)
    for idx in range(n_systems):
        seg = _chain_regime_segment_2d(ds, idx)
        if seg is None:
            continue
        a, b = seg
        dist[idx] = float(_distance_points_to_segment_2d(xy, a, b)[0])
    finite = np.isfinite(dist)
    if not np.any(finite):
        return None, np.inf, dist
    idx = int(np.argmin(dist))
    return idx, float(dist[idx]), dist


def _draw_nonadjacent_region_boundaries_2d(
    ax,
    region_idx_img: np.ndarray,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    linewidth: float = 1.6,
    color: str = "black",
    alpha: float = 0.95,
    zorder: float = 3.0,
):
    region = np.asarray(region_idx_img, dtype=int)
    if region.ndim != 2 or region.shape[0] < 1 or region.shape[1] < 1:
        return None
    x_vec = np.asarray(x_vec, dtype=float).reshape(-1)
    y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
    if x_vec.size != region.shape[1] or y_vec.size != region.shape[0]:
        return None
    if x_vec.size < 2 or y_vec.size < 2:
        return None

    dx = float(x_vec[1] - x_vec[0])
    dy = float(y_vec[1] - y_vec[0])
    segments = []

    left = region[:, :-1]
    right = region[:, 1:]
    mask_v = (left != right) & (np.abs(left - right) > 1)
    if np.any(mask_v):
        ii, jj = np.nonzero(mask_v)
        x_mid = 0.5 * (x_vec[jj] + x_vec[jj + 1])
        y_ctr = y_vec[ii]
        y0 = np.maximum(y_ctr - 0.5 * dy, y_min)
        y1 = np.minimum(y_ctr + 0.5 * dy, y_max)
        for k in range(ii.size):
            segments.append([(x_mid[k], y0[k]), (x_mid[k], y1[k])])

    low = region[:-1, :]
    up = region[1:, :]
    mask_h = (low != up) & (np.abs(low - up) > 1)
    if np.any(mask_h):
        ii, jj = np.nonzero(mask_h)
        y_mid = 0.5 * (y_vec[ii] + y_vec[ii + 1])
        x_ctr = x_vec[jj]
        x0 = np.maximum(x_ctr - 0.5 * dx, x_min)
        x1 = np.minimum(x_ctr + 0.5 * dx, x_max)
        for k in range(ii.size):
            segments.append([(x0[k], y_mid[k]), (x1[k], y_mid[k])])

    if len(segments) == 0:
        return None

    artist = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection(artist)
    return artist


def _default_chain_region_colors(n_systems: int) -> np.ndarray:
    n_systems = int(max(1, n_systems))
    if n_systems <= 10:
        return plt.get_cmap("tab10", n_systems)(np.arange(n_systems))
    if n_systems <= 20:
        return plt.get_cmap("tab20", n_systems)(np.arange(n_systems))

    # For long chains, use evenly spaced hues to avoid repeated tab10/tab20
    # colors that make disconnected regions look like the same subsystem.
    h = (np.arange(n_systems, dtype=float) + 0.5) / float(n_systems)
    s = np.full((n_systems,), 0.70, dtype=float)
    v = np.full((n_systems,), 0.95, dtype=float)
    rgb = hsv_to_rgb(np.column_stack([h, s, v]))
    return np.column_stack([rgb, np.ones((n_systems,), dtype=float)])


def evaluate_chain_regions(ds, points: np.ndarray, mode: str = "line_regions", base_colors: np.ndarray = None):
    """Evaluate chain region ownership and associated field on query points."""
    if not _is_chain_ds_for_region_plot(ds):
        raise TypeError("Chain region evaluation requires a chain DS with transition geometry and _velocity_for_index.")

    mode = resolve_chain_plot_mode(mode)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        n_systems = int(max(1, getattr(ds, "n_systems", 1)))
        return (
            np.zeros_like(points),
            np.zeros((points.shape[0],), dtype=int),
            np.zeros((points.shape[0], n_systems), dtype=float),
            np.zeros((points.shape[0], 4), dtype=float),
        )

    n_points = int(points.shape[0])
    n_systems = int(max(1, ds.n_systems))
    if base_colors is None:
        base_colors = _default_chain_region_colors(n_systems)
    base_colors = np.asarray(base_colors, dtype=float)
    if base_colors.ndim != 2 or base_colors.shape[0] < n_systems or base_colors.shape[1] < 4:
        raise ValueError("base_colors must have shape (>=n_systems, >=4).")

    region_idx = np.zeros((n_points,), dtype=int)
    weights = np.zeros((n_points, n_systems), dtype=float)
    velocities = np.zeros_like(points)

    for i in range(n_points):
        x = points[i]
        # Mean-normal line partition first; ambiguous regions fallback to nearest regime segment.
        idx_line, ambiguous = _chain_line_state(ds, x)
        idx = _chain_nominal_index_from_lines_with_regime_fallback(ds, x)
        idx_closest = int(idx)
        closest_idx_2d, _, _ = _closest_regime_info_2d(ds, x)
        if closest_idx_2d is not None:
            idx_closest = int(closest_idx_2d)
        if mode == "time_blend" and ambiguous and closest_idx_2d is not None:
            idx = int(idx_closest)

        w = np.zeros((n_systems,), dtype=float)
        w[idx] = 1.0
        blended = False
        if mode == "time_blend":
            prev_idx = idx - 1
            if (
                prev_idx >= 0
                and not ambiguous
                and int(idx) == int(idx_line)
                and _boundary_has_transition_zone(ds, prev_idx)
                and _is_local_to_transition_neighborhood(ds, prev_idx, x)
            ):
                alpha = _chain_transition_progress(ds, prev_idx, x)
                w[:] = 0.0
                w[prev_idx] = 1.0 - alpha
                w[idx] = alpha
                blended = True

        region_idx[i] = int(idx)
        weights[i] = w

        nonzero_idx = np.flatnonzero(w > 1e-12)
        for k in nonzero_idx:
            velocities[i] += w[k] * _chain_velocity_for_idx(ds, x, int(k))

    rgba = weights @ base_colors[:, :4]
    rgba = np.clip(rgba, 0.0, 1.0)
    return velocities, region_idx, weights, rgba


def _transition_line_endpoints_2d(center: np.ndarray, normal: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float):
    center = np.asarray(center, dtype=float).reshape(-1)
    normal = np.asarray(normal, dtype=float).reshape(-1)
    if center.shape[0] < 2 or normal.shape[0] < 2:
        return None
    n = normal[:2]
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        return None
    n = n / n_norm
    d = np.array([-n[1], n[0]], dtype=float)
    d_norm = float(np.linalg.norm(d))
    if d_norm <= 1e-12:
        return None
    d = d / d_norm

    L = 2.5 * np.hypot(float(x_max - x_min), float(y_max - y_min))
    c = center[:2]
    p0 = c - L * d
    p1 = c + L * d
    return p0, p1


def _transition_line_segment_2d(center: np.ndarray, normal: np.ndarray, half_length: float):
    center = np.asarray(center, dtype=float).reshape(-1)
    normal = np.asarray(normal, dtype=float).reshape(-1)
    if center.shape[0] < 2 or normal.shape[0] < 2:
        return None
    n = normal[:2]
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        return None
    n = n / n_norm
    d = np.array([-n[1], n[0]], dtype=float)
    d_norm = float(np.linalg.norm(d))
    if d_norm <= 1e-12:
        return None
    d = d / d_norm
    L = float(max(half_length, 1e-12))
    c = center[:2]
    p0 = c - L * d
    p1 = c + L * d
    return p0, p1


def _distance_points_to_segment_2d(points_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    a_xy = np.asarray(a_xy, dtype=float).reshape(2)
    b_xy = np.asarray(b_xy, dtype=float).reshape(2)
    ab = b_xy - a_xy
    ab2 = float(np.dot(ab, ab))
    if ab2 <= 1e-12:
        return np.linalg.norm(points_xy - a_xy.reshape(1, 2), axis=1)
    t = ((points_xy - a_xy.reshape(1, 2)) @ ab) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a_xy.reshape(1, 2) + t[:, None] * ab.reshape(1, 2)
    return np.linalg.norm(points_xy - proj, axis=1)


def _chain_regime_segment_2d(ds, idx: int):
    idx = int(idx)
    n_systems = int(max(1, getattr(ds, "n_systems", 1)))
    idx = int(np.clip(idx, 0, n_systems - 1))

    src = np.asarray(getattr(ds, "node_sources", []), dtype=float)
    tgt = np.asarray(getattr(ds, "node_targets", []), dtype=float)
    seq = np.asarray(getattr(ds, "state_sequence", []), dtype=float)

    def _safe_xy(arr, row_idx):
        if arr.ndim != 2 or arr.shape[0] == 0 or row_idx < 0 or row_idx >= arr.shape[0]:
            return None
        v = np.asarray(arr[row_idx], dtype=float).reshape(-1)
        if v.shape[0] < 2 or not np.all(np.isfinite(v[:2])):
            return None
        return v[:2].copy()

    a = _safe_xy(src, idx)
    if a is None:
        a = _safe_xy(seq, idx)
    if a is None:
        return None

    b = _safe_xy(tgt, idx)
    if b is None and src.ndim == 2 and (idx + 1) < src.shape[0]:
        b = _safe_xy(src, idx + 1)
    if b is None:
        b = _safe_xy(seq, idx + 1)
    if b is None:
        b = a.copy()
    return a, b


def _distance_points_to_assigned_regime_2d(ds, points_xy: np.ndarray, region_idx: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    region_idx = np.asarray(region_idx, dtype=int).reshape(-1)
    n_points = int(points_xy.shape[0])
    if n_points == 0:
        return np.zeros((0,), dtype=float)

    n_systems = int(max(1, getattr(ds, "n_systems", 1)))
    dist_matrix = np.full((n_points, n_systems), np.inf, dtype=float)
    for idx in range(n_systems):
        seg = _chain_regime_segment_2d(ds, idx)
        if seg is None:
            continue
        a, b = seg
        dist_matrix[:, idx] = _distance_points_to_segment_2d(points_xy, a, b)

    idx_clipped = np.clip(region_idx, 0, n_systems - 1)
    return dist_matrix[np.arange(n_points), idx_clipped]


def draw_chain_partition_field_2d(
    ax,
    ds,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    mode: str = "line_regions",
    plot_sample: int = 50,
    arrowsize: float = 1.1,
    stream_color: str = "black",
    stream_width: float = 1.0,
    anchor_state: np.ndarray = None,
    region_alpha: float = 0.26,
    stream_density: float = 2.4,
    show_transition_lines: bool = True,
    path_bandwidth: Optional[float] = 0.9,
):
    """Draw chain DS partition background + field streamlines on a 2D axis."""
    mode = resolve_chain_plot_mode(mode)
    plot_sample = int(max(8, plot_sample))
    x_vec = np.linspace(float(x_min), float(x_max), plot_sample)
    y_vec = np.linspace(float(y_min), float(y_max), plot_sample)

    x_mesh, y_mesh = np.meshgrid(
        x_vec,
        y_vec,
    )
    xy = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    # Build full-dim query points (freeze dimensions >2 at anchor values).
    point_dim = max(2, int(np.asarray(ds.node_sources, dtype=float).shape[1]))
    points = np.zeros((xy.shape[0], point_dim), dtype=float)
    points[:, :2] = xy
    if point_dim > 2:
        if anchor_state is None:
            anchor_state = np.zeros((point_dim,), dtype=float)
        anchor_state = np.asarray(anchor_state, dtype=float).reshape(-1)
        if anchor_state.shape[0] < point_dim:
            padded = np.zeros((point_dim,), dtype=float)
            padded[: anchor_state.shape[0]] = anchor_state
            anchor_state = padded
        points[:, 2:] = anchor_state[2:point_dim]

    velocities, region_idx, _, rgba = evaluate_chain_regions(ds, points, mode=mode)
    corridor_mask_flat = np.ones((points.shape[0],), dtype=bool)
    if path_bandwidth is not None:
        path_bandwidth = float(path_bandwidth)
    if path_bandwidth is not None and np.isfinite(path_bandwidth) and path_bandwidth > 0.0:
        distances = _distance_points_to_assigned_regime_2d(ds, xy, region_idx)
        corridor_mask_flat = distances <= path_bandwidth
    if not np.any(corridor_mask_flat):
        corridor_mask_flat = np.ones((points.shape[0],), dtype=bool)

    velocities_masked = velocities.copy()
    velocities_masked[~corridor_mask_flat] = 0.0
    u = velocities[:, 0].reshape(plot_sample, plot_sample)
    v = velocities[:, 1].reshape(plot_sample, plot_sample)
    rgba_img = rgba.reshape(plot_sample, plot_sample, 4)
    alpha = float(np.clip(region_alpha, 0.0, 1.0))
    corridor_mask_img = corridor_mask_flat.reshape(plot_sample, plot_sample)
    rgba_img[:, :, 3] = np.where(corridor_mask_img, alpha, 0.0)

    u = velocities_masked[:, 0].reshape(plot_sample, plot_sample)
    v = velocities_masked[:, 1].reshape(plot_sample, plot_sample)

    region_image = ax.imshow(
        rgba_img,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest" if mode == "line_regions" else "bilinear",
        zorder=0,
        aspect="auto",
    )
    stream = ax.streamplot(
        x_mesh,
        y_mesh,
        u,
        v,
        density=float(stream_density),
        color=stream_color,
        linewidth=stream_width,
        arrowsize=arrowsize,
        arrowstyle="->",
        zorder=2,
    )

    transition_lines = []
    if bool(show_transition_lines):
        line_half_length = None
        if path_bandwidth is not None and np.isfinite(path_bandwidth) and path_bandwidth > 0.0:
            line_half_length = float(path_bandwidth)
        centers = np.asarray(getattr(ds, "transition_centers", []), dtype=float)
        normals = np.asarray(getattr(ds, "transition_normals", []), dtype=float)
        distances = np.asarray(getattr(ds, "transition_distances", []), dtype=float).reshape(-1)
        times = np.asarray(getattr(ds, "transition_times", []), dtype=float).reshape(-1)
        n_lines = min(len(centers), len(normals))
        for i in range(n_lines):
            if mode == "time_blend":
                has_spatial_transition = (
                    i < len(distances)
                    and np.isfinite(distances[i])
                    and float(distances[i]) > 1e-12
                )
                has_temporal_transition = (
                    i < len(times)
                    and np.isfinite(times[i])
                    and float(times[i]) > 1e-12
                )
                if has_spatial_transition or has_temporal_transition:
                    # Transition zone is visualized through color blending.
                    continue
            if line_half_length is None:
                endpoints = _transition_line_endpoints_2d(
                    centers[i], normals[i], x_min=float(x_min), x_max=float(x_max), y_min=float(y_min), y_max=float(y_max)
                )
            else:
                endpoints = _transition_line_segment_2d(
                    centers[i], normals[i], half_length=line_half_length
                )
            if endpoints is None:
                continue
            p0, p1 = endpoints
            line, = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                linestyle="-",
                linewidth=1.6,
                color="black",
                alpha=0.95,
                zorder=3,
            )
            transition_lines.append(line)
        if mode == "time_blend":
            nonadjacent_boundary_artist = _draw_nonadjacent_region_boundaries_2d(
                ax=ax,
                region_idx_img=region_idx.reshape(plot_sample, plot_sample),
                x_vec=x_vec,
                y_vec=y_vec,
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
                linewidth=1.6,
                color="black",
                alpha=0.95,
                zorder=3.0,
            )
            if nonadjacent_boundary_artist is not None:
                transition_lines.append(nonadjacent_boundary_artist)

    return {
        "region_image": region_image,
        "stream": stream,
        "transition_lines": transition_lines,
        "u": u,
        "v": v,
        "corridor_mask": corridor_mask_img,
    }


def plot_ds_2d(
    x_train,
    x_test_list,
    lpvds,
    title=None,
    ax=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    arrowsize=1.1,
    include_raw_data=True,
    linewidth=2,
    marker_size=100,
    stream_density=3.0,
    stream_color='black',
    stream_width=1.0,
    chain_plot_mode: str = "line_regions",
    chain_plot_resolution: int = 60,
    chain_plot_path_bandwidth: Optional[float] = 0.9,
    show_chain_transition_lines: bool = True,
    chain_region_alpha: float = 0.26,
):
    """ passing lpvds object to plot the streamline of DS (only in 2D)"""
    A = lpvds.A
    att = lpvds.x_att

    if ax is None:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot()

    if include_raw_data:
        ax.scatter(x_train[:, 0], x_train[:, 1], color='k', s=5, label='original data')

    if x_min is None or x_max is None or y_min is None or y_max is None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

    if _is_chain_ds_for_region_plot(lpvds):
        plot_sample = int(max(8, chain_plot_resolution))
        draw_chain_partition_field_2d(
            ax=ax,
            ds=lpvds,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            mode=chain_plot_mode,
            plot_sample=plot_sample,
            arrowsize=arrowsize,
            stream_density=stream_density,
            stream_color=stream_color,
            stream_width=stream_width,
            anchor_state=np.asarray(att, dtype=float).reshape(-1),
            region_alpha=chain_region_alpha,
            show_transition_lines=bool(show_chain_transition_lines),
            path_bandwidth=chain_plot_path_bandwidth,
        )
    elif hasattr(lpvds, "vector_field"):
        plot_sample = 50
        x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, plot_sample), np.linspace(y_min, y_max, plot_sample))
        X = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
        dx = lpvds.vector_field(X.T).T
        u = dx[0, :].reshape((plot_sample, plot_sample))
        v = dx[1, :].reshape((plot_sample, plot_sample))
        ax.streamplot(x_mesh, y_mesh, u, v, linewidth=stream_width, density=stream_density, color=stream_color, arrowsize=arrowsize, arrowstyle="->")
    else:
        plot_sample = 50
        x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, plot_sample), np.linspace(y_min, y_max, plot_sample))
        X = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
        gamma = lpvds.damm.compute_gamma(X.T)
        for k in np.arange(len(A)):
            if k == 0:
                dx = gamma[k].reshape(1, -1) * (
                            A[k] @ (X - att.reshape(1, -1).T))  # gamma[k].reshape(1, -1): [1, num] dim x num
            else:
                dx += gamma[k].reshape(1, -1) * (A[k] @ (X - att.reshape(1, -1).T))
        u = dx[0, :].reshape((plot_sample, plot_sample))
        v = dx[1, :].reshape((plot_sample, plot_sample))
        ax.streamplot(x_mesh, y_mesh, u, v, linewidth=stream_width, density=stream_density, color=stream_color, arrowsize=arrowsize, arrowstyle="->")
    for idx, x_test in enumerate(x_test_list):
        ax.plot(x_test[:, 0], x_test[:, 1], color='r', linewidth=linewidth)
    ax.scatter(att[0], att[1], color='g', s=marker_size, alpha=0.7, zorder=10)
    ax.set_aspect('equal')

    if title is not None:
        ax.set_title(title)

def plot_ds_3d(x_train, x_test_list, ax=None, att=None, title=None):
    N = x_train.shape[1]

    if ax is None:
        fig = plt.figure(figsize=(12, 10))
    if N == 2:
        if ax is None:
            ax = fig.add_subplot()
        ax.scatter(x_train[:, 0], x_train[:, 1], color='k', s=1, alpha=0.4, label="Demonstration")
        for idx, x_test in enumerate(x_test_list):
            ax.plot(x_test[:, 0], x_test[:, 1], color='b')

    elif N >= 3:
        if ax is None:
            ax = fig.add_subplot(projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], 'o', color='k', s=3, alpha=0.4, label="Demonstration")

        for idx, x_test in enumerate(x_test_list):
            ax.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], color='b')
        if att is not None:
            att = np.asarray(att, dtype=float).reshape(-1)
            if att.shape[0] >= 3:
                ax.scatter(att[0], att[1], att[2], color='g', s=70, alpha=0.8)
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

    if title is not None:
        ax.set_title(title)
    return ax
