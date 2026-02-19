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
        plt.savefig(save_folder + save_as + '.pdf')

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
        plt.savefig(save_folder + save_as + '.pdf')

    return ax

def plot_gg_solution(gg, solution_nodes, initial, attractor, config, ax=None, save_as=None, hide_axis=False):
    dim = _infer_graph_dim(gg)
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
        plt.savefig(save_folder + save_as + '.pdf')

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
        plt.savefig(save_folder + save_as + '.pdf')

    return ax

def plot_gaussian_graph(gg, config, ax=None, save_as=None, hide_axis=False):
    """Plots a GaussianGraph.

    Args:
        gg: a GaussianGraph
    """
    dim = _infer_graph_dim(gg)
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
        x = np.asarray(traj.x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
            continue
        if x.shape[1] >= 3:
            ax.plot(x[:, 0], x[:, 1], x[:, 2], color=color, linewidth=linewidth, alpha=alpha)
            ax.scatter(x[0, 0], x[0, 1], x[0, 2], color='green', s=marker_size * 6)
            ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], color='red', s=marker_size * 6)
        else:
            ax.plot(x[:, 0], x[:, 1], color=color, linewidth=linewidth, alpha=alpha)
            # Start point (green)
            ax.plot(x[0, 0], x[0, 1], 'go', markersize=marker_size)
            # End point (red)
            ax.plot(x[-1, 0], x[-1, 1], 'ro', markersize=marker_size)

    return ax

def primitive_plot_gaussian(ax, mu, sigma, color=None, sigma_bound=2, resolution=200, direction=None):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float)

    # Params
    sigma_bound_color = 'black'
    sigma_bound_linewidth = 0.25
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
    x = np.asarray(x, dtype=float).reshape(-1)
    n_systems = int(max(1, getattr(ds, "n_systems", 1)))
    n_trans = min(
        n_systems - 1,
        len(np.asarray(getattr(ds, "transition_centers", []))),
        len(np.asarray(getattr(ds, "transition_normals", []))),
    )
    idx = 0
    for k in range(n_trans):
        center = np.asarray(ds.transition_centers[k], dtype=float).reshape(-1)
        normal = np.asarray(ds.transition_normals[k], dtype=float).reshape(-1)
        dim = min(center.shape[0], normal.shape[0], x.shape[0])
        if dim <= 0:
            break
        signed = float(np.dot(x[:dim] - center[:dim], normal[:dim]))
        if signed >= 0.0:
            idx += 1
        else:
            break
    return int(np.clip(idx, 0, n_systems - 1))


def _chain_transition_progress(ds, boundary_idx: int, x: np.ndarray) -> float:
    centers = np.asarray(ds.transition_centers, dtype=float)
    normals = np.asarray(ds.transition_normals, dtype=float)
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
    if boundary_idx < len(distances) and np.isfinite(distances[boundary_idx]):
        length = float(max(distances[boundary_idx], 1e-12))
    else:
        length = 1.0
    return float(np.clip(signed / length, 0.0, 1.0))


def _chain_velocity_for_idx(ds, x: np.ndarray, idx: int) -> np.ndarray:
    idx = int(np.clip(idx, 0, int(max(1, ds.n_systems)) - 1))
    x = np.asarray(x, dtype=float).reshape(-1)
    velocity = np.asarray(ds._velocity_for_index(x, idx), dtype=float).reshape(-1)
    if velocity.shape[0] != x.shape[0]:
        raise ValueError("Chain DS velocity dimension mismatch.")
    return velocity


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
        base_colors = plt.get_cmap("tab10", n_systems)(np.arange(n_systems))
    base_colors = np.asarray(base_colors, dtype=float)
    if base_colors.ndim != 2 or base_colors.shape[0] < n_systems or base_colors.shape[1] < 4:
        raise ValueError("base_colors must have shape (>=n_systems, >=4).")

    region_idx = np.zeros((n_points,), dtype=int)
    weights = np.zeros((n_points, n_systems), dtype=float)
    velocities = np.zeros_like(points)

    for i in range(n_points):
        x = points[i]
        idx = _chain_nominal_index_from_lines(ds, x)
        region_idx[i] = idx

        w = np.zeros((n_systems,), dtype=float)
        w[idx] = 1.0
        if mode == "time_blend":
            prev_idx = idx - 1
            if prev_idx >= 0:
                alpha = _chain_transition_progress(ds, prev_idx, x)
                w[:] = 0.0
                w[prev_idx] = 1.0 - alpha
                w[idx] = alpha
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


def draw_chain_partition_field_2d(
    ax,
    ds,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    mode: str = "line_regions",
    plot_sample: int = 50,
    anchor_state: np.ndarray = None,
    region_alpha: float = 0.26,
    stream_density: float = 2.4,
    show_transition_lines: bool = True,
):
    """Draw chain DS partition background + field streamlines on a 2D axis."""
    mode = resolve_chain_plot_mode(mode)
    plot_sample = int(max(8, plot_sample))

    x_mesh, y_mesh = np.meshgrid(
        np.linspace(float(x_min), float(x_max), plot_sample),
        np.linspace(float(y_min), float(y_max), plot_sample),
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

    velocities, _, _, rgba = evaluate_chain_regions(ds, points, mode=mode)
    u = velocities[:, 0].reshape(plot_sample, plot_sample)
    v = velocities[:, 1].reshape(plot_sample, plot_sample)
    rgba_img = rgba.reshape(plot_sample, plot_sample, 4)
    rgba_img[:, :, 3] = float(np.clip(region_alpha, 0.0, 1.0))

    region_image = ax.imshow(
        rgba_img,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
        zorder=0,
        aspect="auto",
    )
    stream = ax.streamplot(
        x_mesh,
        y_mesh,
        u,
        v,
        density=float(stream_density),
        color="black",
        arrowsize=1.0,
        arrowstyle="->",
        zorder=2,
    )

    transition_lines = []
    if bool(show_transition_lines):
        centers = np.asarray(getattr(ds, "transition_centers", []), dtype=float)
        normals = np.asarray(getattr(ds, "transition_normals", []), dtype=float)
        n_lines = min(len(centers), len(normals))
        for i in range(n_lines):
            endpoints = _transition_line_endpoints_2d(
                centers[i], normals[i], x_min=float(x_min), x_max=float(x_max), y_min=float(y_min), y_max=float(y_max)
            )
            if endpoints is None:
                continue
            p0, p1 = endpoints
            line, = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                linestyle="--",
                linewidth=1.1,
                color="black",
                alpha=0.75,
                zorder=3,
            )
            transition_lines.append(line)

    return {
        "region_image": region_image,
        "stream": stream,
        "transition_lines": transition_lines,
        "u": u,
        "v": v,
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
    chain_plot_mode: str = "line_regions",
    chain_plot_resolution: int = 60,
    show_chain_transition_lines: bool = True,
    chain_region_alpha: float = 0.26,
):
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
            anchor_state=np.asarray(att, dtype=float).reshape(-1),
            region_alpha=chain_region_alpha,
            stream_density=3.0,
            show_transition_lines=bool(show_chain_transition_lines and resolve_chain_plot_mode(chain_plot_mode) == "line_regions"),
        )
    elif hasattr(lpvds, "vector_field"):
        plot_sample = 50
        x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, plot_sample), np.linspace(y_min, y_max, plot_sample))
        X = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
        dx = lpvds.vector_field(X.T).T
        u = dx[0, :].reshape((plot_sample, plot_sample))
        v = dx[1, :].reshape((plot_sample, plot_sample))
        ax.streamplot(x_mesh, y_mesh, u, v, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")
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
        ax.streamplot(x_mesh, y_mesh, u, v, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")

    ax.scatter(att[0], att[1], color='g', s=100, alpha=0.7)
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
