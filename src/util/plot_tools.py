from typing import List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import random


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": 30
})



def plot_gaussians_with_ds(gg, lpvds, x_test_list, save_folder, i, config): 
    fig, axs = plt.subplots(1, 1, figsize=(8,6), sharex=True, sharey=True)
    mus, sigmas, directions = gg.get_gaussians(gg.shortest_path[1:-1])
    plot_gaussians(mus, sigmas, directions, ax=axs, extent=((config.x_min, config.x_max), (config.y_min, config.y_max)), resolution=1000)
    # gg.plot_shortest_path_gaussians(ax=axs[0])
    axs.set_xlim(config.x_min, config.x_max)
    axs.set_ylim(config.y_min, config.y_max)
    axs.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_folder + "shortest_path_{}.png".format(i), dpi=300)
    plt.close()

    fig, axs = plt.subplots(1, 1, figsize=(8,6), sharex=True, sharey=True)
    plot_ds_2d(lpvds.x, x_test_list, lpvds, ax=axs, x_min=config.x_min, x_max=config.x_max, y_min=config.y_min, y_max=config.y_max)
    axs.set_xlim(config.x_min, config.x_max)
    axs.set_ylim(config.y_min, config.y_max)
    axs.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_folder + "ds_{}.png".format(i), dpi=300)
    plt.close()

    # Plot updates gaussians from lpvds if they were updated
    if hasattr(lpvds.damm, "Mu"):
        fig, axs = plt.subplots(1, 1, figsize=(8,6), sharex=True, sharey=True)
        centers = lpvds.damm.Mu
        assignment_arr = lpvds.assignment_arr
        mean_xdot = np.zeros((lpvds.damm.K, lpvds.x.shape[1]))
        for k in range(lpvds.damm.K):
            mean_xdot[k] = np.mean(lpvds.x_dot[assignment_arr==k], axis=0)
        plot_gaussians(centers, lpvds.damm.Sigma, mean_xdot, ax=axs, extent=((config.x_min, config.x_max), (config.y_min, config.y_max)), resolution=1000)
        axs.set_xlim(config.x_min, config.x_max)
        axs.set_ylim(config.y_min, config.y_max)
        axs.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(save_folder + "updated_gaussians_{}.png".format(i), dpi=300)
        plt.close()


def save_initial_plots(gg, data, save_folder, config):
    # initial plots that only need to be computed once
    fig, axs = plt.subplots(1, 1, figsize=(8,6), sharex=True, sharey=True)
    axs = gg.plot(ax=axs)
    # Override the grid setting from gg.plot() and set proper limits
    axs.grid(False)
    axs.axis("off")
    axs.set_xlim(config.x_min, config.x_max)
    axs.set_ylim(config.y_min, config.y_max)
    axs.set_aspect('equal')
    plt.savefig(save_folder + "graph.png", dpi=300)
    plt.close()

    fig, axs = plt.subplots(1, 1, figsize=(8,6), sharex=True, sharey=True)
    plot_gaussians(data["centers"], data["sigmas"], data["directions"], ax=axs, extent=((config.x_min, config.x_max), (config.y_min, config.y_max)), resolution=1000)
    axs.set_xlim(config.x_min, config.x_max)
    axs.set_ylim(config.y_min, config.y_max)
    axs.set_aspect('equal')
    plt.savefig(save_folder + "gaussians.png", dpi=300)
    plt.close()


def plot_gaussians(mus, sigmas, directions=None, resolution=400, extent=None, ax=None):
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
    # initialize lists for means and covariances
    N = len(mus)
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

    # Reshape back to grid
    heatmap = total_values.reshape(resolution[1], resolution[0])

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(heatmap, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='GnBu', aspect='equal')

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

        # FIXED: Use ymin instead of ymax, and scale appropriately
        arrow_scale = min(xmax - xmin, ymax - ymin) * 0.01

        for mu, dir_vec in zip(mus, directions_norm):
            ax.arrow(mu[0], mu[1],
                     dir_vec[0] * arrow_scale, dir_vec[1] * arrow_scale,
                     head_width=arrow_scale*0.2, head_length=arrow_scale*0.15,
                     fc='white', ec='white', linewidth=2, alpha=0.8)

    # plt.colorbar(im, ax=ax, label='Scaled Gaussian Density')

    # set tight layout
    ax.set_aspect('equal')
    return ax


def plot_trajectories(trajectories: List[np.ndarray], title: str = "Trajectories", save_folder: str = "", config = None, ax = None):
    """Plot trajectories."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), sharex=True, sharey=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.7, label=f'Traj {i+1}')
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
        ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8)
        
    # ax.grid(True, alpha=0.3)
    if config is not None:
        ax.set_xlim(config.x_min, config.x_max)
        ax.set_ylim(config.y_min, config.y_max)

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_folder + "trajectories.png", dpi=300)
    plt.close()


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
            dx = gamma[k].reshape(1, -1) * (A[k] @ (X - att.T))  # gamma[k].reshape(1, -1): [1, num] dim x num
        else:
            dx +=  gamma[k].reshape(1, -1) * (A[k] @ (X - att.T)) 
    u = dx[0,:].reshape((plot_sample,plot_sample))
    v = dx[1,:].reshape((plot_sample,plot_sample))

    ax.streamplot(x_mesh,y_mesh,u,v, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")
    ax.scatter(att[:,0], att[:,1], color='g', s=100, alpha=0.7)
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