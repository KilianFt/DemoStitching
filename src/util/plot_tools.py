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


def plot_2d_gaussian(mu, sigma, ax=None, n_ellipses=3, n_points=100, force_plot=False, print_matrix_info=False):
    """
    Plot a 2D Gaussian distribution with confidence ellipses
    
    Parameters:
    mu: array-like, shape (2,) - mean vector
    sigma: array-like, shape (2,2) - covariance matrix
    n_ellipses: int - number of confidence ellipses to plot
    n_points: int - resolution of the contour plot
    force_plot: bool - if True, attempt to plot even invalid covariance matrices
    print_matrix_info: bool - if True, print matrix properties
    """
    
    # Convert to numpy arrays
    mu = np.array(mu)
    sigma = np.array(sigma)
    
    # Check matrix properties
    eigenvals, eigenvecs = np.linalg.eigh(sigma)
    det = np.linalg.det(sigma)
    is_singular = np.abs(det) < 1e-10
    is_positive_definite = np.all(eigenvals > 1e-10)
    is_negative_definite = np.all(eigenvals < -1e-10)
    
    if print_matrix_info:
        print(f"Matrix properties:")
        print(f"  Determinant: {det:.2e}")
        print(f"  Eigenvalues: {eigenvals}")
        print(f"  Positive definite: {is_positive_definite}")
        print(f"  Negative definite: {is_negative_definite}")
        print(f"  Singular: {is_singular}")
    
    # Handle different cases
    if is_negative_definite:
        print("Warning: Matrix is negative definite - not a valid covariance matrix!")
        if not force_plot:
            print("This cannot represent a probability distribution.")
            print("Possible fixes:")
            print("  1. Take absolute value: sigma_fixed = np.abs(sigma)")
            print("  2. Flip signs: sigma_fixed = -sigma")
            print("  3. Use force_plot=True to plot anyway (will use abs(eigenvalues))")
            return None, None
        else:
            print("Using absolute values of eigenvalues for plotting...")
            eigenvals = np.abs(eigenvals)
            sigma_reg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    elif is_singular:
        print("Warning: Covariance matrix is singular (determinant ≈ 0)")
        print("This represents a degenerate distribution along a line")
        # Add small regularization
        sigma_reg = sigma + np.eye(2) * 1e-6
        print(f"Adding regularization: {1e-6} to diagonal elements")
    elif not is_positive_definite:
        print("Warning: Matrix is not positive definite (has negative eigenvalues)")
        print("Adding regularization to make it positive definite...")
        # Make all eigenvalues positive
        eigenvals_fixed = np.maximum(eigenvals, 1e-6)
        sigma_reg = eigenvecs @ np.diag(eigenvals_fixed) @ eigenvecs.T
    else:
        sigma_reg = sigma
    
    # Create a grid for plotting the contours
    x_range = 4 * np.sqrt(np.abs(sigma[0, 0]))
    y_range = 4 * np.sqrt(np.abs(sigma[1, 1]))
    
    x = np.linspace(mu[0] - x_range, mu[0] + x_range, n_points)
    y = np.linspace(mu[1] - y_range, mu[1] + y_range, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the probability density
    pos = np.dstack((X, Y))
    try:
        rv = multivariate_normal(mu, sigma_reg)
        pdf_values = rv.pdf(pos)
    except Exception as e:
        print(f"Error computing PDF: {e}")
        return None, None
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the contours
    try:
        contours = ax.contour(X, Y, pdf_values, levels=10, colors='blue', alpha=0.6)
        ax.contourf(X, Y, pdf_values, levels=50, cmap='Blues', alpha=0.3)
    except Exception as e:
        print(f"Error plotting contours: {e}")
        # Plot a simple scatter or other visualization
        ax.scatter(mu[0], mu[1], c='black', s=100, alpha=0.7)
    
    # Plot the mean point
    ax.plot(mu[0], mu[1], 'o', markersize=5, color='black', label='Mean')
    
    # Add confidence ellipses using corrected eigenvalues
    eigenvals_corrected, eigenvecs_corrected = np.linalg.eigh(sigma_reg)
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvals_corrected)[::-1]
    eigenvals_corrected = eigenvals_corrected[idx]
    eigenvecs_corrected = eigenvecs_corrected[:, idx]
    
    # Calculate angle of rotation
    angle = np.degrees(np.arctan2(eigenvecs_corrected[1, 0], eigenvecs_corrected[0, 0]))
    
    # Plot ellipses for different confidence levels
    colors = ['red', 'orange', 'yellow']
    confidence_levels = [1, 2, 3]  # 1σ, 2σ, 3σ
    
    for i, (conf_level, color) in enumerate(zip(confidence_levels[:n_ellipses], colors[:n_ellipses])):
        # Calculate ellipse parameters
        width = 2 * conf_level * np.sqrt(np.abs(eigenvals_corrected[0]))
        height = 2 * conf_level * np.sqrt(np.abs(eigenvals_corrected[1]))
        
        ellipse = Ellipse(xy=mu, width=width, height=height, 
                         angle=angle, facecolor='none', 
                         edgecolor=color, linewidth=2,
                         label=f'{conf_level}σ ellipse')
        ax.add_patch(ellipse)
    
    # Add information about the distribution
    # info_text = f'det(Σ) = {det:.2e}\nλ₁ = {eigenvals[0]:.2e}\nλ₂ = {eigenvals[1]:.2e}'
    # if is_negative_definite:
    #     info_text += '\nNegative definite!'
    # elif is_singular:
    #     info_text += '\nSingular matrix!'
    # elif not is_positive_definite:
    #     info_text += '\nNot pos. definite!'
    
    # ax.text(0.02, 0.98, info_text, 
    #         transform=ax.transAxes, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    # ax.set_title('2D Gaussian Distribution with Confidence Ellipses')
    # ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if ax is None:
        plt.tight_layout()
        plt.show()
        return fig, ax

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



def plot_ds_2d(x_train, x_test_list, lpvds, *args):
    """ passing lpvds object to plot the streamline of DS (only in 2D)"""
    A = lpvds.A
    att = lpvds.x_att

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot()

    ax.scatter(x_train[:, 0], x_train[:, 1], color='k', s=5, label='original data')
    for idx, x_test in enumerate(x_test_list):
        ax.plot(x_test[:, 0], x_test[:, 1], color= 'r', linewidth=2)

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

    plt.streamplot(x_mesh,y_mesh,u,v, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")
    plt.gca().set_aspect('equal')

    if len(args) !=0:
        ax.set_title(args[0])




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