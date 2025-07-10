import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class

input_message = '''
Please choose a data input option:
1. X - trajectory sets
2. Intersecting trajectories
Enter the corresponding option number: '''
input_opt  = input(input_message)

# data list with each entry containing x, x_dot, x_att, x_init
data = load_tools.load_data_stitch(int(input_opt))

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# 1) get GMM centers and directionality

node_centers = []
node_directions = []
for i in range(len(data)):
    x, x_dot, x_att, x_init = data[i]
    lpvds = lpvds_class(x, x_dot, x_att)
    lpvds._cluster()

    # get directionality
    centers = lpvds.damm.Mu
    assignment_arr = lpvds.assignment_arr
    # get mean xdot per cluster
    # NOTE could be modified to have weighted average
    mean_xdot = np.zeros((lpvds.damm.K, x.shape[1]))
    for k in range(lpvds.damm.K):
        mean_xdot[k] = np.mean(x_dot[assignment_arr==k], axis=0)

    for k in range(lpvds.damm.K):
        sigma = lpvds.damm.Sigma[k, :, :]
        mu = lpvds.damm.Mu[k, :]
        plot_tools.plot_2d_gaussian(mu, sigma, ax = ax)
    
    # plot mean xdot per cluster
    for k in range(lpvds.damm.K):
        ax.arrow(centers[k, 0],
                 centers[k, 1],
                 mean_xdot[k, 0],
                 mean_xdot[k, 1],
                 head_width=0.5,
                 head_length=0.5,
                 fc='r',
                 ec='r',
                 zorder=10
                 )

    node_centers.extend(centers.tolist())
    node_directions.extend(mean_xdot.tolist())

    # plot original data if wanted
    # ax.scatter(x[:, 0], x[:, 1], color='k', s=5, label='original data')

# 2) build graph
print(node_centers)
print(node_directions)

# given shortest path, run lpvds

# run lpvds
# lpvds.begin()

# evaluate results
# x_test_list = []
# for x_0 in x_init:
#     x_test_list.append(lpvds.sim(x_0, dt=0.01))


# plot results
    # plot_tools.plot_gmm(x, lpvds.assignment_arr, lpvds.damm, axs[1,i])
# if x.shape[1] == 2:
#     plot_tools.plot_ds_2d(x, x_test_list, lpvds)
# else:
#     plot_tools.plot_ds_3d(x, x_test_list)
plt.tight_layout()
plt.show()