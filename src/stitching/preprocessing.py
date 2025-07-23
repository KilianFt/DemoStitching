import numpy as np

from src.util import load_tools
from src.lpvds_class import lpvds_class


def calculate_demo_lpvds(input_opt, data_file=None):
    # data list with each entry containing x, x_dot, x_att, x_init
    data, data_hash = load_tools.load_data_stitch(int(input_opt), data_file=data_file)
    n_demos = len(data)
    node_centers = []
    node_sigmas = []
    node_directions = []
    xs = []
    x_dots = []
    x_atts = []
    x_inits = []
    assignment_arrs = []
    n_centers = 0
    As = []
    Ps = []
    for i in range(n_demos):
        x, x_dot, x_att, x_init = data[i]
        lpvds = lpvds_class(x, x_dot, x_att)
        # lpvds._cluster()
        lpvds.begin()

        # get directionality
        centers = lpvds.damm.Mu
        assignment_arr = lpvds.assignment_arr
        # NOTE could be modified to have weighted average
        mean_xdot = np.zeros((lpvds.damm.K, x.shape[1]))
        for k in range(lpvds.damm.K):
            mean_xdot[k] = np.mean(x_dot[assignment_arr==k], axis=0)

        node_centers.extend(centers.tolist())
        node_directions.extend(mean_xdot.tolist())
        node_sigmas.extend(lpvds.damm.Sigma.tolist())
        xs.append(x)
        x_dots.append(x_dot)
        x_atts.append(np.squeeze(x_att))
        x_inits.append(np.squeeze(x_init).mean(axis=0))
        As.append(lpvds.A)
        Ps.append(lpvds.ds_opt.P)

        # ensures that the assignment array is unique
        assignment_arrs.append(assignment_arr+n_centers)
        n_centers += centers.shape[0]

    return {
        "centers": np.array(node_centers),
        "directions": np.array(node_directions),
        "sigmas": np.array(node_sigmas),
        "xs": xs,
        "x_dots": x_dots,
        "x_atts": x_atts,
        "x_inits": x_inits,
        "assignment_arrs": assignment_arrs,
        "As": np.vstack(As),
        "Ps": Ps,
    }, data_hash
