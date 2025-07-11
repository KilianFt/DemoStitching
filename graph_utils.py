import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.f2py.symbolic import normalize


def create_gaussian_graph(gaussian_mu, gaussian_sigma, gaussian_direction, reverse_gaussians=False, param_dist=1, param_cos=1):
    """ Converts a set of gaussians (with direction) into a networkx graph.

    Args:
        gaussian_mu: N x 2 ndarray of gaussian means
        gaussian_sigma: N x 2 x 2 ndarray of gaussian covariance matrices
        gaussian_direction: N x 2 ndarray of gaussian directions
        reverse_gaussians: True to duplicate the gaussians and reverse the directions
        param_dist: distance exponent

        Returns:
            gaussian_graph: networkx digraph representing the gaussian.

        """

    gaussian_graph = nx.DiGraph()
    N = gaussian_mu.shape[0]

    # Convert gaussians to nodes
    for i in range(N):
        gaussian_graph.add_node(i, mean=gaussian_mu[i], covariance=gaussian_sigma[i], direction=gaussian_direction[i])

        if reverse_gaussians:
            gaussian_graph.add_node(i+N, mean=gaussian_mu[i], covariance=gaussian_sigma[i], direction=-gaussian_direction[i])


    # Connect nodes by weighted edges
    for id1 in gaussian_graph.nodes:
        for id2 in gaussian_graph.nodes:
            mean1 = gaussian_graph.nodes[id1]['mean']
            direction1 = gaussian_graph.nodes[id1]['direction']
            mean2 = gaussian_graph.nodes[id2]['mean']

            # skip nodes with the same mean (same or reverse-pairs)
            if np.array_equal(mean1, mean2):
                continue

            # distance
            d = np.linalg.norm(mean1 - mean2)

            # directionality score
            direction_to = mean2 - mean1
            dir_score = np.dot(direction1, direction_to) / (np.linalg.norm(direction1) * np.linalg.norm(direction_to))
            if dir_score < 0: # skip if node2 is behind node1 (based on node1 direction)
                continue

            # edge weight (less weight = stronger connection)
            edge_weight = d**param_dist / dir_score**param_cos

            # add edge to graph
            gaussian_graph.add_edge(id1, id2, weight=edge_weight)

    return gaussian_graph

def plot_gaussian_graph(gaussian_graph):
    """Plot the gaussian graph with nodes at their mean positions and edges with alpha based on edge weight.
    Higher weight means more transparent (lower alpha)."""

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Draw nodes
    pos = {node: gaussian_graph.nodes[node]['mean'] for node in gaussian_graph.nodes()}
    nx.draw_networkx_nodes(gaussian_graph, pos, node_color='lightblue', node_size=300, ax=ax)
    nx.draw_networkx_labels(gaussian_graph, pos, font_size=12, font_weight='bold', ax=ax)

    # Extract edge weights
    edges = gaussian_graph.edges(data=True)
    weights = [edata['weight'] for _, _, edata in edges]

    # Normalize weights for alpha values (higher weight = lower alpha)
    norm_param = 1.3
    normalize_weights = weights / min(weights)
    normalize_weights = normalize_weights - 1
    normalize_weights = normalize_weights * norm_param / np.median(normalize_weights)
    normalize_weights = normalize_weights + 1
    alphas = [np.exp(1-w) for w in normalize_weights]

    # Draw edges
    for (u, v, edata), alpha in zip(edges, alphas):
        nx.draw_networkx_edges(gaussian_graph, pos, edgelist=[(u, v)],
                               alpha=alpha, edge_color='black',
                               arrows=True, arrowsize=20, ax=ax)

    plt.title('Gaussian Graph with Edge Transparency by Weight')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
