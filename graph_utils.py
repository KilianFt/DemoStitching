import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from numpy.f2py.symbolic import normalize
from src.util import plot_tools

class GaussianGraph:

    def __init__(self, gaussian_mu, gaussian_sigma, gaussian_direction, attractor=None, initial=None, reverse_gaussians=False, param_dist=1, param_cos=1):

        self.param_dist = param_dist
        self.param_cos = param_cos
        self.N = gaussian_mu.shape[0]
        self.graph = self.create_gaussian_graph(gaussian_mu, gaussian_sigma, gaussian_direction,
                                                reverse_gaussians=reverse_gaussians)
        self.gaussian_ids = list(self.graph.nodes.keys())

        self.attractor_id = None
        self.initial_id = None
        if attractor is not None:
            self.set_attractor(attractor)
        if initial is not None:
            self.set_initial(initial)

        self.shortest_path = None

    def create_gaussian_graph(self, gaussian_mu, gaussian_sigma, gaussian_direction, reverse_gaussians=False):
        """ Converts a set of gaussians (with direction) into a networkx graph and saves it to the graph.

        Args:
            gaussian_mu: n x 2 ndarray of gaussian means
            gaussian_sigma: n x 2 x 2 ndarray of gaussian covariance matrices
            gaussian_direction: n x 2 ndarray of gaussian directions
            reverse_gaussians: True to duplicate the gaussians and reverse the directions

        Returns:
            gaussian_graph: networkx Digraph
        """

        gaussian_graph = nx.DiGraph()

        # Convert gaussians to nodes
        for i in range(self.N):
            gaussian_graph.add_node(i, mean=gaussian_mu[i], covariance=gaussian_sigma[i], direction=gaussian_direction[i])

            if reverse_gaussians:
                gaussian_graph.add_node(i+self.N, mean=gaussian_mu[i], covariance=gaussian_sigma[i], direction=-gaussian_direction[i])


        # Connect nodes by weighted edges
        for id1 in gaussian_graph.nodes:
            for id2 in gaussian_graph.nodes:
                edge_weight = self.compute_edge_weight(gaussian_graph.nodes[id1]['mean'],
                                                       gaussian_graph.nodes[id1]['direction'],
                                                       gaussian_graph.nodes[id2]['mean'])

                # add edge to graph
                if edge_weight is not None:
                    gaussian_graph.add_edge(id1, id2, weight=edge_weight)

        return gaussian_graph

    def set_attractor(self, attractor):
        """ Updates/creates an attractor node and adds it to the graph.

        Args:
            attractor position: R^N ndarray
        """

        # remove current attractor if it exists
        if self.attractor_id is not None:
            self.graph.remove_node(self.attractor_id)

        # add attractor
        self.attractor_id = 'attractor'
        self.graph.add_node(self.attractor_id, pos=attractor)
        for gaussian_id in self.gaussian_ids:
            edge_weight = self.compute_edge_weight(self.graph.nodes[gaussian_id]['mean'],
                                                   self.graph.nodes[gaussian_id]['direction'],
                                                   attractor)
            if edge_weight is not None:
                self.graph.add_edge(gaussian_id, self.attractor_id, weight=edge_weight)

    def set_initial(self, initial):
        """ Updates/creates an initial node and adds it to the graph.

        Args:
            initial position: R^N ndarray
        """

        # remove current attractor if it exists
        if self.initial_id is not None:
            self.graph.remove_node(self.initial_id)

        # add attractor
        self.initial_id = 'initial'
        self.graph.add_node(self.initial_id, pos=initial)
        for gaussian_id in self.gaussian_ids:
            edge_weight = self.compute_edge_weight(initial,
                                                   self.graph.nodes[gaussian_id]['direction'],
                                                   self.graph.nodes[gaussian_id]['mean'])
            if edge_weight is not None:
                self.graph.add_edge(self.initial_id, gaussian_id, weight=edge_weight)

    def compute_edge_weight(self, pos1, direction1, pos2):

        # skip nodes with the same mean (same or reverse-pairs)
        if np.array_equal(pos1, pos2):
            return None

        # distance
        d = np.linalg.norm(pos1 - pos2)

        # directionality score
        direction_to = pos2 - pos1
        dir_score = np.dot(direction1, direction_to) / (np.linalg.norm(direction1) * np.linalg.norm(direction_to))
        if dir_score < 0:  # if node2 is behind node1 (by direction) then there is no edge
            return None

        # edge weight (less weight = stronger connection)
        return d ** self.param_dist / dir_score ** self.param_cos

    def compute_shortest_path(self):
        """ Computes the shortest path from the initial to the attractor.
        """
        if self.initial_id is not None and self.attractor_id is not None:
            self.shortest_path = nx.shortest_path(self.graph, source='initial', target='attractor', weight='weight')

    def get_gaussian(self, node_id):
        """ Gets the gaussian parameters of a node.

        Args:
            node_id: id of the node

        Returns:
            mu: Gaussian mean
            sigma: Gaussian covariance
            direction: Gaussian direction
        """
        mu = self.graph.nodes[node_id]['mean']
        sigma = self.graph.nodes[node_id]['covariance']
        direction = self.graph.nodes[node_id]['direction']

        return mu, sigma, direction

    def plot(self, ax=None):
        """Plots a GaussianGraph.

        Args:
            gaussian_graph: a GaussianGraph
        """

        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()

        gg = self.graph

        # Draw nodes (gaussians, attractor, initial)
        pos = {node: gg.nodes[node]['mean'] for node in self.gaussian_ids}
        if self.attractor_id:
            pos[self.attractor_id] = gg.nodes[self.attractor_id]['pos']
        if self.initial_id:
            pos[self.initial_id] = gg.nodes[self.initial_id]['pos']

        colormap = []
        for node in gg.nodes:
            if node is self.attractor_id:
                colormap.append('red')
            elif node is self.initial_id:
                colormap.append('green')
            else:
                colormap.append('blue')

        nx.draw_networkx_nodes(gg, pos, node_color=colormap, node_size=300, ax=ax)
        nx.draw_networkx_labels(gg, pos, font_size=12, font_weight='bold', ax=ax)

        # Extract edge weights
        edges = gg.edges(data=True)
        weights = [edata['weight'] for _, _, edata in edges]

        # Normalize weights for alpha values (higher weight = lower alpha)
        norm_param = 5
        normalize_weights = weights / min(weights)
        normalize_weights = normalize_weights - 1
        normalize_weights = normalize_weights * norm_param / np.median(normalize_weights)
        normalize_weights = normalize_weights + 1
        alphas = [np.exp(1-w) for w in normalize_weights]

        # Draw edges
        for (u, v, edata), alpha in zip(edges, alphas):
            nx.draw_networkx_edges(gg, pos, edgelist=[(u, v)],
                                   alpha=alpha, edge_color='black',
                                   arrows=True, arrowsize=20, ax=ax)

        # Draw shortest path
        if self.shortest_path is not None:

            path_edges = [(self.shortest_path[i], self.shortest_path[i+1]) for i in range(len(self.shortest_path) - 1)]
            nx.draw_networkx_edges(gg, pos, edgelist=path_edges, alpha=0.25, edge_color='magenta',
                                   arrows=True, width=10, arrowsize=30, ax=ax)
        ax.axis('equal')
        # ax.tight_layout()
        ax.grid(True, alpha=0.5)
        # plt.show()

    def plot_shortest_path_gaussians(self, ax=None):
        shortest_path_mus = []
        shortest_path_sigmas = []
        shortest_path_directions = []
        for node_id in self.shortest_path[1:-1]:
            mu, sigma, direction = self.get_gaussian(node_id)
            shortest_path_mus.append(mu)
            shortest_path_sigmas.append(sigma)
            shortest_path_directions.append(direction)
        shortest_path_mus = np.array(shortest_path_mus)
        shortest_path_sigmas = np.array(shortest_path_sigmas)
        shortest_path_directions = np.array(shortest_path_directions)
        
        plot_tools.plot_gaussians(shortest_path_mus,
                                  shortest_path_sigmas,
                                  shortest_path_directions,
                                  ax=ax)
