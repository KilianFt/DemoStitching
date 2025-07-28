import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections.abc import Iterable
from src.util import plot_tools

class GaussianGraph:

    def __init__(self, gaussian_mu, gaussian_sigma, gaussian_direction, attractor=None, initial=None, reverse_gaussians=False, param_dist=1, param_cos=1):

        self.param_dist = param_dist
        self.param_cos = param_cos
        self.n_gaussians = gaussian_mu.shape[0]
        self.gaussian_reversal_map = dict()
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
        for i in range(self.n_gaussians):

            id = i
            gaussian_graph.add_node(id, mean=gaussian_mu[i], covariance=gaussian_sigma[i], direction=gaussian_direction[i])

            # If gaussian reversal is enabled, create a reverse node and mapping to the original
            if reverse_gaussians:
                id_reverse = i + self.n_gaussians
                gaussian_graph.add_node(id_reverse, mean=gaussian_mu[i], covariance=gaussian_sigma[i], direction=-gaussian_direction[i])
                self.gaussian_reversal_map[id_reverse] = id

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
        """Computes edge weight between two nodes based on distance and directionality.

        Args:
            pos1 (array): Position of first node.
            direction1 (array): Direction vector of first node.
            pos2 (array): Position of second node.

        Returns:
            float or None: Edge weight (distance^param_dist / dir_score^param_cos)
                or None if nodes have same position or direction score is negative.
        """

        # skip nodes with the same mean (same or reverse-pairs)
        if np.allclose(pos1, pos2):
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
        """Compute the shortest path from initial to attractor and configure initial node.

        Finds the weighted shortest path and sets the initial node's mean, covariance,
        and direction based on the next node in the path.
        """
        if self.initial_id is None or self.attractor_id is None:
            print("Initial or attractor node not set. Cannot compute shortest path.")

        self.shortest_path = nx.shortest_path(self.graph, source='initial', target='attractor', weight='weight')

        # Give the initial node a mean, covariance, and direction toward the next node in the path
        next_node = self.graph.nodes[self.shortest_path[1]]

        # direction
        init_direction = next_node['mean'] - self.graph.nodes[self.initial_id]['pos']
        init_direction = init_direction / np.linalg.norm(init_direction) * np.linalg.norm(next_node['direction'])

        # covariance
        R = self._rotation_matrix_between_vectors(next_node['direction'], init_direction)
        init_covariance = R @ next_node['covariance'] @ R.T

        self.graph.nodes[self.initial_id]['mean'] = self.graph.nodes[self.initial_id]['pos']
        self.graph.nodes[self.initial_id]['covariance'] = init_covariance
        self.graph.nodes[self.initial_id]['direction'] = init_direction

    def compute_node_wise_shortest_path(self):
        """Computes shortest paths to attractor and removes duplicate nodes at same positions.

        Finds shortest paths from all nodes to the attractor node, then removes duplicate
        nodes (same mean position) by keeping only the one with the shortest path length.

        Returns:
            None: Results stored in self.node_wise_shortest_path as list of node IDs.

        Note:
            Requires self.attractor_id to be set. No-op if attractor_id is None.
        """
        def path_length(path):
            """Calculate the length of a path based on edge weights."""
            return sum(self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

        if self.attractor_id is None:
            print("Attractor node not set. Cannot compute node-wise shortest path.")

        all_paths_to_target = nx.shortest_path(self.graph, target=self.attractor_id, weight='weight')

        # compare gaussians at the same location, remove the one with the longer path
        nodes_to_remove = set()
        for node, path in all_paths_to_target.items():
            if node not in nodes_to_remove:

                # find all other nodes with the same mean
                for other_node, other_path in all_paths_to_target.items():
                    if np.allclose(self.graph.nodes[node]['mean'], self.graph.nodes[other_node]['mean']) and node != other_node:

                        # compare path lengths
                        if path_length(path) > path_length(other_path):
                            nodes_to_remove.add(node)
                            break
                        else:
                            nodes_to_remove.add(other_node)
                            continue

        nodes_to_keep = list(set(all_paths_to_target.keys()) - nodes_to_remove)
        self.node_wise_shortest_path = nodes_to_keep

    @staticmethod
    def _rotation_matrix_between_vectors(a, b, tol=1e-8):
        """Rotation matrix that rotates vector a to align with vector b.

        Args:
            a: Source vector (2D or 3D).
            b: Target vector (2D or 3D).
            tol: Tolerance for parallel/anti-parallel detection (3D only).

        Returns:
            Rotation matrix (2×2 for 2D vectors, 3×3 for 3D vectors).
        """
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same dimension")

        dim = len(a)

        # Normalize input vectors
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)

        if dim == 2:
            # 2D case: calculate rotation angle and build matrix
            angle = np.arctan2(b_norm[1], b_norm[0]) - np.arctan2(a_norm[1], a_norm[0])
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s],
                             [s, c]])

        elif dim == 3:
            # 3D case: use cross product and Rodrigues' formula
            v = np.cross(a_norm, b_norm)  # rotation axis (unnormalized)
            s = np.linalg.norm(v)  # sin(θ)
            c = np.dot(a_norm, b_norm)  # cos(θ)

            if s < tol:  # parallel or anti-parallel
                return np.eye(3) if c > 0 else (2 * np.outer(a_norm, a_norm) - np.eye(3))

            # Skew-symmetric matrix
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])

            return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)

        else:
            raise ValueError(f"Only 2D and 3D vectors supported, got {dim}D")

    def get_gaussians(self, node_id):
        """Gets the Gaussian parameters of one or more nodes.

        Args:
            node_id (int or iterable): Node ID or iterable of node IDs.

        Returns:
            tuple or tuple of ndarrays:
                If node_id is a single ID, returns tuple of (mu, sigma, direction).
                If node_id is an iterable, returns tuple of three ndarrays
                (mus, sigmas, directions), each containing the respective values for all nodes.

                Where:
                    mu: Gaussian mean
                    sigma: Gaussian covariance
                    direction: Gaussian direction
        """
        if isinstance(node_id, Iterable) and not isinstance(node_id, (str, bytes)):
            mus = []
            sigmas = []
            directions = []
            for id in node_id:
                mu = self.graph.nodes[id]['mean']
                sigma = self.graph.nodes[id]['covariance']
                direction = self.graph.nodes[id]['direction']
                mus.append(mu)
                sigmas.append(sigma)
                directions.append(direction)
            return np.array(mus), np.array(sigmas), np.array(directions)
        else:
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

        nx.draw_networkx_nodes(gg, pos, node_color=colormap, node_size=100, ax=ax)
        nx.draw_networkx_labels(gg, pos, font_size=8, font_weight='bold', ax=ax)

        # Extract edge weights
        edges = gg.edges(data=True)
        weights = [edata['weight'] for _, _, edata in edges]

        # Normalize weights for alpha values (higher weight = lower alpha)
        norm_param = 10
        normalize_weights = weights / min(weights)
        normalize_weights = normalize_weights - 1
        normalize_weights = normalize_weights * norm_param / np.median(normalize_weights)
        normalize_weights = normalize_weights + 1
        alphas = [np.exp(1-w) for w in normalize_weights]

        # Draw edges
        for (u, v, edata), alpha in zip(edges, alphas):
            nx.draw_networkx_edges(gg, pos, edgelist=[(u, v)],
                                   alpha=alpha, edge_color='black',
                                   arrows=True, arrowsize=8, ax=ax)

        # Draw shortest path
        if self.shortest_path is not None:

            path_edges = [(self.shortest_path[i], self.shortest_path[i+1]) for i in range(len(self.shortest_path) - 1)]
            nx.draw_networkx_edges(gg, pos, edgelist=path_edges, alpha=0.25, edge_color='magenta',
                                   arrows=True, width=10, arrowsize=30, ax=ax)
        ax.axis('equal')
        # ax.tight_layout()
        ax.grid(True, alpha=0.5)
        # plt.show()
        return ax

    def plot_shortest_path_gaussians(self, ax=None):
        shortest_path_mus = []
        shortest_path_sigmas = []
        shortest_path_directions = []
        for node_id in self.shortest_path[1:-1]:
            mu, sigma, direction = self.get_gaussians(node_id)
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
