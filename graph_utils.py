import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.util import plot_tools

class GaussianGraph:

    def __init__(self, gaussians, attractor=None, initial=None, reverse_gaussians=False, param_dist=1, param_cos=1):
        """Initializes a GaussianGraph with nodes and edge parameters.

        Args:
            gaussians: Dictionary of Gaussian parameters for graph nodes.
            attractor: Optional attractor point to mark in the graph.
            initial: Optional initial point to mark in the graph.
            reverse_gaussians: If True, adds nodes with reversed directions.
            param_dist: Distance scaling parameter for edge computation.
            param_cos: Cosine scaling parameter for edge computation.

        """
        self.param_dist = param_dist
        self.param_cos = param_cos
        self.n_gaussians = len(gaussians)
        self.gaussian_reversal_map = dict()
        self.graph = nx.DiGraph()
        self.create_gaussian_graph(gaussians, reverse_gaussians=reverse_gaussians)
        self.gaussian_ids = list(self.graph.nodes.keys())

        self.attractor_id = None
        self.initial_id = None
        if attractor is not None:
            self.set_attractor(attractor)
        if initial is not None:
            self.set_initial(initial)

        self.shortest_path = None

    def create_gaussian_graph(self, gaussians, reverse_gaussians=False):
        """Builds a directed graph where nodes represent Gaussians with direction and prior.

        Args:
            gaussians: Dictionary mapping node IDs to Gaussian parameters
                       ('mu', 'sigma', 'direction', 'prior').
            reverse_gaussians: If True, adds reversed-direction nodes for each Gaussian.

        Returns:
            None: Modifies the internal graph by adding nodes and weighted edges.
        """

        # Convert gaussians to nodes (duplicate and reverse if enabled)
        for id, gaussian_params in gaussians.items():
            self.graph.add_node(id, mean=gaussian_params['mu'],
                                covariance=gaussian_params['sigma'],
                                direction=gaussian_params['direction'],
                                prior=gaussian_params['prior'])

            if reverse_gaussians:
                id_reversed = (*id, 'reversed')
                self.graph.add_node(id_reversed, mean=gaussian_params['mu'],
                                    covariance=gaussian_params['sigma'],
                                    direction=-gaussian_params['direction'],
                                    prior=gaussian_params['prior'])
                self.gaussian_reversal_map[id_reversed] = id

        # Connect nodes by weighted edges
        for id1 in self.graph.nodes:
            for id2 in self.graph.nodes:
                edge_weight = self.compute_edge_weight(self.graph.nodes[id1]['mean'],
                                                       self.graph.nodes[id1]['direction'],
                                                       self.graph.nodes[id2]['mean'])

                # add edge to graph
                if edge_weight is not None:
                    self.graph.add_edge(id1, id2, weight=edge_weight)

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

        if self.attractor_id is None:
            print("Attractor node not set. Cannot compute node-wise shortest path.")

        all_paths_to_target = nx.shortest_path(self.graph, target=self.attractor_id, weight='weight')
        del all_paths_to_target[self.attractor_id]

        # If two reversed nodes of each other both have a path to the target, remove the one with the longer path.
        nodes_to_remove = set()
        for node in all_paths_to_target.keys():

            if node in self.gaussian_reversal_map:
                original_node = self.gaussian_reversal_map[node]
                if original_node in all_paths_to_target:
                    if self.path_length(all_paths_to_target[node]) < self.path_length(all_paths_to_target[original_node]):
                        nodes_to_remove.add(original_node)
                    else:
                        nodes_to_remove.add(node)

        nodes_to_keep = list(set(all_paths_to_target.keys()) - nodes_to_remove)
        self.node_wise_shortest_path = nodes_to_keep

    def path_length(self, path):
        """Calculate the length of a path based on edge weights."""
        return sum(self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

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

    def get_gaussian(self, node_id):
        """Retrieves the mean, covariance, direction, and prior for one or more Gaussian nodes.

        Args:
            node_id: Node ID (single tuple) or iterable of node IDs (list/set of tuples).

        Returns:
            tuple:
                - If node_id is a single tuple ID: (mu, sigma, direction, prior) for that node.
                - If node_id is iterable: (mus, sigmas, directions, priors) as ndarrays.
        """
        # Accept only lists/sets as collections of node IDs
        if isinstance(node_id, (list, set)):
            mus, sigmas, directions, priors = [], [], [], []
            for nid in node_id:
                node = self.graph.nodes[nid]
                mus.append(node['mean'])
                sigmas.append(node['covariance'])
                directions.append(node['direction'])
                priors.append(node['prior'])
            return (np.array(mus), np.array(sigmas), np.array(directions), np.array(priors))
        else:
            node = self.graph.nodes[node_id]
            return node['mean'], node['covariance'], node['direction'], node['prior']

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
