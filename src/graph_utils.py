import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianGraph:
    """A directed graph where each node represents a Gaussian distribution (with mean, covariance, direction, and prior)
    and edges represent potential transitions between these Gaussians based on distance and directionality.
    """

    def __init__(self, param_dist=1, param_cos=1, bhattacharyya_threshold=0.1):
        """Initializes an empty GaussianGraph.

        Args:
            param_dist: Exponent for distance in edge weight calculation (default=1).
            param_cos: Exponent for directionality score in edge weight calculation (default=1).
            bhattacharyya_threshold: Threshold for pruning edges based on Bhattacharyya coefficient (default=0.1).
        """
        self.param_dist = param_dist
        self.param_cos = param_cos
        self.bhattacharyya_threshold = bhattacharyya_threshold

        self.graph = nx.DiGraph()
        self.gaussian_reversal_map = dict()  # keys = reversed node ids, values = original node ids

    def __str__(self):
        return f'GaussianGraph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.'

    def add_gaussians(self, gaussians, reverse_gaussians=False):
        """ Adds Gaussian nodes to the graph.

        args:
            gaussians: Dictionary mapping node IDs to Gaussian parameters (mean, covariance, direction, prior).
            reverse_gaussians: If True, adds nodes with reversed directions for each Gaussian.
        """

        # create nodes
        existing_nodes = set(self.graph.nodes)
        new_nodes = []
        for id, gaussian_params in gaussians.items():
            new_nodes.append(id)
            self.graph.add_node(id,
                                mean=gaussian_params['mu'],
                                covariance=gaussian_params['sigma'],
                                direction=gaussian_params['direction'],
                                prior=gaussian_params['prior'])

            if reverse_gaussians:
                id_reversed = (*id, 'reversed')
                new_nodes.append(id_reversed)
                self.graph.add_node(id_reversed,
                                    mean=gaussian_params['mu'],
                                    covariance=gaussian_params['sigma'],
                                    direction=-gaussian_params['direction'],
                                    prior=gaussian_params['prior'])
                self.gaussian_reversal_map[id_reversed] = id

        # create edges between new nodes and all existing nodes
        for i, id1 in enumerate(new_nodes):
            for id2 in existing_nodes.union(new_nodes[i+1:]):

                # forward edge
                edge_weight = self.compute_edge_weight(self.graph.nodes[id1]['mean'],
                                                       self.graph.nodes[id1]['direction'],
                                                       self.graph.nodes[id2]['mean'])
                if edge_weight is not None:
                    self.graph.add_edge(id1, id2, weight=edge_weight)

                # reverse edge
                edge_weight = self.compute_edge_weight(self.graph.nodes[id2]['mean'],
                                                       self.graph.nodes[id2]['direction'],
                                                       self.graph.nodes[id1]['mean'])
                if edge_weight is not None:
                    self.graph.add_edge(id2, id1, weight=edge_weight)

        # Prune edges
        #   It is necessary to check all edges again since new edges may have created new shortest paths, leading to
        #   existing edges being prunable.
        self.prune_edges()

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

        # directionality score (if no direction provided, score is based purely on distance)
        if direction1 is not None:
            direction_to = pos2 - pos1
            dir_score = np.dot(direction1, direction_to) / (np.linalg.norm(direction1) * np.linalg.norm(direction_to))
            if dir_score < 0:  # if node2 is behind node1 (by direction) then there is no edge
                return None
        else:
            dir_score = 1

        # edge weight (less weight = stronger connection)
        return d ** self.param_dist / dir_score ** self.param_cos

    def prune_edges(self):
        """ Prunes edges from the graph based on Bhattacharyya distance and shortest paths.
        """

        def bhattacharyya_distance(node1, node2):
            """Computes the Bhattacharyya distance between two Gaussian nodes.

            args:
                node1, node2: Node IDs of the two Gaussian nodes to compare.
            """

            mu1, sigma1 = self.graph.nodes[node1]['mean'], self.graph.nodes[node1]['covariance']
            mu2, sigma2 = self.graph.nodes[node2]['mean'], self.graph.nodes[node2]['covariance']

            sigma_avg = (sigma1 + sigma2) / 2
            mu_diff = mu1 - mu2
            try:
                inv_sigma_avg = np.linalg.inv(sigma_avg)
            except np.linalg.LinAlgError:
                return float('inf')  # If the average covariance is singular, treat distance as infinite

            sigma_avg_det = np.linalg.det(sigma_avg)
            sigma1_det = np.linalg.det(sigma1)
            sigma2_det = np.linalg.det(sigma2)

            DB = (1/8) * mu_diff.T @ inv_sigma_avg @ mu_diff + (1/2) * np.log(sigma_avg_det / np.sqrt(sigma1_det * sigma2_det))
            BC = np.exp(-DB)
            return BC

        # Prune edges if Bhattacharyya coefficient is below threshold
        for n1, n2 in list(self.graph.edges):

            bc = bhattacharyya_distance(n1, n2)

            if bc < self.bhattacharyya_threshold:
                self.graph.remove_edge(n1, n2)

        # Prune edges that are not part of any shortest path between their nodes
        edges_to_check = set(self.graph.edges)
        while edges_to_check:
            n1, n2 = edges_to_check.pop()

            try:
                shortest_path = nx.shortest_path(self.graph, source=n1, target=n2, weight='weight')
            except:
                continue  # no path exists, skip

            # Remove edges if not part of shortest path
            for i in range(len(shortest_path) - 2):
                for j in range(i + 2, len(shortest_path)):
                    edges_to_check -= {(shortest_path[i], shortest_path[j])}  # no need to check them later
                    if self.graph.has_edge(shortest_path[i], shortest_path[j]):
                        self.graph.remove_edge(shortest_path[i], shortest_path[j])

    def shortest_path(self, initial_state, target_state):
        """ Computes the shortest path from initial_state to target_state through the graph.

        Adds temporary nodes for the initial and target states, connects them to existing nodes, computes the shortest
        path, and then removes the temporary nodes before returning the path.

        args:
            initial_state: Starting point for the path.
            target_state: Ending point for the path.

        returns:
            List of node IDs representing the shortest path from initial_state to target_state, excluding the temporary
            nodes. Returns None if no path exists.
        """

        INIT = '__TEMP_INITIAL__'
        TARGET = '__TEMP_TARGET__'

        # Add temporary initial node and connect based on normal edge weight * Gaussian evaluation
        """
        self.graph.add_node(INIT, mean=initial_state)
        neighbor_scores = {
            node: multivariate_normal.pdf(initial_state,
                                         mean=self.graph.nodes[node]['mean'],
                                         cov=self.graph.nodes[node]['covariance'])
            for node in self.graph.nodes
        }
        neighbor = max(self.graph.nodes,
                       key=lambda node: multivariate_normal.pdf(initial_state,
                                                                mean=self.graph.nodes[node]['mean'],
                                                                cov=self.graph.nodes[node]['covariance'])
                       )
        self.graph.add_edge(INIT, neighbor)
        """

        self.graph.add_node(INIT, mean=initial_state)
        for node in self.graph.nodes():
            if node == INIT:
                continue

            edge_weight = self.compute_edge_weight(initial_state, self.graph.nodes[node]['direction'], self.graph.nodes[node]['mean'])
            gaussian_eval = multivariate_normal.pdf(initial_state, mean=self.graph.nodes[node]['mean'], cov=self.graph.nodes[node]['covariance'])
            edge_weight = edge_weight / max(gaussian_eval, 1e-6) if edge_weight is not None else None

            if edge_weight is not None:
                self.graph.add_edge(INIT, node, weight=edge_weight)

        # Add temporary target node
        self.graph.add_node(TARGET, mean=target_state)
        # Connect to all other nodes with edge weights based on distance and directionality
        for node in self.graph.nodes():
            if node == INIT or node == TARGET:
                continue

            edge_weight = self.compute_edge_weight(self.graph.nodes[node]['mean'], self.graph.nodes[node]['direction'], target_state)
            gaussian_eval = multivariate_normal.pdf(target_state, mean=self.graph.nodes[node]['mean'], cov=self.graph.nodes[node]['covariance'])
            edge_weight = edge_weight / max(gaussian_eval, 1e-6) if edge_weight is not None else None

            if edge_weight is not None:
                self.graph.add_edge(node, TARGET, weight=edge_weight)

        # Compute the shortest path from init to target
        try:
            path = nx.shortest_path(self.graph, source=INIT, target=TARGET, weight='weight')
        except:
            path = None

        # Remove temporary nodes
        self.graph.remove_node(INIT)
        self.graph.remove_node(TARGET)

        return path[1:-1] if path is not None else None  # exclude initial and target from returned path

    def shortest_path_tree(self, target_state):
        """Returns a list of node ids for which there exists a path to the target state.

        Adds a temporary target node, connects it to existing nodes, computes the shortest path length from each node
        to the target, and then removes the temporary node before returning the list of nodes that have a path to the
        target.

        args:
            target_state: Ending point for the paths.

        returns:
            List of node IDs that have a path to the target state, excluding the temporary target node
        """

        # Add temporary target node
        TARGET = '__TEMP_TARGET__'
        self.graph.add_node(TARGET, pos=target_state)
        for node in self.graph.nodes():
            if node == TARGET:
                continue

            edge_weight = self.compute_edge_weight(self.graph.nodes[node]['mean'], self.graph.nodes[node]['direction'], target_state)
            gaussian_eval = multivariate_normal.pdf(target_state, mean=self.graph.nodes[node]['mean'], cov=self.graph.nodes[node]['covariance'])
            edge_weight = edge_weight / gaussian_eval if edge_weight is not None else None
            if edge_weight is not None:
                self.graph.add_edge(node, TARGET, weight=edge_weight)

        # Compute the shortest path length from each node to target
        path_lengths = nx.shortest_path_length(self.graph, target=TARGET, weight='weight')
        del path_lengths[TARGET]

        # If two reversed nodes of each other both have a path to the target, remove the one with the longer path.
        nodes_to_remove = set()
        for node in path_lengths.keys():

            if node in self.gaussian_reversal_map:
                original_node = self.gaussian_reversal_map[node]
                if original_node in path_lengths:
                    if path_lengths[node] < path_lengths[original_node]:
                        nodes_to_remove.add(original_node)
                    else:
                        nodes_to_remove.add(node)
        remaining_nodes =  list(set(path_lengths.keys()) - nodes_to_remove)

        # Remove temporary target node and edges
        self.graph.remove_node(TARGET)

        return remaining_nodes

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

    def plot(self, ax=None, nodes=None):
        """Plots the basic gaussian graph.

        Intended to be used as a base for more complex plots that overlay additional info (e.g. DS vector field,
        trajectories, etc.)

        Args:
            ax: Optional matplotlib axis to plot on. If None, a new figure and axis will be created.
        """

        # Params
        node_size = 100
        node_color = 'teal'

        edge_color = 'black'
        min_edge_alpha = 0.3
        edge_start_space = 0.2
        edge_end_space = 0.2
        edge_width = 0.5
        arrow_head_size = 8

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Get positions for nodes
        nodes = nodes if nodes is not None else self.graph.nodes
        pos = {node: self.graph.nodes[node]['mean'] for node in nodes}
        edges = [e for e in self.graph.edges if e[0] in nodes and e[1] in nodes]



        # Extract edge weights and normalize for alpha values
        weights = [self.graph.edges[e]['weight'] for e in edges]
        weights_np = np.array(weights)
        alphas = np.minimum(1/weights_np + min_edge_alpha, 1)

        # Draw edges
        for (u, v), alpha in zip(edges, alphas):
            start_pos = pos[u]
            end_pos = pos[v]

            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx ** 2 + dy ** 2)

            if length > 0:
                # Normalize direction vector
                dx_norm = dx / length
                dy_norm = dy / length

                # Calculate adjusted start and end points based on buffer distances
                adjusted_start_x = start_pos[0] + dx_norm * edge_start_space
                adjusted_start_y = start_pos[1] + dy_norm * edge_start_space
                adjusted_end_x = end_pos[0] - dx_norm * edge_end_space
                adjusted_end_y = end_pos[1] - dy_norm * edge_end_space

                # Draw the line between adjusted points
                ax.plot([adjusted_start_x, adjusted_end_x], [adjusted_start_y, adjusted_end_y],
                        color=edge_color, alpha=alpha, linewidth=edge_width)

                # Draw arrow head at the end of the adjusted line
                # Position arrow slightly before the adjusted end point
                arrow_tail_x = adjusted_end_x - dx_norm * 0.05
                arrow_tail_y = adjusted_end_y - dy_norm * 0.05

                ax.annotate('', xy=(adjusted_end_x, adjusted_end_y),
                            xytext=(arrow_tail_x, arrow_tail_y),
                            arrowprops=dict(arrowstyle='->', color=edge_color,
                                            alpha=alpha, lw=edge_width,
                                            mutation_scale=arrow_head_size))

        # Draw nodes using scatter plot
        pos_np = np.array([p for p in pos.values()])
        ax.scatter(pos_np[:, 0], pos_np[:, 1], c=node_color, s=node_size)

        return ax

    def get_all_simple_paths(self, nr_edges):
        """Returns a list of all simple paths in the graph with a specified number of edges."""

        def get_simple_paths(curr_edges):

            # recursion termination
            if len(curr_edges) == nr_edges:
                return {curr_edges}

            out_edges = [e for e in self.graph.out_edges(curr_edges[-1][1]) if e not in curr_edges]
            new_paths = set()
            for e in out_edges:
                new_paths.update(
                    get_simple_paths(curr_edges + (e,))
                )

            return new_paths

        paths = set()
        for start_edge in self.graph.edges:
            paths.update(
                get_simple_paths((start_edge,))
            )

        return paths