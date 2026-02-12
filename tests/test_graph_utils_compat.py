import unittest

import numpy as np
import networkx as nx

from graph_utils import GaussianGraph
from src.stitching.graph_paths import shortest_path_nodes


class GaussianGraphCompatibilityTests(unittest.TestCase):
    def test_shortest_path_start_candidates_can_force_local_start(self):
        class _DummyGG:
            def __init__(self):
                self.graph = nx.DiGraph()
                self.near = ("near", 0)
                self.far = ("far", 0)
                self.near_mu = np.array([0.0, 0.0])
                self.far_mu = np.array([10.0, 0.0])
                self.target = np.array([20.0, 0.0])
                self.graph.add_node(self.near, mean=self.near_mu, direction=np.array([1.0, 0.0]))
                self.graph.add_node(self.far, mean=self.far_mu, direction=np.array([1.0, 0.0]))

            def compute_edge_weight(self, start, direction, end):
                start = np.asarray(start, dtype=float)
                end = np.asarray(end, dtype=float)
                if direction is None:
                    if np.allclose(end, self.near_mu):
                        return 0.1
                    if np.allclose(end, self.far_mu):
                        return 1.0
                    return None
                if np.allclose(start, self.near_mu) and np.allclose(end, self.target):
                    return 100.0
                if np.allclose(start, self.far_mu) and np.allclose(end, self.target):
                    return 0.0
                return None

        gg = _DummyGG()
        initial = np.array([-1.0, 0.0])
        target = gg.target.copy()

        path_all = shortest_path_nodes(
            gg=gg,
            initial_state=initial,
            target_state=target,
            start_node_candidates=None,
        )
        path_local = shortest_path_nodes(
            gg=gg,
            initial_state=initial,
            target_state=target,
            start_node_candidates=1,
        )

        self.assertEqual(path_all[0], gg.far)
        self.assertEqual(path_local[0], gg.near)

    def test_shortest_path_goal_candidates_can_force_local_goal(self):
        class _DummyGG:
            def __init__(self):
                self.graph = nx.DiGraph()
                self.s = ("s", 0)
                self.near_goal = ("near_goal", 0)
                self.far_goal = ("far_goal", 0)
                self.s_mu = np.array([0.0, 0.0])
                self.near_mu = np.array([1.0, 0.0])
                self.far_mu = np.array([2.0, 0.0])
                self.target = np.array([3.0, 0.0])
                self.graph.add_node(self.s, mean=self.s_mu, direction=np.array([1.0, 0.0]))
                self.graph.add_node(self.near_goal, mean=self.near_mu, direction=np.array([1.0, 0.0]))
                self.graph.add_node(self.far_goal, mean=self.far_mu, direction=np.array([1.0, 0.0]))
                self.graph.add_edge(self.s, self.near_goal, weight=1.0)
                self.graph.add_edge(self.s, self.far_goal, weight=1.0)

            def compute_edge_weight(self, start, direction, end):
                start = np.asarray(start, dtype=float)
                end = np.asarray(end, dtype=float)
                if direction is None:
                    if np.allclose(end, self.s_mu):
                        return 0.0
                    return None
                if np.allclose(start, self.near_mu) and np.allclose(end, self.target):
                    return 0.1
                if np.allclose(start, self.far_mu) and np.allclose(end, self.target):
                    return 10.0
                return None

        gg = _DummyGG()
        initial = np.array([-1.0, 0.0])
        target = gg.target.copy()

        path_all = shortest_path_nodes(
            gg=gg,
            initial_state=initial,
            target_state=target,
            start_node_candidates=1,
            goal_node_candidates=None,
        )
        path_local_goal = shortest_path_nodes(
            gg=gg,
            initial_state=initial,
            target_state=target,
            start_node_candidates=1,
            goal_node_candidates=1,
        )

        self.assertEqual(path_all[-1], gg.near_goal)
        self.assertEqual(path_local_goal[-1], gg.near_goal)

    def test_shortest_path_relaxes_candidates_when_local_pair_disconnected(self):
        class _DummyGG:
            def __init__(self):
                self.graph = nx.DiGraph()
                self.start_near = ("start_near", 0)
                self.start_far = ("start_far", 0)
                self.goal_near = ("goal_near", 0)
                self.goal_far = ("goal_far", 0)
                self.start_near_mu = np.array([0.0, 0.0])
                self.start_far_mu = np.array([10.0, 0.0])
                self.goal_near_mu = np.array([20.0, 0.0])
                self.goal_far_mu = np.array([30.0, 0.0])
                self.target = np.array([21.0, 0.0])
                self.graph.add_node(self.start_near, mean=self.start_near_mu, direction=np.array([1.0, 0.0]))
                self.graph.add_node(self.start_far, mean=self.start_far_mu, direction=np.array([1.0, 0.0]))
                self.graph.add_node(self.goal_near, mean=self.goal_near_mu, direction=np.array([1.0, 0.0]))
                self.graph.add_node(self.goal_far, mean=self.goal_far_mu, direction=np.array([1.0, 0.0]))
                # Only cross connections: strict local-local has no path.
                self.graph.add_edge(self.start_near, self.goal_far, weight=1.0)
                self.graph.add_edge(self.start_far, self.goal_near, weight=1.0)

            def compute_edge_weight(self, start, direction, end):
                start = np.asarray(start, dtype=float)
                end = np.asarray(end, dtype=float)
                if direction is None:
                    if np.allclose(end, self.start_near_mu):
                        return 0.1
                    if np.allclose(end, self.start_far_mu):
                        return 1.0
                    return None
                if np.allclose(start, self.goal_near_mu) and np.allclose(end, self.target):
                    return 0.1
                if np.allclose(start, self.goal_far_mu) and np.allclose(end, self.target):
                    return 1.0
                return None

        gg = _DummyGG()
        path_relaxed = shortest_path_nodes(
            gg=gg,
            initial_state=np.array([-1.0, 0.0]),
            target_state=gg.target.copy(),
            start_node_candidates=1,
            goal_node_candidates=1,
            allow_candidate_relaxation=True,
        )
        path_strict = shortest_path_nodes(
            gg=gg,
            initial_state=np.array([-1.0, 0.0]),
            target_state=gg.target.copy(),
            start_node_candidates=1,
            goal_node_candidates=1,
            allow_candidate_relaxation=False,
        )

        self.assertIsNone(path_strict)
        self.assertIsNotNone(path_relaxed)
        # Relaxation strategy keeps goal local first when possible.
        self.assertEqual(path_relaxed[-1], gg.goal_near)

    def test_shortest_path_with_incremental_graph_api(self):
        gaussians = {
            (0, 0): {
                "mu": np.array([0.0, 0.0]),
                "sigma": 0.02 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 0.5,
            },
            (1, 0): {
                "mu": np.array([1.0, 0.0]),
                "sigma": 0.02 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 0.5,
            },
        }
        gg = GaussianGraph(param_dist=1, param_cos=1)
        gg.add_gaussians(gaussians, reverse_gaussians=False)
        shortest_path = shortest_path_nodes(
            initial_state=np.array([-0.5, 0.0]),
            target_state=np.array([1.5, 0.0]),
            gg=gg,
        )

        self.assertIsNotNone(shortest_path)
        self.assertGreaterEqual(len(shortest_path), 1)
        self.assertTrue(any(node in shortest_path for node in [(0, 0), (1, 0)]))

    def test_shortest_path_tree_returns_reachable_nodes(self):
        gaussians = {
            (0, 0): {
                "mu": np.array([0.0, 0.0]),
                "sigma": 0.02 * np.eye(2),
                "direction": np.array([1.0, 0.0]),
                "prior": 1.0,
            },
        }
        gg = GaussianGraph(param_dist=1, param_cos=1)
        gg.add_gaussians(gaussians, reverse_gaussians=True)
        reachable = gg.shortest_path_tree(target_state=np.array([1.0, 0.0]))

        self.assertIsNotNone(reachable)
        self.assertGreaterEqual(len(reachable), 1)


if __name__ == "__main__":
    unittest.main()
