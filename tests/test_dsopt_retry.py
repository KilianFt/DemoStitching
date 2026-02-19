import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from src.dsopt.dsopt_class import dsopt_class


class _DummyProblem:
    def __init__(self, fail_times=0, status_after_success="optimal"):
        self.fail_times = int(fail_times)
        self.status_after_success = status_after_success
        self.calls = 0
        self.status = None

    def solve(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("solver failed")
        self.status = self.status_after_success


class DSOptRetryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(20, 2))
        self.x_dot = -0.8 * self.x + 0.05 * rng.normal(size=(20, 2))
        self.x_att = np.zeros((1, 2), dtype=float)

        gamma = rng.uniform(low=0.1, high=1.0, size=(3, self.x.shape[0]))
        gamma /= np.sum(gamma, axis=0, keepdims=True)
        self.gamma = gamma
        self.assignment_arr = np.argmax(gamma, axis=0)

    def test_solve_with_retries_succeeds_after_failures(self):
        ds = dsopt_class(self.x, self.x_dot, self.x_att, self.gamma, self.assignment_arr)
        ds.max_solve_retries = 5
        prob = _DummyProblem(fail_times=2, status_after_success="optimal")
        ok = ds._solve_with_retries(prob, "test")
        self.assertTrue(ok)
        self.assertEqual(prob.calls, 3)

    def test_solve_with_retries_returns_false_when_exhausted(self):
        ds = dsopt_class(self.x, self.x_dot, self.x_att, self.gamma, self.assignment_arr)
        ds.max_solve_retries = 3
        prob = _DummyProblem(fail_times=10, status_after_success="optimal")
        ok = ds._solve_with_retries(prob, "test")
        self.assertFalse(ok)
        self.assertEqual(prob.calls, 3)

    def test_optimize_p_raises_when_retries_exhausted(self):
        ds = dsopt_class(self.x, self.x_dot, self.x_att, self.gamma, self.assignment_arr)
        with patch.object(ds, "_solve_with_retries", return_value=False):
            with self.assertRaises(RuntimeError):
                ds._optimize_P()

    def test_optimize_a_raises_when_retries_exhausted(self):
        ds = dsopt_class(self.x, self.x_dot, self.x_att, self.gamma, self.assignment_arr)
        ds.P = np.eye(2)
        with patch.object(ds, "_solve_with_retries", return_value=False):
            with self.assertRaises(RuntimeError):
                ds._optimize_A()


if __name__ == "__main__":
    unittest.main()
