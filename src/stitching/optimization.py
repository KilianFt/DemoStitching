import numpy as np
import cvxpy as cp
from src.util.stitching import is_negative_definite


def compute_valid_A(A_init, P, x, x_att, x_dot, regularization=1e-3, max_iters=1000):
    """
    Compute a valid A matrix using convex optimization that satisfies:
    1. A + A^T is negative definite (stability)
    2. A^T P + P A is negative definite (Lyapunov stability)
    3. Maximizes log-likelihood of observed data
    
    Args:
        A_init: Initial A matrix (n x n)
        P: Lyapunov matrix (n x n, positive definite)
        x: Data points (m x n)
        x_att: Attractor point (n,)
        x_dot: Velocity data (m x n)
        regularization: Regularization parameter for numerical stability
        max_iters: Maximum optimization iterations
        
    Returns:
        A_opt: Optimized A matrix satisfying constraints
    """
    n = A_init.shape[0]  # Dimension
    m = x.shape[0]       # Number of data points
    
    # Ensure x_att is properly shaped
    if x_att.ndim == 1:
        x_att = x_att.reshape(1, -1)
    
    # Center the data
    x_centered = x - x_att  # (m x n)
    
    try:
        # Define optimization variable
        A = cp.Variable((n, n))
        
        # Compute predicted velocities
        x_dot_pred = (A @ x_centered.T).T  # (m x n)
        
        # Log-likelihood objective (negative squared error)
        # Maximize log-likelihood = minimize squared error
        residual = x_dot - x_dot_pred
        log_likelihood = -cp.sum(cp.sum(cp.square(residual), axis=1))
        
        # Add regularization term to keep A close to initial guess
        regularization_term = -regularization * cp.sum(cp.square(A - A_init))
        
        # Combined objective
        objective = cp.Maximize(log_likelihood + regularization_term)
        
        # Constraints
        constraints = []
        
        # Constraint 1: A + A^T ≺ -ε I (negative definite)
        epsilon1 = 1e-4
        constraints.append(A + A.T + epsilon1 * np.eye(n) << 0)
        
        # Constraint 2: A^T P + P A ≺ -ε I (Lyapunov stability)
        epsilon2 = 1e-4
        constraints.append(A.T @ P + P @ A + epsilon2 * np.eye(n) << 0)
        
        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        
        # Solve with different solvers if needed
        solvers_to_try = [cp.MOSEK, cp.SCS, cp.CVXOPT]
        
        for solver in solvers_to_try:
            try:
                if solver == cp.MOSEK:
                    problem.solve(solver=solver, verbose=False)
                elif solver == cp.SCS:
                    problem.solve(solver=solver, verbose=False, max_iters=max_iters)
                else:
                    problem.solve(solver=solver, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    break
                    
            except Exception as e:
                print(f"Solver {solver} failed: {e}")
                continue
        
        if problem.status != cp.OPTIMAL:
            print(f"Optimization failed with status: {problem.status}")
            print("Falling back to regularized least squares solution...")
            return _fallback_solution(A_init, P, x_centered, x_dot, regularization)
        
        A_opt = A.value
        
        # Verify constraints are satisfied
        valid_A = is_negative_definite(A_opt + A_opt.T)
        valid_wrt_p = is_negative_definite(A_opt.T @ P + P @ A_opt)
        
        if not (valid_A and valid_wrt_p):
            print("Warning: Optimized A does not satisfy constraints. Using fallback.")
            return _fallback_solution(A_init, P, x_centered, x_dot, regularization)
        
        print(f"Optimization successful. Objective value: {problem.value:.4f}")
        return A_opt
        
    except Exception as e:
        print(f"Optimization error: {e}")
        print("Using fallback solution...")
        return _fallback_solution(A_init, P, x_centered, x_dot, regularization)


def _fallback_solution(A_init, P, x_centered, x_dot, regularization):
    """
    Fallback solution using regularized least squares with projection onto feasible set.
    """
    n = A_init.shape[0]
    
    try:
        # Regularized least squares solution
        # min ||x_dot - A @ x_centered.T||^2 + λ||A - A_init||^2
        
        # Vectorize the problem
        X = x_centered  # (m x n)
        Y = x_dot       # (m x n)
        
        # For each column of A
        A_opt = np.zeros_like(A_init)
        
        for i in range(n):
            # Solve for i-th column of A
            # Y[:, i] = X @ A[:, i]
            XTX = X.T @ X + regularization * np.eye(n)
            XTY = X.T @ Y[:, i]
            A_opt[:, i] = np.linalg.solve(XTX, XTY)
        
        # Project onto feasible set using eigenvalue modification
        A_opt = _project_to_stable(A_opt, P)
        
        return A_opt
        
    except Exception as e:
        print(f"Fallback solution failed: {e}")
        # Return a simple stable matrix
        return -0.5 * np.eye(n)


def _project_to_stable(A, P, min_eigenvalue=-1e-2):
    """
    Project matrix A to satisfy stability constraints using eigenvalue modification.
    """
    n = A.shape[0]
    
    # Make A + A^T negative definite
    S = A + A.T
    eigvals, eigvecs = np.linalg.eigh(S)
    
    # Modify eigenvalues to be negative
    eigvals_new = np.minimum(eigvals, min_eigenvalue)
    S_new = eigvecs @ np.diag(eigvals_new) @ eigvecs.T
    
    # Reconstruct A (assuming A is approximately symmetric for simplicity)
    A_proj = 0.5 * S_new
    
    # Check Lyapunov constraint and adjust if needed
    L = A_proj.T @ P + P @ A_proj
    eigvals_L, eigvecs_L = np.linalg.eigh(L)
    
    if np.max(eigvals_L) >= 0:
        # Scale A to satisfy Lyapunov constraint
        scale_factor = min_eigenvalue / (np.max(eigvals_L) + 1e-6)
        A_proj *= scale_factor
    
    return A_proj