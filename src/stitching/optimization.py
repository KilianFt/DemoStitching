import numpy as np
import cvxpy as cp
from src.util.benchmarking_tools import is_negative_definite


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

def find_lyapunov_function(As, n_dim, regularization=1e-4):
    """
    Find a Lyapunov function P that satisfies stability conditions for all A matrices.
    
    For 2D systems, P is parameterized as:
    P = [p11  p12]
        [p12  p22]
    
    We solve for p11, p12, p22 such that:
    - P is symmetric and negative definite
    - A^T @ P + P @ A is negative definite for all A in As
    
    Args:
        As: Array of A matrices (num_matrices x n_dim x n_dim)
        n_dim: Dimension of the system
        regularization: Regularization parameter for numerical stability
        
    Returns:
        P: Lyapunov matrix if found, None if infeasible
    """
    if n_dim == 2:
        return _find_lyapunov_2d_parametric(As, regularization)
    else:
        # Fall back to general approach for higher dimensions
        return _find_lyapunov_general(As, n_dim, regularization)

def _find_lyapunov_2d_parametric(As, regularization=1e-4):
    """
    Simplified parametric approach for 2D systems.
    P = [p11  p12]
        [p12  p22]
    """
    try:
        # Define optimization variables for the 3 unique elements
        p11 = cp.Variable()
        p12 = cp.Variable() 
        p22 = cp.Variable()
        
        # Construct P matrix
        P = cp.bmat([[p11, p12], [p12, p22]])
        
        constraints = []
        
        # Constraint 1: P is negative definite
        # For 2x2 matrix: det(P) > 0 and trace(P) < 0
        epsilon_p = regularization
        constraints.append(p11 + p22 + epsilon_p <= 0)  # trace(P) < -ε
        constraints.append(p11 * p22 - p12**2 - epsilon_p >= 0)  # det(P) > ε
        
        # Constraint 2: A^T @ P + P @ A is negative definite for all A
        epsilon_lyap = regularization
        for A in As:
            # Compute A^T @ P + P @ A symbolically
            lyap_matrix = A.T @ P + P @ A
            
            # For 2x2 matrix to be negative definite:
            # trace < 0 and det > 0
            lyap_trace = cp.trace(lyap_matrix)
            lyap_det = lyap_matrix[0,0] * lyap_matrix[1,1] - lyap_matrix[0,1] * lyap_matrix[1,0]
            
            constraints.append(lyap_trace + epsilon_lyap <= 0)
            constraints.append(lyap_det - epsilon_lyap >= 0)
        
        # Objective: Maximize -(p11 + p22) to encourage more negative eigenvalues
        objective = cp.Maximize(-(p11 + p22))
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        
        # Try different solvers
        solvers_to_try = [cp.MOSEK, cp.SCS, cp.CVXOPT]
        
        for solver in solvers_to_try:
            try:
                if solver == cp.MOSEK:
                    problem.solve(solver=solver, verbose=False)
                elif solver == cp.SCS:
                    problem.solve(solver=solver, verbose=False, max_iters=5000)
                else:
                    problem.solve(solver=solver, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    # Construct the solution matrix
                    P_opt = np.array([[p11.value, p12.value], 
                                     [p12.value, p22.value]])
                    
                    # Verify the solution
                    if _verify_lyapunov_solution(P_opt, As):
                        print(f"Found valid 2D parametric Lyapunov function using {solver}")
                        print(f"P = [[{p11.value:.4f}, {p12.value:.4f}], [{p12.value:.4f}, {p22.value:.4f}]]")
                        return P_opt
                    else:
                        print(f"Solution from {solver} failed verification")
                        
            except Exception as e:
                print(f"Solver {solver} failed: {e}")
                continue
        
        # If parametric approach failed, try simplified approach
        print("Parametric approach failed, trying simplified approach...")
        return _find_lyapunov_simplified(As, 2)
        
    except Exception as e:
        print(f"2D parametric approach failed: {e}")
        return None

def _verify_lyapunov_solution(P, As):
    """
    Verify that P satisfies all Lyapunov conditions.
    """
    # Check if P is negative definite
    if not is_negative_definite(P):
        return False
    
    # Check if P is symmetric
    if not np.allclose(P, P.T, atol=1e-6):
        return False
    
    # Check Lyapunov condition for all A matrices
    for A in As:
        lyap_matrix = A.T @ P + P @ A
        if not is_negative_definite(lyap_matrix):
            return False
    
    return True

def _find_lyapunov_general(As, n_dim, regularization=1e-4):
    """
    General LMI approach for higher dimensions (fallback).
    """
    try:
        # Define the optimization variable P (symmetric matrix)
        P = cp.Variable((n_dim, n_dim), symmetric=True)
        
        # Constraints
        constraints = []
        
        # Constraint 1: P is negative definite (P ≺ -ε I)
        epsilon_p = regularization
        constraints.append(P + epsilon_p * np.eye(n_dim) << 0)
        
        # Constraint 2: A^T @ P + P @ A is negative definite for all A
        epsilon_lyap = regularization
        for A in As:
            lyap_matrix = A.T @ P + P @ A
            constraints.append(lyap_matrix + epsilon_lyap * np.eye(n_dim) << 0)
        
        # Objective: Maximize trace of -P
        objective = cp.Maximize(cp.trace(-P))
        
        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        
        # Try different solvers
        solvers_to_try = [cp.MOSEK, cp.SCS, cp.CVXOPT]
        
        for solver in solvers_to_try:
            try:
                if solver == cp.MOSEK:
                    problem.solve(solver=solver, verbose=False)
                elif solver == cp.SCS:
                    problem.solve(solver=solver, verbose=False, max_iters=5000)
                else:
                    problem.solve(solver=solver, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    P_opt = P.value
                    
                    if P_opt is not None and _verify_lyapunov_solution(P_opt, As):
                        print(f"Found valid general Lyapunov function using {solver}")
                        return P_opt
                    
            except Exception as e:
                print(f"Solver {solver} failed: {e}")
                continue
        
        # If general approach failed, try simplified approach
        print("General approach failed, trying simplified approach...")
        return _find_lyapunov_simplified(As, n_dim)
        
    except Exception as e:
        print(f"General LMI solving failed: {e}")
        return None

def _find_lyapunov_simplified(As, n_dim):
    """
    Simplified approach: try to find P as a scaled identity matrix.
    """
    try:
        # Try P = -α * I for different values of α
        alphas = np.logspace(-3, 1, 50)  # From 0.001 to 10
        
        for alpha in alphas:
            P_candidate = -alpha * np.eye(n_dim)
            
            # Check if this P works for all A matrices
            valid_for_all = True
            for A in As:
                lyap_matrix = A.T @ P_candidate + P_candidate @ A
                if not is_negative_definite(lyap_matrix):
                    valid_for_all = False
                    break
            
            if valid_for_all:
                print(f"Found simplified Lyapunov function P = -{alpha:.4f} * I")
                return P_candidate
        
        print("Could not find valid Lyapunov function even with simplified approach")
        return None
        
    except Exception as e:
        print(f"Simplified approach failed: {e}")
        return None