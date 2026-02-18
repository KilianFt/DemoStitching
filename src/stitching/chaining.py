from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal
from src.damm.src.damm_class import DAMM as damm_class
import cvxpy as cp
from configs import ChainConfig, StitchConfig
from src.lpvds_class import lpvds_class



def _lyapunov_constraint_residual(
    A: np.ndarray,
    P: np.ndarray,
    stabilization_margin: float,
) -> float:
    """Return max-eigenvalue residual of A^T P + P A + eps I <= 0 (<=0 means feasible)."""
    eps = abs(float(stabilization_margin))
    dim = A.shape[0]
    lmi = A.T @ P + P @ A + eps * np.eye(dim)
    lmi = 0.5 * (lmi + lmi.T)
    try:
        return float(np.max(np.linalg.eigvalsh(lmi)))
    except np.linalg.LinAlgError:
        return np.inf


def _fit_linear_system(
    x: np.ndarray,
    x_dot: np.ndarray,
    target: np.ndarray,
    stabilization_margin: float = 1e-3,
    lyapunov_P: Optional[np.ndarray] = None,
    lmi_tolerance: float = 5e-5,
) -> Optional[np.ndarray]:
    """LPV-DS-like fit: constrained MSE with Lyapunov stability."""
    if x.shape[0] < x.shape[1]:
        return None

    X = x - target.reshape(1, -1)
    Y = x_dot

    dim = X.shape[1]
    if lyapunov_P is None:
        P = np.eye(dim)
    else:
        P = np.asarray(lyapunov_P, dtype=float)
        if P.shape != (dim, dim) or not np.all(np.isfinite(P)):
            P = np.eye(dim)
        else:
            P = 0.5 * (P + P.T)

    # Match lpvds objective shape: min ||A X - Y||_F with Lyapunov constraint.
    A_var = cp.Variable((dim, dim))
    eps = abs(float(stabilization_margin))
    objective = cp.sum_squares(A_var @ X.T - Y.T)
    constraints = [A_var.T @ P + P @ A_var << -eps * np.eye(dim)]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        problem.solve(solver=cp.SCS, warm_start=True, verbose=False)
    except Exception:
        try:
            problem.solve(warm_start=True, verbose=False)
        except Exception:
            return None

    if A_var.value is None:
        return None
    A_opt = np.asarray(A_var.value, dtype=float)
    if not np.all(np.isfinite(A_opt)):
        return None
    residual = _lyapunov_constraint_residual(A_opt, P=P, stabilization_margin=eps)
    if not np.isfinite(residual) or residual > float(lmi_tolerance):
        return None
    return A_opt


class ChainedLinearDS:
    """DS chain with state-triggered entry and time-triggered transition completion."""

    def __init__(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        attractor: np.ndarray,
        path_nodes,
        node_sources: np.ndarray,
        node_targets: np.ndarray,
        A_seq: np.ndarray,
        damm,
        transition_centers: np.ndarray,
        transition_normals: np.ndarray,
        transition_ratio_nodes: np.ndarray,
        transition_edge_ratios: np.ndarray,
        transition_times: np.ndarray,
        transition_distances: np.ndarray,
        chain_cfg: ChainConfig,
    ) -> None:
        self.x = x
        self.x_dot = x_dot
        self.x_att = np.asarray(attractor, dtype=float)

        # One linear DS per subsystem window.
        self.path_nodes = list(path_nodes)
        self.node_sources = np.asarray(node_sources, dtype=float)  # (S, d)
        self.node_targets = np.asarray(node_targets, dtype=float)  # (S, d)
        self.A_seq = np.asarray(A_seq, dtype=float)  # (N, d, d)
        self.n_systems = self.A_seq.shape[0]
        if self.n_systems > 0:
            self.state_sequence = np.vstack([self.node_sources, self.node_targets[-1]])
        else:
            self.state_sequence = np.zeros((0, x.shape[1]), dtype=float)

        # Transition geometry between consecutive subsystems.
        # The transition from subsystem i to i+1 is triggered when
        # dot(x - transition_centers[i], transition_normals[i]) >= 0.
        self.transition_centers = np.asarray(transition_centers, dtype=float)
        self.transition_normals = np.asarray(transition_normals, dtype=float)
        self.transition_ratio_nodes = np.asarray(transition_ratio_nodes, dtype=float)
        self.transition_edge_ratios = np.asarray(transition_edge_ratios, dtype=float)
        self.transition_plane_points = self.transition_centers
        self.transition_plane_normals = self.transition_normals
        self.transition_times = np.asarray(transition_times, dtype=float)
        self.transition_distances = np.asarray(transition_distances, dtype=float)
        self.chain_cfg = chain_cfg
        self.transition_trigger_method = chain_cfg.transition_trigger_method

        # Compatibility attributes used by existing plotting/eval code.
        self.damm = damm
        self.K = self.damm.K
        self.node_means = self.node_sources
        if self.A_seq.shape[0] == self.K + 1:
            # Legacy layout: first DS was initial->first_gaussian.
            self.A = self.A_seq[1:].copy()
        else:
            # Current layout: one DS per path gaussian edge (+ final gaussian->goal edge).
            self.A = self.A_seq[:self.K].copy()

        try:
            gamma = np.asarray(self.damm.compute_gamma(self.x), dtype=float)
            if gamma.ndim == 2 and gamma.shape[1] == self.x.shape[0]:
                self.assignment_arr = np.argmax(gamma, axis=0).astype(int)
            else:
                self.assignment_arr = np.zeros(self.x.shape[0], dtype=int)
        except Exception:
            self.assignment_arr = np.zeros(self.x.shape[0], dtype=int)

        self.tol = 10e-3
        self.max_iter = 10000
        self.last_sim_indices = None

        self._runtime_idx = 0
        self._runtime_time = 0.0
        self._state_entry_t = 0.0
        self._transition_from_idx = None
        self._transition_t0 = None

    def _state_vec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return x[0]
        return x

    def _velocity_for_index(self, x: np.ndarray, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.n_systems - 1))
        return self.A_seq[idx] @ (x - self.node_targets[idx])

    def _trigger_state_mean_normals(self, idx: int, x: np.ndarray) -> bool:
        if idx >= len(self.transition_centers) or idx >= len(self.transition_normals):
            return False
        signed_distance = float(
            np.dot(
                np.asarray(x, dtype=float) - self.transition_centers[idx],
                self.transition_normals[idx],
            )
        )
        return signed_distance >= 0.0

    def _trigger_state_distance_ratio(self, idx: int, x: np.ndarray) -> bool:
        if (
            idx >= len(self.transition_centers)
            or idx >= len(self.transition_ratio_nodes)
            or idx >= len(self.transition_edge_ratios)
        ):
            return False
        n1 = self.transition_centers[idx]
        n2 = self.transition_ratio_nodes[idx]
        ratio_threshold = float(self.transition_edge_ratios[idx])
        if not np.isfinite(ratio_threshold):
            return False

        x = np.asarray(x, dtype=float)
        d1 = float(np.linalg.norm(x - n1))
        d2 = float(np.linalg.norm(x - n2))
        ratio = d1 / max(d2, 1e-12)
        return ratio >= ratio_threshold

    def trigger_state(self, idx: int, x: np.ndarray) -> bool:
        idx = int(idx)
        if idx < 0 or idx >= self.n_systems:
            return False
        if self.transition_trigger_method == "distance_ratio":
            return self._trigger_state_distance_ratio(idx, x)
        elif self.transition_trigger_method == "mean_normals":
            return self._trigger_state_mean_normals(idx, x)
        else:
            raise ValueError(f"Unknown transition trigger method: {self.transition_trigger_method}")

    def trigger_time(self, idx: int, t: float) -> bool:
        if idx >= len(self.transition_times):
            return True
        if self._transition_t0 is None:
            return False
        return (t - self._transition_t0) >= self.transition_times[idx]

    def select_node_index(self, x: np.ndarray, current_idx: int) -> int:
        x = self._state_vec(x)
        if not self.chain_cfg.enable_recovery:
            return int(current_idx)

        current_idx = int(np.clip(current_idx, 0, self.n_systems - 1))
        dist_to_current = np.linalg.norm(x - self.node_sources[current_idx])
        if dist_to_current <= self.chain_cfg.recovery_distance:
            return current_idx

        distances = np.linalg.norm(self.node_sources - x.reshape(1, -1), axis=1)
        return int(np.argmin(distances))

    def reset_runtime(self, initial_idx: int = 0, start_time: float = 0.0):
        self._runtime_idx = int(np.clip(initial_idx, 0, self.n_systems - 1))
        self._runtime_time = float(start_time)
        self._state_entry_t = float(start_time)
        self._transition_from_idx = None
        self._transition_t0 = None

    @property
    def transition_active(self) -> bool:
        return self._transition_from_idx is not None

    def _clear_transition(self):
        self._transition_from_idx = None
        self._transition_t0 = None

    def _start_transition_if_triggered(self, x: np.ndarray, t: float):
        if self._runtime_idx >= self.n_systems - 1:
            return
        if self._transition_from_idx is not None:
            return
        trigger = self.trigger_state(self._runtime_idx, x)

        if trigger:
            self._transition_from_idx = self._runtime_idx
            self._transition_t0 = t

    def _transition_velocity(self, x: np.ndarray, t: float) -> np.ndarray:
        idx = int(self._transition_from_idx)
        v_current = self._velocity_for_index(x, idx)
        v_next = self._velocity_for_index(x, idx + 1)

        T = self.transition_times[idx] if idx < len(self.transition_times) else 0.0
        if T <= 1e-12:
            alpha = 1.0
        else:
            alpha = np.clip((t - self._transition_t0) / T, 0.0, 1.0)
        v = (1.0 - alpha) * v_current + alpha * v_next

        if self.trigger_time(idx, t):
            self._runtime_idx = min(idx + 1, self.n_systems - 1)
            self._state_entry_t = t
            self._clear_transition()
        return v

    def step_once(self, x: np.ndarray, dt: float, current_idx: Optional[int] = None, current_time: Optional[float] = None):
        x = self._state_vec(x)
        t = self._runtime_time if current_time is None else float(current_time)

        if current_idx is not None and int(current_idx) != self._runtime_idx:
            self._runtime_idx = int(np.clip(current_idx, 0, self.n_systems - 1))
            self._state_entry_t = t
            self._clear_transition()

        # Disturbance recovery: jump to nearest source state.
        recovered_idx = self.select_node_index(x, current_idx=self._runtime_idx)
        if recovered_idx != self._runtime_idx:
            self._runtime_idx = recovered_idx
            self._state_entry_t = t
            self._clear_transition()

        self._start_transition_if_triggered(x, t)
        if self._transition_from_idx is None:
            velocity = self._velocity_for_index(x, self._runtime_idx)
        else:
            velocity = self._transition_velocity(x, t)

        x_next = x + dt * velocity
        if current_time is None:
            self._runtime_time += dt
        return x_next, velocity, self._runtime_idx

    def sim(self, x_init: np.ndarray, dt: float):
        x = self._state_vec(x_init)
        init_idx = int(np.argmin(np.linalg.norm(self.node_sources - x.reshape(1, -1), axis=1)))
        self.reset_runtime(initial_idx=init_idx, start_time=0.0)

        trajectory = [x.copy()]
        gamma_history = []
        index_history = [self._runtime_idx]

        for _ in range(self.max_iter):
            if np.linalg.norm(x - self.x_att) < self.tol:
                break

            x, _, idx = self.step_once(x, dt=dt)
            trajectory.append(x.copy())
            index_history.append(idx)
            gamma_history.append(self.damm.compute_gamma(x.reshape(1, -1))[:, 0])

        self.last_sim_indices = np.array(index_history, dtype=int)
        return np.vstack(trajectory), np.array(gamma_history)

    def predict_velocities(self, x_positions: np.ndarray) -> np.ndarray:
        """Stateless proxy for metrics/visualization."""
        x_positions = np.atleast_2d(x_positions)
        velocities = []
        for x in x_positions:
            idx = int(np.argmin(np.linalg.norm(self.node_sources - x.reshape(1, -1), axis=1)))
            if idx < self.n_systems - 1 and self.trigger_state(idx, x):
                v = 0.5 * self._velocity_for_index(x, idx) + 0.5 * self._velocity_for_index(x, idx + 1)
            else:
                v = self._velocity_for_index(x, idx)
            velocities.append(v)
        return np.vstack(velocities)

    def vector_field(self, x_positions: np.ndarray) -> np.ndarray:
        return self.predict_velocities(x_positions)

class ChainedSegmentedDS:
    """DS chain with state-triggered entry and time-triggered transition completion."""

    def __init__(self, x, x_dot, ds_set, gg, path_nodes, initial, attractor, intermediate_segments, intermediate_DSs, A_init,
                 A_attractor, blend_length_ratio) -> None:
        self.x = x
        self.x_dot = x_dot
        self.ds_set = ds_set
        self.gg = gg
        self.path_nodes = path_nodes
        self.initial = initial
        self.attractor = attractor
        self.intermediate_segments = intermediate_segments
        self.intermediate_DSs = intermediate_DSs
        self.A_init = A_init
        self.A_attractor = A_attractor
        self.blend_length_ratio = blend_length_ratio

        self.tol = 10e-3
        self.max_iter = 10000
        self.last_sim_indices = None

        # ---- Nominal DSs (with dimension shaping to get [1,d] shape)
        self.nominal_DSs = []
        self.nominal_DSs.append(lambda x: np.expand_dims(self.A_init @ (x.squeeze() - self._get_gaussian_center(self.path_nodes[0])), axis=0))
        for ds in self.intermediate_DSs:
            self.nominal_DSs.append(lambda x: ds._step(x, dt=1)[2].T)
        self.nominal_DSs.append(lambda x: np.expand_dims(self.A_attractor @ (x.squeeze() - self.attractor), axis=0))

        # ---- Transition points
        self.transition_points = []
        self.transition_points.append((self.initial,
                                       (self.initial + self._get_gaussian_center(self.path_nodes[0]))/2,
                                       self._get_gaussian_center(self.path_nodes[0])))
        for seg in self.intermediate_segments:
            self.transition_points.append((self._get_gaussian_center(seg[0]),
                                           self._get_gaussian_center(seg[1]),
                                           self._get_gaussian_center(seg[2])))
        self.transition_points.append((self._get_gaussian_center(self.path_nodes[0]),
                                       (self._get_gaussian_center(self.path_nodes[0]) + self.attractor)/2,
                                       self.attractor))

        # Simulation variables
        self.current_state = None
        self.state_time = None
        self.curr_trigger = None
        self.curr_timer = None
        self.curr_transitions_DS = None


    def _state_vec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return x[0]
        return x

    def _velocity_for_index(self, x: np.ndarray, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.n_systems - 1))
        return self.A_seq[idx] @ (x - self.node_targets[idx])

    def _trigger_state_mean_normals(self, idx: int, x: np.ndarray) -> bool:
        if idx >= len(self.transition_centers) or idx >= len(self.transition_normals):
            return False
        signed_distance = float(
            np.dot(
                np.asarray(x, dtype=float) - self.transition_centers[idx],
                self.transition_normals[idx],
            )
        )
        return signed_distance >= 0.0

    def _trigger_state_distance_ratio(self, idx: int, x: np.ndarray) -> bool:
        if (
            idx >= len(self.transition_centers)
            or idx >= len(self.transition_ratio_nodes)
            or idx >= len(self.transition_edge_ratios)
        ):
            return False
        n1 = self.transition_centers[idx]
        n2 = self.transition_ratio_nodes[idx]
        ratio_threshold = float(self.transition_edge_ratios[idx])
        if not np.isfinite(ratio_threshold):
            return False

        x = np.asarray(x, dtype=float)
        d1 = float(np.linalg.norm(x - n1))
        d2 = float(np.linalg.norm(x - n2))
        ratio = d1 / max(d2, 1e-12)
        return ratio >= ratio_threshold

    def trigger_state(self, idx: int, x: np.ndarray) -> bool:
        idx = int(idx)
        if idx < 0 or idx >= self.n_systems:
            return False
        if self.transition_trigger_method == "distance_ratio":
            return self._trigger_state_distance_ratio(idx, x)
        elif self.transition_trigger_method == "mean_normals":
            return self._trigger_state_mean_normals(idx, x)
        else:
            raise ValueError(f"Unknown transition trigger method: {self.transition_trigger_method}")

    def trigger_time(self, idx: int, t: float) -> bool:
        if idx >= len(self.transition_times):
            return True
        if self._transition_t0 is None:
            return False
        return (t - self._transition_t0) >= self.transition_times[idx]

    def select_node_index(self, x: np.ndarray, current_idx: int) -> int:
        x = self._state_vec(x)
        if not self.chain_cfg.enable_recovery:
            return int(current_idx)

        current_idx = int(np.clip(current_idx, 0, self.n_systems - 1))
        dist_to_current = np.linalg.norm(x - self.node_sources[current_idx])
        if dist_to_current <= self.chain_cfg.recovery_distance:
            return current_idx

        distances = np.linalg.norm(self.node_sources - x.reshape(1, -1), axis=1)
        return int(np.argmin(distances))

    def reset_runtime(self, initial_idx: int = 0, start_time: float = 0.0):
        self._runtime_idx = int(np.clip(initial_idx, 0, self.n_systems - 1))
        self._runtime_time = float(start_time)
        self._state_entry_t = float(start_time)
        self._transition_from_idx = None
        self._transition_t0 = None

    @property
    def transition_active(self) -> bool:
        return self._transition_from_idx is not None

    def _clear_transition(self):
        self._transition_from_idx = None
        self._transition_t0 = None

    def _start_transition_if_triggered(self, x: np.ndarray, t: float):
        if self._runtime_idx >= self.n_systems - 1:
            return
        if self._transition_from_idx is not None:
            return
        trigger = self.trigger_state(self._runtime_idx, x)

        if trigger:
            self._transition_from_idx = self._runtime_idx
            self._transition_t0 = t

    def _transition_velocity(self, x: np.ndarray, t: float) -> np.ndarray:
        idx = int(self._transition_from_idx)
        v_current = self._velocity_for_index(x, idx)
        v_next = self._velocity_for_index(x, idx + 1)

        T = self.transition_times[idx] if idx < len(self.transition_times) else 0.0
        if T <= 1e-12:
            alpha = 1.0
        else:
            alpha = np.clip((t - self._transition_t0) / T, 0.0, 1.0)
        v = (1.0 - alpha) * v_current + alpha * v_next

        if self.trigger_time(idx, t):
            self._runtime_idx = min(idx + 1, self.n_systems - 1)
            self._state_entry_t = t
            self._clear_transition()
        return v

    def _get_gaussian_center(self, node_id):
        node_data = self.gg.graph.nodes[node_id]
        if "mean" in node_data:
            return np.asarray(node_data["mean"], dtype=float)
        return None

    def _get_trigger(self):

        # If last DS, trigger is always false (no transition).
        if self.current_state[1] >= len(self.nominal_DSs):
            return lambda x: False


        start, mid, end = self.transition_points[self.current_state[1]]
        if mid is None:
            mid = (start + end) / 2

        e1 = mid - start
        e2 = end - mid

        transition_ratio = np.linalg.norm(e1) / np.linalg.norm(e2)
        trigger = lambda x: np.linalg.norm(x - start) / np.linalg.norm(x - end) >= transition_ratio
        return trigger

    def _get_timer(self):

        start, mid, end = self.transition_points[self.current_state[1]]
        e2 = end - mid

        ds1 = self.nominal_DSs[self.current_state[1]]
        ds2 = self.nominal_DSs[self.current_state[1] + 1]

        mid_x_dot = ds1(mid)
        T = self.blend_length_ratio * np.linalg.norm(e2) / np.linalg.norm(mid_x_dot)

        timer = lambda t: t - self.state_time >= T
        transition_DS = lambda x, t: ds1(x) * (t-self.state_time)/T + ds2(x) * (1 - (t-self.state_time)/T)

        return timer, transition_DS

    def step_once(self, x: np.ndarray, dt: float):

        # Update state
        if self.current_state[0] == 'nominal':
            trigger_eval = self.curr_trigger(x)
            if trigger_eval:
                self.current_state = ('intermediate', self.current_state[1])
                self.curr_timer, self.curr_transitions_DS = self._get_timer()

        elif self.current_state[0] == 'intermediate':
            timer_eval = self.curr_timer(self.state_time)
            if timer_eval:
                self.current_state = ('nominal', self.current_state[1] + 1)
                self.curr_trigger = self._get_trigger()


        # Get x_dot
        if self.current_state[0] == 'nominal':
            x_dot = self.nominal_DSs[self.current_state[1]](x)
        elif self.current_state[0] == 'intermediate':
            x_dot = self.curr_transitions_DS(x, self.state_time)

        # Update variables
        x_next = x + dt * x_dot
        self.state_time += dt

        return x_next

    def sim(self, x_init: np.ndarray, dt: float):

        # Init
        x = np.asarray(x_init, dtype=float)

        # Reset simulation variables
        self.current_state = ('nominal', 0)
        self.state_time = 0  # time since entering current state
        self.curr_trigger = self._get_trigger()
        self.curr_timer = None
        self.curr_transitions_DS = None

        trajectory = [x.copy()]
        gamma_history = []  # TODO sort this out for the segmented case

        for _ in range(self.max_iter):
            if np.linalg.norm(x - self.attractor) < self.tol:
                break

            x = self.step_once(x, dt=dt)
            trajectory.append(x.copy())

        return np.vstack(trajectory), np.array(gamma_history)

    def predict_velocities(self, x_positions: np.ndarray) -> np.ndarray:
        """Stateless proxy for metrics/visualization."""
        x_positions = np.atleast_2d(x_positions)
        velocities = []
        for x in x_positions:
            idx = int(np.argmin(np.linalg.norm(self.node_sources - x.reshape(1, -1), axis=1)))
            if idx < self.n_systems - 1 and self.trigger_state(idx, x):
                v = 0.5 * self._velocity_for_index(x, idx) + 0.5 * self._velocity_for_index(x, idx + 1)
            else:
                v = self._velocity_for_index(x, idx)
            velocities.append(v)
        return np.vstack(velocities)

    def vector_field(self, x_positions: np.ndarray) -> np.ndarray:
        return self.predict_velocities(x_positions)


def _get_source_data_for_node(ds_set, gg, node_id):
    ds_idx = node_id[0]
    gaussian_idx = node_id[1]
    reverse_sign = -1.0 if node_id in gg.gaussian_reversal_map else 1.0
    assigned_x = ds_set[ds_idx].x[ds_set[ds_idx].assignment_arr == gaussian_idx]
    assigned_x_dot = ds_set[ds_idx].x_dot[ds_set[ds_idx].assignment_arr == gaussian_idx] * reverse_sign
    base_A = ds_set[ds_idx].A[gaussian_idx] * reverse_sign
    lyapunov_P = None
    if hasattr(ds_set[ds_idx], "ds_opt") and hasattr(ds_set[ds_idx].ds_opt, "P"):
        lyapunov_P = ds_set[ds_idx].ds_opt.P
    return assigned_x, assigned_x_dot, base_A, lyapunov_P


def _graph_gaussian_ids(gg):
    return [
        node_id
        for node_id, node_data in gg.graph.nodes(data=True)
        if "mean" in node_data and "covariance" in node_data and "prior" in node_data
    ]


def _stack_xy(x_parts, x_dot_parts):
    valid = [
        (np.asarray(xi, dtype=float), np.asarray(vi, dtype=float))
        for xi, vi in zip(x_parts, x_dot_parts)
        if xi is not None and vi is not None and len(xi) > 0 and len(vi) > 0
    ]
    if len(valid) == 0:
        return np.zeros((0, 0), dtype=float), np.zeros((0, 0), dtype=float)
    x = np.vstack([v[0] for v in valid])
    x_dot = np.vstack([v[1] for v in valid])
    return x, x_dot


def _direction_consistency_stats(
    A: np.ndarray,
    fit_x: np.ndarray,
    source_state: np.ndarray,
    target_state: np.ndarray,
):
    fit_x = np.asarray(fit_x, dtype=float)
    n_points = int(fit_x.shape[0]) if fit_x.ndim == 2 else 0
    if n_points == 0:
        return {"n_points": 0, "frac_forward": np.nan, "min_proj": np.nan, "mean_proj": np.nan}

    edge = np.asarray(target_state, dtype=float) - np.asarray(source_state, dtype=float)
    edge_len = np.linalg.norm(edge)
    if edge_len <= 1e-12:
        return {"n_points": n_points, "frac_forward": np.nan, "min_proj": np.nan, "mean_proj": np.nan}

    edge_dir = edge / edge_len
    target_state = np.asarray(target_state, dtype=float)
    pred_vel = (np.asarray(A, dtype=float) @ (fit_x - target_state.reshape(1, -1)).T).T
    proj = pred_vel @ edge_dir
    return {
        "n_points": n_points,
        "frac_forward": float(np.mean(proj > 0.0)),
        "min_proj": float(np.min(proj)),
        "mean_proj": float(np.mean(proj)),
    }

def _fit_window_matrix(
    window_x: np.ndarray,
    window_x_dot: np.ndarray,
    # source_state: np.ndarray,
    target_state: np.ndarray,
    cfg: ChainConfig,
):
    fitted_A = _fit_linear_system(
        window_x,
        window_x_dot,
        target=target_state,
        stabilization_margin=cfg.stabilization_margin,
        lmi_tolerance=cfg.lmi_tolerance,
    )

    return fitted_A, window_x, window_x_dot


def _resolve_transition_profile(
    path_states: np.ndarray,
    system_start_idx: np.ndarray,
    subsystem_edges: int,
    system_targets: np.ndarray,
    A_seq: np.ndarray,
    cfg: ChainConfig,
):
    n_systems = int(len(system_start_idx))
    n_transitions = max(n_systems - 1, 0)
    dim = int(path_states.shape[1])
    if n_transitions == 0:
        return (
            np.zeros((0, dim), dtype=float),
            np.zeros((0, dim), dtype=float),
            np.zeros((0, dim), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
        )

    subsystem_edges = int(max(1, subsystem_edges))

    def _edge_len(a: int, b: int) -> float:
        if a < 0 or b < 0 or a >= len(path_states) or b >= len(path_states):
            return 0.0
        return float(np.linalg.norm(path_states[b] - path_states[a]))

    transition_centers = []
    transition_normals = []
    transition_ratio_nodes = []
    transition_edge_ratios = []
    transition_times = []
    transition_distances = []
    for i in range(n_transitions):
        start_idx = int(system_start_idx[i])

        if subsystem_edges == 1:
            # Use the two edges around the boundary between subsystems:
            # edge_prev = (i -> i+1), edge_next = (i+1 -> i+2),
            # plane passes through i+1.
            prev_idx = start_idx
            anchor_idx = start_idx + 1
            next_idx = start_idx + 2
        else:
            # For m>=2 and subsystem [i..i+m], use its last two edges:
            # edge_prev = (i+m-2 -> i+m-1), edge_next = (i+m-1 -> i+m),
            # plane passes through i+m-1 (second-to-last node).
            prev_idx = start_idx + subsystem_edges - 2
            anchor_idx = start_idx + subsystem_edges - 1
            next_idx = start_idx + subsystem_edges

        anchor = np.asarray(path_states[anchor_idx], dtype=float)
        prev_state = np.asarray(path_states[prev_idx], dtype=float)
        next_state = np.asarray(path_states[next_idx], dtype=float)

        edge_prev = anchor - prev_state
        edge_next = next_state - anchor
        len_prev = float(np.linalg.norm(edge_prev))
        len_next = float(np.linalg.norm(edge_next))

        def _unit(v: np.ndarray) -> np.ndarray:
            nv = float(np.linalg.norm(v))
            if nv <= 1e-12:
                return np.zeros_like(v, dtype=float)
            return np.asarray(v, dtype=float) / nv

        normal_prev = _unit(edge_prev)
        normal_next = _unit(edge_next)
        plane_normal = normal_prev + normal_next
        if np.linalg.norm(plane_normal) <= 1e-12:
            plane_normal = normal_next if np.linalg.norm(normal_next) > 1e-12 else normal_prev
        if np.linalg.norm(plane_normal) <= 1e-12:
            plane_normal = np.zeros((dim,), dtype=float)
            plane_normal[0] = 1.0
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Orient the plane normal so prev-edge side is negative and next-edge side is positive.
        signed_prev = float(np.dot(prev_state - anchor, plane_normal))
        signed_next = float(np.dot(next_state - anchor, plane_normal))
        if signed_next < signed_prev:
            plane_normal = -plane_normal

        edge_ref = max(len_next, 1e-12)
        transition_length = max(cfg.blend_length_ratio * edge_ref, 1e-12)

        v_center = np.asarray(A_seq[i], dtype=float) @ (anchor - np.asarray(system_targets[i], dtype=float))
        speed = float(np.linalg.norm(v_center))
        transition_time = transition_length / max(speed, 1e-6)
        transition_time = float(max(transition_time, 1e-4))

        transition_centers.append(anchor)
        transition_normals.append(plane_normal)
        transition_ratio_nodes.append(next_state)
        transition_edge_ratios.append(len_prev / max(len_next, 1e-12))
        transition_times.append(transition_time)
        transition_distances.append(transition_length)

    return (
        np.vstack(transition_centers),
        np.vstack(transition_normals),
        np.vstack(transition_ratio_nodes),
        np.asarray(transition_edge_ratios, dtype=float),
        np.asarray(transition_times, dtype=float),
        np.asarray(transition_distances, dtype=float),
    )


def _resolve_path_states(gg, path_nodes: list) -> np.ndarray:
    if path_nodes is None or len(path_nodes) == 0:
        return np.zeros((0, 0), dtype=float)
    return np.vstack([np.asarray(gg.graph.nodes[node]["mean"], dtype=float) for node in path_nodes])

def prepare_chaining_edge_lookup(ds_set, gg):
    """Prepare reusable chain config + triplet lookup scaffold for repeated replanning."""
    # cfg = _resolve_chain_config(config)
    source_cache = {
        node: _get_source_data_for_node(ds_set, gg, node)
        for node in _graph_gaussian_ids(gg)
    }
    triplet_lookup = {
        "source_cache": source_cache,
        "triplet_connections": {},
        "subsystem_edges": 2,
    }
    return triplet_lookup

def _compute_segment_DS(ds_set, gg, segment_nodes, config):

    # Collect the gaussians and normalize priors
    gaussians = []
    for i, node_id in enumerate(segment_nodes):
        mu, sigma, direction, prior = gg.get_gaussian(node_id)
        gaussians.append({
            'prior': prior,  # use normalized prior
            'mu': mu,
            'sigma': sigma,
            'rv': multivariate_normal(mu, sigma, allow_singular=True)
        })
    sum_priors = sum(g['prior'] for g in gaussians)
    for g in gaussians:
        g['prior'] /= sum_priors

    # collect the trajectory points that are assigned to the gaussians along the shortest path
    filtered_x = []
    filtered_x_dot = []
    for node_id in segment_nodes:
        ds_idx = node_id[0]
        gaussian_idx = node_id[1]

        assigned_x = ds_set[ds_idx].x[ds_set[ds_idx].assignment_arr == gaussian_idx]
        assigned_x_dot = ds_set[ds_idx].x_dot[ds_set[ds_idx].assignment_arr == gaussian_idx]

        # reverse velocity if gaussian is reversed
        assigned_x_dot = -assigned_x_dot if node_id in gg.gaussian_reversal_map else assigned_x_dot

        filtered_x.append(assigned_x)
        filtered_x_dot.append(assigned_x_dot)

    filtered_x = np.vstack(filtered_x)
    filtered_x_dot = np.vstack(filtered_x_dot)

    # compute DS
    x_att = np.asarray(gg.graph.nodes[segment_nodes[-1]]["mean"], dtype=float)
    try:
        stitched_ds = lpvds_class(filtered_x, filtered_x_dot, x_att,
                                  rel_scale=getattr(config, 'rel_scale', 0.7),
                                  total_scale=getattr(config, 'total_scale', 1.5),
                                  nu_0=getattr(config, 'nu_0', 5),
                                  kappa_0=getattr(config, 'kappa_0', 1),
                                  psi_dir_0=getattr(config, 'psi_dir_0', 1))
        if config.chain.recompute_gaussians:  # compute new gaussians and linear systems (As)
            result = stitched_ds.begin()
            if not result:
                print('Chaining: Failed to construct Stitched DS: DAMM clustering failed')
                stitched_ds = None
        else:  # compute only linear systems (As)
            stitched_ds.init_cluster(gaussians)
            stitched_ds._optimize()

        return stitched_ds

    except Exception as e:
        print(f'Chaining: Failed to construct Stitched DS for a segment: {e}')
        return None


def build_chained_linear_ds(
        ds_set,
        gg,
        initial,
        attractor,
        config: StitchConfig,
        shortest_path_nodes,
        precomputed_edge_lookup: Optional[dict] = None,
) -> Optional[ChainedLinearDS]:
    if initial is None or attractor is None or config is None:
        raise TypeError("build_chained_ds requires initial, attractor, and config.")

    initial = np.asarray(initial, dtype=float)
    attractor = np.asarray(attractor, dtype=float)
    path_nodes = list(shortest_path_nodes)

    # Always prefer 3-node subsystems (2 edges) and fall back only for short paths.
    subsystem_edges = 2
    window_size = subsystem_edges + 1

    if len(path_nodes) < window_size:
        subsystem_edges = len(path_nodes) - 1
        window_size = subsystem_edges + 1
    if subsystem_edges < 0 or len(path_nodes) < window_size:
        print(f"WARN: subsystem_edges={subsystem_edges}, path_nodes={path_nodes}")
        return None

    path_state_sequence = _resolve_path_states(gg, path_nodes)
    n_systems = len(path_nodes) - subsystem_edges
    if n_systems <= 0:
        print(f"WARN: n_systems={n_systems}")
        return None

    lookup = precomputed_edge_lookup if isinstance(precomputed_edge_lookup, dict) else {}
    source_cache = lookup.get("source_cache")
    if source_cache is None:
        source_cache = {
            node: _get_source_data_for_node(ds_set, gg, node)
            for node in _graph_gaussian_ids(gg)
        }
        if isinstance(lookup, dict):
            lookup["source_cache"] = source_cache
    triplet_lookup = lookup.get("triplet_connections")
    if triplet_lookup is None:
        triplet_lookup = {}
        if isinstance(lookup, dict):
            lookup["triplet_connections"] = triplet_lookup

    system_start_idx = np.arange(n_systems, dtype=int)
    system_target_idx = system_start_idx + subsystem_edges
    node_sources = path_state_sequence[system_start_idx]
    fit_node_targets = np.asarray(path_state_sequence[system_target_idx], dtype=float)

    # Data selection/fitting remains graph-node-only, but execution must converge to the
    # actual requested attractor. Only the final runtime target is replaced.
    node_targets = fit_node_targets.copy()
    node_targets[-1] = attractor
    window_nodes_seq = [tuple(path_nodes[i: i + window_size]) for i in range(n_systems)]
    if len(window_nodes_seq) != n_systems:
        return None

    A_seq = []
    fit_points_seq = []
    fit_velocities_seq = []
    direction_stats_seq = []
    for i in range(n_systems):
        window_nodes = tuple(window_nodes_seq[i])
        source_state = np.asarray(node_sources[i], dtype=float)
        fit_target_state = np.asarray(fit_node_targets[i], dtype=float)
        use_triplet_cache = subsystem_edges == 2
        cached = triplet_lookup.get(window_nodes) if use_triplet_cache else None

        # already computes ds
        if cached is not None:
            A_i = np.asarray(cached["A"], dtype=float)
            fit_x = np.asarray(cached["fit_x"], dtype=float)
            fit_x_dot = np.asarray(cached["fit_x_dot"], dtype=float)
            stats = cached["stats"]
        # needs to compute ds
        else:
            if not all(node in source_cache for node in window_nodes):
                return None
            window_x, window_x_dot = _stack_xy(
                [source_cache[node][0] for node in window_nodes],
                [source_cache[node][1] for node in window_nodes],
            )

            assert not window_x.shape[0] == 0

            A_i, fit_x, fit_x_dot = _fit_window_matrix(
                window_x=window_x,
                window_x_dot=window_x_dot,
                # source_state=source_state,
                target_state=fit_target_state,
                cfg=config.chain,
            )

            assert A_i is not None

            stats = _direction_consistency_stats(
                A=A_i,
                fit_x=fit_x,
                source_state=source_state,
                target_state=fit_target_state,
            )
            if use_triplet_cache:
                triplet_lookup[window_nodes] = {
                    "A": np.asarray(A_i, dtype=float).copy(),
                    "fit_x": np.asarray(fit_x, dtype=float).copy(),
                    "fit_x_dot": np.asarray(fit_x_dot, dtype=float).copy(),
                    "stats": dict(stats),
                }

        A_seq.append(A_i.copy())
        fit_points_seq.append(fit_x.copy())
        fit_velocities_seq.append(fit_x_dot.copy())
        direction_stats_seq.append(stats)

    A_seq = np.array(A_seq)

    (
        transition_centers,
        transition_normals,
        transition_ratio_nodes,
        transition_edge_ratios,
        transition_times,
        transition_distances,
    ) = _resolve_transition_profile(
        path_states=path_state_sequence,
        system_start_idx=system_start_idx,
        subsystem_edges=subsystem_edges,
        system_targets=fit_node_targets,
        A_seq=A_seq,
        cfg=config.chain,
    )

    gaussian_path_nodes = list(path_nodes)
    if len(gaussian_path_nodes) == 0:
        gaussian_lists = [{
            "prior": 1.0,
            "mu": np.asarray(initial, dtype=float),
            "sigma": np.eye(len(initial)) * 0.01,
            "rv": multivariate_normal(np.asarray(initial, dtype=float), np.eye(len(initial)) * 0.01,
                                      allow_singular=True),
        }]
        filtered_x = np.asarray(initial, dtype=float).reshape(1, -1)
        filtered_x_dot = np.zeros_like(filtered_x)
    else:
        raw_priors = np.array([gg.graph.nodes[node_id]["prior"] for node_id in gaussian_path_nodes], dtype=float)
        priors = raw_priors / np.sum(raw_priors)
        gaussian_lists = []
        filtered_x = []
        filtered_x_dot = []
        for idx, node_id in enumerate(gaussian_path_nodes):
            mu, sigma, _, _ = gg.get_gaussian(node_id)
            gaussian_lists.append(
                {
                    "prior": float(priors[idx]),
                    "mu": mu,
                    "sigma": sigma,
                    "rv": multivariate_normal(mu, sigma, allow_singular=True),
                }
            )
            assigned_x, assigned_x_dot, _, _ = source_cache[node_id]
            filtered_x.append(assigned_x)
            filtered_x_dot.append(assigned_x_dot)
        filtered_x, filtered_x_dot = _stack_xy(filtered_x, filtered_x_dot)
        if filtered_x.shape[0] == 0:
            return None

    # Reuse the existing DAMM Gaussian machinery (no custom GMM wrapper).
    x_proc, x_dot_proc, x_dir = damm_class.pre_process(filtered_x, filtered_x_dot)
    if x_proc.shape[0] == 0:
        x_proc = filtered_x.copy()
        x_dir = np.zeros_like(x_proc)
        x_dir[:, 0] = 1.0

    damm_cfg = config.damm
    damm = damm_class(
        x_proc,
        x_dir,
        nu_0=damm_cfg.nu_0,
        kappa_0=damm_cfg.kappa_0,
        psi_dir_0=damm_cfg.psi_dir_0,
        rel_scale=damm_cfg.rel_scale,
        total_scale=damm_cfg.total_scale,
    )
    damm.K = len(gaussian_lists)
    damm.gaussian_lists = gaussian_lists
    damm.Mu = np.array([g["mu"] for g in gaussian_lists])
    damm.Sigma = np.array([g["sigma"] for g in gaussian_lists])
    damm.Prior = np.array([g["prior"] for g in gaussian_lists])

    terminal_state = np.asarray(attractor, dtype=float)

    chained_ds = ChainedLinearDS(
        x=filtered_x,
        x_dot=filtered_x_dot,
        attractor=terminal_state,
        path_nodes=path_nodes,
        node_sources=node_sources,
        node_targets=node_targets,
        A_seq=A_seq,
        damm=damm,
        transition_centers=transition_centers,
        transition_normals=transition_normals,
        transition_ratio_nodes=transition_ratio_nodes,
        transition_edge_ratios=transition_edge_ratios,
        transition_times=transition_times,
        transition_distances=transition_distances,
        chain_cfg=config.chain,
    )
    chained_ds.system_start_idx = system_start_idx
    chained_ds.system_target_idx = system_target_idx
    chained_ds.subsystem_edges = subsystem_edges
    chained_ds.path_state_sequence = path_state_sequence
    chained_ds.fit_node_targets = fit_node_targets
    chained_ds.triplet_windows = window_nodes_seq
    chained_ds.edge_fit_points = fit_points_seq
    chained_ds.edge_fit_velocities = fit_velocities_seq
    chained_ds.edge_direction_stats = direction_stats_seq
    chained_ds.edge_forward_fraction = np.array(
        [s.get("frac_forward", np.nan) for s in direction_stats_seq],
        dtype=float,
    )
    chained_ds.edge_forward_min_proj = np.array(
        [s.get("min_proj", np.nan) for s in direction_stats_seq],
        dtype=float,
    )
    chained_ds.transition_ratio_nodes = transition_ratio_nodes
    chained_ds.transition_edge_ratios = transition_edge_ratios
    return chained_ds

def build_chained_segmented_ds(
    ds_set,
    gg,
    initial,
    attractor,
    config: StitchConfig,
    shortest_path_nodes,
    precomputed_edge_lookup: Optional[dict] = None,
) -> Optional[ChainedSegmentedDS]:

    if initial is None or attractor is None or config is None:
        raise TypeError("build_chained_ds requires initial, attractor, and config.")

    initial = np.asarray(initial, dtype=float)
    attractor = np.asarray(attractor, dtype=float)
    path_nodes = list(shortest_path_nodes)

    # Split path into intermediate segments (e.g. 2 edges: (n1->n2->n3), (n2->n3->n4), etc.)
    segment_size = config.chain.subsystem_edges
    if len(path_nodes)-1 < segment_size:
        intermediate_segments = [tuple(path_nodes)]
    else:
        intermediate_segments = []
        for i in range(0, len(path_nodes) - segment_size):
            intermediate_segments.append(tuple(path_nodes[i : i + segment_size + 1]))

    # Fetch precomputed segment DSs for intermediate segments if available.
    intermediate_DSs = [None for _ in range(len(intermediate_segments))]
    if precomputed_edge_lookup is not None:
        for i, segment in enumerate(intermediate_segments):
            if segment in precomputed_edge_lookup:
                intermediate_DSs[i] = precomputed_edge_lookup[segment]

    # ---- Compute DSs for each segment (init to first node, intermediate segments, last node to attractor) ----
    # Gives:
    #     1. A_init: and LTI A-matrix that is GAS to the first node
    #     2. A_attractor: an LTI A-matrix that is GAS to the attractor
    #     3. intermediate_DSs: a list of multi-node-fitted DS objects for each intermediate segment
    # 1. Init to first node is a single-segment DS fit. Use gaussian from the first node as the target.
    assigned_x, assigned_x_dot, _, _ = _get_source_data_for_node(ds_set, gg, path_nodes[0])
    A_init = _fit_linear_system(x=assigned_x,
                                x_dot=assigned_x_dot,
                                target=np.asarray(gg.graph.nodes[path_nodes[0]]["mean"], dtype=float),
                                stabilization_margin=config.chain.stabilization_margin,
                                lmi_tolerance=config.chain.lmi_tolerance)
    if A_init is None:
        raise RuntimeError("Chaining: Failed to fit DS for initial -> first node.")

    # 2. Last node to attractor is a single-segment DS fit. Use attractor as the target.
    assigned_x, assigned_x_dot, _, _ = _get_source_data_for_node(ds_set, gg, path_nodes[-1])
    A_attractor = _fit_linear_system(x=assigned_x,
                                     x_dot=assigned_x_dot,
                                     target=attractor,
                                     stabilization_margin=config.chain.stabilization_margin,
                                     lmi_tolerance=config.chain.lmi_tolerance)
    if A_attractor is None:
        raise RuntimeError("Chaining: Failed to fit DS for last node -> attractor.")

    # 3. Intermediate segments are multi-node DS fits. Use the last node in each segment as the target.
    for i, segment_DS in enumerate(intermediate_DSs):
        if segment_DS is not None:
            continue
        intermediate_DSs[i] = _compute_segment_DS(ds_set, gg, intermediate_segments[i], config)

    # Collect all x_ref and x_dot_ref along the path
    x_list = []
    x_dot_list = []
    for node_id in shortest_path_nodes:
        assigned_x, assigned_x_dot, _, _ = _get_source_data_for_node(ds_set, gg, node_id)
        x_list.append(assigned_x)
        x_dot_list.append(assigned_x_dot)
    x_ref = np.vstack(x_list)
    x_dot_ref = np.vstack(x_dot_list)

    # Build and return segmented DS-chain
    chained_ds = ChainedSegmentedDS(
        x=x_ref,
        x_dot=x_dot_ref,
        ds_set=ds_set,
        gg=gg,
        path_nodes=path_nodes,
        initial=initial,
        attractor=attractor,
        intermediate_segments=intermediate_segments,
        intermediate_DSs=intermediate_DSs,
        A_init=A_init,
        A_attractor=A_attractor,
        blend_length_ratio=config.chain.blend_length_ratio,
    )
    return chained_ds
