from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal
from src.damm.src.damm_class import DAMM as damm_class
import cvxpy as cp
from configs import ChainConfig, StitchConfig
from src.lpvds_class import lpvds_class


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ChainedDS — shared backbone
# ---------------------------------------------------------------------------

class ChainedDS:
    """Base class for chained dynamical systems with trigger-based transitions.

    Stores common attributes (path geometry, transition geometry, runtime
    state) and implements shared logic: triggers, recovery, simulation,
    stateless velocity prediction, and vector-field evaluation.

    Subclasses must implement ``step_once``.
    """

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
        raw_vmax = getattr(chain_cfg, "velocity_max", None)
        if raw_vmax is None:
            self.velocity_max = None
        else:
            raw_vmax = float(raw_vmax)
            self.velocity_max = raw_vmax if np.isfinite(raw_vmax) and raw_vmax > 0.0 else None

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

        gamma = np.asarray(self.damm.compute_gamma(self.x), dtype=float)
        self.assignment_arr = np.argmax(gamma, axis=0).astype(int)

        self.tol = 10e-3
        self.max_iter = 10000
        self.last_sim_indices = None

        # Common runtime state.
        self._runtime_idx = 0
        self._runtime_time = 0.0
        self._state_entry_t = 0.0
        self._transition_from_idx = None
        self._transition_t0 = None
        self._state_dim = int(self.node_sources.shape[1]) if self.node_sources.ndim == 2 else int(self.x_att.shape[0])

    # ---- utilities --------------------------------------------------------

    def _state_vec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return x[0]
        return x

    def _velocity_for_index(self, x: np.ndarray, idx: int) -> np.ndarray:
        """Default linear velocity: A_seq[idx] @ (x - target[idx])."""
        idx = int(np.clip(idx, 0, self.n_systems - 1))
        return self.A_seq[idx] @ (x - self.node_targets[idx])

    def _clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        v = np.asarray(velocity, dtype=float).reshape(-1)
        if v.shape[0] != self._state_dim:
            return np.zeros((self._state_dim,), dtype=float)
        if not np.all(np.isfinite(v)):
            return np.zeros((self._state_dim,), dtype=float)

        if self.velocity_max is not None:
            speed = float(np.linalg.norm(v))
            if np.isfinite(speed) and speed > self.velocity_max and speed > 1e-12:
                v = v * (self.velocity_max / speed)

        return v

    # ---- trigger methods --------------------------------------------------

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

    # ---- recovery ---------------------------------------------------------

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

    # ---- runtime reset / step / sim ---------------------------------------

    def reset_runtime(self, initial_idx: int = 0, start_time: float = 0.0):
        self._runtime_idx = int(np.clip(initial_idx, 0, self.n_systems - 1))
        self._runtime_time = float(start_time)
        self._state_entry_t = float(start_time)
        self._clear_transition()

    def step_once(self, x: np.ndarray, dt: float, current_idx: Optional[int] = None, current_time: Optional[float] = None):
        raise NotImplementedError("Subclasses must implement step_once")

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

    # ---- transition machinery ---------------------------------------------

    @property
    def transition_active(self) -> bool:
        return self._transition_from_idx is not None

    def _clear_transition(self):
        self._transition_from_idx = None
        self._transition_t0 = None

    def trigger_time(self, idx: int, t: float) -> bool:
        if idx >= len(self.transition_times):
            return True
        if self._transition_t0 is None:
            return False
        return (t - self._transition_t0) >= self.transition_times[idx]

    def _start_transition_if_triggered(self, x: np.ndarray, t: float):
        if self._runtime_idx >= self.n_systems - 1:
            return
        if self._transition_from_idx is not None:
            return
        if self.trigger_state(self._runtime_idx, x):
            self._transition_from_idx = self._runtime_idx
            self._transition_t0 = t

    def _transition_velocity(self, x: np.ndarray, t: float) -> np.ndarray:
        idx = int(self._transition_from_idx)
        v_current = self._clip_velocity(self._velocity_for_index(x, idx))
        v_next = self._clip_velocity(self._velocity_for_index(x, idx + 1))

        T = self.transition_times[idx] if idx < len(self.transition_times) else 0.0
        if T <= 1e-12:
            alpha = 1.0
        else:
            alpha = np.clip((t - self._transition_t0) / T, 0.0, 1.0)
        v = self._clip_velocity((1.0 - alpha) * v_current + alpha * v_next)

        if self.trigger_time(idx, t):
            self._runtime_idx = min(idx + 1, self.n_systems - 1)
            self._state_entry_t = t
            self._clear_transition()
        return v

    # ---- stateless velocity queries ---------------------------------------

    def predict_velocities(self, x_positions: np.ndarray) -> np.ndarray:
        """Stateless proxy for metrics/visualization."""
        x_positions = np.atleast_2d(x_positions)
        velocities = []
        for x in x_positions:
            idx = int(np.argmin(np.linalg.norm(self.node_sources - x.reshape(1, -1), axis=1)))
            if idx < self.n_systems - 1 and self.trigger_state(idx, x):
                v = self._clip_velocity(
                    0.5 * self._velocity_for_index(x, idx) + 0.5 * self._velocity_for_index(x, idx + 1)
                )
            else:
                v = self._clip_velocity(self._velocity_for_index(x, idx))
            velocities.append(v)
        return np.vstack(velocities)

    def vector_field(self, x_positions: np.ndarray) -> np.ndarray:
        return self.predict_velocities(x_positions)


# ---------------------------------------------------------------------------
# ChainedLinearDS — one A matrix per segment, linear blending transitions
# ---------------------------------------------------------------------------

class ChainedLinearDS(ChainedDS):
    """DS chain with state-triggered entry and time-triggered transition completion."""

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
            velocity = self._clip_velocity(self._velocity_for_index(x, self._runtime_idx))
        else:
            velocity = self._transition_velocity(x, t)

        x_next = x + dt * velocity
        if current_time is None:
            self._runtime_time += dt
        return x_next, velocity, self._runtime_idx


# ---------------------------------------------------------------------------
# ChainedSegmentedDS — lpvds_class per segment, nominal/intermediate FSM
# ---------------------------------------------------------------------------

class ChainedSegmentedDS(ChainedDS):
    """Segment-wise chain using base-class transition machinery."""

    def __init__(self, *args, segment_ds_seq, transition_point_triples,
                 has_init_boundary=True, has_attractor_boundary=True, **kwargs) -> None:
        self.segment_ds_seq = list(segment_ds_seq)
        self.has_init_boundary = bool(has_init_boundary)
        self.has_attractor_boundary = bool(has_attractor_boundary)
        self.transition_point_triples = [tuple(np.asarray(p, dtype=float) for p in trip) for trip in transition_point_triples]
        super().__init__(*args, **kwargs)
        # Compatibility aliases with previous segmented naming.
        self.intermediate_DSs = self.segment_ds_seq
        self.transition_points = self.transition_point_triples

    def _velocity_for_index(self, x: np.ndarray, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.n_systems - 1))
        # Boundary systems (if present) are explicit linear maps.
        if self.has_init_boundary and idx == 0:
            return super()._velocity_for_index(x, idx)
        if self.has_attractor_boundary and idx == self.n_systems - 1:
            return super()._velocity_for_index(x, idx)

        # Segment DS objects.
        seg_idx = idx - (1 if self.has_init_boundary else 0)
        seg_ds = self.segment_ds_seq[seg_idx]
        x_row = np.asarray(x, dtype=float).reshape(1, -1)
        _, _, x_dot = seg_ds._step(x_row, dt=1.0)
        return np.asarray(x_dot, dtype=float).reshape(-1)

    def step_once(
        self,
        x: np.ndarray,
        dt: float,
        current_idx: Optional[int] = None,
        current_time: Optional[float] = None,
    ):
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
            velocity = self._clip_velocity(self._velocity_for_index(x, self._runtime_idx))
        else:
            velocity = self._transition_velocity(x, t)

        x_next = x + dt * velocity
        if current_time is None:
            self._runtime_time += dt
        return x_next, velocity, self._runtime_idx


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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


def _extract_linear_triples(path_states, system_start_idx, subsystem_edges):
    """Extract (start, center, end) point triples for linear chain transitions."""
    subsystem_edges = int(max(1, subsystem_edges))
    n_transitions = max(len(system_start_idx) - 1, 0)
    triples = []
    for i in range(n_transitions):
        s = int(system_start_idx[i])
        if subsystem_edges == 1:
            prev_idx, anchor_idx, next_idx = s, s + 1, s + 2
        else:
            prev_idx = s + subsystem_edges - 2
            anchor_idx = s + subsystem_edges - 1
            next_idx = s + subsystem_edges
        triples.append((path_states[prev_idx], path_states[anchor_idx], path_states[next_idx]))
    return triples


def _resolve_path_states(gg, path_nodes: list) -> np.ndarray:
    if path_nodes is None or len(path_nodes) == 0:
        return np.zeros((0, 0), dtype=float)
    return np.vstack([np.asarray(gg.graph.nodes[node]["mean"], dtype=float) for node in path_nodes])


def _resolve_transition_profile_from_point_triples(
    point_triples,
    nominal_velocity_fn,
    cfg: ChainConfig,
    dim: int,
):
    n_transitions = int(len(point_triples))
    if n_transitions == 0:
        return (
            np.zeros((0, dim), dtype=float),
            np.zeros((0, dim), dtype=float),
            np.zeros((0, dim), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
        )

    transition_centers = []
    transition_normals = []
    transition_ratio_nodes = []
    transition_edge_ratios = []
    transition_times = []
    transition_distances = []
    raw_min_transition_time = float(getattr(cfg, "min_transition_time", 1e-4))
    if not np.isfinite(raw_min_transition_time):
        min_transition_time = 1e-4
    else:
        min_transition_time = max(raw_min_transition_time, 1e-6)
    raw_velocity_max = getattr(cfg, "velocity_max", None)
    if raw_velocity_max is None:
        velocity_cap = None
    else:
        raw_velocity_max = float(raw_velocity_max)
        velocity_cap = raw_velocity_max if np.isfinite(raw_velocity_max) and raw_velocity_max > 0.0 else None

    def _unit(v: np.ndarray) -> np.ndarray:
        nv = float(np.linalg.norm(v))
        if nv <= 1e-12:
            return np.zeros_like(v, dtype=float)
        return np.asarray(v, dtype=float) / nv

    for i, (start, center, end) in enumerate(point_triples):
        start = np.asarray(start, dtype=float)
        center = np.asarray(center, dtype=float)
        end = np.asarray(end, dtype=float)

        e1 = center - start
        e2 = end - center
        len_e1 = float(np.linalg.norm(e1))
        len_e2 = float(np.linalg.norm(e2))

        n1 = _unit(e1)
        n2 = _unit(e2)
        plane_normal = n1 + n2
        if np.linalg.norm(plane_normal) <= 1e-12:
            plane_normal = n2 if np.linalg.norm(n2) > 1e-12 else n1
        if np.linalg.norm(plane_normal) <= 1e-12:
            plane_normal = np.zeros((dim,), dtype=float)
            plane_normal[0] = 1.0
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Orient the plane normal so start side is negative and end side is positive.
        signed_start = float(np.dot(start - center, plane_normal))
        signed_end = float(np.dot(end - center, plane_normal))
        if signed_end < signed_start:
            plane_normal = -plane_normal

        transition_length = max(cfg.blend_length_ratio * max(len_e2, 1e-12), 1e-12)
        v_center = np.asarray(nominal_velocity_fn(i, center), dtype=float).reshape(-1)
        speed = float(np.linalg.norm(v_center))
        if not np.isfinite(speed):
            speed = 0.0
        if velocity_cap is not None:
            speed = min(speed, velocity_cap)
        transition_time = transition_length / max(speed, 1e-6)
        if not np.isfinite(transition_time):
            transition_time = min_transition_time
        transition_time = float(max(transition_time, min_transition_time))

        transition_centers.append(center)
        transition_normals.append(plane_normal)
        transition_ratio_nodes.append(end)
        transition_edge_ratios.append(len_e1 / max(len_e2, 1e-12))
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


def _build_chain_path_compat_data(ds_set, gg, path_nodes, source_cache, initial, config: StitchConfig):
    gaussian_path_nodes = list(path_nodes)
    if len(gaussian_path_nodes) == 0:
        gaussian_lists = [{
            "prior": 1.0,
            "mu": np.asarray(initial, dtype=float),
            "sigma": np.eye(len(initial)) * 0.01,
            "rv": multivariate_normal(
                np.asarray(initial, dtype=float),
                np.eye(len(initial)) * 0.01,
                allow_singular=True,
            ),
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
            return None, None, None

    x_proc, _, x_dir = damm_class.pre_process(filtered_x, filtered_x_dot)
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
    return filtered_x, filtered_x_dot, damm


def prepare_chaining_edge_lookup(ds_set, gg):
    """Prepare reusable chain config + triplet lookup scaffold for repeated replanning."""
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

def _compute_segment_DS(ds_set, gg, segment_nodes, config, x_att_override=None):

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
    x_att = np.asarray(x_att_override, dtype=float) if x_att_override is not None else np.asarray(gg.graph.nodes[segment_nodes[-1]]["mean"], dtype=float)
    stitched_ds = lpvds_class(filtered_x, filtered_x_dot, x_att,
                              rel_scale=getattr(config, 'rel_scale', 0.7),
                              total_scale=getattr(config, 'total_scale', 1.5),
                              nu_0=getattr(config, 'nu_0', 5),
                              kappa_0=getattr(config, 'kappa_0', 1),
                              psi_dir_0=getattr(config, 'psi_dir_0', 1))
    if config.chain.recompute_gaussians:  # compute new gaussians and linear systems (As)
        result = stitched_ds.begin()
        if not result:
            raise RuntimeError('Chaining: DAMM clustering failed for segment DS')
    else:  # compute only linear systems (As)
        stitched_ds.init_cluster(gaussians)
        stitched_ds._optimize()

    return stitched_ds


# ---------------------------------------------------------------------------
# Builder: ChainedLinearDS
# ---------------------------------------------------------------------------

def _concat_transition_arrays(*parts_list):
    """Concatenate multiple (centers, normals, ratio_nodes, edge_ratios, times, distances) tuples."""
    centers, normals, ratio_nodes = [], [], []
    edge_ratios, times, distances = [], [], []
    for (tc, tn, trn, ter, tt, td) in parts_list:
        if tc.shape[0] > 0:
            centers.append(tc)
            normals.append(tn)
            ratio_nodes.append(trn)
            edge_ratios.append(ter)
            times.append(tt)
            distances.append(td)
    if len(centers) == 0:
        dim = parts_list[0][0].shape[1] if len(parts_list) > 0 and parts_list[0][0].ndim == 2 else 2
        return (
            np.zeros((0, dim), dtype=float),
            np.zeros((0, dim), dtype=float),
            np.zeros((0, dim), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
        )
    return (
        np.vstack(centers),
        np.vstack(normals),
        np.vstack(ratio_nodes),
        np.concatenate(edge_ratios),
        np.concatenate(times),
        np.concatenate(distances),
    )


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
    use_init = config.chain.use_boundary_ds_initial
    use_end = config.chain.use_boundary_ds_end

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
    n_core_systems = len(path_nodes) - subsystem_edges
    if n_core_systems <= 0:
        print(f"WARN: n_core_systems={n_core_systems}")
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

    system_start_idx = np.arange(n_core_systems, dtype=int)
    system_target_idx = system_start_idx + subsystem_edges
    node_sources_core = path_state_sequence[system_start_idx]
    fit_node_targets = np.asarray(path_state_sequence[system_target_idx], dtype=float)

    window_nodes_seq = [tuple(path_nodes[i: i + window_size]) for i in range(n_core_systems)]
    if len(window_nodes_seq) != n_core_systems:
        return None

    # ---- Fit core A matrices ----
    A_seq_core = []
    fit_points_seq = []
    fit_velocities_seq = []
    direction_stats_seq = []
    for i in range(n_core_systems):
        window_nodes = tuple(window_nodes_seq[i])
        source_state = np.asarray(node_sources_core[i], dtype=float)
        fit_target_state = np.asarray(fit_node_targets[i], dtype=float)
        use_triplet_cache = subsystem_edges == 2
        cached = triplet_lookup.get(window_nodes) if use_triplet_cache else None

        if cached is not None:
            A_i = np.asarray(cached["A"], dtype=float)
            fit_x = np.asarray(cached["fit_x"], dtype=float)
            fit_x_dot = np.asarray(cached["fit_x_dot"], dtype=float)
            stats = cached["stats"]
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

        A_seq_core.append(A_i.copy())
        fit_points_seq.append(fit_x.copy())
        fit_velocities_seq.append(fit_x_dot.copy())
        direction_stats_seq.append(stats)

    A_seq_core = np.array(A_seq_core)
    dim = int(path_state_sequence.shape[1])

    # ---- Build full sequences with independent init / end boundary ----
    first_center = path_state_sequence[0]
    last_center = path_state_sequence[-1]

    # When end-boundary is OFF the last core system drives to the attractor directly.
    core_node_targets = fit_node_targets.copy()
    if not use_end:
        core_node_targets[-1] = attractor

    # Core transitions (always present).
    core_triples = _extract_linear_triples(path_state_sequence, system_start_idx, subsystem_edges)
    core_trans = _resolve_transition_profile_from_point_triples(
        point_triples=core_triples,
        nominal_velocity_fn=lambda i, x: np.asarray(A_seq_core[i], dtype=float) @ (np.asarray(x, dtype=float).reshape(-1) - np.asarray(fit_node_targets[i], dtype=float)),
        cfg=config.chain,
        dim=dim,
    )

    # -- Optional init boundary --
    if use_init:
        ax_first, axd_first, _, _ = source_cache[path_nodes[0]]
        A_init = _fit_linear_system(
            ax_first, axd_first,
            target=first_center,
            stabilization_margin=config.chain.stabilization_margin,
            lmi_tolerance=config.chain.lmi_tolerance,
        )
        if A_init is None:
            raise RuntimeError("Chaining: Failed to fit DS for initial -> first node.")

        init_trans = _resolve_transition_profile_from_point_triples(
            point_triples=[(initial, 0.5 * (initial + first_center), first_center)],
            nominal_velocity_fn=lambda _i, x: np.asarray(A_init, dtype=float) @ (np.asarray(x, dtype=float).reshape(-1) - first_center),
            cfg=config.chain,
            dim=dim,
        )
        init_stats = _direction_consistency_stats(A=A_init, fit_x=ax_first, source_state=initial, target_state=first_center)

    # -- Optional end boundary --
    if use_end:
        ax_last, axd_last, _, _ = source_cache[path_nodes[-1]]
        A_att = _fit_linear_system(
            ax_last, axd_last,
            target=attractor,
            stabilization_margin=config.chain.stabilization_margin,
            lmi_tolerance=config.chain.lmi_tolerance,
        )
        if A_att is None:
            raise RuntimeError("Chaining: Failed to fit DS for last node -> attractor.")

        att_trans = _resolve_transition_profile_from_point_triples(
            point_triples=[(last_center, 0.5 * (last_center + attractor), attractor)],
            nominal_velocity_fn=lambda _i, x: np.asarray(A_seq_core[-1], dtype=float) @ (np.asarray(x, dtype=float).reshape(-1) - fit_node_targets[-1]),
            cfg=config.chain,
            dim=dim,
        )
        att_stats = _direction_consistency_stats(A=A_att, fit_x=ax_last, source_state=last_center, target_state=attractor)

    # -- Assemble full sequences --
    A_list = list(A_seq_core)
    src_list = list(node_sources_core)
    tgt_list = list(core_node_targets)
    stats_list = list(direction_stats_seq)
    fp_list = list(fit_points_seq)
    fv_list = list(fit_velocities_seq)
    wn_list = list(window_nodes_seq)
    trans_parts = [core_trans]

    if use_init:
        A_list.insert(0, A_init)
        src_list.insert(0, initial)
        tgt_list.insert(0, first_center)
        stats_list.insert(0, init_stats)
        fp_list.insert(0, ax_first.copy())
        fv_list.insert(0, axd_first.copy())
        wn_list.insert(0, tuple([path_nodes[0]]))
        trans_parts.insert(0, init_trans)

    if use_end:
        A_list.append(A_att)
        src_list.append(last_center)
        tgt_list.append(attractor)
        stats_list.append(att_stats)
        fp_list.append(ax_last.copy())
        fv_list.append(axd_last.copy())
        wn_list.append(tuple([path_nodes[-1]]))
        trans_parts.append(att_trans)

    A_seq = np.array(A_list)
    node_sources = np.array(src_list)
    node_targets = np.array(tgt_list)
    n_systems = len(A_list)
    direction_stats_seq = stats_list
    fit_points_seq = fp_list
    fit_velocities_seq = fv_list
    window_nodes_seq = wn_list

    (
        transition_centers,
        transition_normals,
        transition_ratio_nodes,
        transition_edge_ratios,
        transition_times,
        transition_distances,
    ) = _concat_transition_arrays(*trans_parts)

    # ---- Compat damm ----
    filtered_x, filtered_x_dot, damm = _build_chain_path_compat_data(
        ds_set=ds_set,
        gg=gg,
        path_nodes=path_nodes,
        source_cache=source_cache,
        initial=initial,
        config=config,
    )
    if filtered_x is None or filtered_x_dot is None or damm is None:
        return None

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


# ---------------------------------------------------------------------------
# Builder: ChainedSegmentedDS
# ---------------------------------------------------------------------------

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
    if len(path_nodes) == 0:
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

    if isinstance(lookup, dict) and (
        "source_cache" in lookup or "triplet_connections" in lookup or "segment_ds_lookup" in lookup
    ):
        segment_ds_lookup = lookup.setdefault("segment_ds_lookup", {})
    else:
        segment_ds_lookup = lookup

    # Split path into intermediate segments (e.g. 2 edges: (n1->n2->n3), (n2->n3->n4), etc.)
    segment_size = 2
    if len(path_nodes) - 1 < segment_size:
        segment_size = len(path_nodes) - 1
    if len(path_nodes) - 1 < segment_size:
        intermediate_segments = [tuple(path_nodes)]
    else:
        intermediate_segments = []
        for i in range(0, len(path_nodes) - segment_size):
            intermediate_segments.append(tuple(path_nodes[i: i + segment_size + 1]))

    # Fetch/compute intermediate segment DSs.
    intermediate_DSs = []
    for segment in intermediate_segments:
        segment_ds = segment_ds_lookup.get(segment)
        if segment_ds is None:
            segment_ds = _compute_segment_DS(ds_set, gg, segment, config)
            segment_ds_lookup[segment] = segment_ds
        intermediate_DSs.append(segment_ds)

    use_init = config.chain.use_boundary_ds_initial
    use_end = config.chain.use_boundary_ds_end

    def _center(node_id):
        return np.asarray(gg.graph.nodes[node_id]["mean"], dtype=float)

    first_center = _center(path_nodes[0])
    last_center = _center(path_nodes[-1])
    dim = int(initial.shape[0])

    # When end-boundary is OFF the last segment DS targets the attractor directly.
    # This segment is NOT cached because the attractor changes across replans.
    if not use_end and len(intermediate_DSs) > 0:
        last_seg = intermediate_segments[-1]
        intermediate_DSs[-1] = _compute_segment_DS(
            ds_set, gg, last_seg, config, x_att_override=attractor,
        )

    # ---- Optional init boundary ----
    A_init = None
    if use_init:
        assigned_x, assigned_x_dot, _, _ = _get_source_data_for_node(ds_set, gg, path_nodes[0])
        A_init = _fit_linear_system(
            x=assigned_x,
            x_dot=assigned_x_dot,
            target=first_center,
            stabilization_margin=config.chain.stabilization_margin,
            lmi_tolerance=config.chain.lmi_tolerance,
        )
        if A_init is None:
            raise RuntimeError("Chaining: Failed to fit DS for initial -> first node.")

    # ---- Optional end boundary ----
    A_attractor = None
    if use_end:
        assigned_x, assigned_x_dot, _, _ = _get_source_data_for_node(ds_set, gg, path_nodes[-1])
        A_attractor = _fit_linear_system(
            x=assigned_x,
            x_dot=assigned_x_dot,
            target=attractor,
            stabilization_margin=config.chain.stabilization_margin,
            lmi_tolerance=config.chain.lmi_tolerance,
        )
        if A_attractor is None:
            raise RuntimeError("Chaining: Failed to fit DS for last node -> attractor.")

    # ---- Build nominal system sequence ----
    # Layout: [init_boundary?] + segment_0 .. segment_m-1 + [end_boundary?]
    A_list = []
    src_list = []
    tgt_list = []
    seg_ds_list = []          # parallel to intermediate_DSs but only segment systems
    triplet_windows = []

    if use_init:
        A_list.append(np.asarray(A_init, dtype=float))
        src_list.append(initial)
        tgt_list.append(first_center)
        triplet_windows.append(tuple([path_nodes[0]]))

    for i, seg in enumerate(intermediate_segments):
        seg_start = _center(seg[0])
        seg_end = _center(seg[-1]) if (i < len(intermediate_segments) - 1 or use_end) else attractor
        src_list.append(seg_start)
        tgt_list.append(seg_end)
        seg_ds = intermediate_DSs[i]
        seg_ds_list.append(seg_ds)
        A_raw = np.asarray(seg_ds.A, dtype=float)
        A_list.append(np.asarray(A_raw[0], dtype=float))
        triplet_windows.append(tuple(seg))

    if use_end:
        A_list.append(np.asarray(A_attractor, dtype=float))
        src_list.append(last_center)
        tgt_list.append(attractor)
        triplet_windows.append(tuple([path_nodes[-1]]))

    node_sources = np.asarray(src_list, dtype=float)
    node_targets = np.asarray(tgt_list, dtype=float)
    A_seq = np.asarray(A_list, dtype=float)
    n_systems = len(A_list)

    # ---- Transition point triples ----
    transition_point_triples = []

    if use_init:
        transition_point_triples.append(
            (initial, 0.5 * (initial + first_center), first_center)
        )

    for seg in intermediate_segments:
        s = _center(seg[0])
        c = _center(seg[1]) if len(seg) >= 3 else 0.5 * (s + _center(seg[-1]))
        e = _center(seg[-1])
        transition_point_triples.append((s, c, e))

    transition_point_triples = transition_point_triples[: max(n_systems - 1, 0)]

    # ---- Nominal velocity function for transition timing ----
    _init_offset = 1 if use_init else 0

    def _nominal_velocity_fn(idx: int, x: np.ndarray) -> np.ndarray:
        idx = int(np.clip(idx, 0, n_systems - 1))
        x = np.asarray(x, dtype=float).reshape(-1)
        if use_init and idx == 0:
            return np.asarray(A_init, dtype=float) @ (x - first_center)
        if use_end and idx == n_systems - 1:
            return np.asarray(A_attractor, dtype=float) @ (x - attractor)
        seg_ds = seg_ds_list[idx - _init_offset]
        x_row = x.reshape(1, -1)
        _, _, x_dot = seg_ds._step(x_row, dt=1.0)
        return np.asarray(x_dot, dtype=float).reshape(-1)

    (
        transition_centers,
        transition_normals,
        transition_ratio_nodes,
        transition_edge_ratios,
        transition_times,
        transition_distances,
    ) = _resolve_transition_profile_from_point_triples(
        point_triples=transition_point_triples,
        nominal_velocity_fn=_nominal_velocity_fn,
        cfg=config.chain,
        dim=dim,
    )

    filtered_x, filtered_x_dot, damm = _build_chain_path_compat_data(
        ds_set=ds_set,
        gg=gg,
        path_nodes=path_nodes,
        source_cache=source_cache,
        initial=initial,
        config=config,
    )
    if filtered_x is None or filtered_x_dot is None or damm is None:
        return None

    chained_ds = ChainedSegmentedDS(
        x=filtered_x,
        x_dot=filtered_x_dot,
        attractor=np.asarray(attractor, dtype=float),
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
        segment_ds_seq=seg_ds_list,
        transition_point_triples=transition_point_triples,
        has_init_boundary=use_init,
        has_attractor_boundary=use_end,
    )

    chained_ds.system_start_idx = np.arange(n_systems, dtype=int)
    chained_ds.system_target_idx = np.arange(n_systems, dtype=int)
    chained_ds.subsystem_edges = segment_size
    chained_ds.path_state_sequence = _resolve_path_states(gg, path_nodes)
    chained_ds.fit_node_targets = node_targets.copy()
    chained_ds.triplet_windows = triplet_windows

    edge_fit_points = []
    edge_fit_velocities = []
    edge_direction_stats = []
    for i in range(n_systems):
        is_init_boundary = use_init and i == 0
        is_end_boundary = use_end and i == n_systems - 1
        if is_init_boundary:
            fit_nodes = [path_nodes[0]]
        elif is_end_boundary:
            fit_nodes = [path_nodes[-1]]
        else:
            seg_i = i - _init_offset
            fit_nodes = list(intermediate_segments[seg_i])
        fit_x, fit_x_dot = _stack_xy(
            [source_cache[n][0] for n in fit_nodes],
            [source_cache[n][1] for n in fit_nodes],
        )
        edge_fit_points.append(fit_x.copy())
        edge_fit_velocities.append(fit_x_dot.copy())
        edge_direction_stats.append(
            _direction_consistency_stats(
                A=A_seq[i],
                fit_x=fit_x,
                source_state=node_sources[i],
                target_state=node_targets[i],
            )
        )

    chained_ds.edge_fit_points = edge_fit_points
    chained_ds.edge_fit_velocities = edge_fit_velocities
    chained_ds.edge_direction_stats = edge_direction_stats
    chained_ds.edge_forward_fraction = np.array(
        [s.get("frac_forward", np.nan) for s in edge_direction_stats],
        dtype=float,
    )
    chained_ds.edge_forward_min_proj = np.array(
        [s.get("min_proj", np.nan) for s in edge_direction_stats],
        dtype=float,
    )
    chained_ds.transition_ratio_nodes = transition_ratio_nodes
    chained_ds.transition_edge_ratios = transition_edge_ratios
    chained_ds.intermediate_segments = list(intermediate_segments)
    chained_ds.A_init = A_init
    chained_ds.A_attractor = A_attractor
    return chained_ds


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def build_chained_ds(
    ds_set,
    gg,
    initial,
    attractor,
    config: StitchConfig,
    shortest_path_nodes,
    precomputed_edge_lookup: Optional[dict] = None,
):
    method = str(getattr(config.chain, "ds_method", "linear")).strip().lower()
    if method == "linear":
        return build_chained_linear_ds(
            ds_set=ds_set,
            gg=gg,
            initial=initial,
            attractor=attractor,
            config=config,
            shortest_path_nodes=shortest_path_nodes,
            precomputed_edge_lookup=precomputed_edge_lookup,
        )
    if method == "segmented":
        return build_chained_segmented_ds(
            ds_set=ds_set,
            gg=gg,
            initial=initial,
            attractor=attractor,
            config=config,
            shortest_path_nodes=shortest_path_nodes,
            precomputed_edge_lookup=precomputed_edge_lookup,
        )
    raise NotImplementedError(f"Invalid chain.ds_method: {method}")
