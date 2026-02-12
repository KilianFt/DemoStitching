from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal
from src.damm.src.damm_class import DAMM as damm_class
from src.stitching.graph_paths import shortest_path_nodes as _safe_shortest_path_nodes
import cvxpy as cp


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
        P = 0.5 * (P + P.T)
        try:
            if np.min(np.linalg.eigvalsh(P)) <= 1e-8:
                P = np.eye(dim)
        except np.linalg.LinAlgError:
            P = np.eye(dim)

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


@dataclass
class _ChainConfig:
    trigger_radius_ratio: float = 0.10
    transition_time: float = 0.18
    recovery_distance: float = 0.35
    enable_recovery: bool = True
    stabilization_margin: float = 1e-3
    lmi_tolerance: float = 5e-5
    edge_data_mode: str = "both_all"


class ChainedLinearDS:
    """DS chain with state-triggered entry and time-triggered transition completion."""

    def __init__(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        attractor: np.ndarray,
        path_nodes,
        state_sequence: np.ndarray,
        A_seq: np.ndarray,
        damm,
        trigger_radii: np.ndarray,
        transition_times: np.ndarray,
        chain_cfg: _ChainConfig,
    ) -> None:
        self.x = x
        self.x_dot = x_dot
        self.x_att = np.asarray(attractor, dtype=float)

        # n_1 ... n_{N+1}; each f_i stabilizes towards mu_{i+1}
        self.path_nodes = list(path_nodes)
        self.state_sequence = np.asarray(state_sequence, dtype=float)  # (N+1, d)
        self.A_seq = np.asarray(A_seq, dtype=float)  # (N, d, d)
        self.n_systems = self.A_seq.shape[0]

        self.trigger_radii = np.asarray(trigger_radii, dtype=float)
        self.transition_times = np.asarray(transition_times, dtype=float)
        self.chain_cfg = chain_cfg

        # Compatibility attributes used by existing plotting/eval code.
        self.damm = damm
        self.K = self.damm.K
        self.node_means = self.state_sequence[:-1]
        self.node_targets = self.state_sequence[1:]
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
        self._transition_from_idx = None
        self._transition_t0 = None

    def _state_vec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return x[0]
        return x

    def _velocity_for_index(self, x: np.ndarray, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.n_systems - 1))
        return self.A_seq[idx] @ (x - self.state_sequence[idx + 1])

    def trigger_state(self, idx: int, x: np.ndarray) -> bool:
        idx = int(np.clip(idx, 0, self.n_systems - 1))
        return np.linalg.norm(x - self.state_sequence[idx + 1]) <= self.trigger_radii[idx]

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
        dist_to_current = np.linalg.norm(x - self.state_sequence[current_idx])
        if dist_to_current <= self.chain_cfg.recovery_distance:
            return current_idx

        distances = np.linalg.norm(self.state_sequence[:-1] - x.reshape(1, -1), axis=1)
        return int(np.argmin(distances))

    def reset_runtime(self, initial_idx: int = 0, start_time: float = 0.0):
        self._runtime_idx = int(np.clip(initial_idx, 0, self.n_systems - 1))
        self._runtime_time = float(start_time)
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
        if self.trigger_state(self._runtime_idx, x):
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
            self._clear_transition()
        return v

    def step_once(self, x: np.ndarray, dt: float, current_idx: Optional[int] = None, current_time: Optional[float] = None):
        x = self._state_vec(x)
        t = self._runtime_time if current_time is None else float(current_time)

        if current_idx is not None and int(current_idx) != self._runtime_idx:
            self._runtime_idx = int(np.clip(current_idx, 0, self.n_systems - 1))
            self._clear_transition()

        # Disturbance recovery: jump to nearest source state.
        recovered_idx = self.select_node_index(x, current_idx=self._runtime_idx)
        if recovered_idx != self._runtime_idx:
            self._runtime_idx = recovered_idx
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
        init_idx = int(np.argmin(np.linalg.norm(self.state_sequence[:-1] - x.reshape(1, -1), axis=1)))
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
            idx = int(np.argmin(np.linalg.norm(self.state_sequence[:-1] - x.reshape(1, -1), axis=1)))
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


def _edge_fit_data(
    source_x: np.ndarray,
    source_x_dot: np.ndarray,
    target_x: np.ndarray,
    target_x_dot: np.ndarray,
    source_state: np.ndarray,
    target_state: np.ndarray,
    mode: str,
):
    mode = str(mode).lower()
    x_all, x_dot_all = _stack_xy([source_x, target_x], [source_x_dot, target_x_dot])
    if x_all.size == 0:
        return x_all, x_dot_all

    if mode in {"both_all", "all"}:
        return x_all, x_dot_all

    if mode in {"between_orthogonals", "segment"}:
        edge = np.asarray(target_state, dtype=float) - np.asarray(source_state, dtype=float)
        edge_len = np.linalg.norm(edge)
        if edge_len <= 1e-12:
            return x_all, x_dot_all

        edge_dir = edge / edge_len
        proj_pos = (x_all - source_state.reshape(1, -1)) @ edge_dir
        between_planes = (proj_pos >= 0.0) & (proj_pos <= edge_len)
        # Prefer pointwise progress-to-target over raw edge-axis projection.
        toward_next = np.sum((target_state.reshape(1, -1) - x_all) * x_dot_all, axis=1) > 0.0
        mask = between_planes & toward_next

        if np.any(mask):
            return x_all[mask], x_dot_all[mask]
        return np.zeros((0, x_all.shape[1]), dtype=float), np.zeros((0, x_dot_all.shape[1]), dtype=float)

    raise ValueError(f"Unsupported chain edge_data_mode: {mode}")


def _edge_fit_samples(
    source_data,
    target_data,
    source_state: np.ndarray,
    target_state: np.ndarray,
    cfg: _ChainConfig,
):
    source_x, source_x_dot, _, _ = source_data
    if target_data is None:
        mode = str(cfg.edge_data_mode).lower()
        if mode in {"between_orthogonals", "segment"}:
            edge = np.asarray(target_state, dtype=float) - np.asarray(source_state, dtype=float)
            edge_len = np.linalg.norm(edge)
            if edge_len <= 1e-12:
                # Degenerate segment: keep source-node data instead of dropping everything.
                return source_x, source_x_dot
            edge_dir = edge / edge_len
            proj_pos = (source_x - np.asarray(source_state, dtype=float).reshape(1, -1)) @ edge_dir
            between_planes = (proj_pos >= 0.0) & (proj_pos <= edge_len)
            toward_next = np.sum((np.asarray(target_state, dtype=float).reshape(1, -1) - source_x) * source_x_dot, axis=1) > 0.0
            mask = between_planes & toward_next
            return source_x[mask], source_x_dot[mask]
        return source_x, source_x_dot
    target_x, target_x_dot, _, _ = target_data
    return _edge_fit_data(
        source_x=source_x,
        source_x_dot=source_x_dot,
        target_x=target_x,
        target_x_dot=target_x_dot,
        source_state=source_state,
        target_state=target_state,
        mode=cfg.edge_data_mode,
    )


def _fit_edge_matrix(
    source_data,
    target_data,
    source_state: np.ndarray,
    target_state: np.ndarray,
    cfg: _ChainConfig,
):
    source_x, source_x_dot, _, _ = source_data
    fit_x, fit_x_dot = _edge_fit_samples(
        source_data=source_data,
        target_data=target_data,
        source_state=source_state,
        target_state=target_state,
        cfg=cfg,
    )
    fitted_A = _fit_linear_system(
        fit_x,
        fit_x_dot,
        target=target_state,
        stabilization_margin=cfg.stabilization_margin,
        lmi_tolerance=cfg.lmi_tolerance,
    )
    if fitted_A is None:
        return None, fit_x, fit_x_dot
    return fitted_A, fit_x, fit_x_dot


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


def _build_edge_lookup(ds_set, gg, cfg: _ChainConfig):
    """Pre-compute DS matrices for gaussian->gaussian edges (path-independent cache)."""
    edge_lookup = {}
    gaussian_ids = _graph_gaussian_ids(gg)
    gaussian_id_set = set(gaussian_ids)
    source_cache = {}

    for source in gaussian_ids:
        source_cache[source] = _get_source_data_for_node(ds_set, gg, source)

    for source in gaussian_ids:
        source_data = source_cache[source]
        source_state = np.asarray(gg.graph.nodes[source]["mean"], dtype=float)
        for target in gg.graph.successors(source):
            if target not in gaussian_id_set:
                continue
            target_data = source_cache[target]
            target_state = np.asarray(gg.graph.nodes[target]["mean"], dtype=float)
            A_edge, _, _ = _fit_edge_matrix(
                source_data=source_data,
                target_data=target_data,
                source_state=source_state,
                target_state=target_state,
                cfg=cfg,
            )
            if A_edge is not None:
                edge_lookup[(source, target)] = A_edge
    return edge_lookup


def _resolve_chain_config(config) -> _ChainConfig:
    # chain_trigger_radius is interpreted as percentage of edge length.
    trigger_radius_ratio = float(
        getattr(
            config,
            "chain_trigger_radius",
            getattr(config, "chain_switch_threshold", 0.10),
        )
    )
    if not np.isfinite(trigger_radius_ratio) or trigger_radius_ratio <= 0.0:
        trigger_radius_ratio = 0.10
    transition_time = float(
        getattr(
            config,
            "chain_transition_time",
            getattr(config, "chain_blend_width", 0.18),
        )
    )
    return _ChainConfig(
        trigger_radius_ratio=trigger_radius_ratio,
        transition_time=transition_time,
        recovery_distance=float(getattr(config, "chain_recovery_distance", 0.35)),
        enable_recovery=bool(getattr(config, "chain_enable_recovery", True)),
        stabilization_margin=float(getattr(config, "chain_stabilization_margin", 1e-3)),
        lmi_tolerance=float(getattr(config, "chain_lmi_tolerance", 5e-5)),
        edge_data_mode=str(getattr(config, "chain_edge_data_mode", "both_all")),
    )


def _resolve_state_sequence(gg, path_nodes, initial: np.ndarray, attractor: np.ndarray) -> np.ndarray:
    states = []
    for node in path_nodes:
        if node == "initial":
            states.append(np.asarray(initial, dtype=float))
        elif node == "attractor":
            states.append(np.asarray(attractor, dtype=float))
        else:
            states.append(np.asarray(gg.graph.nodes[node]["mean"], dtype=float))
    return np.vstack(states)


def _resolve_trigger_radii(state_sequence: np.ndarray, cfg: _ChainConfig) -> np.ndarray:
    edge_lengths = np.linalg.norm(np.diff(state_sequence, axis=0), axis=1)
    radii = cfg.trigger_radius_ratio * edge_lengths
    radii = np.maximum(radii, 1e-12)
    return radii


def prepare_chaining_edge_lookup(ds_set, gg, config):
    """Prepare reusable chain config + edge lookup for repeated replanning."""
    cfg = _resolve_chain_config(config)
    edge_lookup = _build_edge_lookup(ds_set, gg, cfg=cfg)
    return cfg, edge_lookup


def build_chained_ds(
    ds_set,
    gg,
    initial: np.ndarray,
    attractor: np.ndarray,
    config,
    precomputed_chain_cfg: Optional[_ChainConfig] = None,
    precomputed_edge_lookup: Optional[dict] = None,
    shortest_path_nodes: Optional[list] = None,
) -> Optional[ChainedLinearDS]:
    if shortest_path_nodes is None:
        shortest_path_attr = getattr(gg, "shortest_path", None)
        if callable(shortest_path_attr):
            shortest_path_nodes = _safe_shortest_path_nodes(gg, initial_state=initial, target_state=attractor)
        elif shortest_path_attr is not None:
            shortest_path_nodes = list(shortest_path_attr)
            if len(shortest_path_nodes) > 0 and shortest_path_nodes[0] == "initial":
                shortest_path_nodes = shortest_path_nodes[1:-1]
        else:
            shortest_path_nodes = None
    if shortest_path_nodes is None or len(shortest_path_nodes) == 0:
        return None
    shortest_path_nodes = list(shortest_path_nodes)
    # Build DSs over graph nodes directly so the first fitted edge uses start-node gaussian data.
    path_nodes = [*shortest_path_nodes, "attractor"]

    cfg = precomputed_chain_cfg if precomputed_chain_cfg is not None else _resolve_chain_config(config)
    state_sequence = _resolve_state_sequence(gg, path_nodes, initial=initial, attractor=attractor)
    n_systems = state_sequence.shape[0] - 1
    if n_systems <= 0:
        return None

    edge_lookup = precomputed_edge_lookup if precomputed_edge_lookup is not None else _build_edge_lookup(ds_set, gg, cfg=cfg)
    gaussian_ids = _graph_gaussian_ids(gg)
    gaussian_id_set = set(gaussian_ids)

    # Build f_1 ... f_N, each stabilizing towards mu_{i+1}.
    A_seq = []
    fit_points_seq = []
    fit_velocities_seq = []
    direction_stats_seq = []
    source_cache = {}
    for node in gaussian_ids:
        source_cache[node] = _get_source_data_for_node(ds_set, gg, node)

    for i in range(n_systems):
        current_node = path_nodes[i]
        next_node = path_nodes[i + 1]
        source_state = state_sequence[i]
        target_state = state_sequence[i + 1]
        fit_source_state = source_state
        fit_target_state = target_state
        if i == 0:
            # First edge should span from the actual initial state to the next node.
            fit_source_state = np.asarray(initial, dtype=float)
        dim = state_sequence.shape[1]
        fit_x = np.zeros((0, dim), dtype=float)
        fit_x_dot = np.zeros((0, dim), dtype=float)

        use_lookup = (
            i > 0
            and i < n_systems - 1
            and current_node in gaussian_id_set
            and next_node in gaussian_id_set
            and (current_node, next_node) in edge_lookup
        )
        if use_lookup:
            fit_x, fit_x_dot = _edge_fit_samples(
                source_data=source_cache[current_node],
                target_data=source_cache[next_node],
                source_state=fit_source_state,
                target_state=fit_target_state,
                cfg=cfg,
            )
            A_i = edge_lookup[(current_node, next_node)]
        else:
            if current_node in gaussian_id_set:
                source_node = current_node
            elif next_node in gaussian_id_set:
                source_node = next_node
            else:
                source_node = None

            if source_node is None:
                return None

            target_data = source_cache[next_node] if (current_node in gaussian_id_set and next_node in gaussian_id_set) else None
            A_i, fit_x, fit_x_dot = _fit_edge_matrix(
                source_data=source_cache[source_node],
                target_data=target_data,
                source_state=fit_source_state,
                target_state=fit_target_state,
                cfg=cfg,
            )
            if A_i is None:
                return None

        A_seq.append(A_i)
        fit_points_seq.append(np.asarray(fit_x, dtype=float).copy())
        fit_velocities_seq.append(np.asarray(fit_x_dot, dtype=float).copy())
        direction_stats_seq.append(
            _direction_consistency_stats(
                A=A_i,
                fit_x=fit_x,
                source_state=source_state,
                target_state=target_state,
            )
        )
    A_seq = np.array(A_seq)

    trigger_radii = _resolve_trigger_radii(state_sequence, cfg=cfg)
    transition_times = np.full(max(n_systems - 1, 0), cfg.transition_time, dtype=float)

    gaussian_path_nodes = [n for n in shortest_path_nodes if n in gaussian_id_set]
    if len(gaussian_path_nodes) == 0:
        # Fallback for degenerate graphs.
        gaussian_lists = [{
            "prior": 1.0,
            "mu": np.asarray(attractor, dtype=float),
            "sigma": np.eye(len(attractor)) * 0.01,
            "rv": multivariate_normal(np.asarray(attractor, dtype=float), np.eye(len(attractor)) * 0.01, allow_singular=True),
        }]
        filtered_x = np.asarray(initial, dtype=float).reshape(1, -1)
        filtered_x_dot = (np.asarray(attractor, dtype=float) - np.asarray(initial, dtype=float)).reshape(1, -1)
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
        filtered_x = np.vstack(filtered_x)
        filtered_x_dot = np.vstack(filtered_x_dot)

    # Reuse the existing DAMM Gaussian machinery (no custom GMM wrapper).
    x_proc, x_dot_proc, x_dir = damm_class.pre_process(filtered_x, filtered_x_dot)
    if x_proc.shape[0] == 0:
        x_proc = filtered_x.copy()
        x_dir = np.zeros_like(x_proc)
        x_dir[:, 0] = 1.0

    damm = damm_class(
        x_proc,
        x_dir,
        nu_0=getattr(config, "nu_0", 5),
        kappa_0=getattr(config, "kappa_0", 1),
        psi_dir_0=getattr(config, "psi_dir_0", 1),
        rel_scale=getattr(config, "rel_scale", 0.7),
        total_scale=getattr(config, "total_scale", 1.5),
    )
    damm.K = len(gaussian_lists)
    damm.gaussian_lists = gaussian_lists
    damm.Mu = np.array([g["mu"] for g in gaussian_lists])
    damm.Sigma = np.array([g["sigma"] for g in gaussian_lists])
    damm.Prior = np.array([g["prior"] for g in gaussian_lists])

    chained_ds = ChainedLinearDS(
        x=filtered_x,
        x_dot=filtered_x_dot,
        attractor=np.asarray(attractor, dtype=float),
        path_nodes=path_nodes,
        state_sequence=state_sequence,
        A_seq=A_seq,
        damm=damm,
        trigger_radii=trigger_radii,
        transition_times=transition_times,
        chain_cfg=cfg,
    )
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
    return chained_ds
