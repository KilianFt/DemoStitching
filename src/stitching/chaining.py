from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal
from src.damm.src.damm_class import DAMM as damm_class


def _stable_matrix(A: np.ndarray, margin: float = 1e-3) -> np.ndarray:
    """Project A so its symmetric part is strictly negative definite."""
    symmetric = 0.5 * (A + A.T)
    skew = 0.5 * (A - A.T)
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    eigvals = np.minimum(eigvals, -abs(margin))
    symmetric_stable = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return symmetric_stable + skew


def _fit_linear_system(
    x: np.ndarray,
    x_dot: np.ndarray,
    target: np.ndarray,
    regularization: float = 1e-4,
) -> Optional[np.ndarray]:
    """Fit y = A (x - target) in least-squares sense."""
    if x.shape[0] < x.shape[1]:
        return None

    X = x - target.reshape(1, -1)
    Y = x_dot
    lhs = X.T @ X + regularization * np.eye(X.shape[1])
    rhs = X.T @ Y
    try:
        B = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    return B.T


@dataclass
class _ChainConfig:
    trigger_radius: float = 0.12
    trigger_radius_scale: float = 0.0
    trigger_radius_min: float = 0.05
    trigger_radius_max: float = 0.35
    transition_time: float = 0.18
    recovery_distance: float = 0.35
    enable_recovery: bool = True
    stabilization_margin: float = 1e-3
    fit_regularization: float = 1e-4
    fit_blend: float = 0.0


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
        self.A = self.A_seq[1:].copy() if self.n_systems > 1 else self.A_seq.copy()
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
    return assigned_x, assigned_x_dot, base_A


def _build_edge_lookup(ds_set, gg, cfg: _ChainConfig):
    """Pre-compute DS matrices for gaussian->gaussian edges (path-independent cache)."""
    edge_lookup = {}
    gaussian_id_set = set(gg.gaussian_ids)
    source_cache = {}

    for source in gg.gaussian_ids:
        source_cache[source] = _get_source_data_for_node(ds_set, gg, source)

    for source in gg.gaussian_ids:
        source_x, source_x_dot, base_A = source_cache[source]
        for target in gg.graph.successors(source):
            if target not in gaussian_id_set:
                continue
            target_state = np.asarray(gg.graph.nodes[target]["mean"], dtype=float)
            fitted_A = _fit_linear_system(
                source_x,
                source_x_dot,
                target=target_state,
                regularization=cfg.fit_regularization,
            )
            if fitted_A is None:
                A_edge = base_A
            else:
                A_edge = cfg.fit_blend * base_A + (1.0 - cfg.fit_blend) * fitted_A
            edge_lookup[(source, target)] = _stable_matrix(A_edge, margin=cfg.stabilization_margin)
    return edge_lookup


def _resolve_chain_config(config) -> _ChainConfig:
    # Backward compatible field mapping.
    trigger_radius = float(
        getattr(
            config,
            "chain_trigger_radius",
            getattr(config, "chain_switch_threshold", 0.12),
        )
    )
    transition_time = float(
        getattr(
            config,
            "chain_transition_time",
            getattr(config, "chain_blend_width", 0.18),
        )
    )
    return _ChainConfig(
        trigger_radius=trigger_radius,
        trigger_radius_scale=float(getattr(config, "chain_trigger_radius_scale", 0.0)),
        trigger_radius_min=float(getattr(config, "chain_trigger_radius_min", 0.05)),
        trigger_radius_max=float(getattr(config, "chain_trigger_radius_max", 0.35)),
        transition_time=transition_time,
        recovery_distance=float(getattr(config, "chain_recovery_distance", 0.35)),
        enable_recovery=bool(getattr(config, "chain_enable_recovery", True)),
        stabilization_margin=float(getattr(config, "chain_stabilization_margin", 1e-3)),
        fit_regularization=float(getattr(config, "chain_fit_regularization", 1e-4)),
        fit_blend=float(getattr(config, "chain_fit_blend", 0.0)),
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
    radii = cfg.trigger_radius + cfg.trigger_radius_scale * edge_lengths
    radii = np.clip(radii, cfg.trigger_radius_min, cfg.trigger_radius_max)
    return radii


def build_chained_ds(ds_set, gg, gg_solution_nodes, initial: np.ndarray, attractor: np.ndarray, config) -> Optional[ChainedLinearDS]:

    path_nodes = gg_solution_nodes
    if len(path_nodes) < 2:
        return None

    cfg = _resolve_chain_config(config)
    state_sequence = _resolve_state_sequence(gg, path_nodes, initial=initial, attractor=attractor)
    n_systems = state_sequence.shape[0] - 1
    if n_systems <= 0:
        return None

    edge_lookup = _build_edge_lookup(ds_set, gg, cfg=cfg)
    gaussian_id_set = set(gg.gaussian_ids)

    # Build f_1 ... f_N, each stabilizing towards mu_{i+1}.
    A_seq = []
    source_cache = {}
    for node in gg.gaussian_ids:
        source_cache[node] = _get_source_data_for_node(ds_set, gg, node)

    for i in range(n_systems):
        current_node = path_nodes[i]
        next_node = path_nodes[i + 1]
        target_state = state_sequence[i + 1]

        use_lookup = (
            i > 0
            and i < n_systems - 1
            and current_node in gaussian_id_set
            and next_node in gaussian_id_set
            and (current_node, next_node) in edge_lookup
        )
        if use_lookup:
            A_i = edge_lookup[(current_node, next_node)]
        else:
            if current_node in gaussian_id_set:
                source_node = current_node
            elif next_node in gaussian_id_set:
                source_node = next_node
            else:
                source_node = None

            if source_node is None:
                dim = state_sequence.shape[1]
                A_i = -np.eye(dim)
            else:
                assigned_x, assigned_x_dot, base_A = source_cache[source_node]
                fitted_A = _fit_linear_system(
                    assigned_x,
                    assigned_x_dot,
                    target=target_state,
                    regularization=cfg.fit_regularization,
                )
                if fitted_A is None:
                    A_i = base_A
                else:
                    A_i = cfg.fit_blend * base_A + (1.0 - cfg.fit_blend) * fitted_A
                A_i = _stable_matrix(A_i, margin=cfg.stabilization_margin)

        A_seq.append(A_i)
    A_seq = np.array(A_seq)

    trigger_radii = _resolve_trigger_radii(state_sequence, cfg=cfg)
    transition_times = np.full(max(n_systems - 1, 0), cfg.transition_time, dtype=float)

    gaussian_path_nodes = [n for n in path_nodes if n in gaussian_id_set]
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
            assigned_x, assigned_x_dot, _ = source_cache[node_id]
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
    return chained_ds
