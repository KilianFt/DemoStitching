import argparse
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import numpy as np

import graph_utils as gu
from main_stitch import Config as StitchConfig
from src.stitching.chaining import build_chained_ds, prepare_chaining_edge_lookup
from src.stitching.ds_stitching import construct_stitched_ds
from src.stitching.graph_paths import shortest_path_nodes
from src.util.benchmarking_tools import initialize_iter_strategy
from src.util.ds_tools import apply_lpvds_demowise, get_gaussian_directions
from src.util.load_tools import get_demonstration_set, resolve_data_scales


@dataclass
class LiveConfig(StitchConfig):
    dt: float = 0.02
    animation_interval_ms: int = 30
    goal_tolerance: float = 0.08
    disturbance_step: float = 0.25
    auto_restart: bool = False
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    figure_width: float = 10.0
    figure_height: float = 10.0
    view_padding_ratio: float = 0.08
    view_padding_abs: float = 0.5


def _predict_velocity_field(ds, points: np.ndarray) -> np.ndarray:
    if hasattr(ds, "predict_velocities"):
        return ds.predict_velocities(points)

    gamma = ds.damm.compute_gamma(points)  # K x M
    vel = np.zeros_like(points)
    for k in range(ds.A.shape[0]):
        vel += gamma[k][:, None] * ((ds.A[k] @ (points - ds.x_att).T).T)
    return vel


def _collect_live_points_2d(demo_set, gaussian_map, start_state: Optional[np.ndarray] = None) -> np.ndarray:
    points = []
    for demo in demo_set:
        for traj in demo.trajectories:
            x = np.asarray(traj.x, dtype=float)
            if x.ndim == 2 and x.shape[0] > 0 and x.shape[1] >= 2:
                points.append(x[:, :2])

    if gaussian_map is not None and len(gaussian_map) > 0:
        mus = []
        for node in gaussian_map.values():
            mu = np.asarray(node["mu"], dtype=float).reshape(-1)
            if mu.shape[0] >= 2 and np.all(np.isfinite(mu[:2])):
                mus.append(mu[:2])
        if len(mus) > 0:
            points.append(np.vstack(mus))

    if start_state is not None:
        start_xy = np.asarray(start_state, dtype=float).reshape(-1)
        if start_xy.shape[0] >= 2 and np.all(np.isfinite(start_xy[:2])):
            points.append(start_xy[:2].reshape(1, 2))

    if len(points) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(points)


def _compute_view_extent(
    points_xy: np.ndarray,
    padding_ratio: float = 0.08,
    padding_abs: float = 0.5,
):
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return -1.0, 1.0, -1.0, 1.0

    pts = pts[:, :2]
    finite_mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite_mask]
    if pts.shape[0] == 0:
        return -1.0, 1.0, -1.0, 1.0

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    max_span = float(np.max(spans))
    margin = max(float(padding_abs), float(padding_ratio) * max_span)
    half_extent = 0.5 * max_span + margin
    center = 0.5 * (mins + maxs)
    return (
        float(center[0] - half_extent),
        float(center[0] + half_extent),
        float(center[1] - half_extent),
        float(center[1] + half_extent),
    )


class LiveStitchController:
    def __init__(self, config: LiveConfig):
        self.config = config
        np.random.seed(config.seed)

        data_position_scale, data_velocity_scale = resolve_data_scales(config)
        self.demo_set = get_demonstration_set(
            config.dataset_path,
            position_scale=data_position_scale,
            velocity_scale=data_velocity_scale,
        )
        self.ds_set, self.reversed_ds_set, self.norm_demo_set = apply_lpvds_demowise(self.demo_set, config)
        self.state_dim = int(self.ds_set[0].x.shape[1])

        self.gaussian_map = {
            (i, j): {"mu": mu, "sigma": sigma, "direction": direction, "prior": prior}
            for i, ds in enumerate(self.ds_set)
            for j, (mu, sigma, direction, prior) in enumerate(
                zip(ds.damm.Mu, ds.damm.Sigma, get_gaussian_directions(ds), ds.damm.Prior)
            )
        }

        self.chain_base_graph = gu.GaussianGraph(
            param_dist=config.param_dist,
            param_cos=config.param_cos,
        )
        self.chain_base_graph.add_gaussians(self.gaussian_map, reverse_gaussians=config.reverse_gaussians)
        self.chain_cfg, self.chain_edge_lookup = prepare_chaining_edge_lookup(
            self.ds_set, self.chain_base_graph, config
        )

        self.start_state = self._select_start_state()
        self.current_state = self.start_state.copy()
        self.goal_state = None

        self.current_ds = None
        self.current_gg = None
        self.current_path_nodes = None
        self.path_anchor_state = self.current_state.copy()
        self.current_chain_idx = None
        self.trajectory = [self.current_state.copy()]

    def _nearest_graph_mu(self, point: np.ndarray) -> np.ndarray:
        centers = np.array([node["mu"] for node in self.gaussian_map.values()])
        idx = int(np.argmin(np.linalg.norm(centers - point.reshape(1, -1), axis=1)))
        return centers[idx]

    def _select_start_state(self) -> np.ndarray:
        combos = initialize_iter_strategy(self.config, self.demo_set)
        candidate = np.asarray(combos[0][0], dtype=float)
        if self.config.start_x is not None and self.config.start_y is not None:
            candidate = np.array([self.config.start_x, self.config.start_y], dtype=float)

        if self.state_dim > 2:
            lift = np.zeros(self.state_dim)
            lift[:2] = candidate[:2]
            candidate = lift
        return self._nearest_graph_mu(candidate)

    def _compose_goal_state(self, goal_xy: np.ndarray) -> np.ndarray:
        goal_xy = np.asarray(goal_xy, dtype=float)
        if self.state_dim <= 2:
            return goal_xy[: self.state_dim]
        goal = self.start_state.copy()
        goal[:2] = goal_xy[:2]
        return goal

    def _build_chain_ds(self, initial: np.ndarray, attractor: np.ndarray):
        gg = gu.GaussianGraph(param_dist=self.config.param_dist, param_cos=self.config.param_cos)
        gg.add_gaussians(self.gaussian_map, reverse_gaussians=self.config.reverse_gaussians)
        path_nodes = shortest_path_nodes(
            gg,
            initial_state=initial,
            target_state=attractor,
            start_node_candidates=getattr(self.config, "chain_start_node_candidates", 1),
            goal_node_candidates=getattr(self.config, "chain_goal_node_candidates", 1),
        )
        if path_nodes is None:
            print("Failed to find path nodes")
            return None, gg, None
        ds = build_chained_ds(
            self.ds_set,
            gg,
            initial=initial,
            attractor=attractor,
            config=self.config,
            precomputed_chain_cfg=self.chain_cfg,
            precomputed_edge_lookup=self.chain_edge_lookup,
            shortest_path_nodes=path_nodes,
        )
        gg.shortest_path_nodes = list(path_nodes)
        return ds, gg, list(path_nodes)

    def _build_other_ds(self, initial: np.ndarray, attractor: np.ndarray):
        try:
            ds, gg, _ = construct_stitched_ds(
                self.config,
                self.norm_demo_set,
                self.ds_set,
                self.reversed_ds_set,
                initial,
                attractor,
            )
        except Exception:
            return None, None, None
        if ds is None or gg is None:
            return None, gg, None
        if hasattr(gg, "shortest_path_nodes"):
            return ds, gg, list(gg.shortest_path_nodes)
        if hasattr(gg, "node_wise_shortest_path"):
            return ds, gg, list(gg.node_wise_shortest_path)
        return ds, gg, None

    def plan_to_goal(
        self,
        goal_xy: np.ndarray,
        keep_trajectory: bool = True,
        initial_override: Optional[np.ndarray] = None,
    ) -> bool:
        attractor = self._compose_goal_state(goal_xy)
        initial = self.current_state.copy() if initial_override is None else np.asarray(initial_override, dtype=float)

        if self.config.ds_method == "chain":
            ds, gg, path_nodes = self._build_chain_ds(initial, attractor)
        else:
            ds, gg, path_nodes = self._build_other_ds(initial, attractor)

        if ds is None:
            print("Failed to build DS")
            return False
        if gg is None:
            print("Failed to build Gaussian Graph")
            return False

        self.goal_state = attractor
        self.current_ds = ds
        self.current_gg = gg
        self.current_path_nodes = path_nodes
        self.path_anchor_state = initial.copy()
        self.current_state = initial.copy()
        if not keep_trajectory:
            self.trajectory = [self.current_state.copy()]
        elif len(self.trajectory) == 0:
            self.trajectory = [self.current_state.copy()]

        if self.config.ds_method == "chain":
            init_idx = int(
                np.argmin(np.linalg.norm(self.current_ds.state_sequence[:-1] - self.current_state.reshape(1, -1), axis=1))
            )
            self.current_ds.reset_runtime(initial_idx=init_idx, start_time=0.0)
            self.current_chain_idx = init_idx
        else:
            self.current_chain_idx = None

        return True

    def reset_to_start(self):
        self.current_state = self.start_state.copy()
        self.trajectory = [self.current_state.copy()]
        if self.goal_state is not None:
            return self.plan_to_goal(
                self.goal_state[:2],
                keep_trajectory=False,
                initial_override=self.start_state.copy(),
            )
        self.current_ds = None
        self.current_gg = None
        self.current_path_nodes = None
        self.path_anchor_state = self.current_state.copy()
        self.current_chain_idx = None
        return True

    def apply_disturbance(self, direction_xy: np.ndarray, magnitude: Optional[float] = None):
        direction_xy = np.asarray(direction_xy, dtype=float)
        direction_norm = np.linalg.norm(direction_xy[:2])
        if direction_norm <= 1e-12:
            return self.current_state.copy()

        step = float(self.config.disturbance_step if magnitude is None else magnitude)
        disturbed_state = self.current_state.copy()
        disturbed_state[:2] += (step / direction_norm) * direction_xy[:2]
        self.current_state = disturbed_state
        self.trajectory.append(self.current_state.copy())

        if self.config.ds_method == "chain" and self.current_ds is not None:
            idx = int(
                np.argmin(
                    np.linalg.norm(
                        self.current_ds.state_sequence[:-1] - self.current_state.reshape(1, -1),
                        axis=1,
                    )
                )
            )
            self.current_chain_idx = idx
        return self.current_state.copy()

    def _chain_step(self):
        x_next, velocity, idx = self.current_ds.step_once(self.current_state, dt=self.config.dt)
        self.current_chain_idx = idx
        return x_next, velocity

    def _generic_step(self):
        velocity = _predict_velocity_field(self.current_ds, self.current_state.reshape(1, -1))[0]
        x_next = self.current_state + self.config.dt * velocity
        return x_next, velocity

    def step_once(self):
        if self.current_ds is None or self.goal_state is None:
            return self.current_state.copy(), np.zeros(self.state_dim), False

        if self.config.ds_method == "chain":
            x_next, velocity = self._chain_step()
        else:
            x_next, velocity = self._generic_step()

        self.current_state = x_next
        self.trajectory.append(self.current_state.copy())
        reached = np.linalg.norm(self.current_state - self.goal_state) <= self.config.goal_tolerance

        if reached and self.config.auto_restart:
            self.current_state = self.start_state.copy()
            self.trajectory = [self.current_state.copy()]
            if self.config.ds_method == "chain":
                init_idx = int(
                    np.argmin(
                        np.linalg.norm(
                            self.current_ds.state_sequence[:-1] - self.current_state.reshape(1, -1), axis=1
                        )
                    )
                )
                self.current_ds.reset_runtime(initial_idx=init_idx, start_time=0.0)
                self.current_chain_idx = init_idx
            reached = False

        return self.current_state.copy(), velocity.copy(), reached

    def path_points_2d(self):
        if self.current_gg is None or self.current_path_nodes is None:
            return None
        points = [np.asarray(self.path_anchor_state, dtype=float)[:2]]
        for node_id in self.current_path_nodes:
            p = self.current_gg.graph.nodes[node_id]["mean"]
            points.append(np.asarray(p, dtype=float)[:2])
        if self.goal_state is not None:
            points.append(np.asarray(self.goal_state, dtype=float)[:2])
        return np.vstack(points)

    def current_chain_direction_stats(self):
        if self.config.ds_method != "chain" or self.current_ds is None:
            return None
        if not hasattr(self.current_ds, "edge_direction_stats"):
            return None
        idx = int(np.clip(self.current_chain_idx, 0, self.current_ds.n_systems - 1))
        if idx >= len(self.current_ds.edge_direction_stats):
            return None
        return self.current_ds.edge_direction_stats[idx]


class LiveStitchApp:
    def __init__(self, controller: LiveStitchController):
        self.ctrl = controller
        self.stream = None
        self.figure, self.ax = plt.subplots(
            1,
            1,
            figsize=(self.ctrl.config.figure_width, self.ctrl.config.figure_height),
        )

        self.traj_line = None
        self.point_artist = None
        self.goal_artist = None
        self.path_line = None
        self.chain_source_artist = None
        self.chain_target_artist = None
        self.chain_fit_points_artist = None
        self.chain_fit_info_text = None
        self.gaussian_center_artist = None

        self.disturbance_keys = {
            "left": np.array([-1.0, 0.0]),
            "right": np.array([1.0, 0.0]),
            "up": np.array([0.0, 1.0]),
            "down": np.array([0.0, -1.0]),
        }

        view_points = _collect_live_points_2d(
            self.ctrl.norm_demo_set,
            self.ctrl.gaussian_map,
            self.ctrl.start_state,
        )
        self.x_min, self.x_max, self.y_min, self.y_max = _compute_view_extent(
            view_points,
            padding_ratio=self.ctrl.config.view_padding_ratio,
            padding_abs=self.ctrl.config.view_padding_abs,
        )

        self._redraw_scene()
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.anim = FuncAnimation(
            self.figure,
            self._animate,
            interval=self.ctrl.config.animation_interval_ms,
            blit=False,
            cache_frame_data=False,
        )

    def _remove_streamplot(self):
        if self.stream is None:
            return
        for artist in (getattr(self.stream, "lines", None), getattr(self.stream, "arrows", None)):
            if artist is None:
                continue
            try:
                artist.remove()
            except Exception:
                # Backends differ in how/if streamplot artists can be detached.
                pass
        self.stream = None

    def _draw_ds_field(self):
        self._remove_streamplot()
        if self.ctrl.current_ds is None:
            return

        plot_sample = 45
        x_mesh, y_mesh = np.meshgrid(
            np.linspace(self.x_min, self.x_max, plot_sample),
            np.linspace(self.y_min, self.y_max, plot_sample),
        )
        xy = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

        dim = self.ctrl.state_dim
        points = np.zeros((xy.shape[0], dim))
        points[:, :2] = xy
        if dim > 2:
            anchor = self.ctrl.current_state.copy()
            for d in range(2, dim):
                points[:, d] = anchor[d]

        if self.ctrl.config.ds_method == "chain":
            idx = int(np.clip(self.ctrl.current_chain_idx, 0, self.ctrl.current_ds.n_systems - 1))
            target = self.ctrl.current_ds.state_sequence[idx + 1]
            velocities = (self.ctrl.current_ds.A_seq[idx] @ (points - target).T).T
        else:
            velocities = _predict_velocity_field(self.ctrl.current_ds, points)

        u = velocities[:, 0].reshape(plot_sample, plot_sample)
        v = velocities[:, 1].reshape(plot_sample, plot_sample)
        self.stream = self.ax.streamplot(
            x_mesh,
            y_mesh,
            u,
            v,
            density=2.2,
            color="black",
            arrowsize=1.0,
            arrowstyle="->",
        )

    def _draw_demos(self):
        demo_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(self.ctrl.norm_demo_set)))
        for i, demo in enumerate(self.ctrl.norm_demo_set):
            for traj in demo.trajectories:
                self.ax.plot(traj.x[:, 0], traj.x[:, 1], color=demo_colors[i], linewidth=1.0, alpha=0.25)

    def _iter_gaussian_params(self):
        if self.ctrl.current_gg is not None:
            for _, node_data in self.ctrl.current_gg.graph.nodes(data=True):
                mu = node_data.get("mean", None)
                sigma = node_data.get("covariance", None)
                if mu is None or sigma is None:
                    continue
                yield np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
            return

        for node in self.ctrl.gaussian_map.values():
            if "mu" not in node or "sigma" not in node:
                continue
            yield np.asarray(node["mu"], dtype=float), np.asarray(node["sigma"], dtype=float)

    def _draw_gaussians(self):
        centers = []
        for mu, sigma in self._iter_gaussian_params():
            if mu.shape[0] < 2 or sigma.shape[0] < 2 or sigma.shape[1] < 2:
                continue
            if not np.all(np.isfinite(mu[:2])) or not np.all(np.isfinite(sigma[:2, :2])):
                continue

            cov_2d = 0.5 * (sigma[:2, :2] + sigma[:2, :2].T)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov_2d)
            except np.linalg.LinAlgError:
                continue
            eigvals = np.maximum(eigvals, 1e-10)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            width = 4.0 * np.sqrt(eigvals[0])  # 2-sigma ellipse width
            height = 4.0 * np.sqrt(eigvals[1])  # 2-sigma ellipse height
            angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
            ellipse = Ellipse(
                xy=mu[:2],
                width=float(width),
                height=float(height),
                angle=angle,
                facecolor="none",
                edgecolor="dimgray",
                linewidth=0.9,
                alpha=0.40,
                zorder=1.5,
            )
            self.ax.add_patch(ellipse)
            centers.append(mu[:2])

        if len(centers) > 0:
            centers = np.vstack(centers)
            self.gaussian_center_artist = self.ax.scatter(
                centers[:, 0],
                centers[:, 1],
                s=12,
                color="dimgray",
                alpha=0.45,
                zorder=1.6,
            )

    def _draw_graph(self):
        if self.ctrl.current_gg is not None:
            self.ctrl.current_gg.plot(ax=self.ax)
        else:
            self.ctrl.chain_base_graph.plot(ax=self.ax)

    def _draw_chain_markers(self):
        if self.ctrl.config.ds_method != "chain" or self.ctrl.current_ds is None:
            return
        idx = int(np.clip(self.ctrl.current_chain_idx, 0, self.ctrl.current_ds.n_systems - 1))
        source = self.ctrl.current_ds.state_sequence[idx][:2]
        target = self.ctrl.current_ds.state_sequence[idx + 1][:2]
        self.chain_source_artist, = self.ax.plot(source[0], source[1], "o", color="orange", markersize=9)
        self.chain_target_artist, = self.ax.plot(target[0], target[1], "o", color="red", markersize=9)

    def _draw_chain_fit_points(self):
        if self.ctrl.config.ds_method != "chain" or self.ctrl.current_ds is None:
            return
        if not hasattr(self.ctrl.current_ds, "edge_fit_points"):
            return

        idx = int(np.clip(self.ctrl.current_chain_idx, 0, self.ctrl.current_ds.n_systems - 1))
        fit_points_list = self.ctrl.current_ds.edge_fit_points
        if idx >= len(fit_points_list):
            return

        points = np.asarray(fit_points_list[idx], dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            return

        self.chain_fit_points_artist = self.ax.scatter(
            points[:, 0],
            points[:, 1],
            s=12,
            color="deepskyblue",
            alpha=0.4,
            edgecolors="none",
        )

        stats = None
        if hasattr(self.ctrl.current_ds, "edge_direction_stats") and idx < len(self.ctrl.current_ds.edge_direction_stats):
            stats = self.ctrl.current_ds.edge_direction_stats[idx]
        if isinstance(stats, dict):
            n_points = int(stats.get("n_points", points.shape[0]))
            frac = float(stats.get("frac_forward", np.nan))
            min_proj = float(stats.get("min_proj", np.nan))
            if np.isfinite(frac) and np.isfinite(min_proj):
                label = f"A[{idx}] fit n={n_points} | forward={frac:.2f} | min proj={min_proj:.3f}"
            else:
                label = f"A[{idx}] fit n={n_points}"
        else:
            label = f"A[{idx}] fit n={points.shape[0]}"

        self.chain_fit_info_text = self.ax.text(
            0.02,
            0.98,
            label,
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    def _update_chain_markers(self):
        if self.ctrl.config.ds_method != "chain" or self.ctrl.current_ds is None:
            return
        idx = int(np.clip(self.ctrl.current_chain_idx, 0, self.ctrl.current_ds.n_systems - 1))
        source = self.ctrl.current_ds.state_sequence[idx][:2]
        target = self.ctrl.current_ds.state_sequence[idx + 1][:2]
        if self.chain_source_artist is not None:
            self.chain_source_artist.set_data([source[0]], [source[1]])
        if self.chain_target_artist is not None:
            self.chain_target_artist.set_data([target[0]], [target[1]])

    def _redraw_scene(self):
        self.stream = None
        self.ax.clear()
        self._draw_ds_field()
        self._draw_gaussians()
        self._draw_demos()
        self._draw_graph()

        if self.ctrl.goal_state is not None:
            self.goal_artist, = self.ax.plot(
                self.ctrl.goal_state[0], self.ctrl.goal_state[1], "o", color="red", markersize=9
            )
        self.ax.plot(self.ctrl.start_state[0], self.ctrl.start_state[1], "o", color="gold", markersize=9)

        path = self.ctrl.path_points_2d()
        if path is not None:
            self.path_line, = self.ax.plot(path[:, 0], path[:, 1], "--", color="magenta", linewidth=2.5, alpha=0.5)

        traj = np.array(self.ctrl.trajectory)
        self.traj_line, = self.ax.plot(traj[:, 0], traj[:, 1], color="magenta", linewidth=3.0)
        self.point_artist, = self.ax.plot(self.ctrl.current_state[0], self.ctrl.current_state[1], "ko", markersize=6)
        self._draw_chain_markers()
        self._draw_chain_fit_points()

        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"Live Stitch ({self.ctrl.config.ds_method}) | click: new goal | arrows: disturb | r: reset | cyan: fit points"
        )
        self.ax.grid(alpha=0.25)
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        goal = np.array([event.xdata, event.ydata], dtype=float)
        success = self.ctrl.plan_to_goal(goal)
        if not success:
            print("No path/DS could be constructed for clicked goal.")
            return
        self._redraw_scene()
        print(
            f"New goal: [{goal[0]:.3f}, {goal[1]:.3f}] | "
            f"path length: {len(self.ctrl.current_path_nodes) if self.ctrl.current_path_nodes is not None else 0}"
        )
        stats = self.ctrl.current_chain_direction_stats()
        if isinstance(stats, dict):
            frac = stats.get("frac_forward", np.nan)
            min_proj = stats.get("min_proj", np.nan)
            n_points = int(stats.get("n_points", 0))
            if np.isfinite(frac) and np.isfinite(min_proj):
                print(
                    f"Active edge fit stats: n={n_points}, "
                    f"forward_ratio={frac:.2f}, min_proj={min_proj:.3f}"
                )

    def _on_key_press(self, event):
        key = event.key.lower() if isinstance(event.key, str) else event.key
        if key == "r":
            self.ctrl.reset_to_start()
            self._redraw_scene()
            print("Reset to start.")
            return
        if key in self.disturbance_keys:
            disturbed = self.ctrl.apply_disturbance(self.disturbance_keys[key])
            self._redraw_scene()
            print(f"Disturbance {key}: [{disturbed[0]:.3f}, {disturbed[1]:.3f}]")

    def _animate(self, _frame):
        if self.ctrl.current_ds is None:
            return []

        prev_chain_idx = self.ctrl.current_chain_idx
        self.ctrl.step_once()
        traj = np.array(self.ctrl.trajectory)
        self.traj_line.set_data(traj[:, 0], traj[:, 1])
        self.point_artist.set_data([self.ctrl.current_state[0]], [self.ctrl.current_state[1]])

        if self.ctrl.config.ds_method == "chain" and self.ctrl.current_chain_idx != prev_chain_idx:
            stats = self.ctrl.current_chain_direction_stats()
            if isinstance(stats, dict):
                frac = stats.get("frac_forward", np.nan)
                min_proj = stats.get("min_proj", np.nan)
                n_points = int(stats.get("n_points", 0))
                if np.isfinite(frac) and np.isfinite(min_proj):
                    print(
                        f"Switched to edge {self.ctrl.current_chain_idx}: "
                        f"n={n_points}, forward_ratio={frac:.2f}, min_proj={min_proj:.3f}"
                    )
            self._redraw_scene()
            return []

        return []

    def run(self):
        plt.show()


def run_headless_smoke(config: LiveConfig, goal_xy: np.ndarray, n_steps: int = 500):
    controller = LiveStitchController(config)
    if not controller.plan_to_goal(goal_xy):
        raise RuntimeError("Headless smoke failed to construct DS.")
    for _ in range(n_steps):
        controller.step_once()
    return np.linalg.norm(controller.current_state - controller.goal_state), controller


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive live stitching with click-to-goal replanning.")
    parser.add_argument("--dataset-path", type=str, default="dataset/stitching/robottasks_workspace_chain")
    parser.add_argument("--ds-method", type=str, default="chain")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--interval-ms", type=int, default=30)
    parser.add_argument("--data-position-scale", "--damm-position-scale", dest="data_position_scale", type=float, default=None)
    parser.add_argument("--data-velocity-scale", "--damm-velocity-scale", dest="data_velocity_scale", type=float, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=0.08)
    parser.add_argument("--disturbance-step", type=float, default=0.25)
    parser.add_argument("--chain-start-node-candidates", type=int, default=None)
    parser.add_argument("--chain-goal-node-candidates", type=int, default=None)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--figure-width", type=float, default=10.0)
    parser.add_argument("--figure-height", type=float, default=10.0)
    parser.add_argument("--view-padding-ratio", type=float, default=0.08)
    parser.add_argument("--view-padding-abs", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-restart", action="store_true")
    parser.add_argument("--headless-smoke", action="store_true")
    parser.add_argument("--smoke-goal-x", type=float, default=None)
    parser.add_argument("--smoke-goal-y", type=float, default=None)
    parser.add_argument("--smoke-steps", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    config = LiveConfig()
    config.dataset_path = args.dataset_path
    config.ds_method = args.ds_method
    config.dt = args.dt
    config.animation_interval_ms = args.interval_ms
    if args.data_position_scale is not None:
        config.data_position_scale = args.data_position_scale
    config.data_velocity_scale = args.data_velocity_scale
    config.goal_tolerance = args.goal_tolerance
    config.disturbance_step = args.disturbance_step
    if args.chain_start_node_candidates is not None:
        config.chain_start_node_candidates = args.chain_start_node_candidates
    if args.chain_goal_node_candidates is not None:
        config.chain_goal_node_candidates = args.chain_goal_node_candidates
    config.start_x = args.start_x
    config.start_y = args.start_y
    config.figure_width = args.figure_width
    config.figure_height = args.figure_height
    config.view_padding_ratio = args.view_padding_ratio
    config.view_padding_abs = args.view_padding_abs
    config.seed = args.seed
    config.auto_restart = args.auto_restart

    if args.headless_smoke:
        if args.smoke_goal_x is None or args.smoke_goal_y is None:
            raise ValueError("Headless smoke requires --smoke-goal-x and --smoke-goal-y.")
        final_distance, controller = run_headless_smoke(
            config=config,
            goal_xy=np.array([args.smoke_goal_x, args.smoke_goal_y], dtype=float),
            n_steps=args.smoke_steps,
        )
        print(f"Headless smoke complete. Final distance to goal: {final_distance:.6f}")
        print(f"Current path nodes: {len(controller.current_path_nodes) if controller.current_path_nodes is not None else 0}")
        return

    app = LiveStitchApp(LiveStitchController(config))
    app.run()


if __name__ == "__main__":
    main()
