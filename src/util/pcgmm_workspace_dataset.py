import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class TaskSpec:
    task_file: str
    start_anchor: np.ndarray
    end_anchor: np.ndarray
    role: str


PCGMM_3D_WORKSPACE_PLAN = (
    TaskSpec(
        task_file="3D_Cshape_top.mat",
        start_anchor=np.array([-1.20, -0.40, 0.25]),
        end_anchor=np.array([-0.25, 0.25, 0.45]),
        role="movement corridor A to central hub",
    ),
    TaskSpec(
        task_file="3D_Cshape_bottom.mat",
        start_anchor=np.array([-1.20, 0.65, 0.85]),
        end_anchor=np.array([-0.25, 0.25, 0.45]),
        role="movement corridor B to central hub",
    ),
    TaskSpec(
        task_file="3D_viapoint_1.mat",
        start_anchor=np.array([-0.25, 0.25, 0.45]),
        end_anchor=np.array([0.55, 0.65, 0.55]),
        role="movement from hub to upper-right workspace",
    ),
    TaskSpec(
        task_file="3D_viapoint_2.mat",
        start_anchor=np.array([-0.25, 0.25, 0.45]),
        end_anchor=np.array([0.70, -0.35, 0.55]),
        role="movement from hub to lower-right workspace",
    ),
    TaskSpec(
        task_file="3D_viapoint_3.mat",
        start_anchor=np.array([0.55, 0.65, 0.55]),
        end_anchor=np.array([0.70, -0.35, 0.55]),
        role="movement bridge upper-right to lower-right",
    ),
    TaskSpec(
        task_file="3D-cube-pick.mat",
        start_anchor=np.array([0.55, 0.65, 0.55]),
        end_anchor=np.array([1.15, 0.95, 0.40]),
        role="pick operation branch from upper-right",
    ),
    TaskSpec(
        task_file="3D-pick-box.mat",
        start_anchor=np.array([0.70, -0.35, 0.55]),
        end_anchor=np.array([1.10, -0.65, 0.95]),
        role="pick-place operation branch from lower-right",
    ),
)


def _resample_positions(positions: np.ndarray, n_points: int) -> np.ndarray:
    if positions.shape[0] == n_points:
        return positions.copy()
    t_src = np.linspace(0.0, 1.0, positions.shape[0])
    t_dst = np.linspace(0.0, 1.0, n_points)
    return np.stack(
        [np.interp(t_dst, t_src, positions[:, dim]) for dim in range(positions.shape[1])],
        axis=1,
    )


def _warp_to_anchors(
    positions: np.ndarray,
    start_anchor: np.ndarray,
    end_anchor: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    rel = positions - positions[0]
    target_delta = end_anchor - start_anchor
    rel_norm = np.linalg.norm(rel[-1])
    target_norm = np.linalg.norm(target_delta)
    scale = 1.0 if rel_norm < eps else target_norm / rel_norm
    warped = start_anchor + rel * scale

    # Affine endpoint correction to enforce exact connectivity between segments.
    correction = np.linspace(0.0, 1.0, warped.shape[0], dtype=warped.dtype)[:, None]
    correction = correction * (end_anchor - warped[-1])
    return warped + correction


def _compute_velocity(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros_like(x)
    x_dot = np.gradient(x, axis=0)
    x_dot[-1] = x_dot[-2]
    return x_dot


def _load_pcgmm_task_positions(task_file: str) -> list[np.ndarray]:
    data = loadmat(task_file)
    if "data" not in data:
        raise ValueError(f"MAT file has no 'data' entry: {task_file}")
    raw = np.asarray(data["data"], dtype=object)

    trajectories = []
    for idx in np.ndindex(raw.shape):
        sample = raw[idx]
        sample_arr = np.asarray(sample, dtype=float)
        if sample_arr.ndim != 2:
            continue
        if sample_arr.shape[0] >= 3:
            x = sample_arr[:3, :].T
        elif sample_arr.shape[1] >= 3:
            x = sample_arr[:, :3]
        else:
            continue
        if x.ndim == 2 and x.shape[0] > 1 and x.shape[1] == 3 and np.all(np.isfinite(x)):
            trajectories.append(x)

    if len(trajectories) == 0:
        raise ValueError(f"No valid 3D trajectories extracted from: {task_file}")
    return trajectories


def _set_equal_3d_axes(ax, points_xyz: np.ndarray):
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return
    mins = np.min(pts[:, :3], axis=0)
    maxs = np.max(pts[:, :3], axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    max_span = float(np.max(spans))
    center = 0.5 * (mins + maxs)
    half = 0.55 * max_span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _visualize_individual_tasks(
    task_data_dir: str,
    task_plan: Iterable[TaskSpec],
    output_path: str,
):
    task_plan = tuple(task_plan)
    n_tasks = len(task_plan)
    ncols = 3
    nrows = int(np.ceil(n_tasks / ncols))
    fig = plt.figure(figsize=(5.5 * ncols, 4.8 * nrows))

    for i, spec in enumerate(task_plan):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        task_file = Path(task_data_dir) / spec.task_file
        trajectories = _load_pcgmm_task_positions(str(task_file))
        all_points = []
        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue", alpha=0.22, linewidth=0.8)
            all_points.append(traj)
        all_points = np.vstack(all_points)
        _set_equal_3d_axes(ax, all_points)
        ax.set_title(spec.task_file.replace(".mat", ""))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.suptitle("PC-GMM Raw Individual 3D Tasks", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _visualize_combined_workspace(
    output_dir: str,
    metadata: dict,
    output_path: str,
):
    root = Path(output_dir)
    demo_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("demonstration_")])
    if len(demo_dirs) == 0:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10", len(demo_dirs))
    all_points = []

    for i, demo_dir in enumerate(demo_dirs):
        color = cmap(i)
        for traj_file in sorted(demo_dir.glob("trajectory_*.json")):
            with open(traj_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            x = np.asarray(payload["x"], dtype=float)
            ax.plot(x[:, 0], x[:, 1], x[:, 2], color=color, alpha=0.35, linewidth=1.1)
            all_points.append(x)

    for i, demo_info in enumerate(metadata["demonstrations"]):
        color = cmap(i)
        start = np.asarray(demo_info["start_anchor"], dtype=float)
        end = np.asarray(demo_info["end_anchor"], dtype=float)
        ax.scatter(start[0], start[1], start[2], color=color, marker="o", s=46, alpha=0.95)
        ax.scatter(end[0], end[1], end[2], color=color, marker="^", s=52, alpha=0.95)

    if len(all_points) > 0:
        _set_equal_3d_axes(ax, np.vstack(all_points))
    ax.set_title("Composed PC-GMM 3D Workspace (Connected Graph)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def build_pcgmm_3d_workspace_dataset(
    output_dir: str,
    task_data_dir: str = "dataset/pc-gmm-data",
    n_trajectories_per_task: int = 4,
    n_points: int = 220,
    seed: int = 13,
    overwrite: bool = True,
    visualize: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for child in output_path.iterdir():
            if child.is_dir() and child.name.startswith("demonstration_"):
                shutil.rmtree(child)
        for file_name in (
            "pcgmm_workspace_plan.json",
            "pcgmm_tasks_individual_3d.png",
            "pcgmm_workspace_combined_3d.png",
        ):
            file_path = output_path / file_name
            if file_path.exists():
                file_path.unlink()

    metadata = {
        "name": "pcgmm_3d_workspace",
        "task_data_dir": str(task_data_dir),
        "n_trajectories_per_task": int(n_trajectories_per_task),
        "n_points": int(n_points),
        "seed": int(seed),
        "demonstrations": [],
        "visualizations": {},
    }

    for demo_idx, task_spec in enumerate(PCGMM_3D_WORKSPACE_PLAN):
        task_file = Path(task_data_dir) / task_spec.task_file
        if not task_file.exists():
            raise FileNotFoundError(f"Missing pc-gmm task file: {task_file}")

        task_trajectories = _load_pcgmm_task_positions(str(task_file))
        n_available = len(task_trajectories)
        n_traj = min(int(n_trajectories_per_task), n_available)
        selected_indices = rng.choice(n_available, size=n_traj, replace=False)

        demo_dir = output_path / f"demonstration_{demo_idx}"
        demo_dir.mkdir(parents=True, exist_ok=True)

        for traj_idx, source_idx in enumerate(selected_indices):
            source_pos = _resample_positions(task_trajectories[int(source_idx)], n_points=n_points)
            warped_pos = _warp_to_anchors(
                source_pos,
                start_anchor=task_spec.start_anchor,
                end_anchor=task_spec.end_anchor,
            )
            warped_vel = _compute_velocity(warped_pos)
            payload = {
                "x": warped_pos.tolist(),
                "x_dot": warped_vel.tolist(),
                "source_task": task_spec.task_file,
                "source_demo_idx": int(source_idx),
                "segment_role": task_spec.role,
            }
            with open(demo_dir / f"trajectory_{traj_idx}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f)

        metadata["demonstrations"].append(
            {
                "demo_dir": str(demo_dir),
                "task_file": task_spec.task_file,
                "role": task_spec.role,
                "start_anchor": task_spec.start_anchor.tolist(),
                "end_anchor": task_spec.end_anchor.tolist(),
                "source_indices": [int(i) for i in selected_indices],
            }
        )

    if visualize:
        individual_plot = output_path / "pcgmm_tasks_individual_3d.png"
        combined_plot = output_path / "pcgmm_workspace_combined_3d.png"
        _visualize_individual_tasks(
            task_data_dir=task_data_dir,
            task_plan=PCGMM_3D_WORKSPACE_PLAN,
            output_path=str(individual_plot),
        )
        _visualize_combined_workspace(
            output_dir=output_dir,
            metadata=metadata,
            output_path=str(combined_plot),
        )
        metadata["visualizations"] = {
            "individual_tasks_3d": str(individual_plot),
            "combined_workspace_3d": str(combined_plot),
        }

    with open(output_path / "pcgmm_workspace_plan.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def build_default_pcgmm_3d_workspace_dataset(
    overwrite: bool = True,
    visualize: bool = True,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "dataset" / "stitching" / "pcgmm_3d_workspace"
    task_data_dir = repo_root / "dataset" / "pc-gmm-data"
    return build_pcgmm_3d_workspace_dataset(
        output_dir=str(output_dir),
        task_data_dir=str(task_data_dir),
        overwrite=overwrite,
        visualize=visualize,
    )
