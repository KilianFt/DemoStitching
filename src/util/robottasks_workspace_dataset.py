import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    start_anchor: np.ndarray
    end_anchor: np.ndarray
    role: str


DEFAULT_TASK_PLAN = (
    TaskSpec(
        task_name="obstaclerotate",
        start_anchor=np.array([-0.55, 0.02, 0.24]),
        end_anchor=np.array([0.20, 0.05, 0.24]),
        role="central transit corridor",
    ),
    TaskSpec(
        task_name="pouring",
        start_anchor=np.array([-0.55, 0.02, 0.24]),
        end_anchor=np.array([-0.10, 0.28, 0.18]),
        role="branch starting at obstacle entry",
    ),
    TaskSpec(
        task_name="pan2stove",
        start_anchor=np.array([0.20, 0.05, 0.24]),
        end_anchor=np.array([0.88, 0.42, 0.46]),
        role="branch starting at obstacle exit",
    ),
    TaskSpec(
        task_name="openbox",
        start_anchor=np.array([-0.175, 0.035, 0.24]),
        end_anchor=np.array([0.18, -0.24, 0.30]),
        role="side branch off obstacle midpoint",
    ),
)


OBSTACLE_TO_BOTTLE2SHELF_SIDE_PLAN = (
    TaskSpec(
        task_name="obstaclerotate",
        start_anchor=np.array([24.25, -66.70, 11.70]),
        end_anchor=np.array([37.00, -25.00, 26.00]),
        role="primary segment ending at side connection",
    ),
    TaskSpec(
        task_name="bottle2shelf",
        start_anchor=np.array([37.00, -25.00, 26.00]),
        end_anchor=np.array([52.00, -33.00, 28.00]),
        role="connected branch from obstacle side to opposite-side shelf region",
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

    # Affine endpoint correction to make each segment connect exactly.
    correction = np.linspace(0.0, 1.0, warped.shape[0], dtype=warped.dtype)[:, None]
    correction = correction * (end_anchor - warped[-1])
    return warped + correction


def _compute_velocity(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros_like(x)
    x_dot = np.gradient(x, axis=0)
    x_dot[-1] = x_dot[-2]
    return x_dot


def _iter_active_task_plan(include_openbox: bool) -> Iterable[TaskSpec]:
    if include_openbox:
        return DEFAULT_TASK_PLAN
    return tuple(spec for spec in DEFAULT_TASK_PLAN if spec.task_name != "openbox")


def build_workspace_composite_dataset(
    output_dir: str,
    task_data_dir: str = "dataset/robottasks/pos_ori",
    n_trajectories_per_task: int = 6,
    n_points: int = 180,
    include_openbox: bool = True,
    task_plan: Optional[Iterable[TaskSpec]] = None,
    seed: int = 7,
    overwrite: bool = True,
) -> dict:
    """Build a connected stitching dataset from multiple robot sub-task demonstrations.

    The resulting dataset follows the existing folder format:
      output_dir/
        demonstration_0/trajectory_*.json
        demonstration_1/trajectory_*.json
        ...
    """
    rng = np.random.default_rng(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for child in output_path.iterdir():
            if child.is_dir() and child.name.startswith("demonstration_"):
                shutil.rmtree(child)
        for file_name in ("workspace_plan.json",):
            file_path = output_path / file_name
            if file_path.exists():
                file_path.unlink()

    if task_plan is None:
        selected_plan = tuple(_iter_active_task_plan(include_openbox=include_openbox))
    else:
        selected_plan = tuple(task_plan)
    if len(selected_plan) == 0:
        raise ValueError("task_plan must contain at least one task specification")

    metadata = {
        "task_data_dir": str(task_data_dir),
        "n_trajectories_per_task": int(n_trajectories_per_task),
        "n_points": int(n_points),
        "seed": int(seed),
        "task_plan_name": "custom" if task_plan is not None else "default_workspace",
        "demonstrations": [],
    }

    for demo_idx, task_spec in enumerate(selected_plan):
        task_file = Path(task_data_dir) / f"{task_spec.task_name}.npy"
        if not task_file.exists():
            raise FileNotFoundError(f"Missing robot-task file: {task_file}")

        task_data = np.load(task_file, allow_pickle=False)
        n_available = task_data.shape[0]
        n_traj = min(int(n_trajectories_per_task), int(n_available))
        selected_indices = rng.choice(n_available, size=n_traj, replace=False)

        demo_dir = output_path / f"demonstration_{demo_idx}"
        demo_dir.mkdir(parents=True, exist_ok=True)

        for traj_idx, source_idx in enumerate(selected_indices):
            source_pos = task_data[source_idx, :, :3]
            source_pos = _resample_positions(source_pos, n_points=n_points)
            warped_pos = _warp_to_anchors(
                source_pos,
                start_anchor=task_spec.start_anchor,
                end_anchor=task_spec.end_anchor,
            )
            warped_vel = _compute_velocity(warped_pos)

            payload = {
                "x": warped_pos.tolist(),
                "x_dot": warped_vel.tolist(),
                "source_task": task_spec.task_name,
                "source_demo_idx": int(source_idx),
                "segment_role": task_spec.role,
            }
            with open(demo_dir / f"trajectory_{traj_idx}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f)

        metadata["demonstrations"].append(
            {
                "demo_dir": str(demo_dir),
                "task_name": task_spec.task_name,
                "role": task_spec.role,
                "start_anchor": task_spec.start_anchor.tolist(),
                "end_anchor": task_spec.end_anchor.tolist(),
                "source_indices": [int(i) for i in selected_indices],
            }
        )

    with open(output_path / "workspace_plan.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def build_default_workspace_dataset(overwrite: bool = True) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "dataset" / "stitching" / "robottasks_workspace_chain"
    task_data_dir = repo_root / "dataset" / "robottasks" / "pos_ori"
    return build_workspace_composite_dataset(
        output_dir=str(output_dir),
        task_data_dir=str(task_data_dir),
        overwrite=overwrite,
    )


def build_obstacle_to_bottle2shelf_side_dataset(
    output_dir: str,
    task_data_dir: str = "dataset/robottasks/pos_ori",
    n_trajectories_per_task: int = 6,
    n_points: int = 180,
    seed: int = 11,
    overwrite: bool = True,
) -> dict:
    """Build a two-segment dataset: obstaclerotate -> bottle2shelf via side anchor."""
    return build_workspace_composite_dataset(
        output_dir=output_dir,
        task_data_dir=task_data_dir,
        n_trajectories_per_task=n_trajectories_per_task,
        n_points=n_points,
        include_openbox=False,
        task_plan=OBSTACLE_TO_BOTTLE2SHELF_SIDE_PLAN,
        seed=seed,
        overwrite=overwrite,
    )


def build_default_obstacle_to_bottle2shelf_side_dataset(overwrite: bool = True) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "dataset" / "stitching" / "robottasks_obstacle_bottle2shelf_side"
    task_data_dir = repo_root / "dataset" / "robottasks" / "pos_ori"
    return build_obstacle_to_bottle2shelf_side_dataset(
        output_dir=str(output_dir),
        task_data_dir=str(task_data_dir),
        overwrite=overwrite,
    )
