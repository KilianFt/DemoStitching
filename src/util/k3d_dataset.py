import json
import shutil
from pathlib import Path

import numpy as np


def _rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return r_z @ r_y @ r_x


def _compute_velocity(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros_like(x)
    x_dot = np.gradient(x, axis=0)
    x_dot[-1] = x_dot[-2]
    return x_dot


def _base_stroke_points(stroke: str, t: np.ndarray) -> np.ndarray:
    """Return a K-like local stroke before global tilt."""
    if stroke == "stem":
        x = 0.03 * np.sin(2.0 * np.pi * t)
        y = -1.0 + 2.0 * t
        z = 0.18 * np.sin(np.pi * t)
    elif stroke == "upper":
        x = t
        y = t
        z = 0.12 * np.cos(np.pi * t) + 0.10 * t
    elif stroke == "lower":
        x = t
        y = -t
        z = -0.10 * np.cos(np.pi * t) + 0.10 * t
    else:
        raise ValueError(f"Unsupported stroke: {stroke}")
    return np.column_stack([x, y, z])


def _sample_stroke_trajectory(
    stroke: str,
    n_points: int,
    rng: np.random.Generator,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_points)
    base = _base_stroke_points(stroke, t)

    # Demo-level variability while preserving global K-like topology.
    jitter = 0.02 * rng.standard_normal(base.shape)
    smooth_mod = np.column_stack(
        [
            0.02 * np.sin(2.0 * np.pi * t + rng.uniform(0.0, 2.0 * np.pi)),
            0.02 * np.cos(np.pi * t + rng.uniform(0.0, 2.0 * np.pi)),
            0.03 * np.sin(3.0 * np.pi * t + rng.uniform(0.0, 2.0 * np.pi)),
        ]
    )
    local = base + jitter + smooth_mod

    # Mild anisotropic scaling variation between demonstrations.
    scale = np.diag(
        [
            1.1 + rng.uniform(-0.08, 0.08),
            0.9 + rng.uniform(-0.08, 0.08),
            1.0 + rng.uniform(-0.10, 0.10),
        ]
    )
    local = local @ scale.T

    world = local @ rotation.T + translation.reshape(1, 3)
    return world


def build_k3d_dataset(
    output_dir: str,
    n_demo_sets: int = 3,
    n_demos_per_set: int = 4,
    n_points: int = 160,
    seed: int = 11,
    overwrite: bool = True,
) -> dict:
    """Build a tilted 3D K-like dataset with demonstration_*/trajectory_*.json format."""
    if int(n_demo_sets) != 3:
        raise ValueError("This dataset design uses exactly 3 demo sets (stem, upper, lower).")
    if int(n_demos_per_set) <= 0:
        raise ValueError("n_demos_per_set must be positive.")
    if int(n_points) < 3:
        raise ValueError("n_points must be at least 3.")

    rng = np.random.default_rng(seed)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("demonstration_"):
                shutil.rmtree(child)
        meta_file = root / "k3d_plan.json"
        if meta_file.exists():
            meta_file.unlink()

    # Fixed global tilt in all directions.
    rotation = _rotation_matrix_xyz(
        rx=np.deg2rad(34.0),
        ry=np.deg2rad(-27.0),
        rz=np.deg2rad(23.0),
    )
    translation = np.array([0.35, -0.45, 0.60], dtype=float)

    stroke_names = ("stem", "upper", "lower")
    metadata = {
        "name": "k3d_tilted",
        "n_demo_sets": int(n_demo_sets),
        "n_demos_per_set": int(n_demos_per_set),
        "n_points": int(n_points),
        "seed": int(seed),
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
        "demonstrations": [],
    }

    for set_idx, stroke in enumerate(stroke_names):
        demo_dir = root / f"demonstration_{set_idx}"
        demo_dir.mkdir(parents=True, exist_ok=True)

        set_info = {
            "demo_dir": str(demo_dir),
            "stroke": stroke,
            "trajectories": [],
        }
        for traj_idx in range(n_demos_per_set):
            x = _sample_stroke_trajectory(
                stroke=stroke,
                n_points=n_points,
                rng=rng,
                rotation=rotation,
                translation=translation,
            )
            x_dot = _compute_velocity(x)
            payload = {
                "x": x.tolist(),
                "x_dot": x_dot.tolist(),
                "stroke": stroke,
            }
            out_file = demo_dir / f"trajectory_{traj_idx}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            set_info["trajectories"].append(str(out_file))

        metadata["demonstrations"].append(set_info)

    with open(root / "k3d_plan.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata

