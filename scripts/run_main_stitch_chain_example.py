import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from main_stitch import Config
from src.stitching.ds_stitching import construct_stitched_ds
from src.util.ds_tools import apply_lpvds_demowise
from src.util.load_tools import get_demonstration_set


def _mean_endpoint_for_task(dataset_path: str, task_name: str, endpoint: str) -> np.ndarray:
    plan_path = os.path.join(dataset_path, "workspace_plan.json")
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    entries = [entry for entry in plan["demonstrations"] if entry["task_name"] == task_name]
    if not entries:
        raise ValueError(f"Task '{task_name}' not found in {plan_path}")

    demo_dir = entries[0]["demo_dir"]
    files = sorted(
        [
            os.path.join(demo_dir, name)
            for name in os.listdir(demo_dir)
            if name.startswith("trajectory_") and name.endswith(".json")
        ]
    )
    points = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        x = np.asarray(payload["x"], dtype=float)
        points.append(x[0] if endpoint == "start" else x[-1])
    return np.mean(np.vstack(points), axis=0)


def run_example(dataset_path: str, output_dir: str):
    config = Config(
        dataset_path=dataset_path,
        ds_method="chain",
        reverse_gaussians=True,
        save_fig=False,
        n_test_simulations=1,
        noise_std=0.0,
        seed=42,
    )
    np.random.seed(config.seed)

    demo_set = get_demonstration_set(dataset_path)
    ds_set, reversed_ds_set, norm_demo_set = apply_lpvds_demowise(demo_set, config)

    # Example route:
    # start from the end of pouring branch and move to pan2stove goal.
    initial = _mean_endpoint_for_task(dataset_path, "pouring", endpoint="end")
    attractor = _mean_endpoint_for_task(dataset_path, "pan2stove", endpoint="end")

    stitched_ds, gg, _, stats = construct_stitched_ds(
        config, norm_demo_set, ds_set, reversed_ds_set, initial, attractor
    )
    if stitched_ds is None:
        raise RuntimeError("Failed to construct chain DS for example.")

    trajectory, _ = stitched_ds.sim(initial[None, :], dt=0.01)
    final_dist = np.linalg.norm(trajectory[-1] - attractor)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "chain_example_trajectory.png")
    gif_path = os.path.join(output_dir, "chain_example_trajectory_restart.gif")

    # Static figure (xy projection)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for i, demo in enumerate(demo_set):
        for traj in demo.trajectories:
            ax.plot(traj.x[:, 0], traj.x[:, 1], color=colors[i % len(colors)], alpha=0.25, linewidth=1)

    ax.plot(trajectory[:, 0], trajectory[:, 1], color="black", linewidth=2.5, label="chain trajectory")
    ax.scatter(initial[0], initial[1], color="gold", edgecolor="black", s=90, zorder=3, label="start")
    ax.scatter(attractor[0], attractor[1], color="red", edgecolor="black", s=90, zorder=3, label="goal")
    ax.set_aspect("equal")
    ax.set_title("Main Stitch Chain Example (xy projection)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    # Looping animation (auto-restart)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, demo in enumerate(demo_set):
        for traj in demo.trajectories:
            ax.plot(traj.x[:, 0], traj.x[:, 1], color=colors[i % len(colors)], alpha=0.18, linewidth=0.8)
    ax.scatter(initial[0], initial[1], color="gold", edgecolor="black", s=70, zorder=3)
    ax.scatter(attractor[0], attractor[1], color="red", edgecolor="black", s=70, zorder=3)
    line, = ax.plot([], [], color="black", linewidth=2.5)
    point, = ax.plot([], [], "ko", markersize=5)
    ax.set_aspect("equal")
    ax.set_title("Chain trajectory (auto-restart)")

    padding = 0.05
    x_all = np.concatenate([trajectory[:, 0]] + [demo.x[:, 0] for demo in demo_set])
    y_all = np.concatenate([trajectory[:, 1]] + [demo.x[:, 1] for demo in demo_set])
    ax.set_xlim(x_all.min() - padding, x_all.max() + padding)
    ax.set_ylim(y_all.min() - padding, y_all.max() + padding)

    n = len(trajectory)

    def _update(frame):
        idx = frame % n
        if idx == 0:
            line.set_data([], [])
        else:
            line.set_data(trajectory[: idx + 1, 0], trajectory[: idx + 1, 1])
        point.set_data([trajectory[idx, 0]], [trajectory[idx, 1]])
        return line, point

    animation = FuncAnimation(fig, _update, frames=3 * n, interval=20, blit=True, repeat=True)
    writer = PillowWriter(fps=30, metadata={"loop": 0})
    animation.save(gif_path, writer=writer)
    plt.close(fig)

    return {
        "initial": initial,
        "attractor": attractor,
        "final_distance": final_dist,
        "path": getattr(gg, "shortest_path_nodes", None),
        "stats": stats,
        "png_path": png_path,
        "gif_path": gif_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a one-route main-stitch chain example and plot it.")
    parser.add_argument(
        "--dataset-path",
        default="dataset/stitching/robottasks_workspace_chain",
        help="Dataset folder with demonstration_*.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/stitching/robottasks_workspace_chain/figures/chain",
        help="Where to save example figures.",
    )
    args = parser.parse_args()

    result = run_example(dataset_path=args.dataset_path, output_dir=args.output_dir)
    print("Main stitch chain example done.")
    print(f"initial: {np.round(result['initial'], 4)}")
    print(f"attractor: {np.round(result['attractor'], 4)}")
    print(f"final_distance: {result['final_distance']:.6f}")
    print(f"path_length_nodes: {len(result['path']) if result['path'] is not None else 0}")
    print(f"png: {os.path.abspath(result['png_path'])}")
    print(f"gif: {os.path.abspath(result['gif_path'])}")
    print(f"timing: {result['stats']}")


if __name__ == "__main__":
    main()
