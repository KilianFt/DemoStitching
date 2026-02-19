import argparse
from pathlib import Path

from src.util.pcgmm_workspace_dataset import build_pcgmm_3d_workspace_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build a connected 3D stitching workspace from PC-GMM 3D tasks."
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/stitching/pcgmm_3d_workspace",
        help="Output directory for demonstration_* folders.",
    )
    parser.add_argument(
        "--task-data-dir",
        default="dataset/pc-gmm-data",
        help="Directory with source PC-GMM .mat files.",
    )
    parser.add_argument("--n-trajectories", type=int, default=4, help="Trajectories per task segment.")
    parser.add_argument("--n-points", type=int, default=220, help="Resampled points per trajectory.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for trajectory sampling.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing demonstration_* folders in output directory.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip writing visualization figures.",
    )
    args = parser.parse_args()

    metadata = build_pcgmm_3d_workspace_dataset(
        output_dir=args.output_dir,
        task_data_dir=args.task_data_dir,
        n_trajectories_per_task=args.n_trajectories,
        n_points=args.n_points,
        seed=args.seed,
        overwrite=not args.no_overwrite,
        visualize=not args.no_visualize,
    )
    print(f"Wrote PC-GMM 3D workspace dataset to: {Path(args.output_dir).resolve()}")
    print(f"Demonstration groups: {len(metadata['demonstrations'])}")
    if metadata.get("visualizations"):
        print(f"Individual-task plot: {metadata['visualizations'].get('individual_tasks_3d', '')}")
        print(f"Combined-workspace plot: {metadata['visualizations'].get('combined_workspace_3d', '')}")


if __name__ == "__main__":
    main()
