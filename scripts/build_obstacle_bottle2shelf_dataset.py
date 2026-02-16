import argparse
from pathlib import Path

from src.util.robottasks_workspace_dataset import build_obstacle_to_bottle2shelf_side_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build a robottasks dataset with obstaclerotate connected to bottle2shelf at a side anchor."
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/stitching/robottasks_obstacle_bottle2shelf_side",
        help="Output directory for demonstration_* folders.",
    )
    parser.add_argument(
        "--task-data-dir",
        default="dataset/robottasks/pos_ori",
        help="Directory with source robot task .npy files.",
    )
    parser.add_argument("--n-trajectories", type=int, default=6, help="Trajectories per task segment.")
    parser.add_argument("--n-points", type=int, default=180, help="Resampled points per trajectory.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for trajectory sampling.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing demonstration_* folders in output directory.",
    )
    args = parser.parse_args()

    metadata = build_obstacle_to_bottle2shelf_side_dataset(
        output_dir=args.output_dir,
        task_data_dir=args.task_data_dir,
        n_trajectories_per_task=args.n_trajectories,
        n_points=args.n_points,
        seed=args.seed,
        overwrite=not args.no_overwrite,
    )
    print(f"Wrote obstacle->bottle2shelf dataset to: {Path(args.output_dir).resolve()}")
    print(f"Demonstration groups: {len(metadata['demonstrations'])}")


if __name__ == "__main__":
    main()
