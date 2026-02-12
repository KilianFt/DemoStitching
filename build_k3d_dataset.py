import argparse
from pathlib import Path

from src.util.k3d_dataset import build_k3d_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build a tilted 3D K-like stitching dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/stitching/k3d_tilted",
        help="Output directory for demonstration_* folders.",
    )
    parser.add_argument(
        "--n-demos-per-set",
        type=int,
        default=4,
        help="Number of trajectories per demo set.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=160,
        help="Number of points per trajectory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed for dataset generation.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing demonstration_* folders in output directory.",
    )
    args = parser.parse_args()

    metadata = build_k3d_dataset(
        output_dir=args.output_dir,
        n_demo_sets=3,
        n_demos_per_set=args.n_demos_per_set,
        n_points=args.n_points,
        seed=args.seed,
        overwrite=not args.no_overwrite,
    )
    print(f"Wrote K-3D dataset to: {Path(args.output_dir).resolve()}")
    print(
        f"Demo sets: {metadata['n_demo_sets']} | "
        f"demos per set: {metadata['n_demos_per_set']}"
    )


if __name__ == "__main__":
    main()

