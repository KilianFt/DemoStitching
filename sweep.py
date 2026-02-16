import argparse
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


RunnerFn = Callable[[Sequence[str], float | None], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SweepConfig:
    datasets: tuple[str, ...]
    ds_methods: tuple[str, ...]
    seeds: tuple[int, ...]
    output_dir: str
    timeout_s: float = 0.0
    copy_figures: bool = True


def _dataset_slug(dataset_path: str) -> str:
    path = Path(dataset_path)
    parts = [
        p
        for p in path.parts
        if p not in ("", ".", "..", "/", "\\", path.anchor)
    ]
    return "__".join(parts) if parts else "dataset"


def _runner_code(dataset_path: str, ds_method: str, seed: int) -> str:
    # Keep main_stitch unchanged: override the config constructor from a wrapper.
    return "\n".join(
        [
            "import main_stitch",
            "from configs import StitchConfig",
            "",
            "class _SweepConfig(StitchConfig):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        self.dataset_path = {dataset_path!r}",
            f"        self.ds_method = {ds_method!r}",
            f"        self.seed = {int(seed)}",
            "        self.save_fig = True",
            "",
            "main_stitch.StitchConfig = _SweepConfig",
            "main_stitch.main()",
            "",
        ]
    )


def _default_runner(cmd: Sequence[str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return math.nan
    arr = np.asarray(df[col], dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def _summarize_run_metrics(results_csv: Path) -> dict[str, float]:
    if not results_csv.exists():
        return {
            "n_result_rows": 0,
            "prediction_rmse_mean": math.nan,
            "cosine_dissimilarity_mean": math.nan,
            "dtw_distance_mean": math.nan,
            "distance_to_attractor_mean": math.nan,
        }

    df = pd.read_csv(results_csv)
    return {
        "n_result_rows": int(len(df)),
        "prediction_rmse_mean": _safe_mean(df, "prediction_rmse"),
        "cosine_dissimilarity_mean": _safe_mean(df, "cosine_dissimilarity"),
        "dtw_distance_mean": _safe_mean(df, "dtw_distance_mean"),
        "distance_to_attractor_mean": _safe_mean(df, "distance_to_attractor_mean"),
    }


def _copy_tree_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_sweep(cfg: SweepConfig, runner: RunnerFn = _default_runner) -> pd.DataFrame:
    output_root = Path(cfg.output_dir)
    logs_dir = output_root / "logs"
    copied_results_dir = output_root / "results"
    copied_figures_dir = output_root / "figures"
    logs_dir.mkdir(parents=True, exist_ok=True)
    copied_results_dir.mkdir(parents=True, exist_ok=True)
    copied_figures_dir.mkdir(parents=True, exist_ok=True)

    timeout_s = cfg.timeout_s if cfg.timeout_s > 0 else None
    rows: list[dict] = []
    total_runs = len(cfg.datasets) * len(cfg.ds_methods) * len(cfg.seeds)
    run_index = 0

    for dataset_path in cfg.datasets:
        for ds_method in cfg.ds_methods:
            for seed in cfg.seeds:
                run_index += 1
                dataset = Path(dataset_path)
                dataset_slug = _dataset_slug(dataset_path)
                combo_tag = f"{dataset_slug}__{ds_method}__seed_{seed}"
                print(f"[{run_index}/{total_runs}] Running {combo_tag}")

                code = _runner_code(dataset_path=dataset_path, ds_method=ds_method, seed=seed)
                cmd = [sys.executable, "-c", code]
                start_t = time.perf_counter()
                timed_out = False
                proc: subprocess.CompletedProcess[str] | None = None
                err_msg = ""
                try:
                    proc = runner(cmd, timeout_s)
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    err_msg = str(exc)
                except Exception as exc:  # pragma: no cover - defensive safety
                    err_msg = repr(exc)
                elapsed = time.perf_counter() - start_t

                log_path = logs_dir / f"{combo_tag}.log"
                with open(log_path, "w", encoding="utf-8") as f:
                    if proc is not None:
                        f.write(proc.stdout or "")
                        if proc.stderr:
                            if proc.stdout:
                                f.write("\n")
                            f.write(proc.stderr)
                    elif err_msg:
                        f.write(err_msg)

                run_ok = (not timed_out) and proc is not None and proc.returncode == 0

                src_method_dir = dataset / "figures" / ds_method
                src_results_csv = src_method_dir / f"results_{seed}.csv"
                copied_results_csv = copied_results_dir / f"{combo_tag}__results.csv"
                if src_results_csv.exists():
                    shutil.copy2(src_results_csv, copied_results_csv)
                else:
                    copied_results_csv = Path("")

                copied_figure_dir = copied_figures_dir / dataset_slug / ds_method / f"seed_{seed}"
                if cfg.copy_figures and src_method_dir.exists():
                    _copy_tree_if_exists(src_method_dir, copied_figure_dir)

                metric_summary = _summarize_run_metrics(src_results_csv)
                rows.append(
                    {
                        "dataset_path": dataset_path,
                        "dataset_slug": dataset_slug,
                        "ds_method": ds_method,
                        "seed": int(seed),
                        "status": "ok" if run_ok else "failed",
                        "timed_out": bool(timed_out),
                        "return_code": proc.returncode if proc is not None else -1,
                        "duration_s": float(elapsed),
                        "log_path": str(log_path),
                        "results_csv_source": str(src_results_csv) if src_results_csv.exists() else "",
                        "results_csv_copy": str(copied_results_csv) if copied_results_csv != Path("") else "",
                        "figures_dir_copy": str(copied_figure_dir) if cfg.copy_figures and src_method_dir.exists() else "",
                        **metric_summary,
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = output_root / "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSweep finished. Saved summary CSV: {out_csv.resolve()}")
    return df


def _parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(
        description="Run a grid sweep over datasets, ds methods, and seeds by invoking main_stitch."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset paths passed to StitchConfig.dataset_path.",
    )
    parser.add_argument(
        "--ds-methods",
        nargs="+",
        required=True,
        help="DS methods passed to StitchConfig.ds_method.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seeds passed to StitchConfig.seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/stitching/sweep_main_stitch",
        help="Directory for summary CSV, copied figures, copied result CSVs, and logs.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=0.0,
        help="Per-run timeout in seconds. 0 disables timeout.",
    )
    parser.add_argument(
        "--no-copy-figures",
        action="store_true",
        help="Disable copying generated figures into output-dir.",
    )
    args = parser.parse_args()
    return SweepConfig(
        datasets=tuple(args.datasets),
        ds_methods=tuple(args.ds_methods),
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
        timeout_s=float(args.timeout_s),
        copy_figures=not args.no_copy_figures,
    )


def main():
    cfg = _parse_args()
    run_sweep(cfg)


if __name__ == "__main__":
    main()
