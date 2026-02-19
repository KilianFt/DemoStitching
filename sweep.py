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
    n_test_simulations: int = 3
    timeout_s: float = 0.0
    copy_figures: bool = True
    mode: str = "standard"
    chain_ds_methods: tuple[str, ...] = ()
    chain_trigger_methods: tuple[str, ...] = ()
    chain_blend_ratios: tuple[float, ...] = ()
    chain_fixed_ds_method: str = "segmented"
    chain_fixed_trigger_method: str = "mean_normals"
    param_dist_values: tuple[float, ...] = ()
    param_cos_values: tuple[float, ...] = ()


def _dataset_slug(dataset_path: str) -> str:
    path = Path(dataset_path)
    parts = [
        p
        for p in path.parts
        if p not in ("", ".", "..", "/", "\\", path.anchor)
    ]
    return "__".join(parts) if parts else "dataset"


def _normalize_chain_ds_method(value: str) -> str:
    method = str(value).strip().lower()
    if method == "segment":
        return "segmented"
    if method in {"segmented", "linear"}:
        return method
    raise ValueError(f"Unsupported chain ds_method: {value}")


def _normalize_trigger_method(value: str) -> str:
    method = str(value).strip().lower()
    if method in {"mean_normals", "distance_ratio"}:
        return method
    raise ValueError(f"Unsupported chain transition_trigger_method: {value}")


def _float_tag(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _runner_code(
    dataset_path: str,
    ds_method: str,
    seed: int,
    n_test_simulations: int = 3,
    results_csv_path: str | None = None,
    chain_ds_method: str | None = None,
    chain_trigger_method: str | None = None,
    chain_blend_ratio: float | None = None,
    param_dist: float | None = None,
    param_cos: float | None = None,
) -> str:
    # Keep main_stitch unchanged: override the config constructor from a wrapper.
    chain_lines = []
    if chain_ds_method is not None:
        chain_lines.append(f"        self.chain.ds_method = {chain_ds_method!r}")
    if chain_trigger_method is not None:
        chain_lines.append(f"        self.chain.transition_trigger_method = {chain_trigger_method!r}")
    if chain_blend_ratio is not None:
        chain_lines.append(f"        self.chain.blend_length_ratio = {float(chain_blend_ratio)}")
    graph_lines = []
    if param_dist is not None:
        graph_lines.append(f"        self.param_dist = {float(param_dist)}")
    if param_cos is not None:
        graph_lines.append(f"        self.param_cos = {float(param_cos)}")

    lines = [
        "import main_stitch",
        "from configs import StitchConfig",
        "",
        "_orig_construct_stitched_ds = main_stitch.construct_stitched_ds",
        "def _compat_construct_stitched_ds(*args, **kwargs):",
        "    kwargs.pop('segment_ds_lookup', None)",
        "    result = _orig_construct_stitched_ds(*args, **kwargs)",
        "    if isinstance(result, tuple) and len(result) == 4:",
        "        stitched_ds, _gg_obj, gg_solution_nodes, stats = result",
        "        return stitched_ds, gg_solution_nodes, stats",
        "    return result",
        "main_stitch.construct_stitched_ds = _compat_construct_stitched_ds",
        "",
    ]
    if str(ds_method).strip().lower() == "chain":
        lines += [
            "def _noop_compute_segment_DS(*args, **kwargs):",
            "    return None",
            "main_stitch._compute_segment_DS = _noop_compute_segment_DS",
            "",
        ]
    if results_csv_path is not None:
        lines += [
            f"__SWEEP_RESULTS_PATH__ = {results_csv_path!r}",
            "_orig_save_results_dataframe = main_stitch.save_results_dataframe",
            "def _sweep_save_results_dataframe(all_results, _save_path):",
            "    return _orig_save_results_dataframe(all_results, __SWEEP_RESULTS_PATH__)",
            "main_stitch.save_results_dataframe = _sweep_save_results_dataframe",
            "",
        ]
    lines += [
        "class _SweepConfig(StitchConfig):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        self.dataset_path = {dataset_path!r}",
        f"        self.ds_method = {ds_method!r}",
        f"        self.seed = {int(seed)}",
        f"        self.n_test_simulations = {int(n_test_simulations)}",
        "        self.save_fig = True",
        *graph_lines,
        *chain_lines,
        "",
        "main_stitch.StitchConfig = _SweepConfig",
        "main_stitch.main()",
        "",
    ]
    return "\n".join(lines)


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


def _safe_mean_any(df: pd.DataFrame, cols: Sequence[str]) -> float:
    for col in cols:
        if col in df.columns:
            return _safe_mean(df, col)
    return math.nan


def _extract_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "combination_id" in out.columns:
        out = out[out["combination_id"].notna()]
    if "ds_method" in out.columns:
        ds_method = out["ds_method"].astype(str).str.strip()
        out = out[out["ds_method"].notna() & (ds_method != "")]
    return out


def _summarize_run_metrics(results_csv: Path) -> dict[str, float]:
    if not results_csv.exists():
        return {
            "n_result_rows": 0,
            "n_eval_rows": 0,
            "prediction_rmse_mean": math.nan,
            "cosine_dissimilarity_mean": math.nan,
            "dtw_distance_mean": math.nan,
            "distance_to_attractor_mean": math.nan,
            "ds_compute_time_mean": math.nan,
            "gg_compute_time_mean": math.nan,
            "total_compute_time_mean": math.nan,
        }

    df = pd.read_csv(results_csv)
    eval_df = _extract_eval_rows(df)
    return {
        "n_result_rows": int(len(df)),
        "n_eval_rows": int(len(eval_df)),
        "prediction_rmse_mean": _safe_mean(eval_df, "prediction_rmse"),
        "cosine_dissimilarity_mean": _safe_mean(eval_df, "cosine_dissimilarity"),
        "dtw_distance_mean": _safe_mean(eval_df, "dtw_distance_mean"),
        "distance_to_attractor_mean": _safe_mean(eval_df, "distance_to_attractor_mean"),
        "ds_compute_time_mean": _safe_mean_any(eval_df, ("ds_compute_time", "ds_compute_time_mean")),
        "gg_compute_time_mean": _safe_mean_any(eval_df, ("gg_compute_time", "gg_compute_time_mean")),
        "total_compute_time_mean": _safe_mean_any(eval_df, ("total_compute_time", "total_compute_time_mean")),
    }


def _copy_tree_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _iter_run_specs(cfg: SweepConfig):
    if cfg.mode == "standard":
        for dataset_path in cfg.datasets:
            for ds_method in cfg.ds_methods:
                for seed in cfg.seeds:
                    yield {
                        "dataset_path": dataset_path,
                        "ds_method": ds_method,
                        "seed": int(seed),
                        "chain_ds_method": None,
                        "chain_trigger_method": None,
                        "chain_blend_ratio": None,
                        "param_dist": None,
                        "param_cos": None,
                    }
        return

    if cfg.mode == "graph_params":
        for dataset_path in cfg.datasets:
            for ds_method in cfg.ds_methods:
                for param_dist in cfg.param_dist_values:
                    for param_cos in cfg.param_cos_values:
                        for seed in cfg.seeds:
                            yield {
                                "dataset_path": dataset_path,
                                "ds_method": ds_method,
                                "seed": int(seed),
                                "chain_ds_method": None,
                                "chain_trigger_method": None,
                                "chain_blend_ratio": None,
                                "param_dist": float(param_dist),
                                "param_cos": float(param_cos),
                            }
        return

    if cfg.mode == "chain_trigger":
        for dataset_path in cfg.datasets:
            for chain_ds_method in cfg.chain_ds_methods:
                for chain_trigger_method in cfg.chain_trigger_methods:
                    for seed in cfg.seeds:
                        yield {
                            "dataset_path": dataset_path,
                            "ds_method": "chain",
                            "seed": int(seed),
                            "chain_ds_method": chain_ds_method,
                            "chain_trigger_method": chain_trigger_method,
                            "chain_blend_ratio": None,
                            "param_dist": None,
                            "param_cos": None,
                        }
        return

    if cfg.mode == "chain_blend":
        for dataset_path in cfg.datasets:
            for chain_blend_ratio in cfg.chain_blend_ratios:
                for seed in cfg.seeds:
                    yield {
                        "dataset_path": dataset_path,
                        "ds_method": "chain",
                            "seed": int(seed),
                            "chain_ds_method": cfg.chain_fixed_ds_method,
                            "chain_trigger_method": cfg.chain_fixed_trigger_method,
                            "chain_blend_ratio": float(chain_blend_ratio),
                            "param_dist": None,
                            "param_cos": None,
                        }
        return

    raise ValueError(f"Unsupported sweep mode: {cfg.mode}")


def _combo_tag(spec: dict[str, object]) -> str:
    dataset_slug = _dataset_slug(str(spec["dataset_path"]))
    ds_method = str(spec["ds_method"])
    seed = int(spec["seed"])

    chain_ds_method = spec["chain_ds_method"]
    chain_trigger_method = spec["chain_trigger_method"]
    chain_blend_ratio = spec["chain_blend_ratio"]
    param_dist = spec["param_dist"]
    param_cos = spec["param_cos"]

    tag = f"{dataset_slug}__{ds_method}__seed_{seed}"
    if chain_ds_method is not None:
        tag += f"__chain_ds_{chain_ds_method}"
    if chain_trigger_method is not None:
        tag += f"__trigger_{chain_trigger_method}"
    if chain_blend_ratio is not None:
        tag += f"__blend_{_float_tag(float(chain_blend_ratio))}"
    if param_dist is not None:
        tag += f"__param_dist_{_float_tag(float(param_dist))}"
    if param_cos is not None:
        tag += f"__param_cos_{_float_tag(float(param_cos))}"
    return tag


def run_sweep(cfg: SweepConfig, runner: RunnerFn = _default_runner) -> pd.DataFrame:
    output_root = Path(cfg.output_dir)
    logs_dir = output_root / "logs"
    raw_results_dir = output_root / "raw_results"
    copied_results_dir = output_root / "results"
    copied_figures_dir = output_root / "figures"
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    copied_results_dir.mkdir(parents=True, exist_ok=True)
    copied_figures_dir.mkdir(parents=True, exist_ok=True)

    timeout_s = cfg.timeout_s if cfg.timeout_s > 0 else None
    rows: list[dict] = []
    specs = list(_iter_run_specs(cfg))
    total_runs = len(specs)

    for run_index, spec in enumerate(specs, start=1):
        dataset_path = str(spec["dataset_path"])
        ds_method = str(spec["ds_method"])
        seed = int(spec["seed"])
        chain_ds_method = spec["chain_ds_method"]
        chain_trigger_method = spec["chain_trigger_method"]
        chain_blend_ratio = spec["chain_blend_ratio"]
        param_dist = spec["param_dist"]
        param_cos = spec["param_cos"]

        dataset = Path(dataset_path)
        dataset_slug = _dataset_slug(dataset_path)
        combo_tag = _combo_tag(spec)
        print(f"[{run_index}/{total_runs}] Running {combo_tag}")
        run_results_csv = raw_results_dir / f"{combo_tag}.csv"
        if run_results_csv.exists():
            run_results_csv.unlink()

        code = _runner_code(
            dataset_path=dataset_path,
            ds_method=ds_method,
            seed=seed,
            n_test_simulations=int(cfg.n_test_simulations),
            results_csv_path=str(run_results_csv),
            chain_ds_method=None if chain_ds_method is None else str(chain_ds_method),
            chain_trigger_method=None if chain_trigger_method is None else str(chain_trigger_method),
            chain_blend_ratio=None if chain_blend_ratio is None else float(chain_blend_ratio),
            param_dist=None if param_dist is None else float(param_dist),
            param_cos=None if param_cos is None else float(param_cos),
        )
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
        failure_reason = ""
        if timed_out:
            failure_reason = "timeout"
        elif proc is None:
            failure_reason = "runner_error"
        elif proc.returncode != 0:
            failure_reason = "nonzero_return_code"

        src_method_dir = dataset / "figures" / ds_method
        src_results_csv = run_results_csv
        copied_results_csv = copied_results_dir / f"{combo_tag}__results.csv"
        if run_results_csv.exists():
            shutil.copy2(src_results_csv, copied_results_csv)
        else:
            copied_results_csv = Path("")

        copied_figure_dir = copied_figures_dir / dataset_slug / ds_method / f"seed_{seed}" / combo_tag
        if cfg.copy_figures and src_method_dir.exists():
            _copy_tree_if_exists(src_method_dir, copied_figure_dir)

        metric_summary = _summarize_run_metrics(src_results_csv)
        if run_ok and int(metric_summary.get("n_eval_rows", 0)) == 0:
            run_ok = False
            failure_reason = "no_evaluation_rows"
        if run_ok and (
            math.isnan(metric_summary["prediction_rmse_mean"])
            and math.isnan(metric_summary["cosine_dissimilarity_mean"])
            and math.isnan(metric_summary["dtw_distance_mean"])
        ):
            run_ok = False
            failure_reason = "no_metric_values"
        rows.append(
            {
                "dataset_path": dataset_path,
                "dataset_slug": dataset_slug,
                "sweep_mode": cfg.mode,
                "ds_method": ds_method,
                "chain_ds_method": "" if chain_ds_method is None else str(chain_ds_method),
                "chain_transition_trigger_method": "" if chain_trigger_method is None else str(chain_trigger_method),
                "chain_blend_length_ratio": np.nan if chain_blend_ratio is None else float(chain_blend_ratio),
                "param_dist": np.nan if param_dist is None else float(param_dist),
                "param_cos": np.nan if param_cos is None else float(param_cos),
                "seed": int(seed),
                "n_test_simulations": int(cfg.n_test_simulations),
                "status": "ok" if run_ok else "failed",
                "failure_reason": "" if run_ok else failure_reason,
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
        description="Run standard or chaining parameter sweeps by invoking main_stitch."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "graph_params", "chain_trigger", "chain_blend"],
        help="Sweep mode.",
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
        default=None,
        help="DS methods for standard mode.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seeds passed to StitchConfig.seed.",
    )
    parser.add_argument(
        "--n-test-simulations",
        type=int,
        default=3,
        help="Number of rollouts per initial/goal combination (StitchConfig.n_test_simulations). Default: 3.",
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
    parser.add_argument(
        "--chain-ds-methods",
        nargs="+",
        default=None,
        help='Chain DS methods for chain_trigger mode (e.g. "segment linear").',
    )
    parser.add_argument(
        "--chain-trigger-methods",
        nargs="+",
        default=None,
        help='Chain trigger methods for chain_trigger mode (e.g. "mean_normals distance_ratio").',
    )
    parser.add_argument(
        "--chain-blend-ratios",
        nargs="+",
        type=float,
        default=None,
        help="Blend ratios for chain_blend mode.",
    )
    parser.add_argument(
        "--chain-fixed-ds-method",
        type=str,
        default="segmented",
        help="Fixed chain ds_method for chain_blend mode.",
    )
    parser.add_argument(
        "--chain-fixed-trigger-method",
        type=str,
        default="mean_normals",
        help="Fixed chain transition_trigger_method for chain_blend mode.",
    )
    parser.add_argument(
        "--param-dist-values",
        nargs="+",
        type=float,
        default=None,
        help="Values for StitchConfig.param_dist in graph_params mode.",
    )
    parser.add_argument(
        "--param-cos-values",
        nargs="+",
        type=float,
        default=None,
        help="Values for StitchConfig.param_cos in graph_params mode.",
    )
    args = parser.parse_args()

    ds_methods = tuple(args.ds_methods or ())
    chain_ds_methods = tuple(_normalize_chain_ds_method(v) for v in (args.chain_ds_methods or ()))
    chain_trigger_methods = tuple(_normalize_trigger_method(v) for v in (args.chain_trigger_methods or ()))
    chain_blend_ratios = tuple(float(v) for v in (args.chain_blend_ratios or ()))
    param_dist_values = tuple(float(v) for v in (args.param_dist_values or ()))
    param_cos_values = tuple(float(v) for v in (args.param_cos_values or ()))

    if args.mode in {"standard", "graph_params"} and len(ds_methods) == 0:
        raise ValueError("--ds-methods is required when --mode standard or --mode graph_params")
    if args.mode == "graph_params":
        if len(param_dist_values) == 0:
            raise ValueError("--param-dist-values is required when --mode graph_params")
        if len(param_cos_values) == 0:
            raise ValueError("--param-cos-values is required when --mode graph_params")
    if args.mode == "chain_trigger":
        if len(chain_ds_methods) == 0:
            raise ValueError("--chain-ds-methods is required when --mode chain_trigger")
        if len(chain_trigger_methods) == 0:
            raise ValueError("--chain-trigger-methods is required when --mode chain_trigger")
    if args.mode == "chain_blend" and len(chain_blend_ratios) == 0:
        raise ValueError("--chain-blend-ratios is required when --mode chain_blend")

    return SweepConfig(
        datasets=tuple(args.datasets),
        ds_methods=ds_methods,
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
        n_test_simulations=int(args.n_test_simulations),
        timeout_s=float(args.timeout_s),
        copy_figures=not args.no_copy_figures,
        mode=args.mode,
        chain_ds_methods=chain_ds_methods,
        chain_trigger_methods=chain_trigger_methods,
        chain_blend_ratios=chain_blend_ratios,
        chain_fixed_ds_method=_normalize_chain_ds_method(args.chain_fixed_ds_method),
        chain_fixed_trigger_method=_normalize_trigger_method(args.chain_fixed_trigger_method),
        param_dist_values=param_dist_values,
        param_cos_values=param_cos_values,
    )


def main():
    cfg = _parse_args()
    run_sweep(cfg)


if __name__ == "__main__":
    main()
