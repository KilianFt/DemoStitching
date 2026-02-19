import argparse
import concurrent.futures as cf
import contextlib
import io
import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from configs import StitchConfig


RunMainFn = Callable[[StitchConfig, str], list[dict] | None]


@dataclass(frozen=True)
class SweepConfig:
    datasets: tuple[str, ...]
    ds_methods: tuple[str, ...]
    seeds: tuple[int, ...]
    output_dir: str
    n_test_simulations: int = 3
    timeout_s: float = 0.0
    workers: int = 1
    copy_figures: bool = False
    save_fig: bool = False
    chain_precompute_segments: bool = False
    mode: str = "standard"
    chain_ds_methods: tuple[str, ...] = ()
    chain_trigger_methods: tuple[str, ...] = ()
    chain_blend_ratios: tuple[float, ...] = ()
    chain_fixed_ds_method: str = "segmented"
    chain_fixed_trigger_method: str = "mean_normals"
    param_dist_values: tuple[float, ...] = ()
    param_cos_values: tuple[float, ...] = ()
    rel_scale_values: tuple[float, ...] = ()


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


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return math.nan
    arr = np.asarray(df[col], dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def _safe_mean_any(df: pd.DataFrame, cols: tuple[str, ...]) -> float:
    for col in cols:
        if col in df.columns:
            return _safe_mean(df, col)
    return math.nan


def _extract_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "combination_id" not in out.columns or "ds_method" not in out.columns:
        return out.iloc[0:0]
    out = out[out["combination_id"].notna()]
    ds_method = out["ds_method"].astype(str).str.strip()
    out = out[out["ds_method"].notna() & (ds_method != "")]
    return out


def _empty_metric_summary() -> dict[str, float]:
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


def _summarize_df_metrics(df: pd.DataFrame) -> dict[str, float]:
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


def _summarize_run_metrics(
    all_results: list[dict] | None,
    results_csv: Path,
) -> dict[str, float]:
    if all_results:
        try:
            return _summarize_df_metrics(pd.DataFrame(all_results))
        except Exception:
            pass

    if results_csv.exists():
        try:
            return _summarize_df_metrics(pd.read_csv(results_csv))
        except Exception:
            pass

    return _empty_metric_summary()


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
                        "rel_scale": None,
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
                                "rel_scale": None,
                            }
        return

    if cfg.mode == "rel_scale":
        for dataset_path in cfg.datasets:
            for ds_method in cfg.ds_methods:
                for rel_scale in cfg.rel_scale_values:
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
                            "rel_scale": float(rel_scale),
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
                            "rel_scale": None,
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
                        "rel_scale": None,
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
    rel_scale = spec["rel_scale"]

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
    if rel_scale is not None:
        tag += f"__rel_scale_{_float_tag(float(rel_scale))}"
    return tag


def _build_stitch_config(cfg: SweepConfig, spec: dict[str, object]) -> StitchConfig:
    stitch_cfg = StitchConfig()
    stitch_cfg.dataset_path = str(spec["dataset_path"])
    stitch_cfg.ds_method = str(spec["ds_method"])
    stitch_cfg.seed = int(spec["seed"])
    stitch_cfg.n_test_simulations = int(cfg.n_test_simulations)
    stitch_cfg.save_fig = bool(cfg.save_fig)

    chain_ds_method = spec["chain_ds_method"]
    chain_trigger_method = spec["chain_trigger_method"]
    chain_blend_ratio = spec["chain_blend_ratio"]
    param_dist = spec["param_dist"]
    param_cos = spec["param_cos"]
    rel_scale = spec["rel_scale"]

    if chain_ds_method is not None:
        stitch_cfg.chain.ds_method = str(chain_ds_method)
    if chain_trigger_method is not None:
        stitch_cfg.chain.transition_trigger_method = str(chain_trigger_method)
    if chain_blend_ratio is not None:
        stitch_cfg.chain.blend_length_ratio = float(chain_blend_ratio)
    if param_dist is not None:
        stitch_cfg.param_dist = float(param_dist)
    if param_cos is not None:
        stitch_cfg.param_cos = float(param_cos)
    if rel_scale is not None:
        stitch_cfg.damm.rel_scale = float(rel_scale)

    setattr(stitch_cfg, "chain_precompute_segments", bool(cfg.chain_precompute_segments))
    return stitch_cfg


def _default_run_main(stitch_cfg: StitchConfig, results_path: str) -> list[dict] | None:
    from main_stitch import main as main_stitch_main

    return main_stitch_main(config=stitch_cfg, results_path=results_path)


def _run_main_worker(
    run_main_fn: RunMainFn,
    stitch_cfg: StitchConfig,
    results_path: str,
    status_queue,
):
    # Child process: suppress verbose output from the stitch pipeline.
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            run_main_fn(stitch_cfg, results_path)
        status_queue.put({"ok": True})
    except Exception as exc:
        status_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_main_with_timeout(
    run_main_fn: RunMainFn,
    stitch_cfg: StitchConfig,
    results_path: str,
    timeout_s: float,
) -> tuple[str, bool]:
    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_main_worker,
        args=(run_main_fn, stitch_cfg, results_path, status_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=float(timeout_s))

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=2.0)
        try:
            status_queue.close()
            status_queue.join_thread()
        except Exception:
            pass
        return f"timeout after {float(timeout_s):.3f}s", True

    payload = None
    try:
        if not status_queue.empty():
            payload = status_queue.get_nowait()
    except Exception:
        payload = None
    finally:
        try:
            status_queue.close()
            status_queue.join_thread()
        except Exception:
            pass

    if payload is None:
        if int(proc.exitcode or 0) == 0:
            return "", False
        return f"child process exited with code {proc.exitcode}", False

    if bool(payload.get("ok", False)):
        return "", False
    return str(payload.get("error", "runner_error")), False


def _run_single_spec(
    cfg: SweepConfig,
    spec: dict[str, object],
    run_main_fn: RunMainFn,
    raw_results_dir: Path,
    run_index: int,
    total_runs: int,
    announce_start: bool = True,
) -> dict:
    dataset_path = str(spec["dataset_path"])
    ds_method = str(spec["ds_method"])
    seed = int(spec["seed"])
    chain_ds_method = spec["chain_ds_method"]
    chain_trigger_method = spec["chain_trigger_method"]
    chain_blend_ratio = spec["chain_blend_ratio"]
    param_dist = spec["param_dist"]
    param_cos = spec["param_cos"]
    rel_scale = spec["rel_scale"]

    dataset_slug = _dataset_slug(dataset_path)
    combo_tag = _combo_tag(spec)
    if announce_start:
        print(f"[{run_index}/{total_runs}] Running {combo_tag}")

    run_results_csv = raw_results_dir / f"{combo_tag}.csv"
    if run_results_csv.exists():
        run_results_csv.unlink()

    stitch_cfg = _build_stitch_config(cfg, spec)
    start_t = time.perf_counter()
    run_error = ""
    all_results: list[dict] | None = None
    timed_out = False
    use_hard_timeout = float(cfg.timeout_s) > 0.0 and run_main_fn is _default_run_main

    if use_hard_timeout:
        run_error, timed_out = _run_main_with_timeout(
            run_main_fn=run_main_fn,
            stitch_cfg=stitch_cfg,
            results_path=str(run_results_csv),
            timeout_s=float(cfg.timeout_s),
        )
    else:
        run_stdout = io.StringIO()
        run_stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(run_stdout), contextlib.redirect_stderr(run_stderr):
                all_results = run_main_fn(stitch_cfg, str(run_results_csv))
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start_t

    metric_summary = _summarize_run_metrics(all_results, run_results_csv)
    run_ok = (run_error == "") and (not timed_out)
    failure_reason = ""
    if timed_out:
        failure_reason = "timeout"
    elif not run_ok:
        failure_reason = "runner_error"
    elif int(metric_summary.get("n_eval_rows", 0)) == 0:
        run_ok = False
        failure_reason = "no_evaluation_rows"
    elif (
        math.isnan(metric_summary["prediction_rmse_mean"])
        and math.isnan(metric_summary["cosine_dissimilarity_mean"])
        and math.isnan(metric_summary["dtw_distance_mean"])
    ):
        run_ok = False
        failure_reason = "no_metric_values"

    # Avoid figure accumulation if a run accidentally opens figures.
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass

    return {
        "run_index": int(run_index),
        "dataset_path": dataset_path,
        "dataset_slug": dataset_slug,
        "sweep_mode": cfg.mode,
        "ds_method": ds_method,
        "chain_ds_method": "" if chain_ds_method is None else str(chain_ds_method),
        "chain_transition_trigger_method": "" if chain_trigger_method is None else str(chain_trigger_method),
        "chain_blend_length_ratio": np.nan if chain_blend_ratio is None else float(chain_blend_ratio),
        "param_dist": np.nan if param_dist is None else float(param_dist),
        "param_cos": np.nan if param_cos is None else float(param_cos),
        "rel_scale": np.nan if rel_scale is None else float(rel_scale),
        "seed": int(seed),
        "n_test_simulations": int(cfg.n_test_simulations),
        "status": "ok" if run_ok else "failed",
        "failure_reason": "" if run_ok else failure_reason,
        "error_message": run_error,
        "timed_out": bool(timed_out),
        "duration_s": float(elapsed),
        "results_csv_source": str(run_results_csv) if run_results_csv.exists() else "",
        **metric_summary,
    }


def _internal_failure_row(cfg: SweepConfig, spec: dict[str, object], run_index: int, error_message: str) -> dict:
    dataset_path = str(spec["dataset_path"])
    chain_ds_method = spec["chain_ds_method"]
    chain_trigger_method = spec["chain_trigger_method"]
    chain_blend_ratio = spec["chain_blend_ratio"]
    param_dist = spec["param_dist"]
    param_cos = spec["param_cos"]
    rel_scale = spec["rel_scale"]
    return {
        "run_index": int(run_index),
        "dataset_path": dataset_path,
        "dataset_slug": _dataset_slug(dataset_path),
        "sweep_mode": cfg.mode,
        "ds_method": str(spec["ds_method"]),
        "chain_ds_method": "" if chain_ds_method is None else str(chain_ds_method),
        "chain_transition_trigger_method": "" if chain_trigger_method is None else str(chain_trigger_method),
        "chain_blend_length_ratio": np.nan if chain_blend_ratio is None else float(chain_blend_ratio),
        "param_dist": np.nan if param_dist is None else float(param_dist),
        "param_cos": np.nan if param_cos is None else float(param_cos),
        "rel_scale": np.nan if rel_scale is None else float(rel_scale),
        "seed": int(spec["seed"]),
        "n_test_simulations": int(cfg.n_test_simulations),
        "status": "failed",
        "failure_reason": "runner_error",
        "error_message": str(error_message),
        "timed_out": False,
        "duration_s": math.nan,
        "results_csv_source": "",
        **_empty_metric_summary(),
    }


def run_sweep(cfg: SweepConfig, run_main_fn: RunMainFn = _default_run_main) -> pd.DataFrame:
    output_root = Path(cfg.output_dir)
    raw_results_dir = output_root / "raw_results"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_dir.mkdir(parents=True, exist_ok=True)

    specs = list(_iter_run_specs(cfg))
    total_runs = len(specs)
    rows_by_index: dict[int, dict] = {}

    max_workers = max(1, int(getattr(cfg, "workers", 1)))
    if max_workers <= 1 or total_runs <= 1:
        for run_index, spec in enumerate(specs, start=1):
            row = _run_single_spec(
                cfg=cfg,
                spec=spec,
                run_main_fn=run_main_fn,
                raw_results_dir=raw_results_dir,
                run_index=run_index,
                total_runs=total_runs,
                announce_start=True,
            )
            rows_by_index[run_index] = row
    else:
        print(f"Using {max_workers} sweep workers")
        future_to_meta = {}
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for run_index, spec in enumerate(specs, start=1):
                combo_tag = _combo_tag(spec)
                print(f"[queued {run_index}/{total_runs}] {combo_tag}")
                future = executor.submit(
                    _run_single_spec,
                    cfg,
                    spec,
                    run_main_fn,
                    raw_results_dir,
                    run_index,
                    total_runs,
                    False,
                )
                future_to_meta[future] = (run_index, spec, combo_tag)

            completed = 0
            for future in cf.as_completed(future_to_meta):
                run_index, spec, combo_tag = future_to_meta[future]
                completed += 1
                try:
                    row = future.result()
                except Exception as exc:
                    row = _internal_failure_row(cfg, spec, run_index, f"{type(exc).__name__}: {exc}")
                rows_by_index[run_index] = row
                print(f"[done {completed}/{total_runs}] {combo_tag} -> {row['status']}")

    rows = [rows_by_index[i] for i in sorted(rows_by_index.keys())]
    for row in rows:
        row.pop("run_index", None)

    df = pd.DataFrame(rows)
    out_csv = output_root / "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSweep finished. Saved summary CSV: {out_csv.resolve()}")
    return df


def _parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(
        description="Run parameter sweeps by invoking main_stitch.main(config=...)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "graph_params", "rel_scale", "chain_trigger", "chain_blend"],
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
        help="DS methods for standard, graph_params, and rel_scale modes.",
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
        help="Number of rollouts per initial/goal combination. Default: 3.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/stitching/sweep_main_stitch",
        help="Directory for summary CSV and per-run raw results.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=900.0,
        help="Per-run hard timeout in seconds; timed-out runs are marked failed and sweep continues.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of sweep workers. Each worker runs one combo at a time.",
    )
    parser.add_argument(
        "--save-fig",
        action="store_true",
        help="Enable figure saving inside each main_stitch run (disabled by default for speed).",
    )
    parser.add_argument(
        "--chain-precompute-segments",
        action="store_true",
        help="Enable chain segment precomputation (disabled by default for speed).",
    )

    # Kept for CLI compatibility; ignored in direct mode.
    parser.add_argument("--no-copy-figures", action="store_true", help=argparse.SUPPRESS)

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
    parser.add_argument(
        "--rel-scale-values",
        nargs="+",
        type=float,
        default=None,
        help="Values for StitchConfig.damm.rel_scale in rel_scale mode.",
    )
    args = parser.parse_args()

    ds_methods = tuple(args.ds_methods or ())
    chain_ds_methods = tuple(_normalize_chain_ds_method(v) for v in (args.chain_ds_methods or ()))
    chain_trigger_methods = tuple(_normalize_trigger_method(v) for v in (args.chain_trigger_methods or ()))
    chain_blend_ratios = tuple(float(v) for v in (args.chain_blend_ratios or ()))
    param_dist_values = tuple(float(v) for v in (args.param_dist_values or ()))
    param_cos_values = tuple(float(v) for v in (args.param_cos_values or ()))
    rel_scale_values = tuple(float(v) for v in (args.rel_scale_values or ()))

    if args.mode in {"standard", "graph_params", "rel_scale"} and len(ds_methods) == 0:
        raise ValueError("--ds-methods is required when --mode standard, --mode graph_params, or --mode rel_scale")
    if args.mode == "graph_params":
        if len(param_dist_values) == 0:
            raise ValueError("--param-dist-values is required when --mode graph_params")
        if len(param_cos_values) == 0:
            raise ValueError("--param-cos-values is required when --mode graph_params")
    if args.mode == "rel_scale" and len(rel_scale_values) == 0:
        raise ValueError("--rel-scale-values is required when --mode rel_scale")
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
        workers=max(1, int(args.workers)),
        copy_figures=not args.no_copy_figures,
        save_fig=bool(args.save_fig),
        chain_precompute_segments=bool(args.chain_precompute_segments),
        mode=args.mode,
        chain_ds_methods=chain_ds_methods,
        chain_trigger_methods=chain_trigger_methods,
        chain_blend_ratios=chain_blend_ratios,
        chain_fixed_ds_method=_normalize_chain_ds_method(args.chain_fixed_ds_method),
        chain_fixed_trigger_method=_normalize_trigger_method(args.chain_fixed_trigger_method),
        param_dist_values=param_dist_values,
        param_cos_values=param_cos_values,
        rel_scale_values=rel_scale_values,
    )


def main():
    cfg = _parse_args()
    run_sweep(cfg)


if __name__ == "__main__":
    main()
