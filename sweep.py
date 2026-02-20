from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import io
import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from configs import StitchConfig


RunMainFn = Callable[[StitchConfig, str], Optional[List[dict]]]


@dataclass(frozen=True)
class SweepConfig:
    datasets: tuple[str, ...]
    ds_methods: tuple[str, ...]
    seeds: tuple[int, ...]
    output_dir: str
    n_test_simulations: int = 3
    timeout_s: float = 0.0
    workers: int = 1
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
    """Yield one spec dict per run.  Only mode-relevant keys are included."""

    def _base(dataset_path, ds_method, seed, **extra):
        return {"dataset_path": dataset_path, "ds_method": ds_method, "seed": int(seed), **extra}

    if cfg.mode == "standard":
        for dp in cfg.datasets:
            for dm in cfg.ds_methods:
                for s in cfg.seeds:
                    yield _base(dp, dm, s)
        return

    if cfg.mode == "graph_params":
        for dp in cfg.datasets:
            for dm in cfg.ds_methods:
                for pd_val in cfg.param_dist_values:
                    for pc_val in cfg.param_cos_values:
                        for s in cfg.seeds:
                            yield _base(dp, dm, s, param_dist=float(pd_val), param_cos=float(pc_val))
        return

    if cfg.mode == "rel_scale":
        for dp in cfg.datasets:
            for dm in cfg.ds_methods:
                for rs in cfg.rel_scale_values:
                    for s in cfg.seeds:
                        yield _base(dp, dm, s, rel_scale=float(rs))
        return

    if cfg.mode == "chain_trigger":
        for dp in cfg.datasets:
            for cdm in cfg.chain_ds_methods:
                for ctm in cfg.chain_trigger_methods:
                    for s in cfg.seeds:
                        yield _base(dp, "chain", s, chain_ds_method=cdm, chain_trigger_method=ctm)
        return

    if cfg.mode == "chain_blend":
        for dp in cfg.datasets:
            for cbr in cfg.chain_blend_ratios:
                for s in cfg.seeds:
                    yield _base(dp, "chain", s,
                                chain_ds_method=cfg.chain_fixed_ds_method,
                                chain_trigger_method=cfg.chain_fixed_trigger_method,
                                chain_blend_ratio=float(cbr))
        return

    raise ValueError(f"Unsupported sweep mode: {cfg.mode}")


# Optional per-run parameter keys (may or may not be present in a spec dict).
_OPTIONAL_SPEC_KEYS = (
    "chain_ds_method", "chain_trigger_method", "chain_blend_ratio",
    "param_dist", "param_cos", "rel_scale",
)


def _combo_tag(spec: dict[str, object]) -> str:
    tag = f"{_dataset_slug(str(spec['dataset_path']))}__{spec['ds_method']}__seed_{spec['seed']}"
    for key in _OPTIONAL_SPEC_KEYS:
        val = spec.get(key)
        if val is not None:
            tag += f"__{key}_{_float_tag(float(val)) if isinstance(val, float) else val}"
    return tag


def _build_stitch_config(cfg: SweepConfig, spec: dict[str, object]) -> StitchConfig:
    stitch_cfg = StitchConfig()
    stitch_cfg.dataset_path = str(spec["dataset_path"])
    stitch_cfg.ds_method = str(spec["ds_method"])
    stitch_cfg.seed = int(spec["seed"])
    stitch_cfg.n_test_simulations = int(cfg.n_test_simulations)
    stitch_cfg.save_fig = bool(cfg.save_fig)
    if stitch_cfg.save_fig:
        combo_tag = _combo_tag(spec)
        stitch_cfg.save_folder_override = str(Path(cfg.output_dir) / "figures" / combo_tag) + "/"

    _setters: dict[str, Callable] = {
        "chain_ds_method": lambda v: setattr(stitch_cfg.chain, "ds_method", str(v)),
        "chain_trigger_method": lambda v: setattr(stitch_cfg.chain, "transition_trigger_method", str(v)),
        "chain_blend_ratio": lambda v: setattr(stitch_cfg.chain, "blend_length_ratio", float(v)),
        "param_dist": lambda v: setattr(stitch_cfg, "param_dist", float(v)),
        "param_cos": lambda v: setattr(stitch_cfg, "param_cos", float(v)),
        "rel_scale": lambda v: setattr(stitch_cfg.damm, "rel_scale", float(v)),
    }
    for key, setter in _setters.items():
        val = spec.get(key)
        if val is not None:
            setter(val)

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
    # Poll in short intervals so KeyboardInterrupt is not blocked.
    deadline = time.perf_counter() + float(timeout_s)
    while proc.is_alive() and time.perf_counter() < deadline:
        proc.join(timeout=0.5)

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


# ── Column names in the summary CSV that map to optional spec keys ──────────
_SPEC_KEY_TO_COL = {
    "chain_ds_method": ("chain_ds_method", str, ""),
    "chain_trigger_method": ("chain_transition_trigger_method", str, ""),
    "chain_blend_ratio": ("chain_blend_length_ratio", float, np.nan),
    "param_dist": ("param_dist", float, np.nan),
    "param_cos": ("param_cos", float, np.nan),
    "rel_scale": ("rel_scale", float, np.nan),
}


def _build_result_row(
    cfg: SweepConfig,
    spec: dict[str, object],
    run_index: int,
    *,
    status: str,
    failure_reason: str = "",
    error_message: str = "",
    timed_out: bool = False,
    duration_s: float = math.nan,
    results_csv_source: str = "",
    metrics: dict[str, float] | None = None,
) -> dict:
    """Build one row of the sweep summary CSV."""
    dataset_path = str(spec["dataset_path"])
    row: dict = {
        "run_index": int(run_index),
        "dataset_path": dataset_path,
        "dataset_slug": _dataset_slug(dataset_path),
        "sweep_mode": cfg.mode,
        "ds_method": str(spec["ds_method"]),
    }
    for spec_key, (col_name, col_type, default) in _SPEC_KEY_TO_COL.items():
        val = spec.get(spec_key)
        row[col_name] = col_type(val) if val is not None else default
    row.update({
        "seed": int(spec["seed"]),
        "n_test_simulations": int(cfg.n_test_simulations),
        "status": status,
        "failure_reason": failure_reason,
        "error_message": error_message,
        "timed_out": timed_out,
        "duration_s": duration_s,
        "results_csv_source": results_csv_source,
        **(metrics or _empty_metric_summary()),
    })
    return row


def _enrich_raw_csv(csv_path: Path, spec: dict[str, object], cfg: SweepConfig) -> None:
    """Inject sweep metadata columns into a per-run result CSV so it is self-describing."""
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    df.insert(0, "dataset_path", str(spec["dataset_path"]))
    df.insert(1, "dataset_slug", _dataset_slug(str(spec["dataset_path"])))
    df.insert(2, "sweep_mode", cfg.mode)
    df.insert(3, "seed", int(spec["seed"]))
    # Add optional parameters that were varied in this sweep.
    for key in _OPTIONAL_SPEC_KEYS:
        val = spec.get(key)
        if val is not None and key not in df.columns:
            df[key] = val
    df.to_csv(csv_path, index=False)


def _run_single_spec(
    cfg: SweepConfig,
    spec: dict[str, object],
    run_main_fn: RunMainFn,
    raw_results_dir: Path,
    run_index: int,
    total_runs: int,
    announce_start: bool = True,
) -> dict:
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

    # Enrich the raw CSV with sweep metadata so it is self-describing.
    _enrich_raw_csv(run_results_csv, spec, cfg)

    # Avoid figure accumulation if a run accidentally opens figures.
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass

    return _build_result_row(
        cfg, spec, run_index,
        status="ok" if run_ok else "failed",
        failure_reason="" if run_ok else failure_reason,
        error_message=run_error,
        timed_out=bool(timed_out),
        duration_s=float(elapsed),
        results_csv_source=str(run_results_csv) if run_results_csv.exists() else "",
        metrics=metric_summary,
    )


def _internal_failure_row(cfg: SweepConfig, spec: dict[str, object], run_index: int, error_message: str) -> dict:
    return _build_result_row(
        cfg, spec, run_index,
        status="failed",
        failure_reason="runner_error",
        error_message=str(error_message),
    )


def run_sweep(cfg: SweepConfig, run_main_fn: RunMainFn = _default_run_main) -> pd.DataFrame:
    output_root = Path(cfg.output_dir)
    raw_results_dir = output_root / "raw_results"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_dir.mkdir(parents=True, exist_ok=True)

    specs = list(_iter_run_specs(cfg))
    total_runs = len(specs)
    rows_by_index: dict[int, dict] = {}

    max_workers = max(1, int(getattr(cfg, "workers", 1)))
    interrupted = False
    if max_workers <= 1 or total_runs <= 1:
        for run_index, spec in enumerate(specs, start=1):
            try:
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
            except KeyboardInterrupt:
                print(f"\nInterrupted at run {run_index}/{total_runs}. Saving partial results...")
                interrupted = True
                break
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
            try:
                for future in cf.as_completed(future_to_meta):
                    run_index, spec, combo_tag = future_to_meta[future]
                    completed += 1
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = _internal_failure_row(cfg, spec, run_index, f"{type(exc).__name__}: {exc}")
                    rows_by_index[run_index] = row
                    print(f"[done {completed}/{total_runs}] {combo_tag} -> {row['status']}")
            except KeyboardInterrupt:
                print(f"\nInterrupted after {completed}/{total_runs} runs. Cancelling pending...")
                interrupted = True
                for f in future_to_meta:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

    rows = [rows_by_index[i] for i in sorted(rows_by_index.keys())]
    for row in rows:
        row.pop("run_index", None)

    df = pd.DataFrame(rows)
    out_csv = output_root / "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSweep finished. Saved summary CSV: {out_csv.resolve()}")
    print(f"Per-run CSVs (with metadata): {raw_results_dir.resolve()}/")
    return df


def load_raw_results(output_dir: str) -> pd.DataFrame:
    """Load and concatenate all per-run CSVs from <output_dir>/raw_results/.

    Each CSV already contains sweep metadata columns (dataset_slug, seed, etc.)
    injected by the sweep runner, so the returned DataFrame is ready for
    groupby / plotting.

    Usage (e.g. in a notebook)::

        from sweep import load_raw_results
        df = load_raw_results("results/sweep_chain_trigger")
    """
    raw_dir = Path(output_dir) / "raw_results"
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


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
