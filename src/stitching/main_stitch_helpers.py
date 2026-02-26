from __future__ import annotations

import os
import signal
import threading

import numpy as np
import pandas as pd

from configs import StitchConfig
from src.stitching.metrics import calculate_ds_metrics


class OperationTimeout(TimeoutError):
    """Raised when a guarded operation exceeds the configured timeout."""


def call_with_timeout(timeout_s, label, fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` with a SIGALRM timeout when available."""
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError):
        timeout = 0.0

    if not np.isfinite(timeout) or timeout <= 0.0:
        return fn(*args, **kwargs)
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return fn(*args, **kwargs)
    if threading.current_thread() is not threading.main_thread():
        return fn(*args, **kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(signum, frame):
        del signum, frame
        raise OperationTimeout(f"{label} timeout after {timeout:.3f}s")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def extract_gaussian_node_indices(node_id):
    """Return ``(ds_idx, gaussian_idx)`` from a graph node id."""
    if isinstance(node_id, np.ndarray):
        node_tuple = tuple(node_id.tolist())
    elif isinstance(node_id, (tuple, list)):
        node_tuple = tuple(node_id)
    else:
        raise TypeError(f"Unsupported Gaussian node id type: {type(node_id)}")

    if len(node_tuple) < 2:
        raise ValueError(f"Gaussian node id must have at least 2 entries, got: {node_tuple!r}")

    return int(node_tuple[0]), int(node_tuple[1])


def simulate_trajectories(ds, initial, config):
    """Simulate multiple trajectories from noisy initial conditions."""
    if ds is None:
        return None

    x_inits = [
        initial + np.random.normal(0, config.noise_std, initial.shape[0])
        for _ in range(config.n_test_simulations)
    ]
    simulated_trajectories = []
    for x_0 in x_inits:
        simulated_trajectories.append(ds.sim(x_0[None, :], dt=0.01)[0])

    return simulated_trajectories


def default_stitching_stats() -> dict[str, float]:
    return {
        "gg_solution_compute_time": np.nan,
        "ds_compute_time": np.nan,
        "total_compute_time": np.nan,
    }


def nan_ds_metrics(initial: np.ndarray, attractor: np.ndarray) -> dict[str, float]:
    return calculate_ds_metrics(
        x_ref=None,
        x_dot_ref=None,
        ds=None,
        sim_trajectories=None,
        initial=np.asarray(initial, dtype=float),
        attractor=np.asarray(attractor, dtype=float),
    )


def checkpoint_results_csv(all_results: list[dict], save_path: str) -> None:
    """Write incremental results so sweeps retain partial progress on long runs."""
    if not all_results:
        return
    try:
        pd.DataFrame(all_results).to_csv(save_path, index=False)
    except Exception as exc:
        print(f"Warning: failed to checkpoint partial results to {save_path}: {exc}")


def resolve_save_figure_indices(config: StitchConfig) -> set[int] | None:
    if not bool(config.save_fig):
        return None

    explicit = getattr(config, "save_fig_indices", None)
    if explicit is not None:
        return {int(i) for i in explicit}

    dataset_path = str(config.dataset_path).replace("\\", "/").rstrip("/")
    dataset_name = os.path.basename(dataset_path)
    by_dataset = getattr(config, "save_fig_indices_by_dataset", {})
    if not isinstance(by_dataset, dict):
        return None

    for key, indices in by_dataset.items():
        key_norm = str(key).replace("\\", "/").rstrip("/")
        if (
            dataset_path == key_norm
            or dataset_path.endswith("/" + key_norm)
            or dataset_name == key_norm
        ):
            return {int(i) for i in indices}
    return None


def extract_chain_segments_from_path_nodes(path_nodes):
    """Match chain segment extraction used by segmented chaining builders."""
    path_nodes = list(path_nodes) if path_nodes is not None else []
    segment_size = 2
    if len(path_nodes) - 1 < segment_size:
        segment_size = len(path_nodes) - 1
    if len(path_nodes) - 1 < segment_size:
        return [tuple(path_nodes)]
    segments = []
    for i in range(0, len(path_nodes) - segment_size):
        segments.append(tuple(path_nodes[i: i + segment_size + 1]))
    return segments
