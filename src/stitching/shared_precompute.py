from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Callable

import src.graph_utils as gu
from src.util.ds_tools import apply_lpvds_demowise, get_gaussian_directions

_SHARED_PRECOMPUTE_SCHEMA_VERSION = 1


def compute_lpvds_and_graph(
    demo_set,
    config,
    *,
    apply_lpvds_demowise_fn: Callable = apply_lpvds_demowise,
    get_gaussian_directions_fn: Callable = get_gaussian_directions,
    gaussian_graph_cls: Callable = gu.GaussianGraph,
) -> dict[str, Any]:
    """Compute the shared LPV-DS + GaussianGraph preprocessing payload."""
    t0 = time.time()
    ds_set, reversed_ds_set, norm_demo_set = apply_lpvds_demowise_fn(demo_set, config.damm)
    ds_compute_time = float(time.time() - t0)

    t0 = time.time()
    direction_method = getattr(config, "gaussian_direction_method", "mean_velocity")
    gaussians = {
        (i, j): {"mu": mu, "sigma": sigma, "direction": direction, "prior": prior}
        for i, ds in enumerate(ds_set)
        for j, (mu, sigma, direction, prior) in enumerate(
            zip(
                ds.damm.Mu,
                ds.damm.Sigma,
                get_gaussian_directions_fn(ds, method=direction_method),
                ds.damm.Prior,
            )
        )
    }
    gg = gaussian_graph_cls(
        param_dist=config.param_dist,
        param_cos=config.param_cos,
        bhattacharyya_threshold=config.bhattacharyya_threshold,
    )
    gg.add_gaussians(gaussians, reverse_gaussians=config.reverse_gaussians)
    gg_compute_time = float(time.time() - t0)

    return {
        "schema_version": _SHARED_PRECOMPUTE_SCHEMA_VERSION,
        "ds_set": ds_set,
        "reversed_ds_set": reversed_ds_set,
        "norm_demo_set": norm_demo_set,
        "gg": gg,
        "ds_compute_time": ds_compute_time,
        "gg_compute_time": gg_compute_time,
    }


def save_shared_precompute(path: str | Path, payload: dict[str, Any]) -> None:
    """Save shared preprocessing payload atomically."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(artifact_path)


def load_shared_precompute(path: str | Path) -> dict[str, Any]:
    """Load shared preprocessing payload and validate schema."""
    artifact_path = Path(path)
    with open(artifact_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid shared precompute payload type: {type(payload)}")
    version = int(payload.get("schema_version", -1))
    if version != _SHARED_PRECOMPUTE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported shared precompute schema version: "
            f"{version} (expected {_SHARED_PRECOMPUTE_SCHEMA_VERSION})"
        )

    required = (
        "ds_set",
        "reversed_ds_set",
        "norm_demo_set",
        "gg",
        "ds_compute_time",
        "gg_compute_time",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Invalid shared precompute payload; missing keys: {missing}")
    return payload


def build_or_load_shared_precompute(
    config,
    demo_set,
    *,
    apply_lpvds_demowise_fn: Callable = apply_lpvds_demowise,
    get_gaussian_directions_fn: Callable = get_gaussian_directions,
    gaussian_graph_cls: Callable = gu.GaussianGraph,
) -> dict[str, Any]:
    """Load precompute artifact when configured; otherwise compute in-process."""
    artifact_path_raw = getattr(config, "shared_precompute_artifact_path", None)
    if artifact_path_raw:
        artifact_path = Path(str(artifact_path_raw))
        if artifact_path.exists():
            payload = load_shared_precompute(artifact_path)
            payload["loaded_from_artifact"] = True
            payload["artifact_path"] = str(artifact_path)
            return payload

        payload = compute_lpvds_and_graph(
            demo_set,
            config,
            apply_lpvds_demowise_fn=apply_lpvds_demowise_fn,
            get_gaussian_directions_fn=get_gaussian_directions_fn,
            gaussian_graph_cls=gaussian_graph_cls,
        )
        save_shared_precompute(artifact_path, payload)
        payload["loaded_from_artifact"] = False
        payload["artifact_path"] = str(artifact_path)
        return payload

    payload = compute_lpvds_and_graph(
        demo_set,
        config,
        apply_lpvds_demowise_fn=apply_lpvds_demowise_fn,
        get_gaussian_directions_fn=get_gaussian_directions_fn,
        gaussian_graph_cls=gaussian_graph_cls,
    )
    payload["loaded_from_artifact"] = False
    payload["artifact_path"] = None
    return payload
