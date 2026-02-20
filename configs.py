from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DammConfig:
    rel_scale: float = 0.1
    total_scale: float = 1.0
    nu_0: int = 5 # NOTE should this be ndim * 2?
    kappa_0: float = 0.1
    psi_dir_0: float = 0.1

@dataclass
class ChainConfig:
    subsystem_edges: int = 2
    blend_length_ratio: float = 0.5
    recompute_gaussians: bool = False # This is automatically set to true for "chain_all"
    ds_method: str = "segmented" # "segmented" or "linear"
    # Supported values:
    # - "mean_normals": transition when crossing the mean-normal plane at n1.
    # - "distance_ratio": transition when d(x,n1)/d(x,n2) >= |e1|/|e2|.
    transition_trigger_method: str = "distance_ratio"
    recovery_distance: float = 0.35
    enable_recovery: bool = False
    stabilization_margin: float = 1e-3
    lmi_tolerance: float = 5e-5
    # Use spatial (position-based) alpha alongside time-based alpha during
    # transitions.  Prevents oscillation when x outruns the time-based blend.
    use_spatial_alpha: bool = False
    # Lower bound on time-based transitions between consecutive subsystems.
    min_transition_time: float = 1e-5
    # Optional speed clip on chain velocity (None disables clipping).
    velocity_max: Optional[float] = None
    # Boundary A matrices (single-node data, separate linear system).
    # When False the first / last segment uses multi-node data (≥3 nodes), like the core systems.
    use_boundary_ds_initial: bool = False
    use_boundary_ds_end: bool = False
    # Chain plotting mode (2D):
    # - "line_regions": hard partitions from nearest path segment
    #   (line-segment Voronoi).
    # - "time_blend": same ownership, with blending across adjacent transition
    #   boundaries; draws black separators where no transition zone exists.
    plot_mode: str = "line_regions"
    plot_grid_resolution: int = 500
    # Display only a corridor around the chain path (2D Euclidean distance).
    # None disables masking and fills the full plotted area.
    plot_path_bandwidth: Optional[float] = None
    # Live-only chain field visualization:
    # - "partition": region partition + lines.
    # - "active_ds": full field of currently active subsystem only.
    live_field_mode: str = "partition"
    plot_show_transition_lines: bool = True
    plot_region_alpha: float = 0.26

# ds_method options:
# - ["lpv-ds_recompute_all"]        Applies LPV-DS to the aggregate of all demonstrations, recomputes all from raw data
# - ["lpv-ds_recompute_ds"]         Applies LPV-DS to the aggregate of all demonstrations, reuses Gaussians.
# - ["sp_recompute_all"]            Uses shortest path, extracts raw traj. points, recomputes Gaussians and DS.
# - ["sp_recompute_ds"]             Uses shortest path, keeps Gaussians but recomputes DS.
# - ["sp_recompute_invalid_As"]     Uses shortest path, selects a P near the attractor, recomputes any incompatible As.
# - ["sp_recompute_P"]              Uses shortest path, keeps Gaussians and As, tries to find a P.
# - ["spt_recompute_all"]           Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["spt_recompute_ds"]            Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["spt_recompute_invalid_As"]    Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["chain"]                       Fit one linear DS per path node and switch/blend between them online (reuses Gaussians).
# - ["chain_all"]                   Same as "chain" but recomputes Gaussians from raw data.

@dataclass
class StitchConfig:
    dataset_path: str = "./dataset/stitching/X"
    force_preprocess: bool = True
    chain_precompute_segments: bool = True
    initial: Optional[np.ndarray] = None
    attractor: Optional[np.ndarray] = None
    ds_method: str = "chain"
    reverse_gaussians: bool = True
    param_dist: int = 2
    param_cos: int = 1
    bhattacharyya_threshold: float = 0.05
    # n_demos: int = 5 # number of demonstrations to generate
    # noise_std: float = 0.05 # standard deviation of noise added to demonstrations
    plot_extent: tuple[float, float, float, float] = (0, 15, 0, 15) # (x_min, x_max, y_min, y_max)
    n_test_simulations: int = 1 # number of test simulations for metrics
    noise_std: float = 0.05
    save_fig: bool = True # standard deviation for initial position for simulation
    save_folder_override: Optional[str] = None
    seed: int = 42 # 42, 100, 3215, 21
    data_position_scale: float = 1.0
    data_velocity_scale: Optional[float] = None
    # Gaussian direction options for graph construction/visualization:
    # - "mean_velocity": normalize mean x_dot of argmax-posterior assigned points.
    # - "a_mu": normalize A_k @ (mu_k - x_att) (legacy behavior).
    gaussian_direction_method: str = "mean_velocity"

    # DAMM settings
    damm: DammConfig = field(default_factory=DammConfig)

    # Chaining settings
    chain: ChainConfig = field(default_factory=ChainConfig)
