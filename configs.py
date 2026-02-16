from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DammConfig:
    rel_scale: float = 0.1 # 0.7
    total_scale: float = 1.0 # 1.5
    nu_0: int = 5 # TODO should this be ndim * 2?
    kappa_0: float = 0.1
    psi_dir_0: float = 0.1 # 1.0

@dataclass
class ChainConfig:
    subsystem_edges: int = 2
    blend_length_ratio: float = 0.1
    # Supported values:
    # - "mean_normals": transition when crossing the mean-normal plane at n1.
    # - "distance_ratio": transition when d(x,n1)/d(x,n2) >= |e1|/|e2|.
    transition_trigger_method: str = "distance_ratio"
    recovery_distance: float = 0.35
    enable_recovery: bool = False
    stabilization_margin: float = 1e-3
    lmi_tolerance: float = 5e-5

# ds_method options:
# - ["sp_recompute_all"]            Uses shortest path, extracts raw traj. points, recomputes Gaussians and DS.
# - ["sp_recompute_ds"]             Uses shortest path, keeps Gaussians but recomputes DS.
# - ["sp_recompute_invalid_As"]     Uses shortest path, selects a P near the attractor, recomputes any incompatible As.
# - ["sp_recompute_P"]              Uses shortest path, keeps Gaussians and As, tries to find a P.
# - ["spt_recompute_all"]           Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["spt_recompute_ds"]            Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["spt_recompute_invalid_As"]    Uses shortest path tree, otherwise same as corresponding "sp" method.
# - ["chain"]                       Fit one linear DS per path node and switch/blend between them online.

@dataclass
class StitchConfig:
    dataset_path: str = "./dataset/stitching/presentation2"
    force_preprocess: bool = True
    initial: Optional[np.ndarray] = None
    attractor: Optional[np.ndarray] = None
    ds_method: str = "chain"
    reverse_gaussians: bool = True
    param_dist: int = 1
    param_cos: int = 1
    bhattacharyya_threshold: float = 0.05
    # n_demos: int = 5 # number of demonstrations to generate
    # noise_std: float = 0.05 # standard deviation of noise added to demonstrations
    plot_extent: tuple[float, float, float, float] = (0, 15, 0, 15) # (x_min, x_max, y_min, y_max)
    n_test_simulations: int = 2 # number of test simulations for metrics
    noise_std: float = 0.05
    save_fig: bool = True
    seed: int = 42 # 42, 100, 3215, 21
    data_position_scale: float = 1.0
    data_velocity_scale: Optional[float] = None

    # DAMM settings
    damm: DammConfig = field(default_factory=DammConfig)

    # Chaining settings
    chain: ChainConfig = field(default_factory=ChainConfig)
