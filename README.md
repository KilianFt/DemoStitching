# Demonstration stitching
This is the official implementation of our stitching framework that enables stitching together demonstrations for different tasks in a shared workspace. We do this by first learning Gaussian mixture models from the demonstrations and then building a graph-based dynamical system that can navigate between different initial and goal positions.

This code builds on top of the following repositories:
1. [LPVDS](https://github.com/sunan-sun/lpvds)
2. [DAMM](https://github.com/SunannnSun/damm)
3. [DSOPT](https://github.com/sunan-sun/dsopt)
<!-- 4. [Robottask dataset](https://github.com/sayantanauddy/clfd-snode) -->

with following references

> [1] Billard, A., Mirrazavi, S., & Figueroa, N. (2022). Learning for adaptive and reactive robot control: a dynamical systems approach. Mit Press.

> [2] Sun, S., Gao, H., Li, T., & Figueroa, N. (2024). "Directionality-aware mixture model parallel sampling for efficient linear parameter varying dynamical system learning". IEEE Robotics and Automation Letters, 9(7), 6248-6255.

> [3] Li, T., Sun, S., Aditya, S. S., & Figueroa, N. (2025). Elastic Motion Policy: An Adaptive Dynamical System for Robust and Efficient One-Shot Imitation Learning. arXiv preprint arXiv:2503.08029.

<!-- > [4] Auddy, S.*, Hollenstein, J.*, Saveriano, M., Rodríguez-Sánchez, A., & Piater, J. (2025). Scalable and Efficient Continual Learning from Demonstration via a Hypernetwork-generated Stable Dynamics Model. arXiv preprint arXiv:2311.03600. -->

Thanks for open sourcing your work!

## Results


### Chain blend ablations
Chain blend ratio ablation on X dataset:
|    |   chain_blend_ratio | prediction_rmse   | cosine_dissimilarity   | dtw_distance_mean    | trajectory_length_mean   |
|---:|--------------------:|:------------------|:-----------------------|:---------------------|:-------------------------|
|  0 |                0    | 0.28 ± 0.33       | 0.20 ± 0.13            | 60398.87 ± 372691.65 | 21904.61 ± 17603.52      |
|  1 |                0.25 | 0.31 ± 0.19       | 0.26 ± 0.15            | 7554.27 ± 6696.75    | 18558.24 ± 8537.02       |
|  2 |                0.5  | 0.22 ± 0.15       | 0.22 ± 0.14            | 6304.31 ± 5646.79    | 22077.16 ± 19668.61      |
|  3 |                0.75 | 0.41 ± 0.54       | 0.22 ± 0.12            | 8695.16 ± 10298.26   | 24602.11 ± 24662.31      |
|  4 |                1    | 0.27 ± 0.37       | 0.23 ± 0.14            | 6934.35 ± 6503.77    | 23049.79 ± 19980.41      |


## Quick Start
```bash
uv run main_stitch.py
```

Otherwise you can use conda
```bash
conda create -n lpvds_env python=3.9
conda activate lpvds_env
pip install -r requirements.txt
python main_stitch.py
```

To run an interactive demo run
```bash
uv run live.py --dataset-path dataset/stitching/X --ds-method chain
```


## How It Works

`main_stitch.py` implements a complete pipeline for trajectory stitching:

1. **Data Loading**: Load trajectory demonstrations from various sources
2. **Gaussian Learning**: Fit Gaussian mixture models to trajectory segments
3. **Graph Construction**: Build a graph connecting Gaussians based on spatial and directional similarity
4. **Path Planning**: Find shortest paths between initial and goal positions
5. **Dynamical System**: Generate smooth trajectories using Linear Parameter Varying Dynamical Systems (LPV-DS)
6. **Evaluation**: Test all combinations of start/goal positions and generate visualizations

The system is configured through the `StitchConfig` dataclass in `configs.py`.



### DS Method Options
The `ds_method` parameter controls how the dynamical system is computed:

- **"lpv-ds_recompute_all"**: Applies LPV-DS to the aggregate of all demonstrations, recomputes all from raw data
- **"lpv-ds_recompute_ds"**: Applies LPV-DS to the aggregate of all demonstrations, reuses Gaussians.
- **"sp_recompute_all"**: Uses shortest path, extracts raw traj. points, recomputes Gaussians and DS.
- **"sp_recompute_ds"**: Uses shortest path, keeps Gaussians but recomputes DS.
- **"sp_recompute_invalid_As"**: Uses shortest path, selects a P near the attractor, recomputes any incompatible As.
- **"sp_recompute_P"**: Uses shortest path, keeps Gaussians and As, tries to find a P.
- **"spt_recompute_all"**: Uses shortest path tree, otherwise same as corresponding "sp" method.
- **"spt_recompute_ds"**: Uses shortest path tree, otherwise same as corresponding "sp" method.
- **"spt_recompute_invalid_As"**: Uses shortest path tree, otherwise same as corresponding "sp" method.
- **"chain"**: Fit one linear DS per path node and switch/blend between them online (reuses Gaussians).
- **"chain_all"**: Same as "chain" but recomputes Gaussians from raw data


### Drawing Custom Trajectories
When a dataset you set is not present, you can draw your own trajectories:

**Interactive Drawing**: 
   - The trajectory drawer will open an interactive matplotlib window
   - Click and drag to draw trajectories
   - Press 'r' to reset current trajectory
   - Press 'u' to undo last point
   - Press 'n' to start a new trajectory
   - Close the window when finished
**Save Your Data**: You'll be prompted to save your trajectories with a custom filename
**Automatic Processing**: The system will:
   - Generate multiple noisy demonstrations from your drawn trajectories
   - Fit Gaussian mixture models to the data
   - Cache the results for future use

#### Tips for Best Results
1. **Draw Smooth Trajectories**: Avoid sharp corners or erratic movements
2. **Multiple Demonstrations**: Draw several similar trajectories to improve learning
3. **Overlap Regions**: Ensure trajectories have some overlapping regions for connectivity
4. **Consistent Speed**: Try to maintain consistent drawing speed for better velocity estimation
5. **Clear Goals**: Make sure trajectories clearly converge to their intended goals


## Sweep
Here the commands to reproduce the results:
### Overall method comparison
```bash
uv run sweep.py \
  --datasets dataset/stitching/X dataset/stitching/2d_large dataset/stitching/pcgmm_3d_workspace_simple \
  --ds-methods lpv-ds_recompute_all lpv-ds_recompute_ds sp_recompute_all sp_recompute_ds chain chain_all \
  --seeds 1 2 3 \
  --output-dir results/methods \
  --timeout-s 2000 --workers 8 --save-fig
```

<!-- Sweep 1: Graph parameters sweep
```bash
uv run sweep.py \
  --mode graph_params \
  --datasets dataset/stitching/X \
  --ds-methods sp_recompute_ds chain \
  --seeds 1 2 \
  --param-dist-values 1 2 3 \
  --param-cos-values 1 2 3 \
  --output-dir results/graph_params \
  --timeout-s 600 --workers 8
``` -->

### Chain blend-length sweep
```bash
uv run sweep.py \
  --mode chain_blend \
  --datasets dataset/stitching/X \
  --seeds 1 2 \
  --chain-blend-ratios 0.0 0.25 0.5 0.75 1.0 \
  --chain-fixed-ds-method segmented \
  --chain-fixed-trigger-method distance_ratio \
  --output-dir results/chain_blend \
  --timeout-s 600 --workers 10
```
