# Demonstration stitching
This code builds on top of the following repositories:
1. [LPVDS](https://github.com/sunan-sun/lpvds)
2. [DAMM](https://github.com/SunannnSun/damm)
3. [DSOPT](https://github.com/sunan-sun/dsopt)
4. [Robottask dataset](https://github.com/sayantanauddy/clfd-snode)

with following references

> [1] Billard, A., Mirrazavi, S., & Figueroa, N. (2022). Learning for adaptive and reactive robot control: a dynamical systems approach. Mit Press.

> [2] Sun, S., Gao, H., Li, T., & Figueroa, N. (2024). "Directionality-aware mixture model parallel sampling for efficient linear parameter varying dynamical system learning". IEEE Robotics and Automation Letters, 9(7), 6248-6255.

> [3] Li, T., Sun, S., Aditya, S. S., & Figueroa, N. (2025). Elastic Motion Policy: An Adaptive Dynamical System for Robust and Efficient One-Shot Imitation Learning. arXiv preprint arXiv:2503.08029.

> [4] Auddy, S.*, Hollenstein, J.*, Saveriano, M., Rodríguez-Sánchez, A., & Piater, J. (2025). Scalable and Efficient Continual Learning from Demonstration via a Hypernetwork-generated Stable Dynamics Model. arXiv preprint arXiv:2311.03600.

Thanks for open sourcing your work!

## Results

### Ablations

#### Stitching ablations


#### Chain ablations
Chain method ablation on X dataset:

| chain_transition_trigger_method   | chain_ds_method   |   prediction_rmse_mean |   cosine_dissimilarity_mean |   dtw_distance_mean |   duration_s |
|:----------------------------------|:------------------|-----------------------:|----------------------------:|--------------------:|-------------:|
| distance_ratio                    | linear            |               0.100918 |                    0.195811 |             2333.61 |      76.808  |
| distance_ratio                    | segmented         |               0.28416  |                    0.21321  |             1075.5  |      91.6978 |
| mean_normals                      | linear            |               0.094193 |                    0.166881 |             2350.73 |      68.7855 |
| mean_normals                      | segmented         |               0.347127 |                    0.19126  |              878.53 |     120.959  |


Chain blend ratio ablation on X dataset:




## Stitching

The stitching framework allows you to connect multiple trajectory demonstrations by learning Gaussian mixture models and building a graph-based dynamical system that can navigate between different initial and goal positions.

### Quick Start
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

### How It Works

`main_stitch.py` implements a complete pipeline for trajectory stitching:

1. **Data Loading**: Load trajectory demonstrations from various sources
2. **Gaussian Learning**: Fit Gaussian mixture models to trajectory segments
3. **Graph Construction**: Build a graph connecting Gaussians based on spatial and directional similarity
4. **Path Planning**: Find shortest paths between initial and goal positions
5. **Dynamical System**: Generate smooth trajectories using Linear Parameter Varying Dynamical Systems (LPV-DS)
6. **Evaluation**: Test all combinations of start/goal positions and generate visualizations

### Configuration Options

The system is configured through the `Config` dataclass in `main_stitch.py`:

```python
@dataclass
class Config:
    input_opt: Optional[int] = 3              # Data source option (1, 2, or 3)
    data_file: Optional[str] = "test_4"       # Custom data file name (for option 3)
    initial: Optional[np.ndarray] = None      # Fixed initial position (if None, uses all combinations)
    attractor: Optional[np.ndarray] = None    # Fixed goal position (if None, uses all combinations)
    ds_method: str = "recompute_all"          # DS computation method
    reverse_gaussians: bool = True            # Duplicate gaussians with reversed directions
    param_dist: int = 3                       # Distance parameter for graph connectivity
    param_cos: int = 3                        # Directionality parameter for graph connectivity
    x_min: float = 0                          # Plot bounds
    x_max: float = 20
    y_min: float = 0
    y_max: float = 20
    save_fig: bool = True                     # Save generated plots
```

### Data Input Options

When running `main_stitch.py`, you'll be prompted to choose a data source:

#### Option 1: X-trajectory Sets
Pre-defined diagonal trajectory patterns for testing.

#### Option 2: Three Separate Trajectories
Pre-defined trajectories with crossings that lead to different goals.

#### Option 3: Custom Drawn Trajectories
Draw your own trajectories interactively using the trajectory drawer tool.

### Drawing Custom Trajectories

To create your own trajectory data:

1. **Choose Option 3** when prompted
2. **Interactive Drawing**: 
   - The trajectory drawer will open an interactive matplotlib window
   - Click and drag to draw trajectories
   - Press 'r' to reset current trajectory
   - Press 'u' to undo last point
   - Press 'n' to start a new trajectory
   - Close the window when finished
3. **Save Your Data**: You'll be prompted to save your trajectories with a custom filename
4. **Automatic Processing**: The system will:
   - Generate multiple noisy demonstrations from your drawn trajectories
   - Fit Gaussian mixture models to the data
   - Cache the results for future use

#### Loading Existing Custom Data

To reuse previously drawn trajectories:

1. Set `data_file` in the Config to your saved filename (without extension)
2. The system will automatically load `./dataset/stitching/{data_file}_traj.pkl`
3. If the file doesn't exist, you'll be prompted to draw new trajectories

### DS Method Options

The `ds_method` parameter controls how the dynamical system is computed:

- **"recompute_all"**: Recompute the entire DS using the shortest path (most accurate)
- **"recompute_ds"**: Only recompute the DS parameters, reuse Gaussian structure
- **"reuse"**: Reuse pre-computed A matrices from individual Gaussians (fastest)
- **"all_paths_all"**: Aggregate all node-wise shortest paths and relearn DS
- **"all_paths_ds"**: Aggregate all node-wise shortest paths and relearn only linear maps
- **"all_paths_reuse"**: Aggregate all node-wise shortest paths with A reuse
- **"chain"**: One linear DS per path node, then online switch/blend to next node target

For chaining, the main control parameters are:
- **"subsystem_edges"**: Number of edges to include in each subsystem
- **"blend_length_ratio"**: Ratio of blend length to subsystem length
- **"transition_trigger_method"**: Method for triggering transitions

### Composite Robot-Task Dataset

Build the connected workspace dataset with `obstaclerotate` as the central corridor:
- `pouring` starts at the obstacle start anchor,
- `pan2stove` starts at the obstacle end anchor,
- `openbox` branches from the obstacle midpoint.

```bash
uv run python build_workspace_dataset.py \
  --output-dir dataset/stitching/robottasks_workspace_chain
```

This generates `demonstration_*` folders and `workspace_plan.json` in:
- `dataset/stitching/robottasks_workspace_chain`

### Output and Visualization

The system generates comprehensive visualizations saved to `./figures/stitching/{data_hash}/{ds_method}/`:

- **graph.png**: The connectivity graph between Gaussians
- **gaussians.png**: Visualization of learned Gaussian mixture components
- **ds_X.png**: Dynamical system trajectories for each start/goal combination

### Example Usage

```python
# Example 1: Use custom drawn trajectories
config = Config(
    input_opt=3,
    data_file="my_custom_trajectories",
    ds_method="recompute_all"
)

# Example 2: Test specific start/goal positions
config = Config(
    input_opt=2,
    initial=np.array([4, 15]),
    attractor=np.array([14, 2]),
    ds_method="reuse"
)

# Example 3: Batch processing with custom bounds
config = Config(
    input_opt=3,
    data_file="workspace_trajectories",
    x_min=-5, x_max=25,
    y_min=-5, y_max=25,
    save_fig=True
)
```

### Interactive Initial/Goal Selection

The system includes an interactive UI for selecting initial and goal positions:

- **Visual Interface**: Shows all learned Gaussians as blue circles
- **Drag-and-Drop**: Green point (initial) and red point (goal) can be moved intuitively
- **Real-time Feedback**: Positions update as you drag
- **Confirmation**: Click "Confirm Selection" to proceed

### Tips for Best Results

1. **Draw Smooth Trajectories**: Avoid sharp corners or erratic movements
2. **Multiple Demonstrations**: Draw several similar trajectories to improve learning
3. **Overlap Regions**: Ensure trajectories have some overlapping regions for connectivity
4. **Consistent Speed**: Try to maintain consistent drawing speed for better velocity estimation
5. **Clear Goals**: Make sure trajectories clearly converge to their intended goals

## Sweep

```bash
python sweep.py \
  --datasets dataset/stitching/X dataset/stitching/robottasks_obstacle_bottle2shelf_side dataset/stitching/robottasks_workspace_chain \
  --ds-methods sp_recompute_all sp_recompute_ds chain \
  --seeds 1 2 3 \
  --output-dir results/sweep_results \
  --timeout-s 600 --workers 12
```

Sweep 1: chaining method comparison (segment vs linear) x (mean_normals vs distance_ratio)
```bash
uv run sweep.py \
  --mode chain_trigger \
  --datasets dataset/stitching/robottasks_workspace_chain dataset/stitching/nodes_1 dataset/stitching/presentation2 \
  --seeds 1 2 3 \
  --chain-ds-methods segment linear \
  --chain-trigger-methods mean_normals distance_ratio \
  --output-dir results/sweep_chain_trigger \
  --timeout-s 600 --workers 12
```

Sweep 2: blend-length sweep
```bash
uv run sweep.py \
  --mode chain_blend \
  --datasets dataset/stitching/X \
  --seeds 1 2 3 \
  --chain-blend-ratios 0.0 0.25 0.5 0.75 1.0 \
  --chain-fixed-ds-method segmented \
  --chain-fixed-trigger-method mean_normals \
  --output-dir results/sweep_chain_blend \
  --timeout-s 600 --workers 12
```

Sweep 3:

```bash
uv run sweep.py \
  --mode graph_params \
  --datasets dataset/stitching/X \
  --ds-methods sp_recompute_ds chain \
  --seeds 1 2 \
  --param-dist-values 1 2 3 \
  --param-cos-values 1 2 3 \
  --output-dir results/sweep_graph_params \
  --timeout-s 600 --workers 12
```