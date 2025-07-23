# Linear Parameter Varying Dynamical Systems (LPV-DS)

Boiler plate code of LPV-DS framework, compatible with any customizing clustering and optimization methods. Providing utilies functions from loading_tools, process_tools, plot_tools, and evaluation_tool to test on any variant of LPV-DS framework.


<!-- ![Picture1](https://github.com/SunannnSun/damm_lpvds/assets/97807687/5a72467b-c771-4e8a-a0e0-7828efa59952) -->


## Stitching

The stitching framework allows you to connect multiple trajectory demonstrations by learning Gaussian mixture models and building a graph-based dynamical system that can navigate between different initial and goal positions.

### Quick Start
```bash
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
- **"chain"**: Fit DS to each node and switch between attractors to reach goal

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

## Usage Example

Fetch the required submodules
```
git submodule update --init --recursive
```

Compile [DAMM](https://github.com/SunannnSun/damm) submodule
```
cd src/damm/build
cmake ../src
make
```

Return to root directory and install all dependencies in a virtual environment
- Make sure to replace `/path/to/python3.8` with the correct path to the Python 3.8 executable on your system. 

```
cd -
virtualenv -p /path/to/python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

Run 
```
python main.py
```
