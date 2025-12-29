# STOFS GNN Surrogate Model - Rollout Workflow

This document describes how to run inference/rollout using the trained Temporal Memory GNN model.

## Prerequisites

### 1. Model Checkpoint
The trained model checkpoint should be located at:
```
outputs/checkpoints/best_temporal_memory_model.pt
```

### 2. Data Files
Processed data files are required in:
```
data/processed_25k/
├── mesh_25k.npz          # Mesh file (nodes, edges, coordinates)
└── processed_YYYYMMDD.npz # Daily data files
```

Each processed data file contains:
- `elevation`: Water level timeseries [timesteps, nodes]
- `u10`: U-component of 10m wind [timesteps, nodes]
- `v10`: V-component of 10m wind [timesteps, nodes]
- `pressure`: Atmospheric pressure [timesteps, nodes]

### 3. Environment
- Python 3.x with PyTorch, NumPy, Matplotlib
- CUDA-enabled GPU (recommended) or CPU

---

## Rollout Scripts

### 1. Temporal Rollout (`rollout_temporal_memory_model.py`)

Runs autoregressive rollout and generates timeseries plots at tide gauge stations.

**Location:** `scripts/rollout_temporal_memory_model.py`

**Usage:**
```bash
# Basic 48-hour rollout
python scripts/rollout_temporal_memory_model.py --date 20251128 --hours 48

# With CO-OPS observations overlay
python scripts/rollout_temporal_memory_model.py --date 20251128 --hours 48 --obs

# Save timeseries to text files
python scripts/rollout_temporal_memory_model.py --date 20251128 --hours 48 --save-ts

# Full example with all options
python scripts/rollout_temporal_memory_model.py --date 20251128 --hours 48 --obs --save-ts
```

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--date` | str | 20251128 | Date in YYYYMMDD format |
| `--hours` | int | 48 | Number of hours to rollout |
| `--obs` | flag | False | Fetch and plot CO-OPS observations |
| `--save-ts` | flag | False | Save timeseries to text files |
| `--ts-dir` | str | None | Custom directory for timeseries output |

**Outputs:**
- `outputs/figures/rollout_temporal_memory_YYYYMMDD.png` - Station timeseries plot
- `outputs/timeseries/YYYYMMDD/*.txt` - Timeseries text files (if `--save-ts`)

**Stations Evaluated:**
- Atlantic City, NJ (CO-OPS: 8534720)
- Sandy Hook, NJ (CO-OPS: 8531680)
- The Battery, NY (CO-OPS: 8518750)
- Lewes, DE (CO-OPS: 8557380)
- Cape May, NJ (CO-OPS: 8536110)

---

### 2. Spatial Rollout (`spatial_rollout_temporal_memory.py`)

Generates spatial visualization plots comparing ground truth vs predictions.

**Location:** `scripts/spatial_rollout_temporal_memory.py`

**Usage:**
```bash
# Default forecast hours (6, 12, 24, 36, 48)
python scripts/spatial_rollout_temporal_memory.py --date 20251128

# Custom forecast hours
python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hours 6 12 24 48

# Only scatter plots (skip error maps)
python scripts/spatial_rollout_temporal_memory.py --date 20251128 --scatter-only

# Only error maps
python scripts/spatial_rollout_temporal_memory.py --date 20251128 --error-only

# Generate hourly snapshots
python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hourly

# Only hourly snapshots
python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hourly-only
```

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--date` | str | Required | Date in YYYYMMDD format |
| `--hours` | int list | 6 12 24 36 48 | Forecast hours to visualize |
| `--checkpoint` | str | outputs/checkpoints/best_temporal_memory_model.pt | Model checkpoint path |
| `--mesh` | str | data/processed_25k/mesh_25k.npz | Mesh file path |
| `--data-dir` | str | data/processed_25k | Processed data directory |
| `--output-dir` | str | outputs/figures | Output directory |
| `--scatter-only` | flag | False | Only generate scatter plots |
| `--error-only` | flag | False | Only generate error maps |
| `--hourly` | flag | False | Also generate hourly snapshots |
| `--hourly-only` | flag | False | Only generate hourly snapshots |

**Outputs:**
- `outputs/figures/spatial_rollout_scatter_temporal_YYYYMMDD_hX_X_X.png` - GT vs Prediction
- `outputs/figures/spatial_error_scatter_temporal_YYYYMMDD_hX_X_X.png` - Error maps
- `outputs/figures/hourly_YYYYMMDD/spatial_hour_XX.png` - Hourly snapshots (if `--hourly`)

#### Time-Varying Water Elevation Snapshots

The `--hourly` or `--hourly-only` flags generate individual PNG files for each forecast hour, showing the spatial evolution of water levels over time. This is useful for:
- Creating animations of storm surge propagation
- Analyzing spatial error patterns at different forecast lead times
- Visualizing tidal propagation through the domain

Each hourly snapshot contains 3 panels:
1. **STOFS Ground Truth** - Reference water levels from STOFS model
2. **GNN Prediction** - Model predicted water levels
3. **Error Map** - Difference (Prediction - Ground Truth) with RMSE and Bias metrics

**Example: Generate hourly snapshots**
```bash
# Generate 49 hourly snapshots (hours 0-48)
python scripts/spatial_rollout_temporal_memory.py --date 20251129 --hourly-only
```

**Output:** `outputs/figures/hourly_YYYYMMDD/spatial_hour_XX.png` (49 files for 48h rollout)

**Creating an Animation (optional):**
```bash
# Using ffmpeg to create MP4 from hourly snapshots
cd outputs/figures/hourly_20251129
ffmpeg -framerate 4 -pattern_type glob -i 'spatial_hour_*.png' -c:v libx264 -pix_fmt yuv420p rollout_animation.mp4

# Or create a GIF
convert -delay 25 -loop 0 spatial_hour_*.png rollout_animation.gif
```

---

### 3. Station Comparison (`plot_station_comparison.py`)

Generates publication-quality station timeseries comparison plots.

**Location:** `scripts/plot_station_comparison.py`

**Prerequisites:** Run `rollout_temporal_memory_model.py` with `--save-ts` first.

**Usage:**
```bash
# Generate all plots (individual + combined)
python scripts/plot_station_comparison.py --date 20251128

# With CO-OPS observations
python scripts/plot_station_comparison.py --date 20251128 --obs

# Specific stations only
python scripts/plot_station_comparison.py --date 20251128 --stations Atlantic_City Sandy_Hook

# Only combined plot
python scripts/plot_station_comparison.py --date 20251128 --combined-only

# Only individual station plots
python scripts/plot_station_comparison.py --date 20251128 --individual-only
```

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--date` | str | Required | Date in YYYYMMDD format |
| `--stations` | str list | All | Specific stations to plot |
| `--obs` | flag | False | Include CO-OPS observations |
| `--combined-only` | flag | False | Only generate combined plot |
| `--individual-only` | flag | False | Only generate individual plots |

**Outputs:**
- `outputs/figures/station_comparison/STATION_comparison_YYYYMMDD.png` - Individual plots
- `outputs/figures/station_comparison/all_stations_comparison_YYYYMMDD.png` - Combined plot

---

## Complete Workflow Example

```bash
# Step 1: Run temporal rollout with timeseries output
python scripts/rollout_temporal_memory_model.py --date 20251129 --hours 48 --save-ts

# Step 2: Run spatial rollout visualization
python scripts/spatial_rollout_temporal_memory.py --date 20251129 --hours 6 12 24 36 48

# Step 3: Generate station comparison plots
python scripts/plot_station_comparison.py --date 20251129
```

### Batch Processing Multiple Dates

```bash
for DATE in 20251128 20251129 20251130; do
    echo "Processing $DATE..."
    python scripts/rollout_temporal_memory_model.py --date $DATE --hours 48 --save-ts
    python scripts/spatial_rollout_temporal_memory.py --date $DATE --hours 6 12 24 36 48
    python scripts/plot_station_comparison.py --date $DATE
done
```

---

## Output Directory Structure

After running the workflow, outputs are organized as:

```
outputs/
├── checkpoints/
│   └── best_temporal_memory_model.pt
├── figures/
│   ├── rollout_temporal_memory_YYYYMMDD.png
│   ├── spatial_rollout_scatter_temporal_YYYYMMDD_hX_X_X.png
│   ├── spatial_error_scatter_temporal_YYYYMMDD_hX_X_X.png
│   ├── hourly_YYYYMMDD/
│   │   └── spatial_hour_XX.png
│   └── station_comparison/
│       ├── STATION_comparison_YYYYMMDD.png
│       └── all_stations_comparison_YYYYMMDD.png
└── timeseries/
    └── YYYYMMDD/
        └── STATION_temporal_memory_rollout.txt
```

---

## Model Configuration

The Temporal Memory GNN model uses the following configuration (stored in checkpoint):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 128 | Hidden layer dimension |
| `num_layers` | 6 | Number of GNN message passing layers |
| `static_features` | 4 | x, y, depth, water_level |
| `forcing_features` | 3 | u10, v10, pressure |
| `temporal_features` | 6 | η(t-1), dη/dt, sin/cos M2, sin/cos S2 |
| `num_nodes` | 25,000 | Number of mesh nodes |

---

## Metrics Reported

The scripts calculate and report:

- **RMSE**: Root Mean Square Error (meters)
- **R**: Pearson correlation coefficient
- **Bias**: Mean prediction - ground truth (meters)

---

## Troubleshooting

### Model not found
```
ERROR: Model not found at outputs/checkpoints/best_temporal_memory_model.pt
```
**Solution:** Train the model first using `train_25k_temporal_memory.py`

### Data file not found
```
Error: Data file not found: data/processed_25k/processed_YYYYMMDD.npz
```
**Solution:** Run preprocessing script for the desired date

### Timeseries directory not found
```
Error: Timeseries directory not found
```
**Solution:** Run `rollout_temporal_memory_model.py` with `--save-ts` flag first

### CUDA out of memory
**Solution:** The model should fit on most GPUs. If issues persist, the scripts will fall back to CPU automatically.
