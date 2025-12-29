# CWL GNN Surrogate Model - Progress Report

**Date**: December 4, 2025
**Status**: Phase 2 Complete - Ensemble Capability + Mid-Atlantic Regional Model

---

## Overview

This document tracks progress on developing a Graph Neural Network (GNN) surrogate model for STOFS 2D Global bias-corrected Coastal Water Level (CWL) predictions. The goal is to enable rapid ensemble generation by replacing computationally expensive ADCIRC simulations with fast GNN inference.

---

## Data Sources

### Available Raw Data (Multi-Cycle)

**Location**: `data/raw/`

#### November 22, 2025 (4 cycles)
| Cycle | File Type | Nodes | Elements | Time Steps | Zeta Range |
|-------|-----------|-------|----------|------------|------------|
| t00z | CWL time series | 12,785,004 | 24,875,336 | 186 | -5.11 to 6.45 m |
| t06z | CWL time series | 12,785,004 | 24,875,336 | 186 | -5.40 to 6.28 m |
| t12z | CWL time series | 12,785,004 | 24,875,336 | 186 | -4.99 to 6.51 m |
| t18z | CWL time series | 12,785,004 | 24,875,336 | 186 | -5.06 to 6.28 m |
| t00z | MAXELE | 12,785,004 | 24,875,336 | - | -0.38 to 7.17 m |
| t06z | MAXELE | 12,785,004 | 24,875,336 | - | -0.38 to 7.33 m |
| t12z | MAXELE | 12,785,004 | 24,875,336 | - | -0.38 to 7.47 m |
| t18z | MAXELE | 12,785,004 | 24,875,336 | - | -0.38 to 7.40 m |

#### November 23, 2025 (4 cycles)
| Cycle | File Type | Nodes | Elements | Time Steps | Zeta Range |
|-------|-----------|-------|----------|------------|------------|
| t00z | CWL time series | 12,785,004 | 24,875,336 | 186 | -4.99 to 6.44 m |
| t06z | CWL time series | 12,785,004 | 24,875,336 | 186 | -5.96 to 6.28 m |
| t12z | CWL time series | 12,785,004 | 24,875,336 | 186 | -4.99 to 6.70 m |
| t18z | CWL time series | 12,785,004 | 24,875,336 | 186 | -5.39 to 6.28 m |
| t00z | MAXELE | 12,785,004 | 24,875,336 | - | -0.38 to 7.41 m |
| t06z | MAXELE | 12,785,004 | 24,875,336 | - | -0.38 to 6.84 m |
| t12z | MAXELE | 12,785,004 | 24,875,336 | - | -0.38 to 7.10 m |
| t18z | MAXELE | 12,785,004 | 24,875,336 | - | -0.38 to 7.23 m |


#### File Structure
```
data/raw/
├── stofs_2d_glo.20251122/
│   ├── stofs_2d_glo.t00z.fields.cwl.nc      # CWL time series (symlink)
│   ├── stofs_2d_glo.t06z.fields.cwl.nc
│   ├── stofs_2d_glo.t12z.fields.cwl.nc
│   ├── stofs_2d_glo.t18z.fields.cwl.nc
│   ├── stofs_2d_glo.t00z.fields.cwl.maxele.nc  # Max elevation (symlink)
│   ├── stofs_2d_glo.t06z.fields.cwl.maxele.nc
│   ├── stofs_2d_glo.t12z.fields.cwl.maxele.nc
│   └── stofs_2d_glo.t18z.fields.cwl.maxele.nc
├── stofs_2d_glo.20251123/
│   ├── stofs_2d_glo.t00z.fields.cwl.nc
│   ├── stofs_2d_glo.t06z.fields.cwl.nc
│   ├── stofs_2d_glo.t12z.fields.cwl.nc
│   ├── stofs_2d_glo.t18z.fields.cwl.nc
│   ├── stofs_2d_glo.t00z.fields.cwl.maxele.nc
│   ├── stofs_2d_glo.t06z.fields.cwl.maxele.nc
│   ├── stofs_2d_glo.t12z.fields.cwl.maxele.nc
│   └── stofs_2d_glo.t18z.fields.cwl.maxele.nc
```

#### NetCDF Variables (CWL Time Series Files)
| Variable | Shape | Description |
|----------|-------|-------------|
| `zeta` | (186, 12785004) | Bias-corrected water level (m) |
| `x` | (12785004,) | Longitude (degrees_east) |
| `y` | (12785004,) | Latitude (degrees_north) |
| `depth` | (12785004,) | Bathymetry below geoid (m) |
| `element` | (24875336, 3) | Triangle connectivity |
| `time` | (186,) | Model time (seconds since base) |

#### NetCDF Variables (MAXELE Files)
| Variable | Shape | Description |
|----------|-------|-------------|
| `zeta_max` | (nodes,) | Maximum water elevation (m) |
| `time_of_zeta_max` | (nodes,) | Time of maximum (seconds) |
| `x`, `y`, `depth`, `element` | - | Same as above |

---

### Meteorological Forcing Data

**Location**: `data/raw/stofs_2d_glo.YYYYMMDD/met_forcing_XXz/`

#### Available Cycles
| Cycle | NCST (Nowcast) | FCST1 (Forecast) | FCST2 | Total Size |
|-------|----------------|------------------|-------|------------|
| Nov 22 00z | 25 hrs (Nov 21 00:00 → Nov 22 00:00) | 121 hrs (Nov 22 00:00 → Nov 27 00:00) | 21 hrs | 44.2 GB |
| Nov 22 12z | 13 hrs (Nov 22 00:00 → Nov 22 12:00) | 121 hrs (Nov 22 12:00 → Nov 27 12:00) | 21 hrs | 41.0 GB |
| Nov 23 00z | 25 hrs (Nov 22 00:00 → Nov 23 00:00) | 121 hrs (Nov 23 00:00 → Nov 28 00:00) | 21 hrs | 44.2 GB |
| Nov 23 12z | 13 hrs (Nov 23 00:00 → Nov 23 12:00) | 121 hrs (Nov 23 12:00 → Nov 28 12:00) | 21 hrs | 41.0 GB |

**Total Met Forcing Data**: ~170 GB

#### File Naming Convention
```
stofs_2d_glo_{type}.{var_code}.nc
  type: ncst (nowcast/analysis), fcst1 (forecast part 1), fcst2 (forecast part 2)
  var_code: 221 (pressure), 222 (wind u/v), 225 (ice concentration)
```

#### Met Forcing Variables
| File Code | Variable | Long Name | Units |
|-----------|----------|-----------|-------|
| `.221.nc` | `pressfc` | Surface pressure | Pa |
| `.222.nc` | `ugrd10m` | 10m U-wind component | m/s |
| `.222.nc` | `vgrd10m` | 10m V-wind component | m/s |
| `.225.nc` | `icec` | Ice concentration | fraction |

#### Met Forcing Grid
- **Resolution**: ~0.117° (~13 km at equator)
- **Size**: 3072 × 1536 (global)
- **Coverage**: 0° to 360° lon, -90° to 90° lat

#### File Structure
```
data/raw/
├── stofs_2d_glo.20251122/
│   ├── met_forcing_00z/
│   │   ├── stofs_2d_glo_ncst.221.nc   # Nowcast pressure
│   │   ├── stofs_2d_glo_ncst.222.nc   # Nowcast wind
│   │   ├── stofs_2d_glo_ncst.225.nc   # Nowcast ice
│   │   ├── stofs_2d_glo_fcst1.221.nc  # Forecast pressure
│   │   ├── stofs_2d_glo_fcst1.222.nc  # Forecast wind
│   │   ├── stofs_2d_glo_fcst1.225.nc  # Forecast ice
│   │   ├── stofs_2d_glo_fcst2.221.nc  # Extended forecast pressure
│   │   ├── stofs_2d_glo_fcst2.222.nc  # Extended forecast wind
│   │   └── stofs_2d_glo_fcst2.225.nc  # Extended forecast ice
│   └── met_forcing_12z/
│       └── ... (same structure)
└── stofs_2d_glo.20251123/
    ├── met_forcing_00z/
    │   └── ... (9 files)
    └── met_forcing_12z/
        └── ... (9 files)
```

---

### Total Available Training Data
- **8 CWL cycles** (Nov 22-23): 8 × 186 = **1,488 hourly time steps**
- **Nodes**: 12,785,004 (global mesh)
- **Potential for multi-cycle training**: Combine cycles for more diverse conditions

### Currently Used Training Data
- **Source File**: `stofs_2d_glo.t00z.fields.cwl.nc` (Nov 22, 00z)
- **Variable**: `zeta` - Bias-corrected coastal water level above xGEOID20B (meters)
- **Original Grid**: 12,785,004 nodes (global ADCIRC mesh)
- **Time Steps**: 186 hourly outputs

### Regional Subset (US East Coast)
- **Bounding Box**: Longitude [-82°, -65°], Latitude [24°, 46°]
- **Target Nodes**: 50,000 (subsampled from ~200K nodes in bbox)
- **Final Nodes**: 27,852 (after filtering dry nodes with >20% missing data)
- **Edges**: 222,816 (Delaunay triangulation connectivity)

### Processed Data Files
```
data/processed/
├── us_east_coast_cwl_mesh.npz      # Mesh geometry and connectivity
│   ├── lon: (27852,)               # Longitude coordinates
│   ├── lat: (27852,)               # Latitude coordinates
│   ├── depth: (27852,)             # Bathymetry
│   ├── edge_index: (2, 222816)     # Graph connectivity
│   └── original_indices: (27852,)  # Mapping to global mesh
│
└── us_east_coast_cwl_elevation.npz # Time series data
    ├── elevation: (186, 27852)     # CWL values (meters)
    └── times: (186,)               # Timestamps
```

### Data Statistics
- **Elevation Range**: [-4.16, 4.19] meters
- **Elevation Mean**: 0.008 m
- **Elevation Std**: 0.496 m
- **Normalization**: eta_scale = 2.0 (divide by 2 for model input)

---

## Model Architecture

### CWLGNN (MeshGraphNet-style)
```
Input:
  - x: Current state (normalized CWL) - shape (nodes, 1)
  - node_features: [x_norm, y_norm, depth_norm] - shape (nodes, 3)
  - edge_index: Graph connectivity - shape (2, edges)
  - edge_attr: [dx/L, dy/L, dist/L] - shape (edges, 3)

Architecture:
  Node Encoder:  Linear(4 → 64) → ReLU → Linear(64 → 64)
  Edge Encoder:  Linear(3 → 64) → ReLU → Linear(64 → 64)

  Processor: 6 × MeshGraphNetBlock
    - Edge MLP: Linear(192 → 64) → ReLU → Linear(64 → 64)
    - Node MLP: Linear(128 → 64) → ReLU → Linear(64 → 64)
    - Residual connections on nodes

  Decoder: Linear(64 → 64) → ReLU → Linear(64 → 1)

Output: Next state prediction (normalized CWL)

Parameters: 186,689 trainable weights
```

### Feature Normalization
```python
# Node features (Cartesian coordinates)
ref_lon, ref_lat = lon.mean(), lat.mean()
R = 6371000.0  # Earth radius in meters
x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
y_cart = R * np.radians(lat - ref_lat)

x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min()) - 1  # [-1, 1]
y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min()) - 1  # [-1, 1]
depth_norm = (log10(|depth|) - mean) / std  # z-score of log depth

# Edge features (relative positions)
char_length = median(edge_distances)
edge_attr = [dx/char_length, dy/char_length, dist/char_length]

# State normalization
eta_scale = 2.0
x_input = elevation / eta_scale
```

---

## Training Configuration

### Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 5e-4
- **Batch Size**: 2 (limited by GPU memory)
- **Epochs**: 100 (early stopping at epoch 34)
- **Gradient Clipping**: 1.0
- **Loss Function**: MSE
- **Train/Val Split**: 80/20 (148 train, 37 validation samples)

### Hardware
- **GPU**: NVIDIA RTX 3050 Ti (4GB VRAM)
- **Training Time**: ~10-15 minutes

### Model Checkpoint
```
outputs/checkpoints/best_cwl_model.pt
├── epoch: 34
├── model_state_dict: {...}
├── val_loss: 0.013712
└── eta_scale: 2.0
```

---

## Validation Results

### Single-Step Prediction
| Metric | Value | Interpretation |
|--------|-------|----------------|
| RMSE | 0.2246 m | Borderline operational |
| MAE | 0.1463 m | Acceptable |
| Correlation | 0.8935 | Good |

### Multi-Step Rollout (Autoregressive)
| Forecast Hour | RMSE (m) | MAE (m) | Correlation |
|---------------|----------|---------|-------------|
| t+0 | 0.000 | 0.000 | 1.000 |
| t+6 | 0.767 | 0.404 | -0.365 |
| t+12 | 0.405 | 0.300 | 0.647 |
| t+24 | 0.643 | 0.537 | 0.356 |
| t+36 | 0.524 | 0.447 | 0.388 |

### Spatial Error Analysis
- **Best Performance**: Open ocean, shallow coastal areas
- **Worst Performance**: Complex coastal geometry (New England, Chesapeake Bay)
- **Error Pattern**: Model tends to underpredict amplitude of water level variations

---

## Scripts Reference

### Data Processing
```bash
# Extract US East Coast subset and train model
python scripts/train_cwl_bias_corrected.py
```
**Location**: `scripts/train_cwl_bias_corrected.py`
- Extracts regional subset from CWL NetCDF file
- Handles dry node filtering (ADCIRC fill value -99999)
- Builds graph connectivity
- Trains CWLGNN model
- Saves checkpoint and processed data

### Validation
```bash
# Run comprehensive validation
python scripts/validate_cwl_model.py
```
**Location**: `scripts/validate_cwl_model.py`
- Single-step prediction accuracy
- Multi-step rollout analysis
- Spatial error distribution
- Generates `outputs/figures/cwl_validation.png`

### Prediction Comparison
```bash
# Generate prediction vs ground truth plots
python scripts/plot_prediction_comparison.py
```
**Location**: `scripts/plot_prediction_comparison.py`
- Side-by-side comparison at multiple forecast hours
- Spatial error maps
- Outputs:
  - `outputs/figures/cwl_snapshot_comparison.png`
  - `outputs/figures/cwl_multi_timestep.png`

### Ensemble Generation
```bash
# Generate ensemble forecasts
python scripts/generate_ensemble.py --members 20 --steps 48 --ic-noise 0.05

# Options:
#   --members       Number of ensemble members (default: 20)
#   --steps         Forecast hours (default: 48)
#   --start         Starting timestep index (default: 100)
#   --ic-noise      Initial condition noise std in meters (default: 0.05)
#   --model-noise   Model noise during rollout in meters (default: 0.0)
#   --corr-length   Spatial correlation length in degrees (default: 1.0)
#   --save-nc       Save results to NetCDF file
```
**Location**: `scripts/generate_ensemble.py`
- Initial condition perturbation (Gaussian or spatially correlated)
- Optional model noise during rollout
- Computes ensemble statistics (mean, std, percentiles)
- Generates plots:
  - `outputs/figures/ensemble_spatial_t12.png` - Spatial ensemble statistics
  - `outputs/figures/ensemble_spatial_t24.png`
  - `outputs/figures/ensemble_timeseries.png` - Time series at sample nodes
  - `outputs/figures/ensemble_spread_skill.png` - Spread-skill analysis

### Exceedance Probability
```bash
# Compute exceedance probabilities
python scripts/compute_exceedance.py --members 50 --thresholds 0.5 1.0 1.5 2.0

# Options:
#   --members       Number of ensemble members (default: 50)
#   --steps         Forecast hours (default: 48)
#   --thresholds    Water level thresholds in meters (default: 0.5 1.0 1.5 2.0)
```
**Location**: `scripts/compute_exceedance.py`
- Computes P(CWL > threshold) at each node/time
- Identifies high-risk coastal areas
- Generates plots:
  - `outputs/figures/exceedance_maps_t24.png` - Probability maps
  - `outputs/figures/exceedance_timeseries.png` - Time series
  - `outputs/figures/max_exceedance_1m.png` - Max probability map

### Station Ensemble Extraction
```bash
# Extract ensemble at tide gauge locations
python scripts/extract_station_ensemble.py --members 30 --save-csv

# Options:
#   --members        Number of ensemble members (default: 30)
#   --steps          Forecast hours (default: 48)
#   --stations-file  JSON file with custom station definitions
#   --save-csv       Save results to CSV file
```
**Location**: `scripts/extract_station_ensemble.py`
- Pre-defined East Coast tide gauge locations (12 stations)
- Finds nearest model nodes to stations
- Extracts ensemble time series
- Generates plots:
  - `outputs/figures/station_<name>_ensemble.png` - Individual station plots
  - `outputs/figures/station_summary.png` - All stations summary
- Optional CSV output: `outputs/station_ensemble.csv`

**Default Stations**:
| Station | ID | Longitude | Latitude |
|---------|-----|-----------|----------|
| Boston | 8443970 | -71.05 | 42.36 |
| New York | 8518750 | -74.01 | 40.70 |
| Atlantic City | 8534720 | -74.42 | 39.36 |
| Philadelphia | 8545240 | -75.14 | 39.93 |
| Baltimore | 8574680 | -76.58 | 39.27 |
| Norfolk | 8638610 | -76.33 | 36.95 |
| Wilmington NC | 8658120 | -77.95 | 34.23 |
| Charleston | 8665530 | -79.93 | 32.78 |
| Savannah | 8670870 | -80.90 | 32.03 |
| Jacksonville | 8720218 | -81.43 | 30.40 |
| Miami | 8723214 | -80.13 | 25.77 |
| Key West | 8724580 | -81.81 | 24.55 |

---

## Mid-Atlantic Regional Model (NEW)

### Overview
A focused regional model for the Mid-Atlantic coast with:
- Smaller domain = better resolution and faster training
- Larger model capacity (hidden_dim=128, num_layers=8)
- Key tide gauge coverage for validation

### Domain
```
Bounding Box: [-76°, -73°] × [38°, 41°]
Coverage: New York, New Jersey, Delaware Bay, Philadelphia
```

### Model Improvements
| Parameter | East Coast | Mid-Atlantic |
|-----------|------------|--------------|
| Domain | 27,852 nodes | ~5,000-10,000 nodes |
| Hidden Dim | 64 | 128 |
| Layers | 6 | 8 |
| Layer Norm | No | Yes |
| Batch Size | 2 | 4 |
| Epochs | 100 | 200 |
| LR Schedule | Fixed | Cosine Annealing |

### Mid-Atlantic Tide Gauges
| Station | ID | Longitude | Latitude |
|---------|-----|-----------|----------|
| Sandy Hook | 8531680 | -74.01 | 40.47 |
| The Battery (NYC) | 8518750 | -74.01 | 40.70 |
| Kings Point | 8516945 | -73.77 | 40.81 |
| Montauk | 8510560 | -71.96 | 41.05 |
| Atlantic City | 8534720 | -74.42 | 39.36 |
| Cape May | 8536110 | -74.96 | 38.97 |
| Lewes | 8557380 | -75.12 | 38.78 |
| Philadelphia | 8545240 | -75.14 | 39.93 |

### Scripts
```bash
# Train Mid-Atlantic model
CUDA_LAUNCH_BLOCKING=1 python scripts/train_midatlantic.py

# Validate model
python scripts/validate_midatlantic.py

# Generate ensemble
python scripts/generate_ensemble_midatlantic.py --members 30 --steps 48
```

### Output Files
```
data/processed/
├── midatlantic_mesh.npz        # Regional mesh
└── midatlantic_elevation.npz   # Regional time series

outputs/checkpoints/
└── best_midatlantic_model.pt   # Trained model

outputs/figures/
├── midatlantic_domain.png      # Domain visualization
├── midatlantic_training.png    # Training curves
├── midatlantic_rollout.png     # Prediction comparison
├── midatlantic_validation.png  # Validation summary
└── midatlantic_ensemble_*.png  # Ensemble plots
```

---

## Output Files

### Figures
```
outputs/figures/
├── cwl_validation.png           # 4-panel validation summary
├── cwl_snapshot_comparison.png  # Single timestep comparison
└── cwl_multi_timestep.png       # Multi-hour evolution
```

### Model Checkpoint
```
outputs/checkpoints/
└── best_cwl_model.pt            # Trained model weights (~769 KB)
```

---

## Known Issues

1. **Amplitude Underprediction**: Model tends to smooth out extreme values
2. **Coastal Errors**: Higher errors near complex coastlines
3. **Correlation Oscillation**: Correlation varies during rollout (possibly tidal signal)
4. **Limited Training Data**: Only 186 timesteps from single STOFS cycle

---

## Next Steps

### Phase 2: Ensemble Generation (COMPLETED)
- [x] **Initial condition perturbation**: Gaussian and spatially correlated perturbations
- [x] **Ensemble statistics**: Mean, spread, percentiles (P10, P25, P50, P75, P90)
- [x] **Spread-skill analysis**: Verification of ensemble calibration
- [x] **Exceedance probabilities**: P(CWL > threshold) computation
- [x] **Station extraction**: Tide gauge ensemble time series

### Short-term Improvements (Model Quality)
- [ ] **Train longer**: Increase epochs to 200-300 with learning rate scheduling
- [ ] **Larger hidden dimension**: Try hidden_dim=128 or 256
- [ ] **Add more message passing layers**: Try num_layers=8 or 10
- [ ] **Curriculum learning**: Start with 1-step, gradually increase rollout length
- [ ] **Ensemble calibration**: Tune IC noise to match observation uncertainty (~5-10 cm)

### Medium-term Enhancements
- [ ] **Multi-cycle training**: Download and train on 5-10 STOFS cycles for more diverse conditions
- [ ] **Add forcing inputs**: Include wind (u10, v10), pressure (mslp) as node features
- [ ] **Tidal constituents**: Add astronomical tide phase as periodic features
- [ ] **Residual/delta prediction**: Train model to predict change rather than absolute value
- [ ] **Ensemble model noise**: Implement stochastic model perturbation during rollout

### Long-term Goals
- [ ] **Full US coastline**: Expand to include Gulf of Mexico, West Coast
- [ ] **Real-time API**: FastAPI/Flask service for operational forecasting
- [ ] **Verification system**: Compare ensemble forecasts against tide gauge observations
- [ ] **Alert generation**: Automated high-water alerts based on exceedance probabilities

### Data Acquisition
- [ ] Download additional STOFS cycles from AWS S3:
  ```
  https://noaa-gestofs-pds.s3.amazonaws.com/
  ```
- [ ] Include storm events (hurricanes) for extreme condition training
- [ ] Collect corresponding atmospheric forcing data (GFS)
- [ ] Download historical tide gauge data for verification

---

## Dependencies

```bash
pip install torch torch-geometric numpy xarray netCDF4 matplotlib scipy
```

### Key Packages
- PyTorch 2.x with CUDA
- PyTorch Geometric (torch_geometric)
- xarray / netCDF4 for STOFS data
- matplotlib for visualization

---

## References

1. Pfaff et al. (2021) "Learning Mesh-Based Simulation with Graph Networks"
2. NVIDIA PhysicsNeMo - MeshGraphNet
3. NOAA STOFS: https://tidesandcurrents.noaa.gov/stofs/
4. ADCIRC Model: https://adcirc.org/

---

## Contact

Project for rapid ensemble storm surge forecasting using AI/ML methods.
