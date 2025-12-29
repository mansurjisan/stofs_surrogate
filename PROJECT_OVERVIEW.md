# STOFS Surrogate Model - GNN for Rapid Ensemble Generation

## Background

### What is STOFS?
STOFS (Surge and Tide Operational Forecast System) is NOAA's operational storm surge forecasting system. It provides water level predictions for coastal areas to support emergency management, navigation, and public safety.

- **STOFS 2D Global**: Based on ADCIRC (ADvanced CIRCulation model), uses unstructured triangular mesh with ~13.4 million nodes globally
- **STOFS 3D Atlantic**: Based on SCHISM, provides 3D water level and current predictions

### The Problem
Running a single STOFS deterministic forecast takes significant computational resources (hours on HPC clusters). For probabilistic forecasting, ensemble runs are needed (20-50+ members), which becomes computationally prohibitive for operational use.

### The Solution
Train a Graph Neural Network (GNN) surrogate model that:
1. Learns the physics of storm surge propagation from deterministic STOFS runs
2. Can generate predictions in **seconds** instead of hours
3. Enables rapid ensemble generation by perturbing initial conditions

## Project Goals

### Phase 1: Surrogate Model Development (COMPLETED)
- [x] Set up project structure and dependencies
- [x] Download and process real STOFS data from AWS S3
- [x] Extract US East Coast regional subset (50K nodes from 13.4M global)
- [x] Implement MeshGraphNet-style GNN architecture
- [x] Train on real STOFS water elevation data
- [x] Validate rollout predictions

### Phase 2: Ensemble Generation (NEXT)
- [ ] Implement initial condition perturbation strategies
- [ ] Generate N ensemble members from single initial state
- [ ] Compute ensemble statistics (mean, spread, percentiles)
- [ ] Validate against STOFS ensemble runs (if available)

### Phase 3: Operational Integration (FUTURE)
- [ ] Expand to full US coastline or global domain
- [ ] Add forcing inputs (wind, pressure, tides)
- [ ] Optimize inference speed for real-time operations
- [ ] Create API for operational forecasting systems

## Data Sources

### AWS S3 STOFS Archive
```
https://noaa-gestofs-pds.s3.amazonaws.com/index.html
```

Key files:
- `stofs_2d_glo_maxele.63.nc` - Maximum water elevation (contains mesh topology)
- `stofs_2d_glo_surf.63.nc` - Time series of water elevation (~14GB for 126 timesteps)

### Data Downloaded
```
data/raw/stofs_2d_glo.20251127/
├── stofs_2d_glo_maxele.63.nc    # 812 MB - mesh + max elevation
└── stofs_2d_glo_surf.63.nc      # 14 GB - elevation time series
```

### Processed Data
```
data/processed/
├── us_east_coast_mesh.npz       # 50K node subset mesh
└── us_east_coast_elevation.npz  # Elevation time series for subset
```

## Project Structure

```
stofs_surrogate/
├── PROJECT_OVERVIEW.md          # This file
├── config/                      # Configuration files
├── data/
│   ├── raw/                     # Downloaded STOFS NetCDF files
│   ├── processed/               # Extracted subsets (.npz)
│   └── cache/                   # Temporary cache
├── src/                         # Core library code
│   ├── mesh.py                  # ADCIRC mesh loading & graph conversion
│   ├── dataset.py               # PyTorch datasets for training
│   ├── model.py                 # GNN model architectures
│   ├── trainer.py               # Training pipeline
│   └── stofs_loader.py          # STOFS NetCDF data loading
├── scripts/                     # Executable scripts
│   ├── download_stofs.py        # Download from AWS S3
│   ├── extract_us_east_coast.py # Extract regional mesh subset
│   ├── extract_elevation_subset.py # Extract elevation time series
│   ├── train_us_east_coast.py   # Train with synthetic dynamics
│   └── train_us_east_coast_real.py # Train with real STOFS data
├── notebooks/                   # Jupyter notebooks for analysis
└── outputs/
    ├── checkpoints/             # Saved model weights
    └── figures/                 # Training curves, rollout plots
```

## Scripts Reference

### 1. Download STOFS Data
```bash
python scripts/download_stofs.py --date 20251127 --files elevation
```
Downloads STOFS output files from NOAA AWS S3 bucket.

### 2. Extract US East Coast Mesh
```bash
python scripts/extract_us_east_coast.py
```
- Loads global mesh from `maxele.63.nc`
- Filters to US East Coast bounding box: [-82°, -65°] x [24°, 46°]
- Subsamples to 50,000 nodes using stratified random sampling
- Saves mesh subset with node mapping

### 3. Extract Elevation Time Series
```bash
python scripts/extract_elevation_subset.py
```
- Opens 14GB `surf.63.nc` file
- Extracts elevation only for US East Coast subset nodes
- Saves compact 12.9MB file with 126 timesteps

### 4. Train with Real Data
```bash
CUDA_LAUNCH_BLOCKING=1 python scripts/train_us_east_coast_real.py
```
- Loads mesh and elevation data
- Creates PyTorch Geometric graph
- Trains MeshGraphNet-style GNN for 100 epochs
- Generates training curves and rollout visualizations
- Saves best model checkpoint

## Model Architecture

### RealSTOFSGNN (MeshGraphNet-style)
```
Input: Node features (elevation at time t)
       Edge features (relative position, distance)

Encoder:  Linear(node_dim -> hidden)
          Linear(edge_dim -> hidden)

Processor: 6 MessagePassing layers
           - Edge update MLP
           - Node update MLP
           - Residual connections

Decoder:   Linear(hidden -> node_dim)

Output: Predicted change in elevation (delta)
```

Parameters: ~187K trainable weights

## Training Results

### Real STOFS Data Training
- **Data**: 126 timesteps, 50K nodes, Nov 26 - Dec 2, 2025
- **Train/Val split**: 100/25 samples
- **Final train loss**: 0.0064 MSE
- **Best val loss**: 0.0072 MSE
- **Training time**: ~2 minutes on RTX 3050 Ti

### Model Outputs
```
outputs/checkpoints/best_real_stofs.pt    # Trained model
outputs/figures/real_stofs_training.png   # Loss curves
outputs/figures/real_stofs_rollout.png    # 20-hour prediction
```

## Dependencies

```bash
pip install torch torch-geometric numpy xarray netCDF4 matplotlib scipy
```

Key packages:
- PyTorch 2.x with CUDA support
- PyTorch Geometric (PyG) for graph neural networks
- xarray/netCDF4 for STOFS data loading
- matplotlib for visualization

## Technical Notes

### Memory Management
- Full STOFS global mesh (13.4M nodes) doesn't fit in GPU memory
- Solution: Extract regional subset (50K nodes) for training
- Chunk-based loading for large NetCDF files

### CUDA Issues
- If CUDA errors occur, use `CUDA_LAUNCH_BLOCKING=1`
- Reduce batch size if GPU memory is limited (batch_size=2 works on 4GB GPU)

### Node Index Mapping
- `original_indices` in mesh.npz maps subset nodes to global mesh
- Essential for extracting corresponding data from full STOFS files

## Future Work

### Ensemble Generation Strategy
1. **Initial Condition Perturbation**
   - Add Gaussian noise to initial water elevation
   - Scale based on observation uncertainty (~5-10 cm)

2. **Parameter Perturbation**
   - Vary model internal representations (dropout, noise injection)

3. **Multi-step Rollout**
   - Autoregressively predict 24-48 hours ahead
   - Accumulate ensemble spread over time

### Scaling Up
- Train on multiple STOFS cycles (more training data)
- Expand to full coastline or multi-region models
- Add atmospheric forcing as input features (wind, pressure)

### Validation
- Compare ensemble spread to STOFS hindcast errors
- Verify probabilistic calibration (reliability diagrams)
- Test on extreme events (hurricanes)

## References

1. NVIDIA PhysicsNeMo - MeshGraphNet for CFD
2. Pfaff et al. (2021) "Learning Mesh-Based Simulation with Graph Networks"
3. NOAA STOFS: https://tidesandcurrents.noaa.gov/stofs/
4. ADCIRC Model: https://adcirc.org/

## Contact

Project developed for rapid ensemble storm surge forecasting using AI/ML methods.
