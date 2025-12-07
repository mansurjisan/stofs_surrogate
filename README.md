# STOFS Surrogate Model

A Graph Neural Network (GNN) based surrogate model for NOAA's Storm Surge Operational Forecast System (STOFS). This model provides fast ensemble forecasts of coastal water levels for the Mid-Atlantic region.

## Overview

This project implements a physics-informed GNN that learns to predict coastal water level (CWL) changes from STOFS output data. The model can generate 48-hour forecasts approximately 100x faster than the full numerical model, enabling ensemble forecasting with meteorological perturbations.

### Key Features

- **Graph Neural Network Architecture**: Uses message-passing on unstructured triangular mesh
- **Physics-Informed Design**: Incorporates shallow water equation inspired blocks
- **Ensemble Forecasting**: Generates probabilistic forecasts with wind/pressure perturbations
- **CO-OPS Validation**: Compares predictions against NOAA tide gauge observations
- **GPU Accelerated**: Optimized for CUDA with mixed precision support

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- NumPy, SciPy, Matplotlib
- netCDF4
- searvey (optional, for CO-OPS observations)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install PyTorch Geometric

```bash
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Project Structure

```
stofs_surrogate/
├── scripts/
│   ├── train_cwl_gnn_optimized_v3.py   # Main training script
│   ├── ensemble_inference.py            # Ensemble forecasting
│   ├── plot_rollout_timeseries.py       # Visualization
│   └── preprocess_stofs.py              # Data preprocessing
├── data/
│   ├── raw/                             # Raw STOFS NetCDF files
│   ├── processed/                       # Processed training data
│   └── processed_optimized/             # Optimized mesh data
├── outputs/
│   ├── checkpoints/                     # Model checkpoints
│   ├── ensemble/                        # Ensemble forecast outputs
│   └── figures/                         # Generated plots
└── README.md
```

## Usage

### 1. Data Preprocessing

First, preprocess STOFS output files:

```bash
python3 scripts/train_cwl_gnn_optimized_v3.py --preprocess
```

This will:
- Extract the Mid-Atlantic domain from global STOFS data
- Subsample to ~15,000 nodes for memory efficiency
- Compute mesh connectivity and edge features
- Save preprocessed data as `.npz` files

### 2. Training

Train the GNN model:

```bash
python3 scripts/train_cwl_gnn_optimized_v3.py --train
```

Training parameters can be configured in the script:
- `MAX_NODES`: Number of mesh nodes (default: 15000)
- `HIDDEN_DIM`: GNN hidden dimension (default: 96)
- `NUM_LAYERS`: Number of message-passing layers (default: 6)
- `BATCH_SIZE`: Training batch size (default: 1-2 for 4GB VRAM)
- `NUM_EPOCHS`: Training epochs (default: 150)

### 3. Ensemble Inference

Generate ensemble forecasts:

```bash
python3 scripts/ensemble_inference.py \
    --preprocessed data/processed_optimized/processed_20251128.npz \
    --n_members 50 \
    --forecast_hours 48
```

### 4. Visualization

Plot rollout timeseries at tide gauge stations:

```bash
python3 scripts/plot_rollout_timeseries.py \
    --checkpoint outputs/checkpoints/best_optimized_model.pt \
    --date 20251130
```

## Model Architecture

The model uses a physics-informed GNN architecture:

```
Input: [cwl_t, static_features, forcing] -> Node Encoder ->
       6x SWE-Inspired Graph Blocks -> Decoder -> cwl_{t+1}
```

### Components:

1. **Node Encoder**: MLP that encodes node features (water level, position, depth, forcing)
2. **Edge Encoder**: MLP that encodes edge features (distance, direction)
3. **SWE-Inspired Blocks**: Message-passing layers with gradient-aware updates
4. **Decoder**: MLP that predicts next-timestep water level

### Static Features:
- Normalized x, y coordinates (Cartesian)
- Log-normalized depth
- Water level (depth + current CWL)

### Forcing Features:
- U10 wind component (normalized)
- V10 wind component (normalized)
- Surface pressure

## Performance

### Training Results (v3 Optimized)
- Best validation loss: 0.002031
- Final MSE: 0.001105
- Training time: ~9 hours (150 epochs on RTX 3050 Ti)

### Rollout RMSE
| Lead Time | RMSE |
|-----------|------|
| t+1h | 0.057m |
| t+6h | 0.195m |
| t+12h | 0.317m |
| t+24h | 0.449m |
| t+48h | 0.296m |

### Station Validation (48h forecast)
| Station | RMSE | Correlation |
|---------|------|-------------|
| Atlantic City | 0.40m | 0.45 |
| Sandy Hook | 0.59m | 0.36 |
| The Battery | 0.46m | 0.53 |
| Lewes, DE | 0.51m | 0.30 |
| Cape May | 0.63m | 0.01 |

## Data Sources

- **STOFS Global**: NOAA operational storm surge forecasts
- **CO-OPS**: NOAA tide gauge observations for validation

## Known Limitations

1. **Phase Errors**: Model predictions may have timing offsets from truth
2. **Amplitude Damping**: Tidal amplitude tends to decrease during long rollouts
3. **Limited Training Data**: Currently trained on 3 days of data
4. **Mesh Resolution**: 15,000 node subsampling may miss coastal details

## Future Work

- [ ] Incorporate tidal harmonics as additional features
- [ ] Multi-step loss for better long-range forecasts
- [ ] Higher resolution coastal refinement
- [ ] Training on more diverse storm events

## License

MIT License

## Acknowledgments

- NOAA/NOS for STOFS operational data
- PyTorch Geometric team for GNN framework

## Citation

If you use this code, please cite:

```bibtex
@software{stofs_surrogate,
  title = {STOFS Surrogate Model},
  year = {2024},
  url = {https://github.com/mansurjisan/stofs_surrogate}
}
```
