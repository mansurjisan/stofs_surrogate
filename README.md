# STOFS-GNN: Physics-Informed Graph Neural Network for Storm Surge Forecasting

A deep learning surrogate model for NOAA's Surge and Tide Operational Forecast System (STOFS), enabling rapid ensemble storm surge predictions using Graph Neural Networks.

## Highlights

- **~100x faster** than full numerical model for 48-hour forecasts
- **50-member ensemble** in ~4 minutes on RTX 3050 Ti
- **Physics-informed** architecture with SWE-inspired message passing
- **Full 2D spatial fields** (15,000 nodes), not just station predictions

## Results

### Rollout Performance

| Lead Time | RMSE (m) |
|-----------|----------|
| t+1h      | 0.057    |
| t+6h      | 0.195    |
| t+12h     | 0.317    |
| t+24h     | 0.449    |
| t+48h     | 0.296    |

### Station Validation (48h forecast)

| Station | RMSE (m) | Correlation |
|---------|----------|-------------|
| Atlantic City | 0.40 | 0.45 |
| Sandy Hook | 0.59 | 0.36 |
| The Battery | 0.46 | 0.53 |
| Lewes, DE | 0.51 | 0.30 |

## Installation

```bash
# Clone repository
git clone https://github.com/mansurjisan/stofs_surrogate.git
cd stofs_surrogate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install package (optional)
pip install -e .
```

## Quick Start

### Training

```bash
# Preprocess STOFS data
python scripts/train_cwl_gnn_optimized_v3.py --preprocess

# Train model
python scripts/train_cwl_gnn_optimized_v3.py --train
```

### Ensemble Inference

```bash
python scripts/ensemble_inference.py \
    --preprocessed data/processed/processed_20251128.npz \
    --n_members 50 \
    --forecast_hours 48
```

### Visualization

```bash
python scripts/plot_rollout_timeseries.py \
    --checkpoint outputs/checkpoints/best_optimized_model.pt \
    --date 20251130
```

## Project Structure

```
stofs_surrogate/
├── stofs_surrogate/           # Main package
│   ├── models/                # GNN architectures
│   │   └── gnn.py             # PhysicsInformedCWLModel, SWEInspiredGraphBlock
│   ├── data/                  # Data utilities
│   │   ├── dataset.py         # PyTorch dataset classes
│   │   ├── mesh.py            # Mesh processing
│   │   └── preprocessing.py   # STOFS data preprocessing
│   ├── training/              # Training utilities
│   │   └── trainer.py
│   ├── inference/             # Inference utilities
│   │   ├── ensemble.py        # Ensemble forecaster
│   │   └── stations.py        # Station extraction
│   └── visualization/         # Plotting utilities
│       └── plots.py
├── scripts/                   # Executable scripts
│   ├── train_cwl_gnn_optimized_v3.py
│   ├── ensemble_inference.py
│   └── plot_rollout_timeseries.py
├── configs/                   # Configuration files
│   ├── train_default.yaml
│   └── ensemble.yaml
├── docs/                      # Documentation
│   ├── TRAINING.md
│   └── ENSEMBLE.md
├── tests/                     # Unit tests
├── setup.py
├── pyproject.toml
└── requirements.txt
```

## Model Architecture

```
Input: [cwl_t, static_features, forcing]
    → Node Encoder (MLP)
    → 6× SWE-Inspired Graph Blocks (message passing)
    → Decoder (MLP)
    → cwl_{t+1}
```

### Features

**Static Features (4):**
- Normalized x, y coordinates
- Log-normalized depth
- Water level (depth + CWL)

**Forcing Features (3):**
- U10 wind component (normalized)
- V10 wind component (normalized)
- Surface pressure

### Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dim | 96 |
| Num layers | 6 |
| Mesh nodes | ~15,000 |
| ETA_SCALE | 2.0 |
| WIND_SCALE | 15.0 |

## Data Sources

- **STOFS Global**: NOAA operational storm surge forecasts
- **CO-OPS**: NOAA tide gauge observations for validation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- CUDA-capable GPU (4GB+ VRAM recommended)

## License

MIT License

## Citation

```bibtex
@software{stofs_surrogate,
  author = {Jisan, Mansur},
  title = {STOFS-GNN: Physics-Informed Graph Neural Network for Storm Surge Forecasting},
  year = {2024},
  url = {https://github.com/mansurjisan/stofs_surrogate}
}
```

## Acknowledgments

- NOAA/NOS for STOFS operational data
- PyTorch Geometric team for GNN framework
