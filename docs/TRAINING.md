# Training Guide

## Overview

This guide covers training the STOFS GNN surrogate model for coastal water level prediction.

## Prerequisites

- STOFS NetCDF data files (cwl, hvel, prmsl)
- GPU with 4GB+ VRAM (RTX 3050 Ti or better)
- Python 3.10+ with PyTorch and PyTorch Geometric

## Quick Start

### 1. Preprocess Data

```bash
python scripts/train_cwl_gnn_optimized_v3.py --preprocess
```

This extracts the Mid-Atlantic domain and creates `.npz` files.

### 2. Train Model

```bash
python scripts/train_cwl_gnn_optimized_v3.py --train
```

## Training Configuration

Key parameters in the training script:

```python
# Model architecture
HIDDEN_DIM = 96
NUM_LAYERS = 6
STATIC_FEATURES = 4  # x, y, depth, water_level
FORCING_FEATURES = 3  # u10, v10, pressure

# Training
BATCH_SIZE = 1-2  # For 4GB VRAM
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3

# Normalization
ETA_SCALE = 2.0
WIND_SCALE = 15.0
```

## Multi-date Training

For better generalization, train on multiple dates:

```bash
python scripts/train_cwl_gnn_multidate.py --preprocess
python scripts/train_cwl_gnn_multidate.py --train
```

## Monitoring Training

Training outputs:
- Checkpoints: `outputs/checkpoints/best_*.pt`
- Logs: `outputs/training_*.log`
- Figures: `outputs/figures/`

## Expected Results

After 150 epochs on 3 days of data:
- Validation loss: ~0.002
- 1-hour RMSE: ~0.06m
- 48-hour RMSE: ~0.30m
