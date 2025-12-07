# STOFS Surrogate Model Training Workflow

## Overview

This document describes the workflow for training the Mid-Atlantic Coastal Water Level (CWL) Graph Neural Network (GNN) surrogate model with meteorological forcing.

## System Requirements

| Component | Specification |
|-----------|---------------|
| RAM | ~24 GB available |
| GPU | NVIDIA RTX 3050 Ti (4 GB VRAM) |
| OS | Windows 11 + WSL2 (Ubuntu) |
| Python | 3.12 |
| CUDA | 12.9 |

## Directory Structure

```
stofs_surrogate/
├── scripts/
│   ├── train_midatlantic_with_forcing.py   # Main training script
│   └── MEMORY_OPTIMIZATIONS.md             # Memory optimization guide
├── outputs/
│   ├── data/
│   │   └── processed/
│   │       └── midatlantic_mesh_v5.npz     # Cached mesh data
│   ├── figures/
│   │   ├── midatlantic_domain_forcing.png  # Domain visualization
│   │   ├── midatlantic_forcing_training.png # Training curves
│   │   └── midatlantic_forcing_rollout.png # Rollout predictions
│   └── models/
│       └── midatlantic_forcing_best.pt     # Best model checkpoint
└── data/
    └── stofs/
        └── stofs_2d_glo.YYYYMMDD/          # STOFS output data
            ├── tXXz/
            │   └── cwl.nc                  # Coastal water level
            └── met_forcing_XXz/
                └── stofs_2d_glo.*.grib2    # Met forcing files
```

## Training Pipeline

### Phase 1: Data Loading

```
┌─────────────────────────────────────────────────────────────┐
│                    MESH EXTRACTION                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Load global STOFS ADCIRC mesh from cwl.nc                │
│ 2. Filter nodes within bounding box:                        │
│    - Longitude: [-76.0, -73.0]                              │
│    - Latitude:  [38.0, 41.0]                                │
│ 3. Build Delaunay triangulation for connectivity            │
│ 4. Cache mesh to .npz file for reuse                        │
│                                                             │
│ Result: 7,743 valid nodes (max available in bbox)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              CYCLE DATA EXTRACTION (per cycle)              │
├─────────────────────────────────────────────────────────────┤
│ For each forecast cycle (t00z, t12z):                       │
│                                                             │
│ 1. Extract CWL (Coastal Water Level):                       │
│    - Read zeta variable from cwl.nc                         │
│    - Extract 186 timesteps at mesh node locations           │
│    - Store as float16 to save memory                        │
│                                                             │
│ 2. Load Met Forcing:                                        │
│    - Read GRIB2 files (u10, v10, pressure)                  │
│    - Subsample spatial grid by 2x                           │
│    - Interpolate temporal: native → 186 timesteps           │
│    - Interpolate spatial: regular grid → mesh nodes         │
│    - Store as float16                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATASET CREATION                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Identify valid nodes (no NaN values across all cycles)   │
│ 2. Create training samples:                                 │
│    - Input: CWL(t), forcing(t), static features             │
│    - Target: CWL(t+1)                                       │
│ 3. Split: 80% train, 20% validation                         │
│                                                             │
│ Result: 370 samples from 2 cycles                           │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              MidAtlanticGNNWithForcing                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Node Features (7 dims):                                     │
│   - State: current CWL (1)                                  │
│   - Static: lon, lat, depth (3)                             │
│   - Forcing: u10, v10, pressure (3)                         │
│                                                             │
│ Edge Features (3 dims):                                     │
│   - dx, dy, distance                                        │
│                                                             │
│ Architecture:                                               │
│   ┌──────────────┐                                          │
│   │ Node Encoder │ Linear(7 → 64)                           │
│   └──────┬───────┘                                          │
│          ▼                                                  │
│   ┌──────────────┐                                          │
│   │ Edge Encoder │ Linear(3 → 64)                           │
│   └──────┬───────┘                                          │
│          ▼                                                  │
│   ┌──────────────┐                                          │
│   │  GNN Layers  │ 4x TransformerConv(64, 64)               │
│   │  + Residual  │ with skip connections                    │
│   └──────┬───────┘                                          │
│          ▼                                                  │
│   ┌──────────────┐                                          │
│   │   Decoder    │ Linear(64 → 64 → 1)                      │
│   └──────┬───────┘                                          │
│          ▼                                                  │
│   Output: Δη (water level change)                           │
│   Prediction: η(t+1) = η(t) + Δη                            │
│                                                             │
│ Total Parameters: 130,305                                   │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING CONFIG                          │
├─────────────────────────────────────────────────────────────┤
│ Epochs:        200                                          │
│ Batch Size:    4 (limited by 4GB VRAM)                      │
│ Learning Rate: 5e-4 (cosine decay)                          │
│ Optimizer:     AdamW (weight_decay=1e-5)                    │
│ Loss:          MSE                                          │
│ DataLoader:    num_workers=2, pin_memory=True               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ for epoch in range(200):                                    │
│     # Training                                              │
│     for batch in train_loader:                              │
│         pred = model(batch)                                 │
│         loss = MSE(pred, target)                            │
│         loss.backward()                                     │
│         optimizer.step()                                    │
│                                                             │
│     # Validation                                            │
│     val_loss = evaluate(val_loader)                         │
│                                                             │
│     # Save best model                                       │
│     if val_loss < best_val_loss:                            │
│         save_checkpoint()                                   │
│                                                             │
│     # Learning rate decay                                   │
│     scheduler.step()                                        │
│                                                             │
│ Time: ~7.6 sec/epoch (~25 min total)                        │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4: Evaluation (Rollout)

```
┌─────────────────────────────────────────────────────────────┐
│                    ROLLOUT EVALUATION                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Autoregressive prediction over 24 hours:                    │
│                                                             │
│ η(t=0) ──► Model ──► η(t=1) ──► Model ──► η(t=2) ──► ...   │
│                                                             │
│ Metrics computed at:                                        │
│   - t+1h:  Single-step accuracy                             │
│   - t+6h:  Short-term forecast                              │
│   - t+12h: Medium-term forecast                             │
│   - t+24h: Full-day forecast                                │
│                                                             │
│ RMSE targets:                                               │
│   - t+1h:  < 0.10m (good)                                   │
│   - t+24h: < 0.50m (acceptable)                             │
└─────────────────────────────────────────────────────────────┘
```

## Running the Training

### Command

```bash
# From WSL2 terminal
cd /mnt/d/AI_4_STOFS/stofs_surrogate
python3 scripts/train_midatlantic_with_forcing.py
```

### Monitoring GPU Usage

```bash
# Check GPU utilization (should be 80-90%)
nvidia-smi

# Note: Windows Task Manager may show incorrect values for WSL2 CUDA workloads
# Always trust nvidia-smi for accurate metrics
```

### Expected Output

```
============================================================
Mid-Atlantic CWL GNN Training WITH MET FORCING
(Memory-optimized version)
============================================================
Domain: [-76.0, -73.0] x [38.0, 41.0]
Model: hidden_dim=64, num_layers=4
Forcing: u10, v10, pressure (normalized)

Loading existing mesh...
Loading cycle data with met forcing...
  Processing stofs_2d_glo.20251122 t00z...
  Processing stofs_2d_glo.20251122 t12z...

Creating multi-cycle dataset...
Valid nodes (all cycles): 7,743 / 25,000
Dataset: 370 samples from 2 cycles
Nodes: 7743, Edges: 31022

Starting training...
Epoch   1: train=0.026173, val=0.009957, lr=5.00e-04, best=0.009957
Epoch  10: train=0.007749, val=0.008302, lr=4.97e-04, best=0.007053
...
Epoch 200: train=0.00XXXX, val=0.00XXXX, lr=X.XXe-05, best=0.00XXXX

Rollout RMSE: t+1h=0.XXXm, t+6h=0.XXXm, t+12h=0.XXXm, t+24h=0.XXXm
```

## Output Files

| File | Description |
|------|-------------|
| `midatlantic_mesh_v5.npz` | Cached mesh (lon, lat, depth, edges, indices) |
| `midatlantic_forcing_best.pt` | Best model checkpoint |
| `midatlantic_domain_forcing.png` | 4-panel domain visualization |
| `midatlantic_forcing_training.png` | Training/validation loss curves |
| `midatlantic_forcing_rollout.png` | 24h rollout prediction comparison |

## Memory Optimization Summary

| Optimization | Impact |
|--------------|--------|
| Float16 storage | 50% reduction in array memory |
| max_nodes limit | Controls mesh size |
| num_workers=2 | Parallel data loading |
| pin_memory=True | Faster GPU transfers |
| Garbage collection | Prevents memory accumulation |
| Batch cleanup | Frees GPU memory per iteration |

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` (try 2 or 1)
- Reduce `max_nodes` (reduces mesh resolution)
- Set `num_workers=0` (reduces RAM, slower loading)

### WSL Crash
- Check `/proc/meminfo` for available RAM
- Increase WSL memory limit in `.wslconfig`
- Reduce number of cycles in `CYCLES` list

### Low GPU Utilization
- Increase `BATCH_SIZE` if VRAM allows
- Set `num_workers=2` or higher
- Enable `pin_memory=True`

## Resolution Comparison

| max_nodes | Valid Nodes | Notes |
|-----------|-------------|-------|
| 5,000 | 1,519 | Fast training, coarse resolution |
| 10,000 | 3,072 | Moderate resolution |
| 25,000+ | 7,743 | Maximum available in bbox |

Note: 7,743 is the maximum number of STOFS ADCIRC mesh nodes within the Mid-Atlantic bounding box. Setting `max_nodes` higher will not increase resolution.
