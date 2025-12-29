# Ensemble Inference Guide

This guide explains how to run ensemble storm surge forecasts using the STOFS GNN surrogate model.

## Prerequisites

- Python environment with required packages: `torch`, `torch_geometric`, `scipy`, `matplotlib`, `numpy`
- Trained model checkpoint (e.g., `best_temporal_memory_model.pt`)
- Preprocessed data files in `data/processed_25k/`
- Mesh file (`mesh_25k.npz`)

## Basic Usage

```bash
MPLBACKEND=Agg python scripts/ensemble_inference.py \
    --checkpoint /path/to/best_temporal_memory_model.pt \
    --n_members 50 \
    --forecast_hours 24
```

**Note:** Use `MPLBACKEND=Agg` for headless environments (WSL, SSH, servers without display).

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Auto-detect | Path to model checkpoint file |
| `--n_members` | 50 | Number of ensemble members |
| `--forecast_hours` | 48 | Forecast length in hours |
| `--seed` | None | Random seed for reproducibility |
| `--perturb_ic` | False | Also perturb initial conditions |
| `--output_dir` | Auto-generated | Output directory for results |
| `--device` | auto | Device: `auto`, `cuda`, or `cpu` |
| `--cache_clear_interval` | 5 | Clear GPU cache every N members |
| `--no_float16` | False | Disable float16 storage (uses more memory) |
| `--fetch_obs` | True | Fetch CO-OPS observations for comparison |
| `--no_obs` | False | Disable observation fetching |
| `--datum` | MSL | Vertical datum: `MSL`, `MLLW`, `NAVD`, `STND` |
| `--forecast_date` | 20251128 | Forecast date (YYYYMMDD format) |
| `--forecast_cycle` | 00 | Forecast cycle hour: `00`, `06`, `12`, `18` |
| `--preprocessed` | Auto | Path to preprocessed .npz file |
| `--mesh` | Auto | Path to mesh .npz file |

## Example Commands

### Standard 50-member, 24-hour forecast
```bash
MPLBACKEND=Agg python scripts/ensemble_inference.py \
    --checkpoint outputs/checkpoints/best_temporal_memory_model.pt \
    --n_members 50 \
    --forecast_hours 24
```

### Large 100-member ensemble
```bash
MPLBACKEND=Agg python scripts/ensemble_inference.py \
    --checkpoint outputs/checkpoints/best_temporal_memory_model.pt \
    --n_members 100 \
    --forecast_hours 24
```

### 48-hour forecast with initial condition perturbations
```bash
MPLBACKEND=Agg python scripts/ensemble_inference.py \
    --checkpoint outputs/checkpoints/best_temporal_memory_model.pt \
    --n_members 50 \
    --forecast_hours 48 \
    --perturb_ic
```

### Memory-safe options for limited VRAM (e.g., RTX 3050 4GB)
```bash
MPLBACKEND=Agg python scripts/ensemble_inference.py \
    --checkpoint outputs/checkpoints/best_temporal_memory_model.pt \
    --n_members 30 \
    --forecast_hours 24 \
    --cache_clear_interval 3
```

### CPU-only (slower but stable)
```bash
MPLBACKEND=Agg python scripts/ensemble_inference.py \
    --checkpoint outputs/checkpoints/best_temporal_memory_model.pt \
    --n_members 50 \
    --forecast_hours 24 \
    --device cpu
```

### Without observation fetching
```bash
MPLBACKEND=Agg python scripts/ensemble_inference.py \
    --checkpoint outputs/checkpoints/best_temporal_memory_model.pt \
    --n_members 50 \
    --forecast_hours 24 \
    --no_obs
```

### Reproducible run with fixed seed
```bash
MPLBACKEND=Agg python scripts/ensemble_inference.py \
    --checkpoint outputs/checkpoints/best_temporal_memory_model.pt \
    --n_members 50 \
    --forecast_hours 24 \
    --seed 42
```

## Output Structure

Results are saved to `outputs/ensemble/run_YYYYMMDD_HHMMSS/`:

```
run_20251211_221142/
├── ensemble_results.npz      # Raw ensemble data (predictions, statistics)
├── ensemble_metadata.json    # Run configuration and summary statistics
├── ensemble_dashboard.png    # Overview visualization
├── timeseries/               # Station time series plots
│   ├── Atlantic_City_ensemble.png
│   ├── Sandy_Hook_ensemble.png
│   ├── The_Battery_ensemble.png
│   ├── Lewes_DE_ensemble.png
│   └── Cape_May_ensemble.png
└── maps/                     # Spatial maps
    ├── exceedance_0.3m.png   # Probability of exceeding 0.3m
    ├── exceedance_0.5m.png   # Probability of exceeding 0.5m
    ├── exceedance_1.0m.png   # Probability of exceeding 1.0m
    ├── exceedance_1.5m.png   # Probability of exceeding 1.5m
    ├── exceedance_2.0m.png   # Probability of exceeding 2.0m
    └── ensemble_spread.png   # Spatial uncertainty map
```

## Perturbation Configuration

The ensemble generates spread through meteorological forcing perturbations:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Wind speed | 30% std | Multiplicative perturbation |
| Wind direction | 25° std | Additive perturbation |
| Pressure | 600 Pa (6 hPa) std | Additive perturbation |
| Initial CWL | 8 cm std | Initial condition perturbation |
| Spatial correlation | 3.0 | Gaussian smoothing sigma |

These can be modified in the `PERTURBATION_CONFIG` dictionary in `ensemble_inference.py` (lines 145-152).

## Key Stations

The script automatically extracts time series at these Mid-Atlantic stations:

| Station | Location | CO-OPS ID |
|---------|----------|-----------|
| Atlantic City | -74.42°W, 39.36°N | 8534720 |
| Sandy Hook | -74.01°W, 40.47°N | 8531680 |
| The Battery | -74.01°W, 40.70°N | 8518750 |
| Lewes, DE | -75.12°W, 38.78°N | 8557380 |
| Cape May | -74.96°W, 38.97°N | 8536110 |

## Performance Guidelines

| GPU VRAM | Recommended Settings |
|----------|---------------------|
| 4 GB | `--n_members 30 --cache_clear_interval 3` |
| 6 GB | `--n_members 50 --cache_clear_interval 5` |
| 8+ GB | `--n_members 100 --cache_clear_interval 10` |

Typical runtime: ~5 seconds per ensemble member on RTX 3050 Ti.

## Troubleshooting

### Display errors (`couldn't connect to display`)
Use `MPLBACKEND=Agg` environment variable before the python command.

### Out of memory errors
- Reduce `--n_members`
- Lower `--cache_clear_interval` (e.g., 3)
- Use `--device cpu`

### Missing observations
- Check internet connectivity
- Use `--no_obs` to skip observation fetching
- Verify CO-OPS station IDs are valid

### Module not found errors
Install required packages:
```bash
pip install torch torch_geometric scipy matplotlib numpy
```
