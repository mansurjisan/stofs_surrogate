# Ensemble Inference Quick Start Guide

## Overview

The `ensemble_inference.py` script enables rapid probabilistic storm surge forecasting using your trained GNN model. Generate 50 ensemble members in ~2 minutes instead of 100+ CPU-hours with ADCIRC.

---

## Quick Start

```bash
# Basic usage (50 members, 48-hour forecast)
python ensemble_inference.py

# Custom settings
python ensemble_inference.py --n_members 100 --forecast_hours 72

# With reproducible random seed
python ensemble_inference.py --n_members 50 --seed 42

# Also perturb initial conditions
python ensemble_inference.py --n_members 50 --perturb_ic
```

---

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | `best_physics_informed_model.pt` | Model checkpoint file |
| `--n_members` | 50 | Number of ensemble members |
| `--forecast_hours` | 48 | Forecast length in hours |
| `--seed` | None | Random seed for reproducibility |
| `--perturb_ic` | False | Also perturb initial conditions |
| `--output_dir` | Auto-generated | Output directory |

---

## What Gets Perturbed

### Meteorological Forcing (Default)

| Parameter | Perturbation | Physical Basis |
|-----------|--------------|----------------|
| Wind speed | ±15% (multiplicative) | NWP model uncertainty |
| Wind direction | ±10° (rotation) | Track/flow uncertainty |
| Pressure | ±3 hPa (additive) | Intensity uncertainty |

### Initial Conditions (Optional with `--perturb_ic`)

| Parameter | Perturbation | Physical Basis |
|-----------|--------------|----------------|
| Water level | ±2 cm (spatially correlated) | Observation/analysis error |

---

## Output Structure

```
outputs/ensemble/run_YYYYMMDD_HHMMSS/
├── ensemble_predictions.npz      # Raw predictions [n_members, times, nodes]
├── ensemble_statistics.npz       # Mean, std, percentiles, exceedance probs
├── metadata.json                 # Run configuration and timing
├── ensemble_dashboard.png        # Summary visualization
├── timeseries/
│   ├── spaghetti_Atlantic_City.png
│   ├── spaghetti_Sandy_Hook.png
│   ├── spaghetti_The_Battery.png
│   └── ...
└── maps/
    ├── exceedance_prob_0.5m.png
    ├── exceedance_prob_1.0m.png
    ├── ensemble_spread.png
    └── ...
```

---

## Output Files Explained

### `ensemble_predictions.npz`

```python
import numpy as np

data = np.load('ensemble_predictions.npz')
predictions = data['predictions']  # Shape: [n_members, n_times, n_nodes]
lon = data['lon']                  # Shape: [n_nodes]
lat = data['lat']                  # Shape: [n_nodes]

# Example: Get member 0, hour 24, all nodes
surge_map = predictions[0, 24, :]
```

### `ensemble_statistics.npz`

```python
stats = np.load('ensemble_statistics.npz')

# Available keys:
# - mean, median          : Central tendency [n_times, n_nodes]
# - std, min, max         : Spread measures [n_times, n_nodes]
# - p5, p10, p25, p75, p90, p95 : Percentiles [n_times, n_nodes]
# - iqr                   : Interquartile range [n_times, n_nodes]
# - prob_exceed_0.3m      : P(surge > 0.3m) [n_times, n_nodes]
# - prob_exceed_0.5m      : P(surge > 0.5m) [n_times, n_nodes]
# - prob_exceed_1.0m      : P(surge > 1.0m) [n_times, n_nodes]
# - prob_exceed_1.5m      : P(surge > 1.5m) [n_times, n_nodes]
# - prob_exceed_2.0m      : P(surge > 2.0m) [n_times, n_nodes]

# Example: Probability of exceeding 1m at hour 24
prob_1m = stats['prob_exceed_1.0m'][24, :]
```

---

## Visualization Guide

### Spaghetti Plot (Time Series)

Shows individual ensemble members as light gray lines with uncertainty bounds:
- Blue shading: 10-90th and 25-75th percentiles
- Blue line: Ensemble mean
- Red dashed: Median

### Exceedance Probability Map

Shows spatial probability that surge exceeds threshold (0.3m, 0.5m, 1.0m, 1.5m, 2.0m):
- Yellow: Low probability
- Red: High probability

### Ensemble Spread Map

Shows standard deviation across ensemble members:
- Dark: Low uncertainty
- Bright: High uncertainty

### Summary Dashboard

Combined view with:
- Maps at peak surge time
- Time series at key stations
- Distribution of peak surge
- Run metadata

---

## Key Stations

Default stations for time series output (Mid-Atlantic domain):

| Station | Longitude | Latitude |
|---------|-----------|----------|
| Atlantic City | -74.42 | 39.36 |
| Sandy Hook | -74.01 | 40.47 |
| The Battery (NYC) | -74.01 | 40.70 |
| Delaware Bay | -75.12 | 38.95 |
| Cape May | -74.96 | 38.93 |

To customize, edit `KEY_STATIONS` in the script.

---

## Performance Expectations

On RTX 3050 Laptop (4GB VRAM):

| Metric | Value |
|--------|-------|
| Single forecast (48h) | ~1-2 seconds |
| 50-member ensemble | ~1-2 minutes |
| 100-member ensemble | ~2-4 minutes |

Compare to ADCIRC:
- Single 48h forecast: 2-4 hours on HPC
- 50-member ensemble: 100-200 CPU-hours

**Speedup: ~1000-10000×**

---

## Programmatic Usage

```python
from ensemble_inference import (
    EnsembleForecaster,
    PhysicsInformedCWLModel,
    load_mesh_data,
    load_forcing_for_ensemble,
    load_initial_condition,
)
import torch
import numpy as np

# Load model
checkpoint = torch.load('best_physics_informed_model.pt')
model = PhysicsInformedCWLModel(
    hidden_dim=checkpoint['config']['hidden_dim'],
    num_layers=checkpoint['config']['num_layers'],
    # ... other params
)
model.load_state_dict(checkpoint['model_state_dict'])

# Load data
mesh_data = load_mesh_data('midatlantic_mesh_v5.npz')
forcing = load_forcing_for_ensemble(...)
initial_cwl = load_initial_condition(...)

# Initialize forecaster
device = torch.device('cuda')
forecaster = EnsembleForecaster(model, mesh_data, device)

# Run ensemble
results = forecaster.run_ensemble(
    initial_cwl=initial_cwl,
    base_forcing=forcing,
    n_members=50,
    forecast_hours=48,
    seed=42,
)

# Access results
predictions = results['predictions']       # [50, 49, n_nodes]
mean_surge = results['statistics']['mean'] # [49, n_nodes]
prob_1m = results['statistics']['prob_exceed_1.0m']  # [49, n_nodes]
```

---

## Connecting to Real-Time Data

For operational use, replace `load_forcing_for_ensemble()` with real-time data:

```python
# Option 1: NOMADS (GFS/HRRR)
# Download from https://nomads.ncep.noaa.gov/

# Option 2: AWS Open Data
# s3://noaa-gfs-bdp-pds/

# Option 3: Google Earth Engine
# GFS data available through EE

# Example structure for real-time forcing:
forcing = {
    'u10': np.array(...),      # [forecast_hours, n_nodes]
    'v10': np.array(...),      # [forecast_hours, n_nodes]  
    'pressure': np.array(...), # [forecast_hours, n_nodes], normalized
}
```

---

## Limitations

1. **Model trained on limited data** - Validate before operational use
2. **Domain-specific** - Only works for Mid-Atlantic region
3. **Simplified perturbations** - Consider using GEFS for proper met uncertainty
4. **No tidal forcing** - Current model doesn't include tides

---

## Next Steps

1. **Validate ensemble spread** - Compare to historical forecast errors
2. **Calibrate perturbations** - Tune to match observed uncertainty
3. **Add more training data** - Before operational deployment
4. **Connect to real-time forcing** - GFS/HRRR data pipeline
5. **Add tidal forcing** - Improve accuracy in tidal areas

---

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch processing in forecaster
# Or use CPU (slower but works)
device = torch.device('cpu')
```

### Missing Forcing Files

```
FileNotFoundError: .../stofs_2d_glo_ncst.222.nc
```

Make sure you have the forcing data downloaded and paths are correct in the configuration section.

### Checkpoint Not Found

```
FileNotFoundError: .../best_physics_informed_model.pt
```

Ensure training completed successfully and checkpoint was saved.

---

*For questions or issues, check the training logs and model configuration.*
