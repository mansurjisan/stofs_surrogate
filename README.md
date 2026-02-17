# STOFS-GNN: Graph Neural Network Surrogate for Storm Surge Forecasting

![STOFS-GNN Banner](docs/figures/banner.png)

A physics-informed deep learning surrogate for NOAA's Surge and Tide Operational Forecast System (STOFS-2D Global), enabling 48h storm surge forecasts over the Mid-Atlantic region in ~3 seconds on a single GPU. Built on MeshGraphNet with shallow water equation (SWE)-inspired message passing.

## Highlights

- **~4,000x speedup** over numerical model (48h forecast in ~3 seconds on a single GPU)
- **25,000-node subsampled mesh** from the STOFS-2D Global unstructured grid (7.4% of ~340K native nodes in domain)
- **Physics-informed GNN** with SWE-inspired gradient scaling, 6-constituent tidal encoding, and temporal memory
- **Long-range edge augmentation** (+262K edges) for tidal/surge signal propagation across estuaries

## Study Domain

![Study Domain](docs/figures/study_domain.png)

Mid-Atlantic Bight (-77 to -72 W, 37 to 42 N): Chesapeake Bay, Delaware Bay, New York Harbor, and coastal New Jersey. The native STOFS-2D Global mesh has 12.8M nodes globally; the surrogate operates on a ~25K-node regional subset with median edge spacing of ~0.9 km (vs ~0.17 km native).

## Model Architecture

A physics-informed GNN built on [MeshGraphNet](https://arxiv.org/abs/2010.03409) (Pfaff et al., 2021). The graph blocks incorporate shallow water equation (SWE) physics through bathymetric gradient scaling, which modulates message passing strength based on local depth gradients — amplifying information flow in regions of steep bathymetry where surge dynamics are most active.

```
Input (27 features per node)
├── State:    η(t), η(t-1), dη/dt                       [3]
├── Tidal:    sin/cos of M2, S2, N2, K1, O1, M4         [12]
├── Static:   x, y, depth, water_level                   [4]
└── Forcing:  u10, v10, |V|, |V|², θ, P, ∂P/∂x, ∂P/∂y  [8]
    → Node Encoder (MLP: 27 → 128)
    → 6× SWE Graph Blocks with gradient scaling: m × (1 + tanh(γ·∇h))
    → Decoder (MLP: 128 → 1)
    → η(t+1) = η(t) + Δη
```

~1.6M parameters.

## Training

| | |
|---|---|
| **Training data** | STOFS-2D Global daily forecasts, Jan 2023 – Dec 2024 (~700 days) |
| **Validation data** | Jan 2025 – present (held-out year) |
| **Hardware** | NVIDIA H100 80GB (NOAA URSA HPC) |
| **Training time** | ~2–3 weeks (100 epochs with curriculum rollout) |
| **Optimizer** | AdamW (lr=2e-4, weight decay=1e-5) |
| **Scheduler** | Cosine annealing (η_min=1e-6) |
| **Precision** | Mixed (AMP) |
| **Effective batch** | 64 (batch=4 × grad accumulation=16) |
| **Curriculum** | Rollout steps 1→2→3→6→12 over 100 epochs |
| **Inference** | ~3 seconds for 48h forecast (single GPU) |

## Results

### Validation (2025 held-out data, epoch 95)

| Lead Time | RMSE (cm) |
|-----------|-----------|
| t+1h      | 4.1       |
| t+6h      | 16.2      |
| t+12h     | 21.7      |
| t+24h     | 28.9      |
| t+48h     | 38.4      |

### Spatial Predictions

![Spatial t+6h](docs/figures/spatial_comparison_h06.png)
*STOFS ground truth (left) vs GNN prediction (right) at t+6h. RMSE: 16.2 cm, R: 0.982.*

![Spatial t+24h](docs/figures/spatial_comparison_h24.png)
*Same at t+24h. RMSE: 28.9 cm, R: 0.930.*

### Curriculum Learning Progression

![Curriculum Learning](docs/figures/rollout_rmse_curriculum.png)
*Progressive rollout training reduces 48h RMSE from 69 cm (early) to 38 cm (epoch 95).*

![RMSE Table](docs/figures/rmse_table_v2.png)

### Station Validation (48h rollout, Jan 20 2025)

![Station Validation](docs/figures/station_timeseries_v2.png)
*Green: STOFS ground truth. Blue dashed: GNN prediction. Strong tidal phase capture in protected bays (Baltimore R=0.99, Philadelphia R=0.98, Annapolis R=0.95).*

### Ensemble Uncertainty Quantification (20 members)

![Ensemble Forecasts](docs/figures/ensemble_station_panel.png)
*20-member ensemble via perturbed meteorological forcing (wind, pressure) and initial conditions. Blue: GNN control forecast. Green: STOFS truth. Shading: ensemble spread. Strong skill in protected bays (Baltimore R=0.99, Annapolis R=0.95); wider spread at exposed coastal stations reflects forcing sensitivity.*

## Repository Structure

- `stofs_surrogate/` — Python package (model, data, training, inference, visualization)
- `scripts/` — Training, preprocessing, rollout, and visualization scripts
- `scripts/archived/` — Historical development scripts
- `docs/figures/` — Result figures used in README

Training requires STOFS-2D Global output ([NOAA S3](https://noaa-nos-stofs2d-pds.s3.amazonaws.com/index.html)) and GFS forcing ([NOAA NOMADS](https://nomads.ncep.noaa.gov/)). Preprocessed data available upon request.

## License

Public domain (17 U.S.C. § 105). See [LICENSE](LICENSE).

## Citation

```bibtex
@software{jisan2025stofsgnn,
  author = {Jisan, Mansur Ali},
  title = {STOFS-GNN: Graph Neural Network Surrogate for Operational Storm Surge Forecasting},
  year = {2025},
  institution = {NOAA/NOS/CO-OPS},
  url = {https://github.com/mansurjisan/stofs_surrogate}
}
```