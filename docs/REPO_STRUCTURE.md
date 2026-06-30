# Repository Structure

How this repo is laid out and where to find the training, inference, and configuration
code.

## Top level

```
stofs_surrogate/      Python package (importable: `import stofs_surrogate`)
configs/              Training and ensemble configuration (YAML)
scripts/              CLI scripts: download, preprocess, train, rollout, visualize
tests/                Unit tests (pytest)
docs/                 Documentation and README figures
data/                 Local data (git-ignored)
outputs/              Checkpoints, figures, logs (git-ignored)
```

## Package (`stofs_surrogate/`)

| Module | Contents |
|---|---|
| `models/gnn.py` | GNN architectures: `STOFSSurrogateGNN` (MeshGraphNet encoder/processor/decoder), `SimpleMeshGraphNet`, the `GraphNetworkBlock` message-passing layer, and the `create_model` factory. |
| `data/` | `mesh.py` (ADCIRC mesh reader / graph conversion), `dataset.py` (synthetic + STOFS datasets), `preprocessing.py`. |
| `training/` | `trainer.py` (training loop, checkpointing), `tracking.py` (MLflow/W&B tracking abstraction), `registry.py` (MLflow Model Registry + lineage). |
| `inference/` | `Predictor` (load checkpoint + autoregressive rollout) and `EnsemblePredictor` (perturbed-forcing ensemble + statistics). Some production rollout scripts in `scripts/` still embed their model inline. |
| `serving/` | `app.py` — FastAPI inference service (`/health`, `/metadata`, `/predict`, `/predict/batch`) over the `Predictor`. |
| `visualization/` | `plots.py` — rollout and station time-series plotting. |

The package `__init__` re-exports the model classes and `create_model`, so
`from stofs_surrogate import STOFSSurrogateGNN` works after `pip install -e .`.

## Configuration (`configs/`)

- `train_default.yaml` — canonical architecture + training recipe; the single source of
  truth that mirrors the model documented in `README.md`.
- `ensemble.yaml` — ensemble-forecast settings; inherits common keys from
  `train_default.yaml`.

> Note: these YAML files currently document the configuration. The training/inference
> scripts take their parameters as command-line arguments rather than loading these files.

## Scripts (`scripts/`)

Representative entry points (run with `--help` for arguments):

- **Download:** `download_stofs.py`, `download_gfs_forcing.py`
- **Preprocess:** `preprocess_25k_v2.py`, `preprocess_25k_gfs_v2.py`
- **Mesh:** `create_longrange_mesh.py`, `visualize_longrange_edges.py`
- **Train:** `train_25k_ursa_h100_v2.py` (+ `run_25k_v2_ursa.sh` launcher)
- **Rollout / inference:** `rollout_25k_model.py`, `spatial_rollout_25k.py`, `generate_rollout.py`
- **Ensemble:** `ensemble_v2.py`, `extract_station_ensemble.py`, `plot_ensemble_v2.py`
- **Visualize:** `visualize_spatial_v2.py`, `visualize_stations_v2.py`, `plot_prediction_comparison.py`

Earlier development scripts (historical training/preprocessing variants) are preserved on
the `dev-history` branch rather than on `main`.

## Tests (`tests/`)

`test_model.py` (architecture), `test_data.py` (mesh/normalization/features), and
`test_inference.py` (perturbation/station/ensemble math). Run with `pytest` from the repo
root.
