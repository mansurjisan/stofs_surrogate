# STOFS-GNN Surrogate Model - Experiment Log

## Project Overview
Building a Graph Neural Network surrogate for NOAA's STOFS (Surge and Tide Operational Forecast System) to enable rapid storm surge predictions in the Mid-Atlantic region.

---

## Completed Experiments

### Phase 1: Initial Development (November - Early December 2025)

#### 1a. Synthetic & Basic STOFS Training
**Dates:** November 2025
**Scripts:** `train_synthetic.py`, `train_stofs.py`, `train_us_east_coast.py`

**Purpose:** Initial proof-of-concept GNN training on STOFS data
- Started with synthetic data to validate architecture
- Moved to real STOFS data on US East Coast
- Established baseline training pipeline

---

#### 1b. Mid-Atlantic Regional Model
**Dates:** December 4-5, 2025
**Scripts:** `train_midatlantic.py`, `train_midatlantic_with_forcing.py`

**Configuration:**
- Domain: [-76, -73] × [38, 41] (NY, NJ, Delaware Bay, Philadelphia)
- Added meteorological forcing (wind u10/v10, pressure)
- Interpolation of met forcing from regular grid to ADCIRC mesh

**Key Learning:** Meteorological forcing significantly improves storm surge prediction

---

#### 1c. CWL GNN Development (Physics-Informed)
**Dates:** December 5-6, 2025
**Scripts:** `train_cwl_gnn_enhanced.py`, `train_cwl_gnn_multidate.py`, `train_cwl_gnn_optimized*.py`

**Enhancements implemented:**
1. Gradient term in message passing (critical for physics)
2. Curriculum learning (faster convergence)
3. Water level in static features
4. Physics-informed loss with mass conservation
5. Multi-date training support

**Hardware:** RTX 3050 (4GB VRAM) - required aggressive memory optimization

**Key Finding:** 15,000 nodes optimal for 4GB VRAM (5-7x faster than 50k)

---

#### 1d. 25K 15-Day Model (A10G)
**Dates:** December 7, 2025
**Script:** `train_25k_15day.py`
**Hardware:** NVIDIA A10G (24GB VRAM)

**Configuration:**
- Nodes: 25,000 (higher resolution)
- Training data: 15 days (Nov 15-29, 2025)
- ~1,650 training samples
- Expected training time: 3-4 hours

**Purpose:** Scale up to higher resolution with cloud GPU

---

#### 1e. Temporal Memory Model (Phase Lag Fix)
**Dates:** December 10, 2025
**Scripts:** `train_25k_temporal_memory.py`, `train_25k_temporal_memory_v3.py`

**Problem:** Model showed phase lag in tidal predictions - couldn't distinguish rising vs falling tide

**Solution:** Added temporal context to model inputs:
- η(t-1): Previous water level
- dη/dt: Rate of change

**Configuration:**
- TEMPORAL_FEATURES = 2 (was 0)
- Increased multi-step training horizon (4-6 steps)

**Result:** Phase lag issue resolved

---

#### 1f. 80K Full Domain Experiments
**Dates:** December 27-30, 2025
**Scripts:** `train_80k_option_a.py`, `train_80k_batched.py`, `train_80k_optimized.py`, `train_80k_inmemory.py`

**Configuration:**
- Nodes: 80,000 (full STOFS domain)
- Multiple optimization iterations for memory efficiency

**Challenges:**
- Memory constraints with large graph
- Required batched processing and in-memory optimizations

---

### Phase 2: Production Training (January 2026)

### 2. 25K V2 Model Training (Current Best)
**Status:** Completed
**Dates:** December 2025 - January 2026
**Location:** URSA H100 Cluster

**Configuration:**
- Nodes: 25,000
- Edges: 185,092
- Hidden dim: 128, Layers: 6
- Training data: 2023 (360 days)
- Validation data: 2025 (held out)

**Training Schedule:**
| Epochs | Rollout Steps | Batch Size |
|--------|---------------|------------|
| 1-15   | 1-step        | 4          |
| 16-30  | 2-step        | 2          |
| 31-50  | 3-step        | 2          |
| 51-100 | 6-step        | 1          |

**Results (Epoch 60, 2025 validation):**
| Lead Time | RMSE (cm) |
|-----------|-----------|
| t+6h      | 21.4      |
| t+12h     | 32.9      |
| t+24h     | 50.7      |
| t+48h     | 58.2      |

**Checkpoints saved:** epoch 15, 30, 50, 55, 60 (training ongoing to epoch 100)

---

### 2. 80K Model Training
**Status:** Completed (initial)
**Dates:** January 2026

**Configuration:**
- Nodes: 80,000
- Full STOFS domain with GFS forcing
- Multiple training script iterations (train_80k_h100.py, v2, fixed, improved)

**Notes:**
- Higher resolution but slower training
- Memory constraints required batch size adjustments

---

### 3. Long-Range Mesh Creation
**Status:** Completed
**Date:** January 17, 2026

**Purpose:** Add strategic long-range edges to improve information propagation for longer forecasts (24-48h)

**Edge Types Added:**
1. Bay mouth → inner bay connections (tidal signal propagation)
2. Along-coast connections (storm surge propagation)
3. Sparse global k=5 nearest neighbors
4. Coastal enhancement edges

**Results:**
- Original edges: 185,092
- New long-range edges: 262,449 (+141.8%)
- Total edges: 447,541

**Files:**
- `scripts/create_longrange_mesh.py`
- Output: `/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2_longrange/mesh.npz`

---

## Currently Running Experiments

### 4. Long-Range Fine-Tuning
**Status:** Starting (needs to be resubmitted)
**Date:** January 18, 2026
**Location:** URSA H100 Cluster

**Purpose:** Fine-tune the 25K V2 model (epoch 60) with the enhanced long-range mesh to improve 24-48h forecasts.

**Configuration:**
- Starting checkpoint: `checkpoint_epoch_60.pt` (best performing)
- Learning rate: 2e-5 (10x lower for fine-tuning)
- Batch size: 1 (due to 2.4x more edges)
- Gradient accumulation: 32 steps

**Fine-Tuning Schedule:**
| Epochs | Rollout Steps | Notes |
|--------|---------------|-------|
| 1-15   | 6-step        | ~8.3 hrs/epoch |
| 16-30  | 12-step       | |
| 31-50  | 24-step       | |

**Estimated Time:**
- ~8.3 hours per epoch (vs ~4 hours for original)
- 72-hour job limit = ~8-9 epochs per submission
- Total 50 epochs = ~6 job submissions

**Script:** `scripts/train_25k_longrange.py`

**Action Required:**
1. Cancel any running job from ep55
2. Ensure `RESUME_FROM` points to `checkpoint_epoch_60.pt`
3. Clear any existing longrange checkpoints
4. Resubmit job

---

## Planned Experiments

### 5. Long-Range Model Evaluation
**Status:** Pending (after long-range training completes)
**Estimated Date:** Late January 2026

**Tasks:**
- [ ] Compare long-range model vs original at 24-48h lead times
- [ ] Generate spatial error maps for both models
- [ ] Station time series comparison
- [ ] Quantify improvement in bay regions (Chesapeake, Delaware)

**Expected Improvements:**
- Better tidal signal propagation into bays
- Reduced error accumulation at longer lead times
- Improved coastal correlation

---

### 6. Scheduled Sampling Training
**Status:** Planned
**Script:** `scripts/train_25k_v2_scheduled_sampling.py`

**Purpose:** Gradually transition from teacher forcing to autoregressive during training to reduce exposure bias.

---

### 7. Multi-Scale Model
**Status:** Planned
**Script:** `scripts/train_25k_multiscale.py`

**Purpose:** Hierarchical approach with different resolution levels for local vs global dynamics.

---

## Key Findings

### Architecture Insights (Phase 1)
1. **Temporal memory is critical**: Adding η(t-1) and dη/dt resolved phase lag issues
2. **Gradient term in message passing**: Essential for physics-informed learning
3. **Curriculum learning**: Start with 1-step, gradually increase rollout horizon
4. **15k nodes**: Sweet spot for 4GB VRAM; 25k needs 24GB+

### Training Insights (Phase 2)
1. **6-step rollout** significantly improves long-term forecasts vs 1-3 step
2. **Epoch 60 > Epoch 55**: 10-14% improvement across all lead times
3. **2025 validation** shows true generalization (2023 was training data)
4. **Memory constraints**: 447k edges requires batch=1 on H100
5. **Met forcing matters**: Wind and pressure significantly improve storm surge prediction

### Model Performance
- Best performance in protected bays (Baltimore R=0.99)
- Challenging areas: open coast, bay mouths
- Error grows roughly linearly with lead time

### Visualization
- Spatial error maps show systematic patterns (not random noise)
- Largest errors at bay mouths and coastal boundaries
- Long-range edges target these problem areas

---

## File Structure

```
stofs_surrogate/
├── scripts/
│   ├── train_25k_ursa_h100_v2.py    # Main 25k training
│   ├── train_25k_longrange.py        # Long-range fine-tuning
│   ├── create_longrange_mesh.py      # Create enhanced mesh
│   ├── visualize_*.py                # Visualization scripts
│   └── ...
├── outputs/
│   ├── checkpoints_25k_v2/           # Original model checkpoints
│   ├── checkpoints_25k_longrange/    # Long-range checkpoints (URSA)
│   └── figures_25k_v2/               # Visualization outputs
└── EXPERIMENT_LOG.md                 # This file
```

---

## Data Locations

| Data | Path |
|------|------|
| 25K V2 processed | `/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2/` |
| 25K longrange mesh | `/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2_longrange/` |
| 80K processed | `/mnt/f/STOFS_TRAINING_DATA/processed_80k_option_a/` |
| Checkpoints (local) | `/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_v2/` |
| Checkpoints (URSA) | `/scratch5/purged/Mansur.Jisan/stofs_surrogate/outputs/` |

---

## Next Actions (Priority Order)

1. **[HIGH]** Submit long-range training job on URSA (from epoch 60)
2. **[HIGH]** Continue monitoring original 25K V2 training to epoch 100
3. **[MEDIUM]** Evaluate long-range checkpoints as they become available
4. **[MEDIUM]** Generate comparison plots (long-range vs original)
5. **[LOW]** Explore scheduled sampling approach
6. **[LOW]** Consider ensemble of original + long-range models

---

*Last updated: January 18, 2026*
