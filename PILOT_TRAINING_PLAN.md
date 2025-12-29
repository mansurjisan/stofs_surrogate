# STOFS Surrogate Model - Pilot Training Plan

**Date:** December 27, 2025
**Domain:** Mid-Atlantic + New England (Norfolk, VA to Portland, ME)
**GPU:** NVIDIA A10G (23GB VRAM)
**Status:** Preprocessing in progress → Pilot training ready today

---

## Executive Summary

This document outlines the pilot training plan for a STOFS (Storm Surge) surrogate model using Graph Neural Networks (GNN). The model will predict coastal water levels (storm surge) across the Mid-Atlantic and New England regions using atmospheric forcing from GFS.

**Key Decision:** Run 30-day pilot training with 40k nodes to validate pipeline before full-scale training with 253 dates.

---

## 1. Domain Configuration

### Previous Model (NY Bight Only)
- **Domain:** 37-42°N, 77-72°W
- **Area:** 235,875 km²
- **Coastline:** ~500 km
- **Nodes:** 25,000
- **Resolution:** 3.1 km grid spacing
- **Training:** 30 dates, 24 hours on A10G

### Current Model (Mid-Atlantic + New England)
- **Domain:** 37-45°N, 77-66°W
- **Area:** 830,280 km² (3.5x larger)
- **Coastline:** ~1,100 km (Norfolk to Portland ME)
- **Nodes:** 40,000
- **Resolution:** 4.6 km grid spacing
- **Total wet nodes available:** 550,203 (using 7.3%)

### Resolution Comparison

| Configuration | Domain Size | Nodes | Grid Spacing | Notes |
|---------------|-------------|-------|--------------|-------|
| Previous | 235,875 km² | 25,000 | 3.1 km | NY Bight only |
| **Current (Pilot)** | **830,280 km²** | **40,000** | **4.6 km** | **Mid-Atl + New England** |
| Production option | 830,280 km² | 100,000 | 2.9 km | Matches previous resolution |

**Key Insight:** Despite having 60% more nodes, current setup has coarser resolution because domain is 3.5x larger. To match previous 3.1 km resolution on the larger domain would require ~88,000-100,000 nodes.

---

## 2. Training Data

### Dataset Overview
- **Total dates:** 360 (Nov-Mar for 2023, 2024, 2025)
- **Season:** Winter storm season only
- **Temporal coverage:** 179 forecast hours per date (skip 7-hour nowcast)
- **Training split:** 253 dates (2023-2024)
- **Validation split:** 107 dates (2025)

### Data Sources
1. **STOFS CWL (Coastal Water Level)**
   - Source: NOAA STOFS operational model
   - Location: `/mnt/f/STOFS_TRAINING_DATA/stofs_data/`
   - Size: 12 GB per date × 360 dates = 4.3 TB
   - Variables: Water elevation (zeta)
   - Temporal resolution: Hourly

2. **GFS Atmospheric Forcing**
   - Source: NOAA GFS (via AWS S3)
   - Location: `/mnt/f/STOFS_TRAINING_DATA/gfs_forcing/`
   - Size: 1.8 MB per date × 360 dates = 648 MB (compressed NPZ)
   - Variables: u10, v10, surface pressure
   - Temporal resolution: 3-hourly (interpolated to hourly)
   - Spatial resolution: 0.25° grid

### Temporal Alignment (Verified)
- **STOFS Hour 7** = **GFS f000** (forecast initialization)
- Skip first 7 hours of STOFS (nowcast period)
- Use hours 7-185 (179 timesteps) aligned with GFS f000-f178

---

## 3. Preprocessing Pipeline

### Status
- **Completed:** 114/360 dates
- **In progress:** Processing at ~1.5 min/file
- **ETA:** ~5 hours (completion ~3 PM today)

### Performance Optimization
**Original script:** 4.0 min/file (179 separate netCDF reads)
**Optimized script:** 1.5 min/file (single contiguous read)
**Speedup:** 2.7x faster

**Key optimization:** Changed from:
```python
# SLOW - 179 separate I/O operations
for t in timesteps:
    data[t] = nc[t, nodes]
```

To:
```python
# FAST - single contiguous read, subset in memory
block = nc[t_start:t_end, idx_min:idx_max]
data = block[:, local_indices]
```

### Output Format
- **Location:** `/mnt/f/STOFS_TRAINING_DATA/processed/`
- **Format:** Compressed NPZ files
- **Size:** ~73 MB per date
- **Variables per file:**
  - `elevation`: (179, 40000) - water level [meters]
  - `u10`: (179, 40000) - 10m u-wind [m/s]
  - `v10`: (179, 40000) - 10m v-wind [m/s]
  - `pressure`: (179, 40000) - normalized surface pressure

### Mesh Structure
- **File:** `mesh.npz`
- **Nodes:** 40,000 wet nodes
- **Edges:** 8,177 edges (bidirectional graph)
- **Node attributes:**
  - `lon`, `lat`: Coordinates
  - `depth`: Bathymetry [meters]
  - `global_indices`: Original STOFS mesh indices

---

## 4. Pilot Training Plan

### Objective
Validate the training pipeline and estimate timing for full production training.

### Configuration
- **Dates:** 30 dates from 2023
- **Nodes:** 40,000
- **Mesh resolution:** 4.6 km grid spacing
- **GPU:** NVIDIA A10G (23GB VRAM)
- **Expected duration:** ~24-30 hours (based on previous training)

### Training Timeline Estimate
Based on previous training (25k nodes, 30 dates, 24 hours):

| Configuration | Dates | Nodes | Domain Size | Estimated Time |
|---------------|-------|-------|-------------|----------------|
| Previous | 30 | 25k | Smaller | 24 hours |
| **Pilot** | **30** | **40k** | **3.5x larger** | **~30 hours** |
| Full (2023-2024) | 253 | 40k | Same | **~14 days** |

**Scaling factor:** 40k/25k = 1.6x more nodes, expect ~25-30% longer training time.

### Success Criteria
1. ✓ Model trains without OOM errors
2. ✓ Training loss decreases consistently
3. ✓ Validation metrics show reasonable predictions
4. ✓ Timing estimates confirmed for production run

### After Pilot Validation
If successful, proceed to full training:
- **Training set:** 253 dates (2023-2024)
- **Validation set:** 107 dates (2025)
- **Estimated time:** ~14 days on A10G

---

## 5. Production Training Considerations

### Option 1: Current Configuration (40k nodes)
- **Pros:**
  - Validated in pilot
  - Faster training iterations
  - Safe memory usage (~4 GB)
- **Cons:**
  - Coarser resolution (4.6 km) than previous model

### Option 2: High Resolution (100k nodes) - Recommended
- **Pros:**
  - Matches previous 3.1 km resolution
  - Better coastal feature representation
  - Still fits on A10G (~15 GB VRAM)
- **Cons:**
  - Longer preprocessing (~2-3x)
  - Longer training (~2-3x)
  - Requires reprocessing all 360 dates

### GPU Memory Estimates

| Nodes | Memory | Grid Spacing | Status |
|-------|--------|--------------|--------|
| 40,000 | ~4 GB | 4.6 km | ✓ Current (pilot) |
| 75,000 | ~7 GB | 3.3 km | ✓ Safe upgrade |
| 100,000 | ~15 GB | 2.9 km | ✓ Recommended production |
| 150,000 | ~22 GB | 2.4 km | ⚠ Risky (batch_size=1 only) |

**Recommendation:** After successful pilot, upgrade to 100k nodes for production training to match previous model's resolution on the larger domain.

---

## 6. Technical Infrastructure

### Hardware
- **GPU:** NVIDIA A10G, 23GB VRAM
- **Storage:**
  - F: drive (external) - 4.3 TB STOFS data
  - D: drive (local) - 253 GB free

### Software Stack
- **Python 3.x**
- **PyTorch** (GPU)
- **PyTorch Geometric** (GNN)
- **netCDF4** (STOFS data)
- **scipy** (interpolation)
- **numpy** (numerical operations)

### Key Scripts
1. `scripts/preprocess_npz.py` - Preprocessing with optimized I/O
2. `scripts/train_25k_temporal_memory.py` - Training script (needs update for 40k)
3. `scripts/rollout_temporal_memory_model.py` - Inference/evaluation

---

## 7. Known Issues and Solutions

### Issue 1: Slow Preprocessing (~4 min/file)
**Solution:** Implemented contiguous read optimization → 1.5 min/file (2.7x faster)

### Issue 2: F: Drive Bottleneck
- **Problem:** CWL files on slow external drive (12 GB each)
- **Impact:** 93% of preprocessing time is reading from F: drive
- **Attempted fixes:**
  - Writing to D: drive - No improvement (read is bottleneck, not write)
  - Parallel processing - Would cause I/O contention
  - Batch copying - Not feasible (need 4.3 TB local storage)
- **Conclusion:** 1.5 min/file is optimal given hardware constraints

### Issue 3: Temporal Alignment
- **Initial assumption:** STOFS Hour 6 = GFS f000 ❌
- **Verified:** STOFS Hour 7 = GFS f000 ✓
- **Solution:** Skip first 7 hours (nowcast), use hours 7-185

---

## 8. Next Steps

### Immediate (Today)
- [x] Download all GFS forcing (360 dates) - **COMPLETED**
- [x] Optimize preprocessing script - **COMPLETED**
- [ ] Complete preprocessing (114/360 done) - **IN PROGRESS (~5 hrs remaining)**

### Pilot Training (Next 2-3 days)
- [ ] Run 30-day pilot training
- [ ] Monitor GPU memory usage
- [ ] Validate training metrics
- [ ] Estimate full training timeline

### Production Training (After Pilot)
- [ ] Decision: 40k vs 100k nodes
- [ ] If 100k: Reprocess all dates with new mesh
- [ ] Run full training (253 dates, ~14 days)
- [ ] Evaluate on 2025 validation set

### Model Deployment
- [ ] Export trained model
- [ ] Create inference pipeline
- [ ] Benchmark vs operational STOFS
- [ ] Document results

---

## 9. File Locations

```
/mnt/d/AI_4_STOFS/stofs_surrogate/
├── scripts/
│   ├── preprocess_npz.py              # Optimized preprocessing
│   ├── train_25k_temporal_memory.py   # Training script
│   └── rollout_temporal_memory_model.py
├── data/
│   └── processed_temp/                # Temporary (unused)
└── outputs/                           # Training outputs

/mnt/f/STOFS_TRAINING_DATA/
├── stofs_data/                        # 4.3 TB CWL data
│   └── YYYYMMDD/
│       └── stofs_2d_glo.t00z.fields.cwl.nc
├── gfs_forcing/                       # 648 MB GFS data (NPZ)
│   └── YYYYMMDD/
│       └── gfs_YYYYMMDD_regional.npz
└── processed/                         # Training-ready data
    ├── mesh.npz                       # Graph structure
    └── processed_YYYYMMDD.npz         # 360 files × 73 MB
```

---

## 10. Contact and Documentation

**Project:** STOFS GNN Surrogate Model
**Domain:** Mid-Atlantic + New England Storm Surge Prediction
**Model Type:** Graph Neural Network (GraphCast-style)
**Purpose:** Fast surrogate for operational STOFS model

**Key References:**
- STOFS: https://polar.ncep.noaa.gov/stofs/
- GFS: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
- Previous model: NY Bight domain (25k nodes, 3.1 km resolution)

---

## Appendix A: Training Data Summary

### Winter Storm Season Coverage
- **2023:** 114 dates (Nov 1 - Mar 1)
- **2024:** 139 dates (Nov 1 - Mar 1)
- **2025:** 107 dates (Nov 1 - Mar 1)
- **Total:** 360 dates

### Data Statistics
- **Total nodes available:** 550,203 wet nodes in domain
- **Selected nodes:** 40,000 (7.3% utilization)
- **Timesteps per date:** 179 hours
- **Input features:** 4 (elevation, u10, v10, pressure)
- **Total samples:** 360 dates × 179 timesteps = 64,440 timesteps

---

## Appendix B: Resolution Sensitivity

To understand resolution requirements for storm surge:

| Feature | Scale | Resolution Needed |
|---------|-------|-------------------|
| Large storms | 100-500 km | 10+ km (coarse) |
| Coastal boundaries | 1-10 km | 1-5 km (medium) |
| Harbor/inlet dynamics | 100m-1km | <1 km (fine) |
| Local bathymetry | 10-100m | <500m (very fine) |

**Current 4.6 km resolution:**
- ✓ Captures large-scale storm patterns
- ✓ Resolves major coastal features
- ⚠ May miss small harbor/inlet details
- ✗ Cannot resolve fine bathymetric features

**Recommended 3 km resolution (100k nodes):**
- ✓ Better coastal boundary representation
- ✓ Improved harbor/inlet features
- ⚠ Still approximates fine bathymetry

**For very high resolution:** Would need >500k nodes (requires A100 80GB GPU)

---

*Last Updated: December 27, 2025*
*Status: Preprocessing in progress, pilot training planned for today*
