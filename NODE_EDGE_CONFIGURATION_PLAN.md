# Node and Edge Configuration Plan for STOFS Surrogate Model

**Date:** December 27, 2025
**GPU:** NVIDIA A10G (23GB VRAM)
**Domain:** Mid-Atlantic + New England (37-45°N, 77-66°W, 830,280 km²)

---

## 1. Problem Analysis: Why 40k Has Fewer Edges Than 25k

### Current Edge Count Comparison
| Model | Nodes | Edges | Edges/Node | Val Loss |
|-------|-------|-------|------------|----------|
| 25k (NY Bight) | 25,000 | 149,944 | **6.0** | 0.00041 |
| 40k (Mid-Atl+NE) | 40,000 | 16,354 | **0.4** | 0.00083 |

### Root Cause: Triangle-Based Edge Construction

The current preprocessing (`preprocess_npz.py:220-238`) builds edges from STOFS mesh triangles:

```python
def build_edges(element, global_indices):
    """Build edge connectivity from triangular elements."""
    for tri in element:
        local_nodes = [global_to_local[n] for n in tri if n in selected_set]
        if len(local_nodes) >= 2:  # Only if triangle has 2+ selected nodes
            # Add edge...
```

**The problem:** When randomly sampling 40k nodes from 550k+ wet nodes (7.3% selection rate):
- Probability a triangle has ≥2 selected nodes ≈ 1.5%
- Most triangles contribute zero edges
- Result: Very sparse connectivity (0.4 edges/node)

**The 25k model likely used:**
- Higher node density in smaller domain
- Or k-nearest neighbor edge construction
- Or radius-based edge construction

---

## 2. Edge Construction Strategies

### Strategy A: K-Nearest Neighbors (Recommended)
Connect each node to its k nearest neighbors.

**Pros:**
- Consistent edge density regardless of subsampling
- Simple to implement
- Intuitive control: k=6 gives 6 edges/node

**Cons:**
- May create long edges across domain if nodes sparse
- Slightly more computation at preprocessing

**Typical values:** k=6-10 for 2D coastal meshes

### Strategy B: Radius-Based
Connect all nodes within radius r (in km).

**Pros:**
- Physically intuitive (reflects spatial correlation)
- Natural multi-scale if combined with distance weighting

**Cons:**
- Edge count varies with local node density
- Need to tune radius per configuration

**Typical values:** r=10-30 km depending on node spacing

### Strategy C: Hybrid (KNN + Radius)
Use KNN for base connectivity, add long-range edges for fast information propagation.

**Example:**
- k=6 nearest neighbors (local physics)
- Add edges to k=2 nodes within 50km band (regional transport)

---

## 3. GPU Memory Analysis

### Memory Components

For GNN training, GPU memory usage includes:

1. **Node features:** `nodes × hidden_dim × 4 bytes × 2 (forward+backward)`
2. **Edge features:** `edges × 3 × 4 bytes × 2`
3. **Message passing:** `edges × hidden_dim × 4 bytes × num_layers`
4. **Activations:** Proportional to nodes × layers
5. **Optimizer states:** ~2x model parameters

### Memory Formula (Approximate)

```
Total Memory (GB) ≈
    nodes × 0.00004   (node overhead)
  + edges × 0.00006   (edge overhead per layer)
  + base_overhead     (~2 GB for model, optimizer)
```

### Configuration Matrix

| Nodes | Edge Strategy | Edges | Edges/Node | Est. Memory | Grid Spacing | Status |
|-------|---------------|-------|------------|-------------|--------------|--------|
| 40,000 | Triangle (current) | 16,354 | 0.4 | ~3 GB | 4.6 km | ✓ Running |
| 40,000 | KNN k=6 | 240,000 | 6.0 | ~5 GB | 4.6 km | **Recommended** |
| 40,000 | KNN k=8 | 320,000 | 8.0 | ~6 GB | 4.6 km | ✓ Safe |
| 60,000 | KNN k=6 | 360,000 | 6.0 | ~7 GB | 3.7 km | ✓ Safe |
| 80,000 | KNN k=6 | 480,000 | 6.0 | ~9 GB | 3.2 km | ✓ Safe |
| 100,000 | KNN k=6 | 600,000 | 6.0 | ~12 GB | 2.9 km | ✓ Production |
| 100,000 | KNN k=8 | 800,000 | 8.0 | ~15 GB | 2.9 km | ✓ High quality |
| 150,000 | KNN k=6 | 900,000 | 6.0 | ~18 GB | 2.4 km | ⚠ Tight |
| 150,000 | KNN k=8 | 1,200,000 | 8.0 | ~22 GB | 2.4 km | ⚠ Batch=1 only |

---

## 4. Recommendations

### Immediate Fix (Quick Win)
**Reprocess 40k with KNN k=6 edges**

This is the fastest path to improved accuracy:
- Keep current 40k nodes (preprocessed data reusable for node features)
- Only rebuild edge connectivity with KNN
- Expected improvement: Loss from 0.00083 → ~0.0005 (based on 25k performance)
- Training time: ~3x longer per epoch (but still faster than 25k's 41 hours)

### Production Configuration (Recommended)
**100k nodes with KNN k=6**

| Metric | 40k (current) | 100k (production) |
|--------|---------------|-------------------|
| Nodes | 40,000 | 100,000 |
| Edges | 16,354 → 240,000 | 600,000 |
| Grid spacing | 4.6 km | 2.9 km |
| Memory | ~3 GB | ~12 GB |
| Training time | ~12 hrs (30 days) | ~36-48 hrs (30 days) |
| Expected loss | ~0.0005 (with KNN) | ~0.0003 |

### High Resolution Option
**100k nodes with KNN k=8**

For best accuracy when training time is not critical:
- 800k edges provides dense connectivity
- Better representation of coastal features
- ~15 GB memory (safe margin on A10G)

---

## 5. Implementation Plan

### Step 1: Quick Fix for Current 40k (Today)

Add KNN edge construction to preprocessing:

```python
from scipy.spatial import cKDTree

def build_edges_knn(lon, lat, k=6):
    """Build edges using K-nearest neighbors."""
    # Convert to Cartesian for distance calculation
    coords = np.column_stack([
        lon * np.cos(np.radians(lat.mean())),  # x ≈ lon * cos(lat_center)
        lat
    ]) * 111.0  # degrees to km

    tree = cKDTree(coords)

    edges = set()
    for i in range(len(lon)):
        # Find k+1 neighbors (includes self)
        dists, neighbors = tree.query(coords[i], k=k+1)
        for j in neighbors[1:]:  # Skip self
            edges.add((min(i, j), max(i, j)))

    # Convert to bidirectional edge_index
    edge_list = list(edges)
    src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]

    return np.array([src, dst], dtype=np.int64)
```

### Step 2: Rebuild Mesh Only

No need to reprocess all 360 dates. Just rebuild mesh.npz with new edges:

```python
# Load existing mesh
mesh = np.load('data/processed_40k/mesh.npz')
lon, lat = mesh['lon'], mesh['lat']

# Build new edges with KNN
edge_index = build_edges_knn(lon, lat, k=6)

# Save updated mesh
np.savez_compressed('data/processed_40k/mesh.npz',
    lon=lon, lat=lat,
    depth=mesh['depth'],
    global_indices=mesh['global_indices'],
    edge_index=edge_index
)
```

### Step 3: Production 100k (After Pilot Validation)

1. Modify `preprocess_npz.py`:
   - Change `MAX_NODES = 100000`
   - Replace `build_edges()` with `build_edges_knn(k=6)`

2. Reprocess all 360 dates:
   ```bash
   python scripts/preprocess_npz.py --max-nodes 100000 --force
   ```
   Estimated time: ~9 hours (1.5 min/file × 360 files)

3. Run pilot training with 100k:
   ```bash
   sbatch scripts/run_100k_pilot_training.sh
   ```

---

## 6. Training Time Estimates

### Per-Epoch Scaling

Training time per epoch scales with:
- **Nodes:** O(n) for node updates
- **Edges:** O(e) for message passing (dominant factor)
- **Layers:** O(L) linear with depth

Formula: `time_per_epoch ≈ base_time × (edges / ref_edges) × (layers / ref_layers)`

### Comparison Table

| Config | Nodes | Edges | Est. Time/Epoch | 100 Epochs (30 days) |
|--------|-------|-------|-----------------|---------------------|
| 40k-triangle | 40k | 16k | ~6 sec | ~12 hours |
| 25k-dense | 25k | 150k | ~17 sec | ~41 hours |
| 40k-knn6 | 40k | 240k | ~25 sec | ~70 hours |
| 100k-knn6 | 100k | 600k | ~65 sec | ~180 hours (7.5 days) |

### Full Training (253 dates)

| Config | 30-day Pilot | 253-day Full |
|--------|--------------|--------------|
| 40k-triangle | 12 hrs | ~4 days |
| 40k-knn6 | 70 hrs | ~25 days |
| 100k-knn6 | 7.5 days | ~63 days |

**Recommendation:** For production, consider:
- Multi-GPU training (if available)
- Gradient accumulation to reduce memory overhead
- Mixed precision (already enabled)

---

## 7. Quick Reference: Preprocessing Commands

### Current 40k (sparse edges)
```bash
python scripts/preprocess_npz.py --max-nodes 40000
# Result: 40k nodes, ~16k edges, 4.6 km spacing
```

### 40k with KNN edges (recommended quick fix)
```bash
python scripts/preprocess_npz.py --max-nodes 40000 --edge-method knn --k 6
# Result: 40k nodes, ~240k edges, 4.6 km spacing
```

### 100k production
```bash
python scripts/preprocess_npz.py --max-nodes 100000 --edge-method knn --k 6
# Result: 100k nodes, ~600k edges, 2.9 km spacing
```

---

## 8. Summary

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| **High** | Add KNN edges to 40k mesh | +50% accuracy | 1 hour |
| **Medium** | Upgrade to 100k nodes | +30% accuracy, 3x resolution | 1 day |
| **Low** | Increase k to 8 | +10% accuracy | Minimal |

**Immediate next step after pilot completes:**
1. Rebuild 40k mesh with KNN k=6 edges
2. Re-run training to validate improvement
3. If successful, proceed to 100k production

---

*Last Updated: December 27, 2025*
