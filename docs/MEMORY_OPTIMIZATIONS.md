# Memory Optimizations for train_midatlantic_with_forcing.py

This document describes the memory optimizations applied to prevent WSL crashes when running the training script on systems with limited RAM.

## System Requirements

- RAM: ~24 GB available
- GPU: RTX 3050 Laptop (4 GB VRAM)

## Optimizations Applied

### 1. Reduced Data Size

| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| `max_nodes` | 8000 | 5000 | ~40% fewer mesh nodes |
| `CYCLES` | 4 | 2 | 50% fewer training cycles |
| Met forcing spatial subsampling | 1x | 2x | 75% less met grid data |

### 2. Float16 Storage

- All elevation and forcing data stored as `float16` instead of `float32`
- Reduces memory usage by 50% for large arrays
- Data converted to `float32` only when accessed for computation
- Controlled by `USE_FLOAT16_STORAGE = True` flag at top of script

### 3. Aggressive Garbage Collection

- `gc.collect()` called after loading each cycle
- Cleanup after intermediate processing steps
- Temporary variables (`forcing_raw`, `cycles_data`) deleted after use
- `gc.collect()` in training loop after each epoch

### 4. DataLoader Optimization

```python
DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,      # Avoids memory duplication from multiprocessing
    pin_memory=False    # Reduces memory overhead
)
```

### 5. Training Loop Cleanup

- Batch tensors deleted after each iteration
- Periodic `torch.cuda.empty_cache()` calls (every 10 batches)
- Garbage collection after each epoch

### 6. Processing Optimizations

- Met forcing files loaded one at a time with immediate cleanup
- Interpolation done in batches of 20 timesteps
- New mesh file version (`v3`) to use smaller node count

## If It Still Crashes

You can further reduce memory by adjusting these parameters in `main()`:

### Option 1: Temporal Subsampling (line ~924)
```python
elevation, times = extract_cycle_data(
    cwl_file,
    mesh_data['global_indices'],
    temporal_subsample=2  # Change from 1 to 2 (halves time steps)
)
```

### Option 2: Increase Met Forcing Subsampling (line ~932)
```python
forcing_raw = load_met_forcing_for_cycle(
    date_dir, met_dir, num_cwl_times,
    subsample_factor=3  # Change from 2 to 3
)
```

### Option 3: Reduce Mesh Nodes (line ~364)
```python
def extract_midatlantic_mesh(nc_file, bbox, max_nodes=3000):  # Reduce from 5000
```

### Option 4: Reduce Training Epochs (line ~75)
```python
EPOCHS = 100  # Reduce from 200
```

### Option 5: Use Only 1 Cycle (lines ~58-63)
```python
CYCLES = [
    ('stofs_2d_glo.20251122', 't00z', 'met_forcing_00z'),
    # ('stofs_2d_glo.20251122', 't12z', 'met_forcing_12z'),  # Comment out
]
```

## Estimated Memory Usage

With current optimizations (2 cycles, 5000 nodes, float16):
- Elevation data: ~2 cycles × 200 timesteps × 5000 nodes × 2 bytes ≈ 4 MB
- Forcing data: ~2 cycles × 200 timesteps × 5000 nodes × 3 vars × 2 bytes ≈ 12 MB
- Model parameters: ~200K parameters × 4 bytes ≈ 0.8 MB
- GPU memory during training: ~1-2 GB

Total peak RAM usage should stay under 8 GB with these optimizations.

## Reverting to Full Training

Once you confirm the script runs without crashing, you can gradually increase:

1. Uncomment additional cycles in `CYCLES`
2. Increase `max_nodes` to 8000 or higher
3. Set `subsample_factor=1` for full met forcing resolution
4. Set `USE_FLOAT16_STORAGE = False` if precision issues arise
