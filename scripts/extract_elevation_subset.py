#!/usr/bin/env python3
"""
Extract US East Coast elevation time series from STOFS surf.63.nc

This script:
1. Loads the US East Coast mesh subset (node indices)
2. Extracts corresponding elevation time series from surf.63.nc
3. Saves processed data for GNN training
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import xarray as xr
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MESH_SUBSET_PATH = Path("data/processed/us_east_coast_mesh.npz")
ELEVATION_PATH = Path("data/raw/stofs_2d_glo.20251127/stofs_2d_glo_surf.63.nc")
OUTPUT_DIR = Path("data/processed")


def main():
    print("=" * 70)
    print("Extract US East Coast Elevation Time Series")
    print("=" * 70)

    # Check files exist
    if not MESH_SUBSET_PATH.exists():
        logger.error(f"Mesh subset not found: {MESH_SUBSET_PATH}")
        logger.info("Run: python scripts/extract_us_east_coast.py first")
        return

    if not ELEVATION_PATH.exists():
        logger.error(f"Elevation file not found: {ELEVATION_PATH}")
        logger.info("Download with: python scripts/download_stofs.py --date 20251127 --files elevation")
        return

    # Load mesh subset to get original indices
    print("\n1. Loading mesh subset...")
    mesh = np.load(MESH_SUBSET_PATH)
    original_indices = mesh['original_indices']
    num_nodes = len(original_indices)
    print(f"   Subset nodes: {num_nodes:,}")
    print(f"   Index range: [{original_indices.min():,}, {original_indices.max():,}]")

    # Open elevation file
    print(f"\n2. Opening elevation file: {ELEVATION_PATH}")
    print(f"   File size: {ELEVATION_PATH.stat().st_size / 1e9:.2f} GB")

    ds = xr.open_dataset(ELEVATION_PATH)

    print(f"\n   Dimensions:")
    for dim, size in ds.sizes.items():
        print(f"     {dim}: {size:,}")

    print(f"\n   Variables:")
    for var in ds.data_vars:
        print(f"     {var}: {ds[var].dims}")

    # Find elevation variable
    elev_var = None
    for name in ['zeta', 'elevation', 'surf_el', 'sea_surface_height']:
        if name in ds:
            elev_var = name
            break

    if elev_var is None:
        elev_var = list(ds.data_vars)[0]
        logger.warning(f"Using first variable as elevation: {elev_var}")

    print(f"\n   Using elevation variable: {elev_var}")

    # Get dimensions
    elev_data = ds[elev_var]
    num_times = elev_data.shape[0]
    total_nodes = elev_data.shape[1]
    print(f"   Time steps: {num_times}")
    print(f"   Total nodes in file: {total_nodes:,}")

    # Extract subset
    print(f"\n3. Extracting subset for {num_nodes:,} nodes...")
    print("   This may take a few minutes for large files...")

    # Load in chunks to manage memory
    chunk_size = 100  # timesteps per chunk
    num_chunks = (num_times + chunk_size - 1) // chunk_size

    elevation_subset = []

    for i in range(num_chunks):
        start_t = i * chunk_size
        end_t = min((i + 1) * chunk_size, num_times)

        # Extract time chunk
        chunk = elev_data.isel(time=slice(start_t, end_t))

        # Extract node subset
        if 'node' in chunk.dims:
            chunk_subset = chunk.isel(node=original_indices).values
        else:
            # Assume second dimension is nodes
            chunk_subset = chunk.values[:, original_indices]

        elevation_subset.append(chunk_subset)

        if (i + 1) % 10 == 0 or i == num_chunks - 1:
            print(f"   Processed {end_t}/{num_times} timesteps ({100*end_t/num_times:.1f}%)")

    # Concatenate
    elevation_subset = np.concatenate(elevation_subset, axis=0)
    print(f"\n   Extracted shape: {elevation_subset.shape}")

    # Handle fill values / NaN
    elevation_subset = np.nan_to_num(elevation_subset, nan=0.0)

    # Get time coordinates
    times = ds['time'].values
    print(f"   Time range: {times[0]} to {times[-1]}")

    # Statistics
    print(f"\n4. Elevation statistics:")
    print(f"   Min: {elevation_subset.min():.3f} m")
    print(f"   Max: {elevation_subset.max():.3f} m")
    print(f"   Mean: {elevation_subset.mean():.3f} m")
    print(f"   Std: {elevation_subset.std():.3f} m")

    # Save
    print(f"\n5. Saving extracted data...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "us_east_coast_elevation.npz"
    np.savez_compressed(
        output_file,
        elevation=elevation_subset.astype(np.float32),
        times=times,
        original_indices=original_indices,
    )

    print(f"   Saved: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1e6:.1f} MB")

    ds.close()

    # Summary
    print("\n" + "=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print(f"""
Data Summary:
  - Nodes: {num_nodes:,}
  - Time steps: {num_times}
  - Shape: {elevation_subset.shape}
  - File: {output_file}

Ready for training with real STOFS data!
Run: python scripts/train_us_east_coast_real.py
""")


if __name__ == '__main__':
    main()
