#!/usr/bin/env python3
"""
Extract 25k node subset from 80k processed data.

This script uses the global_indices from the 25k mesh to extract
the corresponding nodes from the 80k processed files.

Usage:
    python scripts/extract_25k_from_80k.py

Paths can be overridden with environment variables:
    STOFS_80K_DIR: Path to 80k processed data
    STOFS_25K_DIR: Path to output 25k data
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Default paths
DEFAULT_80K_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_80k_option_a')
DEFAULT_25K_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_full')
DEFAULT_25K_MESH = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k/mesh_25k.npz')


def extract_25k_from_80k(src_dir: Path, dst_dir: Path, mesh_25k_path: Path):
    """
    Extract 25k node subset from 80k processed data.

    Args:
        src_dir: Path to 80k processed data
        dst_dir: Path to output 25k data
        mesh_25k_path: Path to 25k mesh file with global_indices
    """
    # Load 25k mesh to get global indices
    print(f"Loading 25k mesh from {mesh_25k_path}")
    mesh_25k = np.load(mesh_25k_path)
    global_indices = mesh_25k['global_indices']
    print(f"  25k nodes: {len(global_indices):,}")
    print(f"  Index range: [{global_indices.min()}, {global_indices.max()}]")

    # Verify 80k mesh exists and check compatibility
    mesh_80k_path = src_dir / 'mesh.npz'
    if mesh_80k_path.exists():
        mesh_80k = np.load(mesh_80k_path)
        print(f"\n80k mesh: {len(mesh_80k['lon']):,} nodes")

        # Verify indices are within range
        if global_indices.max() >= len(mesh_80k['lon']):
            print(f"ERROR: global_indices max ({global_indices.max()}) >= 80k nodes ({len(mesh_80k['lon'])})")
            return
    else:
        print(f"Warning: 80k mesh not found at {mesh_80k_path}")

    # Create output directory
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy 25k mesh to output directory
    mesh_dst = dst_dir / 'mesh_25k.npz'
    if not mesh_dst.exists():
        print(f"\nCopying 25k mesh to {mesh_dst}")
        np.savez_compressed(mesh_dst, **{k: mesh_25k[k] for k in mesh_25k.files})

    # Find all processed files in 80k directory
    src_files = sorted(src_dir.glob('processed_*.npz'))
    print(f"\nFound {len(src_files)} processed files in {src_dir}")

    # Process each file
    processed = 0
    skipped = 0

    for src_file in tqdm(src_files, desc="Extracting 25k subset"):
        date_str = src_file.stem.replace('processed_', '')
        dst_file = dst_dir / f'processed_{date_str}.npz'

        # Skip if already exists
        if dst_file.exists():
            skipped += 1
            continue

        try:
            # Load 80k data
            data_80k = np.load(src_file)

            # Extract 25k subset for each array
            data_25k = {}
            for key in data_80k.files:
                arr = data_80k[key]
                if arr.ndim == 2 and arr.shape[1] == len(mesh_80k['lon']):
                    # Shape is [time, nodes] - extract node subset
                    data_25k[key] = arr[:, global_indices]
                elif arr.ndim == 1 and len(arr) == len(mesh_80k['lon']):
                    # Shape is [nodes] - extract node subset
                    data_25k[key] = arr[global_indices]
                else:
                    # Keep as-is (e.g., time arrays)
                    data_25k[key] = arr

            # Save 25k data
            np.savez_compressed(dst_file, **data_25k)
            processed += 1

        except Exception as e:
            print(f"\nError processing {src_file.name}: {e}")
            continue

    print(f"\nDone!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Output directory: {dst_dir}")


def verify_extraction(src_dir: Path, dst_dir: Path, mesh_25k_path: Path):
    """Verify the extraction by comparing a sample file."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    mesh_25k = np.load(mesh_25k_path)
    global_indices = mesh_25k['global_indices']

    # Find a common file
    dst_files = sorted(dst_dir.glob('processed_*.npz'))
    if not dst_files:
        print("No files to verify!")
        return

    sample_file = dst_files[0]
    date_str = sample_file.stem.replace('processed_', '')
    src_file = src_dir / f'processed_{date_str}.npz'

    print(f"\nComparing {date_str}:")

    data_80k = np.load(src_file)
    data_25k = np.load(sample_file)

    for key in data_25k.files:
        arr_25k = data_25k[key]
        arr_80k = data_80k[key]

        if arr_25k.ndim == 2 and arr_80k.ndim == 2:
            # Check extraction is correct
            expected = arr_80k[:, global_indices]
            match = np.allclose(arr_25k, expected, equal_nan=True)
            print(f"  {key}: shape {arr_80k.shape} -> {arr_25k.shape}, match={match}")
        else:
            print(f"  {key}: shape {arr_25k.shape}")

    # Check elevation statistics
    elev_25k = data_25k['elevation']
    valid = ~np.isnan(elev_25k)
    print(f"\n  Elevation stats:")
    print(f"    Valid values: {valid.sum():,} / {elev_25k.size:,} ({100*valid.mean():.1f}%)")
    print(f"    Range: [{np.nanmin(elev_25k):.3f}, {np.nanmax(elev_25k):.3f}] m")


def main():
    parser = argparse.ArgumentParser(description='Extract 25k from 80k processed data')
    parser.add_argument('--src-dir', type=str,
                        default=os.environ.get('STOFS_80K_DIR', str(DEFAULT_80K_DIR)),
                        help='Path to 80k processed data')
    parser.add_argument('--dst-dir', type=str,
                        default=os.environ.get('STOFS_25K_DIR', str(DEFAULT_25K_DIR)),
                        help='Path to output 25k data')
    parser.add_argument('--mesh-25k', type=str,
                        default=str(DEFAULT_25K_MESH),
                        help='Path to 25k mesh file')
    parser.add_argument('--verify', action='store_true',
                        help='Verify extraction after completion')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    mesh_25k_path = Path(args.mesh_25k)

    print("=" * 60)
    print("EXTRACT 25K SUBSET FROM 80K PROCESSED DATA")
    print("=" * 60)
    print(f"\nSource (80k): {src_dir}")
    print(f"Destination (25k): {dst_dir}")
    print(f"25k Mesh: {mesh_25k_path}")

    # Validate paths
    if not src_dir.exists():
        print(f"\nERROR: Source directory not found: {src_dir}")
        return
    if not mesh_25k_path.exists():
        print(f"\nERROR: 25k mesh file not found: {mesh_25k_path}")
        return

    extract_25k_from_80k(src_dir, dst_dir, mesh_25k_path)

    if args.verify:
        verify_extraction(src_dir, dst_dir, mesh_25k_path)


if __name__ == '__main__':
    main()
