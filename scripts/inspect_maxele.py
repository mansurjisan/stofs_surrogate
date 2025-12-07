#!/usr/bin/env python3
"""
Inspect the maxele.63.nc file - it's smaller and should contain mesh info.
"""

import subprocess
import tempfile
import os

# URL for smaller file with mesh info
url = "https://noaa-gestofs-pds.s3.amazonaws.com/_para/stofs_2d_glo.20251127/00/rerun/stofs_2d_glo_maxele.63.nc"

print("=" * 70)
print("Inspecting STOFS 2D Global maxele.63.nc")
print("=" * 70)
print(f"\nURL: {url}")
print("Size: ~851 MB (contains mesh + max elevation)\n")

with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
    tmp_path = tmp.name

try:
    print("Downloading (this may take a few minutes)...")
    cmd = ["curl", "-s", "--progress-bar", "-o", tmp_path, url]
    subprocess.run(cmd, timeout=600)

    if os.path.exists(tmp_path):
        file_size = os.path.getsize(tmp_path) / 1e6
        print(f"Downloaded: {file_size:.1f} MB")

        import xarray as xr
        ds = xr.open_dataset(tmp_path)

        print("\n" + "=" * 70)
        print("DIMENSIONS")
        print("=" * 70)
        for dim, size in ds.sizes.items():
            print(f"  {dim}: {size:,}")

        print("\n" + "=" * 70)
        print("COORDINATES")
        print("=" * 70)
        for coord in ds.coords:
            var = ds.coords[coord]
            print(f"  {coord}: {var.dims} {var.dtype}")

        print("\n" + "=" * 70)
        print("DATA VARIABLES")
        print("=" * 70)
        for var_name in ds.data_vars:
            var = ds[var_name]
            attrs = dict(var.attrs)
            units = attrs.get('units', 'N/A')
            long_name = attrs.get('long_name', var_name)
            print(f"\n  {var_name}:")
            print(f"    dims: {var.dims}")
            print(f"    shape: {var.shape}")
            print(f"    dtype: {var.dtype}")
            print(f"    long_name: {long_name}")
            if 'units' in attrs:
                print(f"    units: {units}")

            # Show range for coordinate-like variables
            if var_name in ['x', 'y', 'lon', 'lat', 'depth']:
                try:
                    print(f"    range: [{float(var.min()):.4f}, {float(var.max()):.4f}]")
                except:
                    pass

        # Check for mesh connectivity
        print("\n" + "=" * 70)
        print("MESH INFORMATION")
        print("=" * 70)

        if 'element' in ds.data_vars:
            elem = ds['element']
            print(f"  Element connectivity found!")
            print(f"  Shape: {elem.shape} (elements x nodes_per_element)")
            print(f"  This IS the mesh topology!")
        elif 'nele' in ds.dims:
            print(f"  Number of elements: {ds.dims['nele']:,}")
            print("  Element connectivity variable may have different name")
        else:
            print("  No element connectivity found in this file")
            print("  May need separate fort.14 mesh file")

        if 'x' in ds.data_vars and 'y' in ds.data_vars:
            num_nodes = len(ds['x'])
            print(f"\n  Number of nodes: {num_nodes:,}")
            print(f"  Longitude range: [{float(ds['x'].min()):.2f}, {float(ds['x'].max()):.2f}]")
            print(f"  Latitude range: [{float(ds['y'].min()):.2f}, {float(ds['y'].max()):.2f}]")

        if 'depth' in ds.data_vars:
            print(f"\n  Bathymetry available!")
            print(f"  Depth range: [{float(ds['depth'].min()):.2f}, {float(ds['depth'].max()):.2f}] m")

        print("\n" + "=" * 70)
        print("GLOBAL ATTRIBUTES (Selected)")
        print("=" * 70)
        important_attrs = ['model', 'version', 'agrid', 'grid_type', 'Conventions']
        for attr in important_attrs:
            if attr in ds.attrs:
                print(f"  {attr}: {ds.attrs[attr]}")

        ds.close()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY FOR AI TRAINING")
        print("=" * 70)
        print("""
This file contains:
  - Node coordinates (x, y in lon/lat)
  - Bathymetry (depth)
  - Maximum water elevation (zeta_max)
  - Time of maximum elevation (time_of_zeta_max)

For GNN training, we need from this bucket:
  1. stofs_2d_glo_maxele.63.nc - Get mesh coordinates + bathymetry (this file)
  2. stofs_2d_glo_surf.63.nc - Water elevation time series
  3. stofs_2d_glo_surf.64.nc - Velocity time series (optional)
  4. stofs_2d_glo_surf.68.nc - Meteorological forcing (optional)

The mesh connectivity (triangles) can be:
  - Extracted if 'element' variable exists
  - Reconstructed via Delaunay triangulation from (x, y) coordinates
  - Obtained from NOAA as separate fort.14 file
""")

finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
