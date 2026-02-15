#!/usr/bin/env python3
"""
Inspect STOFS NetCDF file structure without downloading the full file.

This script downloads just the header/metadata of a STOFS NetCDF file
to understand its structure and whether mesh info is embedded.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import subprocess
import tempfile
import os

# URLs to inspect
STOFS_BASE = "https://noaa-gestofs-pds.s3.amazonaws.com/_para/stofs_2d_glo.20251127/00/rerun"

FILES_TO_INSPECT = [
    "stofs_2d_glo_maxele.63.nc",   # Smaller file - max elevation
    "stofs_2d_glo_surf.61.nc",     # Station output - very small
]


def download_partial(url: str, output_path: str, bytes_range: str = "0-1048576"):
    """Download partial file (first 1MB) to inspect header."""
    cmd = [
        "curl", "-s", "-r", bytes_range,
        "-o", output_path,
        url
    ]
    subprocess.run(cmd, check=True)


def inspect_netcdf(path: str):
    """Inspect NetCDF file structure using ncdump -h."""
    try:
        result = subprocess.run(
            ["ncdump", "-h", path],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    except FileNotFoundError:
        # ncdump not available, try Python
        try:
            import xarray as xr
            ds = xr.open_dataset(path)
            info = f"Dimensions: {dict(ds.dims)}\n"
            info += f"Variables: {list(ds.data_vars.keys())}\n"
            info += f"Coordinates: {list(ds.coords.keys())}\n"
            for var in ds.data_vars:
                info += f"  {var}: {ds[var].dims} {ds[var].dtype}\n"
            ds.close()
            return info
        except Exception as e:
            return f"Error: {e}"


def main():
    print("=" * 60)
    print("STOFS NetCDF Structure Inspector")
    print("=" * 60)

    # Try to download and inspect a small file
    url = f"{STOFS_BASE}/stofs_2d_glo_surf.61.nc"

    print(f"\nDownloading: {url}")
    print("(This is a small ~17MB station file)")

    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Download full small file
        cmd = ["curl", "-s", "-o", tmp_path, url]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, timeout=120)

        if result.returncode == 0 and os.path.exists(tmp_path):
            print(f"\nFile size: {os.path.getsize(tmp_path) / 1e6:.1f} MB")

            # Inspect structure
            print("\n" + "=" * 60)
            print("NetCDF Structure:")
            print("=" * 60)

            try:
                import xarray as xr
                ds = xr.open_dataset(tmp_path)

                print(f"\nDimensions:")
                for dim, size in ds.dims.items():
                    print(f"  {dim}: {size}")

                print(f"\nCoordinates:")
                for coord in ds.coords:
                    var = ds.coords[coord]
                    print(f"  {coord}: {var.dims} {var.dtype}")
                    if coord in ['x', 'y', 'lon', 'lat']:
                        print(f"    range: [{float(var.min()):.4f}, {float(var.max()):.4f}]")

                print(f"\nData Variables:")
                for var_name in ds.data_vars:
                    var = ds[var_name]
                    attrs = dict(var.attrs)
                    units = attrs.get('units', 'N/A')
                    long_name = attrs.get('long_name', 'N/A')
                    print(f"  {var_name}:")
                    print(f"    dims: {var.dims}")
                    print(f"    shape: {var.shape}")
                    print(f"    dtype: {var.dtype}")
                    print(f"    units: {units}")
                    print(f"    long_name: {long_name}")

                print(f"\nGlobal Attributes:")
                for attr, val in ds.attrs.items():
                    print(f"  {attr}: {val}")

                ds.close()

            except Exception as e:
                print(f"Error inspecting: {e}")
        else:
            print("Download failed")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
STOFS 2D Global NetCDF files typically contain:

1. COORDINATES (mesh geometry):
   - x, y: Node coordinates (lon/lat or projected)
   - element: Triangle connectivity (if included)

2. TIME:
   - time: Forecast/analysis times

3. DATA VARIABLES:
   - zeta / elevation: Water surface elevation (m)
   - u-vel, v-vel: Velocity components (m/s)
   - Various forcing fields

For training the GNN surrogate, we need:
- Node coordinates (from any .63.nc file)
- Time series of elevation (surf.63.nc)
- Time series of velocity (surf.64.nc)
- Optionally: forcing (surf.68.nc)

The mesh connectivity (triangles) may need to be obtained
separately from NOAA or reconstructed via Delaunay triangulation.
""")


if __name__ == '__main__':
    main()
