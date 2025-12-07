#!/usr/bin/env python3
"""
Download STOFS 2D Global data from NOAA AWS S3 bucket.

Usage:
    python scripts/download_stofs.py --date 20251127 --hour 00
    python scripts/download_stofs.py --date 20251127 --hour 00 --files mesh,elevation
    python scripts/download_stofs.py --list  # List available dates
"""

import argparse
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys

# S3 bucket base URL
S3_BASE = "https://noaa-gestofs-pds.s3.amazonaws.com"

# File types and their names
FILE_TYPES = {
    'mesh': 'stofs_2d_glo_maxele.63.nc',      # ~851 MB - Contains mesh + max elevation
    'elevation': 'stofs_2d_glo_surf.63.nc',   # ~14 GB - Water elevation time series
    'velocity': 'stofs_2d_glo_surf.64.nc',    # ~28 GB - Velocity time series
    'forcing': 'stofs_2d_glo_surf.68.nc',     # ~1.5 GB - Met forcing
    'tide': 'stofs_2d_glo_tide.63.nc',        # ~14 GB - Tidal elevation
    'stations': 'stofs_2d_glo_surf.61.nc',    # ~17 MB - Station output (validation)
}

# Smaller files for quick testing
QUICK_TEST_FILES = ['mesh', 'stations']


def get_s3_url(date: str, hour: str, filename: str, para: str = '_para') -> str:
    """Construct S3 URL for STOFS file."""
    return f"{S3_BASE}/{para}/stofs_2d_glo.{date}/{hour}/rerun/{filename}"


def download_file(url: str, output_path: Path, resume: bool = True) -> bool:
    """Download file using curl with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["curl", "-L", "--progress-bar"]

    if resume and output_path.exists():
        cmd.extend(["-C", "-"])  # Resume download

    cmd.extend(["-o", str(output_path), url])

    print(f"\nDownloading: {output_path.name}")
    print(f"URL: {url}")

    try:
        result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Download timed out!")
        return False
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run again to resume.")
        return False


def list_available_dates():
    """List some recent available dates (checking S3 is slow, so we provide guidance)."""
    print("\nSTOFS 2D Global Data Availability")
    print("=" * 50)
    print("""
The NOAA GESTOFS S3 bucket contains STOFS 2D Global data under:
  s3://noaa-gestofs-pds/_para/stofs_2d_glo.YYYYMMDD/

Data is organized by:
  - Date: YYYYMMDD format
  - Cycle: 00, 06, 12, 18 (forecast cycles)
  - Subdirectory: 'rerun/' contains the output files

To check if a specific date exists, try:
  curl -I https://noaa-gestofs-pds.s3.amazonaws.com/_para/stofs_2d_glo.20251127/00/rerun/stofs_2d_glo_maxele.63.nc

Recent dates in the bucket:
  - 2024: January through present (operational)
  - Files are retained for approximately 30-60 days

For training, we recommend downloading multiple dates to capture
different weather conditions and storm events.
""")


def main():
    parser = argparse.ArgumentParser(
        description='Download STOFS 2D Global data from NOAA AWS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download mesh file only (recommended first step)
  python scripts/download_stofs.py --date 20251127 --hour 00 --files mesh

  # Download mesh and elevation for training
  python scripts/download_stofs.py --date 20251127 --hour 00 --files mesh,elevation

  # Download all files for a date
  python scripts/download_stofs.py --date 20251127 --hour 00 --files all

  # Quick test with small files
  python scripts/download_stofs.py --date 20251127 --hour 00 --quick
        """
    )

    parser.add_argument('--date', type=str, help='Date in YYYYMMDD format')
    parser.add_argument('--hour', type=str, default='00', choices=['00', '06', '12', '18'],
                        help='Forecast cycle hour')
    parser.add_argument('--files', type=str, default='mesh',
                        help='Files to download: mesh,elevation,velocity,forcing,all')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--list', action='store_true',
                        help='List available dates')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: download only small files')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh download (no resume)')

    args = parser.parse_args()

    if args.list:
        list_available_dates()
        return

    if not args.date:
        parser.print_help()
        print("\nError: --date is required (e.g., --date 20251127)")
        sys.exit(1)

    # Determine which files to download
    if args.quick:
        files_to_download = QUICK_TEST_FILES
    elif args.files == 'all':
        files_to_download = list(FILE_TYPES.keys())
    else:
        files_to_download = [f.strip() for f in args.files.split(',')]

    # Validate file types
    for f in files_to_download:
        if f not in FILE_TYPES:
            print(f"Unknown file type: {f}")
            print(f"Available: {', '.join(FILE_TYPES.keys())}")
            sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output) / f"stofs_2d_glo.{args.date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STOFS 2D Global Data Download")
    print("=" * 60)
    print(f"Date: {args.date}")
    print(f"Cycle: {args.hour}Z")
    print(f"Output: {output_dir}")
    print(f"Files: {', '.join(files_to_download)}")

    # Estimate sizes
    sizes = {
        'mesh': '851 MB',
        'elevation': '14.2 GB',
        'velocity': '27.7 GB',
        'forcing': '1.5 GB',
        'tide': '14.2 GB',
        'stations': '17 MB',
    }
    print("\nEstimated download sizes:")
    total_mb = 0
    for f in files_to_download:
        size_str = sizes.get(f, 'unknown')
        print(f"  {f}: {size_str}")
        if 'GB' in size_str:
            total_mb += float(size_str.replace(' GB', '')) * 1024
        elif 'MB' in size_str:
            total_mb += float(size_str.replace(' MB', ''))

    print(f"  Total: ~{total_mb/1024:.1f} GB")
    print()

    # Download files
    success_count = 0
    for file_type in files_to_download:
        filename = FILE_TYPES[file_type]
        url = get_s3_url(args.date, args.hour, filename)
        output_path = output_dir / filename

        if download_file(url, output_path, resume=not args.no_resume):
            success_count += 1
            print(f"  ✓ Downloaded: {filename}")
        else:
            print(f"  ✗ Failed: {filename}")

    print("\n" + "=" * 60)
    print(f"Download complete: {success_count}/{len(files_to_download)} files")
    print("=" * 60)

    if success_count > 0:
        print(f"\nFiles saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Train on this data:")
        print(f"     python scripts/train_stofs.py --mesh {output_dir}/stofs_2d_glo_maxele.63.nc")
        print("  2. Or inspect the data:")
        print(f"     python -c \"import xarray as xr; print(xr.open_dataset('{output_dir}/stofs_2d_glo_maxele.63.nc'))\"")


if __name__ == '__main__':
    main()
