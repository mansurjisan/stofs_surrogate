#!/usr/bin/env python3
"""
Download GFS atmospheric forcing data for STOFS training.

Downloads 10m wind (u, v) and surface pressure from GFS to match CWL dates.
Supports multiple data sources for different time periods.

Data Sources:
- AWS noaa-gfs-bdp-pds: Recent ~2 weeks (fastest)
- NOMADS: Recent ~2 weeks
- Google Cloud ARCO: 2020-present (via Zarr)
- AWS NOAA GFS Archive: Historical (requires setup)

Usage:
    # Download for specific dates
    python scripts/download_gfs_forcing.py --dates 20230108 20230109

    # Download for all CWL dates in a directory
    python scripts/download_gfs_forcing.py --cwl-dir /path/to/stofs_data

    # Dry run to check availability
    python scripts/download_gfs_forcing.py --dates 20230108 --dry-run
"""

import os
import argparse
import subprocess
import requests
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Output directory for downloaded GFS data
DEFAULT_OUTPUT_DIR = '/mnt/f/STOFS_TRAINING_DATA/gfs_forcing'
DEFAULT_CWL_DIR = '/mnt/f/STOFS_TRAINING_DATA/stofs_data'

# Region of interest (with margin for interpolation)
BBOX = {
    'lon_min': -80.0,  # Extended for interpolation margin
    'lon_max': -69.0,
    'lat_min': 34.0,
    'lat_max': 45.0
}

# GFS parameters to download
# Using standard GFS variable names
GFS_VARIABLES = [
    'UGRD:10 m above ground',   # u-component of wind at 10m
    'VGRD:10 m above ground',   # v-component of wind at 10m
    'PRES:surface',             # surface pressure
    'PRMSL:mean sea level',     # mean sea level pressure (backup)
]

# Forecast hours to download
# STOFS CWL: 7hr nowcast + 179hr forecast = 186 total hours
# We skip nowcast (hours 0-6), use forecast (hours 7-185) = 179 timesteps
# VERIFIED: CWL hour 7 = GFS f000 (tested with avg diff 0.04 m/s)
# So we need GFS f000-f178
#
# GFS availability:
#   f000-f120: hourly (121 files)
#   f123-f180: 3-hourly (will interpolate to hourly)
FORECAST_HOURS = list(range(0, 121, 1)) + list(range(123, 180, 3))

# Data source URLs
AWS_GFS_BUCKET = 'noaa-gfs-bdp-pds'
NOMADS_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod'
NOMADS_FILTER = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl'

# Historical GFS archive on AWS (NOAA Open Data)
# This contains GFS data from 2021-present
AWS_GFS_ARCHIVE = 'noaa-gfs-bdp-pds'  # Same bucket, historical data available

# NCEI (NCEP archive) - for very old data
NCEI_BASE = 'https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/forecast'

# RDA (Research Data Archive) - requires registration
# https://rda.ucar.edu/datasets/ds084.1/
RDA_BASE = 'https://data.rda.ucar.edu/ds084.1'

# Google Cloud ARCO-ERA5 (alternative for historical)
GCS_ARCO = 'gs://gcp-public-data-arco-era5/ar/'


def check_tools():
    """Check available download tools."""
    tools = {}
    for tool in ['wget', 'curl', 'aws']:
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, timeout=5)
            tools[tool] = True
        except:
            tools[tool] = False
    return tools


def get_cwl_dates(cwl_dir: Path) -> list:
    """Get list of dates that have CWL data."""
    dates = []
    for d in sorted(cwl_dir.iterdir()):
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            cwl_file = d / 'stofs_2d_glo.t00z.fields.cwl.nc'
            if cwl_file.exists():
                dates.append(d.name)
    return dates


def build_nomads_filter_url(date_str: str, forecast_hour: int, bbox: dict) -> str:
    """
    Build NOMADS filter URL for subsetting GFS data.

    This uses the NOMADS filter service to download only the needed
    variables and region, significantly reducing download size.
    """
    # NOMADS filter parameters
    params = {
        'file': f'gfs.t00z.pgrb2.0p25.f{forecast_hour:03d}',
        'dir': f'/gfs.{date_str}/00/atmos',
        # Variables
        'var_UGRD': 'on',
        'var_VGRD': 'on',
        'var_PRES': 'on',
        'var_PRMSL': 'on',
        # Levels
        'lev_10_m_above_ground': 'on',
        'lev_surface': 'on',
        'lev_mean_sea_level': 'on',
        # Region subsetting
        'subregion': '',
        'leftlon': str(bbox['lon_min']),
        'rightlon': str(bbox['lon_max']),
        'toplat': str(bbox['lat_max']),
        'bottomlat': str(bbox['lat_min']),
    }

    query = '&'.join(f'{k}={v}' for k, v in params.items())
    return f'{NOMADS_FILTER}?{query}'


def build_aws_url(date_str: str, forecast_hour: int) -> str:
    """Build AWS S3 URL for GFS file."""
    return f's3://{AWS_GFS_BUCKET}/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{forecast_hour:03d}'


def build_aws_https_url(date_str: str, forecast_hour: int) -> str:
    """Build HTTPS URL for AWS GFS file (no auth needed)."""
    return f'https://{AWS_GFS_BUCKET}.s3.amazonaws.com/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{forecast_hour:03d}'


def check_file_exists_aws(date_str: str) -> bool:
    """Check if GFS data exists on AWS for a given date."""
    url = build_aws_https_url(date_str, 0)
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False


def check_file_exists_nomads(date_str: str) -> bool:
    """Check if GFS data exists on NOMADS for a given date."""
    url = f'{NOMADS_BASE}/gfs.{date_str}/00/atmos/'
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False


def download_file_wget(url: str, output_path: Path, timeout: int = 300) -> bool:
    """Download file using wget."""
    try:
        cmd = ['wget', '-q', '--timeout=60', '-O', str(output_path), url]
        result = subprocess.run(cmd, timeout=timeout, capture_output=True)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.debug(f"wget failed: {e}")
        return False


def download_file_curl(url: str, output_path: Path, timeout: int = 300) -> bool:
    """Download file using curl."""
    try:
        cmd = ['curl', '-s', '-f', '--connect-timeout', '30', '-o', str(output_path), url]
        result = subprocess.run(cmd, timeout=timeout, capture_output=True)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.debug(f"curl failed: {e}")
        return False


def download_file_requests(url: str, output_path: Path, timeout: int = 300, auth: tuple = None) -> bool:
    """Download file using requests library."""
    try:
        response = requests.get(url, timeout=timeout, stream=True, auth=auth)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path.exists() and output_path.stat().st_size > 0
        return False
    except Exception as e:
        logger.debug(f"requests failed: {e}")
        return False


def build_rda_url(date_str: str, forecast_hour: int) -> str:
    """
    Build RDA (Research Data Archive) URL for historical GFS data.

    RDA d084001 structure:
    https://data.rda.ucar.edu/d084001/YYYY/YYYYMMDD/gfs.0p25.YYYYMMDDHH.fFFF.grib2
    """
    year = date_str[:4]
    return f'https://data.rda.ucar.edu/d084001/{year}/{date_str}/gfs.0p25.{date_str}00.f{forecast_hour:03d}.grib2'


def download_from_rda(date_str: str, forecast_hour: int, output_path: Path,
                      email: str = None, password: str = None) -> bool:
    """
    Download GFS data from UCAR RDA.

    RDA allows direct downloads without authentication for data files.

    Args:
        date_str: Date in YYYYMMDD format
        forecast_hour: Forecast hour
        output_path: Output file path
        email: RDA account email (optional, not required for downloads)
        password: RDA account password (optional)
    """
    url = build_rda_url(date_str, forecast_hour)

    # Try wget first (handles large files better)
    if download_file_wget(url, output_path):
        return True

    # Try curl
    if download_file_curl(url, output_path):
        return True

    # Try requests
    try:
        response = requests.get(url, stream=True, timeout=600)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
            return output_path.exists() and output_path.stat().st_size > 1000
    except Exception as e:
        logger.debug(f"RDA download failed: {e}")

    return False


def download_gfs_hour(date_str: str, forecast_hour: int, output_dir: Path,
                      bbox: dict, use_filter: bool = True, use_rda: bool = False) -> bool:
    """
    Download a single GFS forecast hour.

    Args:
        date_str: Date in YYYYMMDD format
        forecast_hour: Forecast hour (0-384)
        output_dir: Output directory
        bbox: Bounding box for subsetting
        use_filter: If True, use NOMADS filter for subsetting
        use_rda: If True, try RDA for historical data

    Returns:
        True if successful
    """
    output_file = output_dir / f'gfs.{date_str}.f{forecast_hour:03d}.grib2'

    # Skip if already exists
    if output_file.exists() and output_file.stat().st_size > 1000:
        return True

    # Try NOMADS filter first (smaller download, recent data only)
    if use_filter:
        url = build_nomads_filter_url(date_str, forecast_hour, bbox)
        if download_file_wget(url, output_file) or download_file_curl(url, output_file):
            return True

    # Try AWS HTTPS (recent data)
    url = build_aws_https_url(date_str, forecast_hour)
    if download_file_wget(url, output_file) or download_file_curl(url, output_file):
        return True

    # Try NOMADS direct (recent data)
    url = f'{NOMADS_BASE}/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{forecast_hour:03d}'
    if download_file_wget(url, output_file) or download_file_curl(url, output_file):
        return True

    # Try RDA for historical data (requires credentials)
    if use_rda:
        if download_from_rda(date_str, forecast_hour, output_file):
            return True

    return False


def download_gfs_date(date_str: str, output_dir: Path, bbox: dict,
                      forecast_hours: list = None, max_workers: int = 4,
                      use_filter: bool = True, use_rda: bool = False) -> dict:
    """
    Download all GFS forecast hours for a single date.

    Args:
        date_str: Date in YYYYMMDD format
        output_dir: Output directory
        bbox: Bounding box
        forecast_hours: List of forecast hours to download
        max_workers: Number of parallel downloads
        use_filter: Use NOMADS filter for subsetting

    Returns:
        dict with download statistics
    """
    if forecast_hours is None:
        forecast_hours = FORECAST_HOURS

    date_dir = output_dir / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading GFS for {date_str} ({len(forecast_hours)} hours)...")

    success = 0
    failed = 0
    skipped = 0

    # Check which files already exist
    existing = set()
    for fhr in forecast_hours:
        fpath = date_dir / f'gfs.{date_str}.f{fhr:03d}.grib2'
        if fpath.exists() and fpath.stat().st_size > 1000:
            existing.add(fhr)
            skipped += 1

    to_download = [fhr for fhr in forecast_hours if fhr not in existing]

    if not to_download:
        logger.info(f"  All {len(forecast_hours)} files already exist")
        return {'success': 0, 'failed': 0, 'skipped': len(forecast_hours)}

    logger.info(f"  Downloading {len(to_download)} files, skipping {skipped} existing")

    # Download in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_gfs_hour, date_str, fhr, date_dir, bbox, use_filter, use_rda): fhr
            for fhr in to_download
        }

        for future in as_completed(futures):
            fhr = futures[future]
            try:
                if future.result():
                    success += 1
                else:
                    failed += 1
                    logger.warning(f"  Failed: f{fhr:03d}")
            except Exception as e:
                failed += 1
                logger.error(f"  Error f{fhr:03d}: {e}")

    logger.info(f"  Completed: {success} success, {failed} failed, {skipped} skipped")

    return {'success': success, 'failed': failed, 'skipped': skipped}


def check_gfs_availability(dates: list) -> dict:
    """
    Check GFS data availability for a list of dates.

    Returns dict mapping dates to availability status.
    """
    logger.info(f"Checking GFS availability for {len(dates)} dates...")

    availability = {}

    for i, date_str in enumerate(dates):
        if i % 50 == 0:
            logger.info(f"  Checked {i}/{len(dates)}...")

        # Check AWS first (faster)
        if check_file_exists_aws(date_str):
            availability[date_str] = 'aws'
        elif check_file_exists_nomads(date_str):
            availability[date_str] = 'nomads'
        else:
            availability[date_str] = None

        time.sleep(0.1)  # Rate limiting

    available = sum(1 for v in availability.values() if v is not None)
    logger.info(f"Available: {available}/{len(dates)} dates")

    return availability


def create_combined_netcdf(date_dir: Path, output_file: Path, bbox: dict):
    """
    Combine GRIB2 files into a single NetCDF with extracted variables.

    Requires wgrib2 or cfgrib/xarray.
    """
    try:
        import xarray as xr
        import cfgrib

        grib_files = sorted(date_dir.glob('gfs.*.f*.grib2'))
        if not grib_files:
            logger.warning(f"No GRIB files found in {date_dir}")
            return False

        logger.info(f"Converting {len(grib_files)} GRIB files to NetCDF...")

        datasets = []
        for grib_file in grib_files:
            try:
                # Read 10m wind
                ds_wind = xr.open_dataset(
                    grib_file, engine='cfgrib',
                    backend_kwargs={'filter_by_keys': {
                        'typeOfLevel': 'heightAboveGround',
                        'level': 10
                    }}
                )

                # Read surface pressure
                ds_pres = xr.open_dataset(
                    grib_file, engine='cfgrib',
                    backend_kwargs={'filter_by_keys': {
                        'typeOfLevel': 'surface',
                        'shortName': 'sp'
                    }}
                )

                # Merge
                ds = xr.merge([ds_wind, ds_pres])
                datasets.append(ds)

            except Exception as e:
                logger.debug(f"Could not read {grib_file}: {e}")

        if datasets:
            combined = xr.concat(datasets, dim='time')
            combined.to_netcdf(output_file)
            logger.info(f"Saved: {output_file}")
            return True

    except ImportError:
        logger.warning("cfgrib/xarray not available for NetCDF conversion")

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Download GFS forcing data for STOFS training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download for specific dates
    python download_gfs_forcing.py --dates 20230108 20230109 20230110

    # Download for all dates in CWL directory
    python download_gfs_forcing.py --cwl-dir /mnt/f/STOFS_TRAINING_DATA/stofs_data

    # Check availability without downloading
    python download_gfs_forcing.py --dates 20230108 --dry-run

    # Download with more parallel connections
    python download_gfs_forcing.py --dates 20230108 --workers 8
        """
    )

    parser.add_argument('--dates', nargs='+', help='Dates to download (YYYYMMDD)')
    parser.add_argument('--cwl-dir', type=str, help='Directory with CWL data (auto-detect dates)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for GFS data')
    parser.add_argument('--workers', type=int, default=4, help='Parallel download workers')
    parser.add_argument('--dry-run', action='store_true', help='Check availability only')
    parser.add_argument('--no-filter', action='store_true',
                        help='Download full files instead of using NOMADS filter')
    parser.add_argument('--forecast-hours', type=str, default='0-186',
                        help='Forecast hours to download (e.g., "0-186" or "0,3,6,9")')
    parser.add_argument('--convert-nc', action='store_true',
                        help='Convert downloaded GRIB to NetCDF')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check which dates have GFS available')
    parser.add_argument('--rda', action='store_true',
                        help='Use RDA archive for historical data (requires credentials)')
    parser.add_argument('--rda-email', type=str, help='RDA account email')
    parser.add_argument('--rda-password', type=str, help='RDA account password')

    args = parser.parse_args()

    # Check tools
    tools = check_tools()
    logger.info(f"Available tools: wget={tools.get('wget')}, curl={tools.get('curl')}, aws={tools.get('aws')}")

    if not tools.get('wget') and not tools.get('curl'):
        logger.error("Neither wget nor curl available. Please install one.")
        return

    # Determine dates to process
    if args.cwl_dir:
        cwl_dir = Path(args.cwl_dir)
        if not cwl_dir.exists():
            logger.error(f"CWL directory not found: {cwl_dir}")
            return
        dates = get_cwl_dates(cwl_dir)
        logger.info(f"Found {len(dates)} dates with CWL data")
    elif args.dates:
        dates = args.dates
    else:
        logger.error("Specify --dates or --cwl-dir")
        return

    # Parse forecast hours
    if '-' in args.forecast_hours:
        start, end = map(int, args.forecast_hours.split('-'))
        # GFS outputs hourly for f000-f120, 3-hourly after
        forecast_hours = list(range(start, min(end+1, 121), 1))
        if end > 120:
            forecast_hours += list(range(123, end+1, 3))
    else:
        forecast_hours = [int(x) for x in args.forecast_hours.split(',')]

    logger.info(f"Forecast hours: {len(forecast_hours)} (f{min(forecast_hours):03d} to f{max(forecast_hours):03d})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check availability only
    if args.check_only or args.dry_run:
        availability = check_gfs_availability(dates)

        # Print summary
        aws_count = sum(1 for v in availability.values() if v == 'aws')
        nomads_count = sum(1 for v in availability.values() if v == 'nomads')
        unavailable = [d for d, v in availability.items() if v is None]

        logger.info(f"\nAvailability Summary:")
        logger.info(f"  AWS: {aws_count} dates")
        logger.info(f"  NOMADS: {nomads_count} dates")
        logger.info(f"  Unavailable: {len(unavailable)} dates")

        if unavailable and len(unavailable) <= 20:
            logger.info(f"  Unavailable dates: {unavailable}")

        # Save availability report
        report_file = output_dir / 'availability_report.json'
        with open(report_file, 'w') as f:
            json.dump(availability, f, indent=2)
        logger.info(f"Saved availability report: {report_file}")

        if args.dry_run:
            return

    # Download data
    logger.info("=" * 70)
    logger.info("DOWNLOADING GFS FORCING DATA")
    logger.info("=" * 70)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Dates: {len(dates)}")
    logger.info(f"Forecast hours per date: {len(forecast_hours)}")
    logger.info(f"Using NOMADS filter: {not args.no_filter}")
    logger.info(f"Using RDA for historical: {args.rda}")

    # Set RDA credentials if provided
    if args.rda_email:
        os.environ['RDA_EMAIL'] = args.rda_email
    if args.rda_password:
        os.environ['RDA_PASSWORD'] = args.rda_password

    total_success = 0
    total_failed = 0
    total_skipped = 0

    for i, date_str in enumerate(dates):
        logger.info(f"\n[{i+1}/{len(dates)}] Processing {date_str}")

        result = download_gfs_date(
            date_str, output_dir, BBOX,
            forecast_hours=forecast_hours,
            max_workers=args.workers,
            use_filter=not args.no_filter,
            use_rda=args.rda
        )

        total_success += result['success']
        total_failed += result['failed']
        total_skipped += result['skipped']

        # Optional: convert to NetCDF
        if args.convert_nc and result['success'] > 0:
            date_dir = output_dir / date_str
            nc_file = output_dir / f'gfs_{date_str}.nc'
            create_combined_netcdf(date_dir, nc_file, BBOX)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total files: {total_success + total_failed + total_skipped}")
    logger.info(f"  Downloaded: {total_success}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Skipped (existing): {total_skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
