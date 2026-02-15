#!/usr/bin/env python3
"""
Batch download GFS forcing data for all STOFS training dates.

Downloads from RDA, extracts US East Coast region, saves as compressed NPZ.
This reduces storage from ~500MB to ~5MB per file.

Usage:
    python scripts/download_gfs_batch.py --start-date 20201101 --end-date 20201110
    python scripts/download_gfs_batch.py --all  # Download for all STOFS dates
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
STOFS_DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/stofs_data')
GFS_OUTPUT_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/gfs_forcing')

# RDA URL pattern (no auth required for direct downloads)
RDA_URL = "https://data.rda.ucar.edu/d084001/{year}/{date}/gfs.0p25.{date}00.f{fhr:03d}.grib2"

# Region of interest (US East Coast with margin)
BBOX = {
    'lon_min': -82.0,
    'lon_max': -64.0,
    'lat_min': 24.0,
    'lat_max': 46.0
}

# GFS forecast hours needed
# VERIFIED: STOFS Hour 7 = GFS f000
# Need f000-f178 for STOFS Hours 7-185 (179 hours)
GFS_HOURS_HOURLY = list(range(0, 121))      # f000-f120 (hourly)
GFS_HOURS_3HOURLY = list(range(123, 180, 3)) # f123-f177 (3-hourly)
GFS_HOURS_ALL = GFS_HOURS_HOURLY + GFS_HOURS_3HOURLY

# Download settings
MAX_WORKERS = 4  # Parallel downloads
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds


def get_stofs_dates():
    """Get all STOFS dates from the data directory."""
    dates = []
    for d in STOFS_DATA_DIR.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            dates.append(d.name)
    return sorted(dates)


def download_gfs_file(date: str, fhr: int, output_dir: Path) -> bool:
    """Download a single GFS file from RDA."""
    year = date[:4]
    url = RDA_URL.format(year=year, date=date, fhr=fhr)

    output_file = output_dir / f"gfs.{date}.f{fhr:03d}.grib2"

    if output_file.exists():
        return True

    for attempt in range(RETRY_ATTEMPTS):
        try:
            result = subprocess.run(
                ['curl', '-s', '-f', '-o', str(output_file), url],
                capture_output=True,
                timeout=300
            )
            if result.returncode == 0 and output_file.exists():
                return True
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")

        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(RETRY_DELAY)

    return False


def extract_region_pygrib(grib_file: Path, bbox: dict) -> dict:
    """Extract regional subset using pygrib."""
    import pygrib

    grbs = pygrib.open(str(grib_file))

    data = {}
    for grb in grbs:
        if grb.shortName == '10u':
            lats, lons = grb.latlons()
            # Convert to -180 to 180
            lons = np.where(lons > 180, lons - 360, lons)

            # Find indices for region
            lat_mask = (lats[:, 0] >= bbox['lat_min']) & (lats[:, 0] <= bbox['lat_max'])
            lon_mask = (lons[0, :] >= bbox['lon_min']) & (lons[0, :] <= bbox['lon_max'])

            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                continue

            lat_slice = slice(lat_idx[0], lat_idx[-1]+1)
            lon_slice = slice(lon_idx[0], lon_idx[-1]+1)

            data['u10'] = grb.values[lat_slice, lon_slice].astype(np.float32)
            data['lat'] = lats[lat_slice, lon_slice].astype(np.float32)
            data['lon'] = lons[lat_slice, lon_slice].astype(np.float32)

        elif grb.shortName == '10v':
            if 'lat' in data:
                lat_mask = (data['lat'][:, 0] >= bbox['lat_min']) & (data['lat'][:, 0] <= bbox['lat_max'])
                lat_idx = np.where(lat_mask)[0]
                lat_slice = slice(0, len(data['lat']))
                lon_slice = slice(0, data['lat'].shape[1])
                data['v10'] = grb.values[lat_idx[0]:lat_idx[0]+data['lat'].shape[0],
                                         :data['lat'].shape[1]].astype(np.float32)

        elif grb.shortName in ['sp', 'pres'] and grb.typeOfLevel == 'surface':
            if 'lat' in data:
                data['sp'] = grb.values[:data['lat'].shape[0],
                                        :data['lat'].shape[1]].astype(np.float32)

        elif grb.shortName == 'prmsl':
            if 'lat' in data and 'sp' not in data:
                data['mslp'] = grb.values[:data['lat'].shape[0],
                                          :data['lat'].shape[1]].astype(np.float32)

    grbs.close()
    return data


def process_date(date: str, temp_dir: Path) -> bool:
    """Download and process all GFS files for a single date."""
    output_dir = GFS_OUTPUT_DIR / date
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    output_file = output_dir / f"gfs_{date}_regional.npz"
    if output_file.exists():
        logger.info(f"{date}: Already processed, skipping")
        return True

    logger.info(f"{date}: Starting download...")

    # Storage for all forecast hours
    all_data = {
        'u10': [],
        'v10': [],
        'sp': [],
        'fhr': [],
    }
    lat = None
    lon = None

    success_count = 0

    for fhr in GFS_HOURS_ALL:
        grib_file = temp_dir / f"gfs.{date}.f{fhr:03d}.grib2"

        # Download
        if download_gfs_file(date, fhr, temp_dir):
            try:
                # Extract region
                data = extract_region_pygrib(grib_file, BBOX)

                if 'u10' in data and 'v10' in data:
                    all_data['u10'].append(data['u10'])
                    all_data['v10'].append(data['v10'])
                    all_data['sp'].append(data.get('sp', data.get('mslp', np.zeros_like(data['u10']))))
                    all_data['fhr'].append(fhr)

                    if lat is None:
                        lat = data['lat']
                        lon = data['lon']

                    success_count += 1

                # Delete full file to save space
                grib_file.unlink()

            except Exception as e:
                logger.warning(f"{date} f{fhr:03d}: Extract failed - {e}")
                if grib_file.exists():
                    grib_file.unlink()
        else:
            logger.warning(f"{date} f{fhr:03d}: Download failed")

    if success_count > 0:
        # Save as NPZ
        np.savez_compressed(
            output_file,
            u10=np.array(all_data['u10']),
            v10=np.array(all_data['v10']),
            sp=np.array(all_data['sp']),
            fhr=np.array(all_data['fhr']),
            lat=lat,
            lon=lon,
            date=date
        )
        logger.info(f"{date}: Saved {success_count}/{len(GFS_HOURS_ALL)} hours to {output_file.name}")
        return True
    else:
        logger.error(f"{date}: No data downloaded")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch download GFS forcing data')
    parser.add_argument('--start-date', type=str, help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYYMMDD)')
    parser.add_argument('--dates', nargs='+', help='Specific dates to download')
    parser.add_argument('--all', action='store_true', help='Download for all STOFS dates')
    parser.add_argument('--workers', type=int, default=1, help='Parallel date processing')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded')
    args = parser.parse_args()

    # Get dates to process
    if args.all:
        dates = get_stofs_dates()
    elif args.dates:
        dates = args.dates
    elif args.start_date and args.end_date:
        all_dates = get_stofs_dates()
        dates = [d for d in all_dates if args.start_date <= d <= args.end_date]
    else:
        parser.print_help()
        return

    logger.info(f"Processing {len(dates)} dates")
    logger.info(f"GFS hours per date: {len(GFS_HOURS_ALL)}")
    logger.info(f"Output directory: {GFS_OUTPUT_DIR}")

    if args.dry_run:
        logger.info("Dry run - would download:")
        for d in dates[:10]:
            logger.info(f"  {d}")
        if len(dates) > 10:
            logger.info(f"  ... and {len(dates)-10} more")
        return

    # Create output directory
    GFS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process dates
    success = 0
    failed = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, date in enumerate(dates):
            logger.info(f"[{i+1}/{len(dates)}] Processing {date}")
            try:
                if process_date(date, temp_path):
                    success += 1
                else:
                    failed.append(date)
            except Exception as e:
                logger.error(f"{date}: Failed - {e}")
                failed.append(date)

    logger.info(f"\nComplete: {success}/{len(dates)} dates processed")
    if failed:
        logger.info(f"Failed dates: {failed}")


if __name__ == '__main__':
    main()
