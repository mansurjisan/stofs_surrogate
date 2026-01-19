#!/usr/bin/env python3
"""
FAST GFS Download - Uses NOMADS filtering to download only needed variables.

Instead of downloading full 300MB GRIB files, this downloads only:
- 10m U wind
- 10m V wind
- Surface pressure (or MSLP)

This reduces download from ~300MB to ~3MB per file (100x faster!)

Usage:
    python download_gfs_fast.py --date-range 20230108 20251217 --workers 16
"""

import numpy as np
import requests
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
import logging
import tempfile
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress connection pool warnings
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Configuration
GFS_OUTPUT_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/gfs_forcing')

# NOMADS filter URL - downloads only specific variables
# This is MUCH faster than downloading full files from S3
NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Region (US East Coast)
BBOX = {'lon_min': -82.0, 'lon_max': -64.0, 'lat_min': 24.0, 'lat_max': 46.0}

# Convert to NOMADS format (0-360 for longitude)
NOMADS_BBOX = {
    'leftlon': 360 + BBOX['lon_min'],  # -82 -> 278
    'rightlon': 360 + BBOX['lon_max'],  # -64 -> 296
    'bottomlat': BBOX['lat_min'],
    'toplat': BBOX['lat_max'],
}

# Forecast hours
GFS_HOURS = list(range(0, 121, 3)) + list(range(123, 180, 3))

# Variables to download
VARIABLES = [
    'UGRD:10 m above ground',   # U wind at 10m
    'VGRD:10 m above ground',   # V wind at 10m
    'PRES:surface',             # Surface pressure
    'PRMSL:mean sea level',     # Mean sea level pressure (backup)
]


def generate_date_range(start_date, end_date):
    """Generate list of dates between start and end (inclusive)."""
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    return dates


def download_filtered_gfs(date_str, fhr, output_dir, timeout=60):
    """
    Download filtered GFS data from NOMADS.
    Only downloads the variables and region we need (~3MB vs ~300MB).
    """
    # Build filter URL
    file_name = f"gfs.t00z.pgrb2.0p25.f{fhr:03d}"

    params = {
        'file': file_name,
        'dir': f'/gfs.{date_str}/00/atmos',
        'subregion': '',
        'leftlon': NOMADS_BBOX['leftlon'],
        'rightlon': NOMADS_BBOX['rightlon'],
        'toplat': NOMADS_BBOX['toplat'],
        'bottomlat': NOMADS_BBOX['bottomlat'],
    }

    # Add variable filters
    for i, var in enumerate(VARIABLES):
        # NOMADS uses 'var_X=on' format
        var_key = f'var_{var.split(":")[0]}'
        params[var_key] = 'on'

    # Add level filters
    params['lev_10_m_above_ground'] = 'on'
    params['lev_surface'] = 'on'
    params['lev_mean_sea_level'] = 'on'

    try:
        response = requests.get(NOMADS_FILTER_URL, params=params, timeout=timeout)

        if response.status_code == 200 and len(response.content) > 1000:
            # Save to temp file
            tmp_file = output_dir / f"gfs_f{fhr:03d}.grib2"
            with open(tmp_file, 'wb') as f:
                f.write(response.content)
            return fhr, tmp_file, len(response.content)
        else:
            return fhr, None, f"HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        return fhr, None, "timeout"
    except Exception as e:
        return fhr, None, str(e)


def extract_from_grib(grib_file, bbox):
    """Extract variables from filtered GRIB file."""
    import pygrib

    grbs = pygrib.open(str(grib_file))

    data = {}

    for grb in grbs:
        name = grb.shortName
        level_type = grb.typeOfLevel

        if data.get('lat') is None:
            lats, lons = grb.latlons()
            # Convert longitudes to -180 to 180
            lons = np.where(lons > 180, lons - 360, lons)
            data['lat'] = lats.astype(np.float32)
            data['lon'] = lons.astype(np.float32)

        if name == '10u' or (name == 'UGRD' and level_type == 'heightAboveGround'):
            data['u10'] = grb.values.astype(np.float32)
        elif name == '10v' or (name == 'VGRD' and level_type == 'heightAboveGround'):
            data['v10'] = grb.values.astype(np.float32)
        elif name in ['sp', 'pres', 'PRES'] and level_type == 'surface':
            if 'sp' not in data:  # Don't overwrite if already found
                data['sp'] = grb.values.astype(np.float32)
        elif name in ['prmsl', 'PRMSL', 'msl'] and level_type == 'meanSea':
            if 'sp' not in data:  # Use MSL as fallback
                data['sp'] = grb.values.astype(np.float32)

    grbs.close()

    return data


def process_date(date_str, output_dir, workers=8, force=False):
    """Download and process all forecast hours for one date."""
    date_output_dir = output_dir / date_str
    output_file = date_output_dir / f"gfs_{date_str}_regional.npz"

    # Skip if already done
    if not force and output_file.exists():
        try:
            existing = np.load(output_file)
            sp = existing['sp']
            if not (np.all(sp == 0) or len(np.unique(sp)) < 10):
                logger.info(f"{date_str}: Already done with valid pressure")
                return True, "skipped"
        except:
            pass

    date_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{date_str}: Downloading {len(GFS_HOURS)} filtered files ({workers} workers)...")

    all_data = {}
    failed_hours = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_filtered_gfs, date_str, fhr, tmpdir): fhr
                for fhr in GFS_HOURS
            }

            completed = 0
            for future in as_completed(futures):
                fhr = futures[future]
                completed += 1

                if completed % 20 == 0:
                    logger.info(f"  {date_str}: {completed}/{len(GFS_HOURS)} downloaded...")

                try:
                    fhr_result, tmp_file, info = future.result()

                    if tmp_file is not None and tmp_file.exists():
                        try:
                            data = extract_from_grib(tmp_file, BBOX)
                            if 'u10' in data and 'v10' in data:
                                all_data[fhr_result] = data
                            tmp_file.unlink()
                        except Exception as e:
                            logger.debug(f"Extract error f{fhr:03d}: {e}")
                            failed_hours.append(fhr_result)
                    else:
                        failed_hours.append(fhr_result)

                except Exception as e:
                    logger.debug(f"Error f{fhr:03d}: {e}")
                    failed_hours.append(fhr)

    if len(all_data) > 0:
        # Sort by forecast hour
        sorted_fhrs = sorted(all_data.keys())

        # Stack arrays
        lat = all_data[sorted_fhrs[0]]['lat']
        lon = all_data[sorted_fhrs[0]]['lon']

        u10 = np.stack([all_data[f]['u10'] for f in sorted_fhrs])
        v10 = np.stack([all_data[f]['v10'] for f in sorted_fhrs])

        # Handle pressure (may be missing for some hours)
        sp_list = []
        for f in sorted_fhrs:
            if 'sp' in all_data[f]:
                sp_list.append(all_data[f]['sp'])
            else:
                # Use mean sea level pressure as fallback, or zeros
                sp_list.append(np.full_like(u10[0], 101325.0))
        sp = np.stack(sp_list)

        fhr_arr = np.array(sorted_fhrs, dtype=np.int16)

        # Validate pressure
        pressure_valid = sp.mean() > 50000 and sp.mean() < 110000

        np.savez_compressed(
            output_file,
            u10=u10,
            v10=v10,
            sp=sp,
            fhr=fhr_arr,
            lat=lat,
            lon=lon,
            date=date_str,
        )

        size_mb = output_file.stat().st_size / 1e6
        status = "VALID" if pressure_valid else "NO_PRESSURE"
        logger.info(f"{date_str}: Saved {len(sorted_fhrs)}/{len(GFS_HOURS)} hours ({size_mb:.1f} MB) - Pressure: {status}")

        if failed_hours:
            logger.warning(f"  Failed hours: {sorted(failed_hours)[:10]}...")

        return True, "success" if pressure_valid else "no_pressure"
    else:
        logger.error(f"{date_str}: FAILED - no data")
        return False, "failed"


def main():
    parser = argparse.ArgumentParser(
        description='FAST GFS Download using NOMADS filtering (100x faster)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('dates', nargs='*', help='Dates to process (YYYYMMDD)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='Process date range (YYYYMMDD YYYYMMDD)')
    parser.add_argument('--date-file', type=str, help='File with dates (one per line)')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--workers', type=int, default=8, help='Parallel downloads (default: 8)')
    parser.add_argument('--force', action='store_true', help='Reprocess existing files')

    args = parser.parse_args()

    # Set output directory
    global GFS_OUTPUT_DIR
    if args.output_dir:
        GFS_OUTPUT_DIR = Path(args.output_dir)

    # Get dates
    if args.date_range:
        dates = generate_date_range(args.date_range[0], args.date_range[1])
    elif args.date_file:
        with open(args.date_file, 'r') as f:
            dates = [line.strip() for line in f if line.strip().isdigit() and len(line.strip()) == 8]
    elif args.dates:
        dates = args.dates
    else:
        parser.print_help()
        print("\nExample:")
        print("  python download_gfs_fast.py --date-range 20230108 20251217 --workers 16")
        return

    logger.info("=" * 70)
    logger.info("FAST GFS DOWNLOAD - NOMADS Filtered (100x faster)")
    logger.info("=" * 70)
    logger.info(f"Dates: {len(dates)}")
    logger.info(f"Output: {GFS_OUTPUT_DIR}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Downloads ~3MB per file instead of ~300MB")
    logger.info("=" * 70)

    GFS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {'success': 0, 'no_pressure': 0, 'failed': 0, 'skipped': 0}
    start_time = time.time()

    for i, date_str in enumerate(dates):
        logger.info(f"\n[{i+1}/{len(dates)}] {date_str}")
        success, status = process_date(date_str, GFS_OUTPUT_DIR, workers=args.workers, force=args.force)
        results[status] = results.get(status, 0) + 1

    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Time: {elapsed/60:.1f} minutes ({elapsed/len(dates):.1f}s per date)")
    logger.info(f"Results: {results}")


if __name__ == '__main__':
    main()
