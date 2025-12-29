#!/usr/bin/env python3
"""
Download GFS forcing data for STOFS training dates (AWS S3 source).

Only downloads for dates that have CWL data in stofs_data folder.
Uses parallel downloads for speed.

Usage:
    python scripts/download_gfs_simple.py 20251208
    python scripts/download_gfs_simple.py --all
    python scripts/download_gfs_simple.py --all --workers 8
"""

import numpy as np
import tempfile
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Configuration
STOFS_DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/stofs_data')
GFS_OUTPUT_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/gfs_forcing')

# AWS S3 bucket (public, no credentials needed)
S3_BUCKET = 'noaa-gfs-bdp-pds'
S3_KEY_TEMPLATE = 'gfs.{date}/00/atmos/gfs.t00z.pgrb2.0p25.f{fhr:03d}'

# Create S3 client for public bucket
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Region (US East Coast)
BBOX = {'lon_min': -82.0, 'lon_max': -64.0, 'lat_min': 24.0, 'lat_max': 46.0}

# Forecast hours: 3-hourly throughout (60 files instead of 140)
# Can interpolate forcing between timesteps for ML if needed
# STOFS Hr7 = GFS f000 (verified alignment)
GFS_HOURS = list(range(0, 121, 3)) + list(range(123, 180, 3))  # 60 files

# Parallel downloads
MAX_WORKERS = 4


def get_cwl_dates():
    """Get all dates that have CWL data."""
    dates = []
    for d in STOFS_DATA_DIR.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            dates.append(d.name)
    return sorted(dates)


def download_file(date_str, fhr, output_path):
    """Download file from AWS S3 using boto3 (~500MB, ~20s at 25MB/s)."""
    key = S3_KEY_TEMPLATE.format(date=date_str, fhr=fhr)
    try:
        s3_client.download_file(S3_BUCKET, key, str(output_path))
        return output_path.exists() and output_path.stat().st_size > 1000
    except Exception:
        return False


def extract_region(grib_path, bbox):
    """Extract regional subset using pygrib."""
    import pygrib

    grbs = pygrib.open(str(grib_path))
    data = {}

    for grb in grbs:
        name = grb.shortName

        if name == '10u':
            lats, lons = grb.latlons()
            lons = np.where(lons > 180, lons - 360, lons)

            # Find region indices
            lat_mask = (lats[:, 0] >= bbox['lat_min']) & (lats[:, 0] <= bbox['lat_max'])
            lon_mask = (lons[0, :] >= bbox['lon_min']) & (lons[0, :] <= bbox['lon_max'])

            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                continue

            i0, i1 = lat_idx[0], lat_idx[-1] + 1
            j0, j1 = lon_idx[0], lon_idx[-1] + 1

            data['u10'] = grb.values[i0:i1, j0:j1].astype(np.float32)
            data['lat'] = lats[i0:i1, j0:j1].astype(np.float32)
            data['lon'] = lons[i0:i1, j0:j1].astype(np.float32)
            data['shape'] = (i1-i0, j1-j0)
            data['i0'] = i0
            data['j0'] = j0

        elif name == '10v' and 'shape' in data:
            data['v10'] = grb.values[data['i0']:data['i0']+data['shape'][0],
                                     data['j0']:data['j0']+data['shape'][1]].astype(np.float32)

        elif name in ['sp', 'pres'] and grb.typeOfLevel == 'surface' and 'shape' in data:
            data['sp'] = grb.values[data['i0']:data['i0']+data['shape'][0],
                                    data['j0']:data['j0']+data['shape'][1]].astype(np.float32)

    grbs.close()
    return data


def download_and_extract(args):
    """Download and extract one forecast hour (for parallel execution)."""
    date_str, fhr, tmpdir = args
    tmp_file = tmpdir / f"gfs_f{fhr:03d}.grib2"

    # Download
    if not download_file(date_str, fhr, tmp_file):
        return fhr, None

    # Extract region
    try:
        data = extract_region(tmp_file, BBOX)
        tmp_file.unlink(missing_ok=True)

        if 'u10' in data and 'v10' in data:
            return fhr, data
    except Exception:
        pass

    tmp_file.unlink(missing_ok=True)
    return fhr, None


def process_date(date_str):
    """Download and process all GFS files for one date using parallel downloads."""
    output_dir = GFS_OUTPUT_DIR / date_str
    output_file = output_dir / f"gfs_{date_str}_regional.npz"

    # Skip if already done
    if output_file.exists() and output_file.stat().st_size > 1000:
        print(f"{date_str}: Already done", flush=True)
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{date_str}: Downloading {len(GFS_HOURS)} files ({MAX_WORKERS} parallel)...", flush=True)

    all_u10, all_v10, all_sp, all_fhr = [], [], [], []
    lat, lon = None, None
    failed_hours = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Prepare download tasks
        tasks = [(date_str, fhr, tmpdir) for fhr in GFS_HOURS]

        # Process with parallel downloads
        completed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_and_extract, task): task[1] for task in tasks}

            for future in as_completed(futures):
                fhr = futures[future]
                completed += 1

                # Progress every 10 files
                if completed % 10 == 0:
                    print(f"  {date_str}: {completed}/{len(GFS_HOURS)}...", flush=True)

                try:
                    fhr_result, data = future.result()
                    if data is not None:
                        all_fhr.append(fhr_result)
                        all_u10.append(data['u10'])
                        all_v10.append(data['v10'])
                        all_sp.append(data.get('sp', np.zeros_like(data['u10'])))
                        if lat is None:
                            lat = data['lat']
                            lon = data['lon']
                    else:
                        failed_hours.append(fhr_result)
                except Exception:
                    failed_hours.append(fhr)

    # Sort by forecast hour before saving
    if len(all_u10) > 0:
        sort_idx = np.argsort(all_fhr)
        all_fhr = [all_fhr[i] for i in sort_idx]
        all_u10 = [all_u10[i] for i in sort_idx]
        all_v10 = [all_v10[i] for i in sort_idx]
        all_sp = [all_sp[i] for i in sort_idx]

        np.savez_compressed(
            output_file,
            u10=np.array(all_u10, dtype=np.float32),
            v10=np.array(all_v10, dtype=np.float32),
            sp=np.array(all_sp, dtype=np.float32),
            fhr=np.array(all_fhr, dtype=np.int16),
            lat=lat, lon=lon, date=date_str
        )
        size_mb = output_file.stat().st_size / 1e6
        print(f"{date_str}: Saved {len(all_fhr)}/{len(GFS_HOURS)} hours ({size_mb:.1f} MB)", flush=True)
        if failed_hours:
            print(f"  Failed: {sorted(failed_hours)[:10]}{'...' if len(failed_hours) > 10 else ''}", flush=True)
        return True
    else:
        print(f"{date_str}: FAILED - no data", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dates', nargs='*', help='Dates (YYYYMMDD)')
    parser.add_argument('--all', action='store_true', help='All CWL dates')
    parser.add_argument('--workers', type=int, default=4, help='Parallel downloads (default: 4)')
    args = parser.parse_args()

    global MAX_WORKERS
    MAX_WORKERS = args.workers

    if args.all:
        dates = get_cwl_dates()
    elif args.dates:
        # Filter to only dates that have CWL data
        cwl_dates = set(get_cwl_dates())
        dates = [d for d in args.dates if d in cwl_dates]
        if len(dates) < len(args.dates):
            print(f"Note: {len(args.dates) - len(dates)} dates skipped (no CWL data)")
    else:
        parser.print_help()
        return

    print(f"Processing {len(dates)} dates (only dates with CWL data)")
    print(f"Output: {GFS_OUTPUT_DIR}")
    print(f"Forecast hours: {len(GFS_HOURS)} (3-hourly)")
    print(f"Parallel workers: {MAX_WORKERS}")

    success, failed = 0, []
    start_time = time.time()

    for i, date in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] ", end="", flush=True)
        if process_date(date):
            success += 1
        else:
            failed.append(date)

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Complete: {success}/{len(dates)} in {elapsed/60:.1f} minutes")
    if failed:
        print(f"Failed: {failed[:20]}{'...' if len(failed) > 20 else ''}")


if __name__ == '__main__':
    main()
