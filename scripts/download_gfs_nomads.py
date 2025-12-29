#!/usr/bin/env python3
"""
Download GFS forcing data using NOMADS filter service (fast, server-side subsetting).

Downloads only u10, v10, sp for US East Coast region (~50KB vs 500MB per file).

Usage:
    python scripts/download_gfs_nomads.py 20251208
    python scripts/download_gfs_nomads.py --all
"""

import numpy as np
import subprocess
import tempfile
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
STOFS_DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/stofs_data')
GFS_OUTPUT_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/gfs_forcing')

# NOMADS filter URL (server-side subsetting)
NOMADS_FILTER_URL = (
    "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
    "file=gfs.t00z.pgrb2.0p25.f{fhr:03d}&"
    "lev_10_m_above_ground=on&lev_surface=on&"
    "var_UGRD=on&var_VGRD=on&var_PRES=on&"
    "subregion=&leftlon=-82&rightlon=-64&toplat=46&bottomlat=24&"
    "dir=%2Fgfs.{date}%2F00%2Fatmos"
)

# Region (must match NOMADS filter params)
BBOX = {'lon_min': -82.0, 'lon_max': -64.0, 'lat_min': 24.0, 'lat_max': 46.0}

# Forecast hours: f000-f120 hourly, f123-f177 3-hourly
GFS_HOURS = list(range(0, 121)) + list(range(123, 180, 3))

# Parallel downloads
MAX_WORKERS = 4


def get_cwl_dates():
    """Get all dates that have CWL data."""
    dates = []
    for d in STOFS_DATA_DIR.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            dates.append(d.name)
    return sorted(dates)


def download_file(date_str, fhr, output_path, timeout=60):
    """Download file from NOMADS filter (~50KB per file)."""
    url = NOMADS_FILTER_URL.format(date=date_str, fhr=fhr)
    cmd = ['curl', '-s', '-f', '-o', str(output_path),
           '--connect-timeout', '30', '--max-time', str(timeout), url]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 500


def extract_region(grib_path):
    """Extract data from regional GRIB file using pygrib."""
    import pygrib

    grbs = pygrib.open(str(grib_path))
    data = {}

    for grb in grbs:
        name = grb.shortName

        if name == '10u':
            lats, lons = grb.latlons()
            lons = np.where(lons > 180, lons - 360, lons)
            data['u10'] = grb.values.astype(np.float32)
            data['lat'] = lats.astype(np.float32)
            data['lon'] = lons.astype(np.float32)
        elif name == '10v':
            data['v10'] = grb.values.astype(np.float32)
        elif name in ['sp', 'pres']:
            data['sp'] = grb.values.astype(np.float32)

    grbs.close()
    return data


def download_and_extract(args):
    """Download and extract one forecast hour."""
    date_str, fhr, tmpdir = args
    tmp_file = tmpdir / f"gfs_f{fhr:03d}.grib2"

    # Download
    if not download_file(date_str, fhr, tmp_file):
        return fhr, None

    # Extract
    try:
        data = extract_region(tmp_file)
        tmp_file.unlink(missing_ok=True)

        if 'u10' in data and 'v10' in data:
            return fhr, data
    except Exception as e:
        pass

    tmp_file.unlink(missing_ok=True)
    return fhr, None


def process_date(date_str):
    """Download and process all GFS files for one date."""
    output_dir = GFS_OUTPUT_DIR / date_str
    output_file = output_dir / f"gfs_{date_str}_regional.npz"

    # Skip if already done
    if output_file.exists() and output_file.stat().st_size > 1000:
        print(f"{date_str}: Already done", flush=True)
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{date_str}: Downloading {len(GFS_HOURS)} forecast hours ({MAX_WORKERS} parallel)...", flush=True)

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

                # Progress every 20 files
                if completed % 20 == 0:
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
                except Exception as e:
                    failed_hours.append(fhr)

    # Sort by forecast hour
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
            print(f"  Failed hours: {sorted(failed_hours)[:10]}{'...' if len(failed_hours) > 10 else ''}", flush=True)
        return True
    else:
        print(f"{date_str}: FAILED - no data downloaded", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dates', nargs='*', help='Dates (YYYYMMDD)')
    parser.add_argument('--all', action='store_true', help='All CWL dates')
    parser.add_argument('--workers', type=int, default=4, help='Parallel downloads')
    args = parser.parse_args()

    global MAX_WORKERS
    MAX_WORKERS = args.workers

    if args.all:
        dates = get_cwl_dates()
    elif args.dates:
        cwl_dates = set(get_cwl_dates())
        dates = [d for d in args.dates if d in cwl_dates]
        if len(dates) < len(args.dates):
            print(f"Note: {len(args.dates) - len(dates)} dates skipped (no CWL data)")
    else:
        parser.print_help()
        return

    print(f"Processing {len(dates)} dates (only dates with CWL data)")
    print(f"Output: {GFS_OUTPUT_DIR}")
    print(f"Using NOMADS filter (server-side subsetting, ~50KB/file)")

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
