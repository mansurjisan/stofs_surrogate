#!/usr/bin/env python3
"""
FIXED GFS Download and Extraction Script

Fixes the surface pressure extraction issue where older dates had sp=0.

Key fixes:
1. Tries multiple variable names for surface pressure (sp, pres, prmsl, msl)
2. Logs what variables are actually found in GRIB files
3. Warns if pressure not found instead of silently using zeros
4. Has diagnostic mode to inspect GRIB file contents
5. Validates extracted data before saving

Usage:
    # Download and extract for specific dates
    python scripts/download_gfs_fixed.py 20230108 20230109

    # Download all dates with CWL data
    python scripts/download_gfs_fixed.py --all

    # Diagnostic mode - inspect what's in a GRIB file
    python scripts/download_gfs_fixed.py --diagnose 20240115

    # Re-extract only (don't download, just re-process existing GRIB files)
    python scripts/download_gfs_fixed.py --reextract 20230108

    # Force reprocessing even if NPZ exists
    python scripts/download_gfs_fixed.py --force 20230108
"""

import numpy as np
import tempfile
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - UPDATE THESE FOR YOUR SYSTEM
STOFS_DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/stofs_data')  # Optional: only needed for --all
GFS_OUTPUT_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/gfs_forcing')
GRIB_CACHE_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/gfs_grib_cache')  # Optional: keep GRIB files

# Date range for --date-range option (doesn't need STOFS data)
DEFAULT_START_DATE = '20230108'
DEFAULT_END_DATE = '20251217'

# AWS S3 bucket (public, no credentials needed)
S3_BUCKET = 'noaa-gfs-bdp-pds'
S3_KEY_TEMPLATE = 'gfs.{date}/00/atmos/gfs.t00z.pgrb2.0p25.f{fhr:03d}'

# Region (US East Coast - covers both 80k and 25k domains)
BBOX = {'lon_min': -82.0, 'lon_max': -64.0, 'lat_min': 24.0, 'lat_max': 46.0}

# Forecast hours: 3-hourly (60 files)
GFS_HOURS = list(range(0, 121, 3)) + list(range(123, 180, 3))

# Parallel downloads
MAX_WORKERS = 4

# Pressure variable detection - try these in order
PRESSURE_VARS = [
    {'shortName': 'sp', 'typeOfLevel': 'surface'},           # Surface pressure (Pa)
    {'shortName': 'pres', 'typeOfLevel': 'surface'},         # Pressure at surface
    {'shortName': 'prmsl', 'typeOfLevel': 'meanSea'},        # Mean sea level pressure
    {'shortName': 'msl', 'typeOfLevel': 'meanSea'},          # MSL pressure
    {'shortName': 'PRES', 'typeOfLevel': 'surface'},         # Uppercase variant
    {'shortName': 'SP', 'typeOfLevel': 'surface'},           # Uppercase variant
]


def get_s3_client():
    """Create S3 client for public bucket."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client('s3', config=Config(signature_version=UNSIGNED))


def generate_date_range(start_date, end_date):
    """Generate list of dates between start and end (inclusive)."""
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    return dates


def get_cwl_dates():
    """Get all dates that have CWL data."""
    dates = []
    for d in STOFS_DATA_DIR.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            dates.append(d.name)
    return sorted(dates)


def download_file(s3_client, date_str, fhr, output_path):
    """Download file from AWS S3."""
    key = S3_KEY_TEMPLATE.format(date=date_str, fhr=fhr)
    try:
        s3_client.download_file(S3_BUCKET, key, str(output_path))
        return output_path.exists() and output_path.stat().st_size > 1000
    except Exception as e:
        logger.debug(f"Download failed for {key}: {e}")
        return False


def diagnose_grib_file(grib_path):
    """
    Diagnose what variables are in a GRIB file.
    Useful for debugging pressure extraction issues.
    """
    import pygrib

    logger.info(f"Diagnosing GRIB file: {grib_path}")

    grbs = pygrib.open(str(grib_path))

    variables = {}
    for grb in grbs:
        key = (grb.shortName, grb.typeOfLevel, grb.level)
        if key not in variables:
            variables[key] = {
                'name': grb.name,
                'shortName': grb.shortName,
                'typeOfLevel': grb.typeOfLevel,
                'level': grb.level,
                'units': grb.units,
                'shape': grb.values.shape,
                'min': float(grb.values.min()),
                'max': float(grb.values.max()),
                'mean': float(grb.values.mean()),
            }

    grbs.close()

    logger.info(f"Found {len(variables)} unique variables:")
    logger.info("-" * 80)

    # Group by type
    wind_vars = []
    pressure_vars = []
    other_vars = []

    for key, info in sorted(variables.items()):
        line = f"  {info['shortName']:8s} | {info['typeOfLevel']:20s} | lev={info['level']:5} | {info['units']:10s} | range=[{info['min']:.1f}, {info['max']:.1f}]"

        if info['shortName'] in ['10u', '10v', 'u10', 'v10', 'UGRD', 'VGRD']:
            wind_vars.append(line)
        elif info['shortName'].lower() in ['sp', 'pres', 'prmsl', 'msl', 'pressure']:
            pressure_vars.append(line)
        else:
            other_vars.append(line)

    logger.info("WIND VARIABLES:")
    for v in wind_vars:
        logger.info(v)

    logger.info("\nPRESSURE VARIABLES:")
    for v in pressure_vars:
        logger.info(v)

    logger.info(f"\nOTHER VARIABLES ({len(other_vars)} total):")
    for v in other_vars[:10]:  # Show first 10
        logger.info(v)
    if len(other_vars) > 10:
        logger.info(f"  ... and {len(other_vars) - 10} more")

    return variables


def extract_region_fixed(grib_path, bbox, verbose=False):
    """
    Extract regional subset from GRIB file with FIXED pressure extraction.

    Improvements over original:
    1. Tries multiple variable names for pressure
    2. Validates pressure values (not all zeros)
    3. Logs what was found
    4. Returns metadata about extraction success
    """
    import pygrib

    grbs = pygrib.open(str(grib_path))
    data = {
        'extracted_vars': [],
        'pressure_source': None,
    }

    # First pass: find U10 to get grid info
    for grb in grbs:
        name = grb.shortName

        if name == '10u' or (name == 'UGRD' and grb.typeOfLevel == 'heightAboveGround' and grb.level == 10):
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
            data['i0'], data['i1'] = i0, i1
            data['j0'], data['j1'] = j0, j1
            data['extracted_vars'].append('u10')
            break

    if 'shape' not in data:
        grbs.close()
        return None

    # Second pass: find V10
    grbs.seek(0)
    for grb in grbs:
        name = grb.shortName
        if name == '10v' or (name == 'VGRD' and grb.typeOfLevel == 'heightAboveGround' and grb.level == 10):
            data['v10'] = grb.values[data['i0']:data['i1'], data['j0']:data['j1']].astype(np.float32)
            data['extracted_vars'].append('v10')
            break

    # Third pass: find surface pressure (try multiple variable names)
    grbs.seek(0)
    pressure_found = False

    # Collect all potential pressure variables
    pressure_candidates = []
    for grb in grbs:
        for pvar in PRESSURE_VARS:
            if grb.shortName == pvar['shortName'] and grb.typeOfLevel == pvar['typeOfLevel']:
                values = grb.values[data['i0']:data['i1'], data['j0']:data['j1']].astype(np.float32)
                pressure_candidates.append({
                    'shortName': grb.shortName,
                    'typeOfLevel': grb.typeOfLevel,
                    'level': grb.level,
                    'values': values,
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'mean': float(values.mean()),
                })

    # Select best pressure variable (prefer surface pressure in Pa range)
    for candidate in pressure_candidates:
        # Surface pressure should be around 100000 Pa (1000 hPa)
        if candidate['mean'] > 50000 and candidate['mean'] < 110000:
            data['sp'] = candidate['values']
            data['pressure_source'] = f"{candidate['shortName']}:{candidate['typeOfLevel']}"
            data['extracted_vars'].append('sp')
            pressure_found = True
            if verbose:
                logger.info(f"    Using pressure: {candidate['shortName']} ({candidate['typeOfLevel']}), "
                           f"range=[{candidate['min']:.0f}, {candidate['max']:.0f}] Pa")
            break

    # If no good surface pressure, try MSL pressure
    if not pressure_found:
        for candidate in pressure_candidates:
            if candidate['mean'] > 90000 and candidate['mean'] < 110000:
                data['sp'] = candidate['values']
                data['pressure_source'] = f"{candidate['shortName']}:{candidate['typeOfLevel']} (MSL)"
                data['extracted_vars'].append('sp (msl)')
                pressure_found = True
                if verbose:
                    logger.info(f"    Using MSL pressure: {candidate['shortName']}, "
                               f"range=[{candidate['min']:.0f}, {candidate['max']:.0f}] Pa")
                break

    # If still no pressure, check what we found
    if not pressure_found and pressure_candidates:
        logger.warning(f"    Found {len(pressure_candidates)} pressure variables but none in valid range:")
        for c in pressure_candidates:
            logger.warning(f"      {c['shortName']}:{c['typeOfLevel']} range=[{c['min']:.0f}, {c['max']:.0f}]")

    if not pressure_found:
        logger.warning(f"    NO PRESSURE FOUND - will use zeros")
        data['sp'] = np.zeros(data['shape'], dtype=np.float32)
        data['pressure_source'] = 'MISSING (zeros)'

    grbs.close()
    return data


def download_and_extract_fixed(args):
    """Download and extract one forecast hour with fixed pressure extraction."""
    date_str, fhr, tmpdir, s3_client, verbose = args
    tmp_file = tmpdir / f"gfs_f{fhr:03d}.grib2"

    # Download
    if not download_file(s3_client, date_str, fhr, tmp_file):
        return fhr, None, "download_failed"

    # Extract region
    try:
        data = extract_region_fixed(tmp_file, BBOX, verbose=verbose)
        tmp_file.unlink(missing_ok=True)

        if data is not None and 'u10' in data and 'v10' in data:
            return fhr, data, data.get('pressure_source', 'unknown')
    except Exception as e:
        logger.debug(f"Extraction failed for f{fhr:03d}: {e}")

    tmp_file.unlink(missing_ok=True)
    return fhr, None, "extraction_failed"


def process_date_fixed(date_str, force=False, verbose=False, workers=4):
    """
    Download and process all GFS files for one date with FIXED pressure extraction.
    """
    output_dir = GFS_OUTPUT_DIR / date_str
    output_file = output_dir / f"gfs_{date_str}_regional.npz"

    # Skip if already done (unless force)
    if not force and output_file.exists() and output_file.stat().st_size > 1000:
        # Check if existing file has valid pressure
        try:
            existing = np.load(output_file)
            sp = existing['sp']
            if np.all(sp == 0) or len(np.unique(sp)) == 1:
                logger.info(f"{date_str}: Existing file has invalid pressure, reprocessing...")
            else:
                logger.info(f"{date_str}: Already done with valid pressure")
                return True, "skipped"
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{date_str}: Downloading {len(GFS_HOURS)} files ({workers} parallel)...")

    s3_client = get_s3_client()

    all_u10, all_v10, all_sp, all_fhr = [], [], [], []
    lat, lon = None, None
    failed_hours = []
    pressure_sources = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Prepare download tasks
        tasks = [(date_str, fhr, tmpdir, s3_client, verbose) for fhr in GFS_HOURS]

        # Process with parallel downloads
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_and_extract_fixed, task): task[1] for task in tasks}

            for future in as_completed(futures):
                fhr = futures[future]
                completed += 1

                if completed % 10 == 0:
                    logger.info(f"  {date_str}: {completed}/{len(GFS_HOURS)}...")

                try:
                    fhr_result, data, pressure_source = future.result()
                    if data is not None:
                        all_fhr.append(fhr_result)
                        all_u10.append(data['u10'])
                        all_v10.append(data['v10'])
                        all_sp.append(data['sp'])
                        pressure_sources.add(pressure_source)
                        if lat is None:
                            lat = data['lat']
                            lon = data['lon']
                    else:
                        failed_hours.append(fhr_result)
                except Exception as e:
                    logger.debug(f"Error processing f{fhr:03d}: {e}")
                    failed_hours.append(fhr)

    # Sort by forecast hour before saving
    if len(all_u10) > 0:
        sort_idx = np.argsort(all_fhr)
        all_fhr = [all_fhr[i] for i in sort_idx]
        all_u10 = [all_u10[i] for i in sort_idx]
        all_v10 = [all_v10[i] for i in sort_idx]
        all_sp = [all_sp[i] for i in sort_idx]

        # Convert to arrays
        u10_arr = np.array(all_u10, dtype=np.float32)
        v10_arr = np.array(all_v10, dtype=np.float32)
        sp_arr = np.array(all_sp, dtype=np.float32)
        fhr_arr = np.array(all_fhr, dtype=np.int16)

        # Validate pressure
        pressure_valid = not (np.all(sp_arr == 0) or len(np.unique(sp_arr)) < 10)

        np.savez_compressed(
            output_file,
            u10=u10_arr,
            v10=v10_arr,
            sp=sp_arr,
            fhr=fhr_arr,
            lat=lat,
            lon=lon,
            date=date_str,
            pressure_sources=list(pressure_sources),
        )

        size_mb = output_file.stat().st_size / 1e6
        pressure_status = "VALID" if pressure_valid else "INVALID (zeros)"
        logger.info(f"{date_str}: Saved {len(all_fhr)}/{len(GFS_HOURS)} hours ({size_mb:.1f} MB)")
        logger.info(f"  Pressure: {pressure_status}, sources: {pressure_sources}")
        logger.info(f"  sp range: [{sp_arr.min():.0f}, {sp_arr.max():.0f}] Pa")

        if failed_hours:
            logger.warning(f"  Failed hours: {sorted(failed_hours)[:10]}{'...' if len(failed_hours) > 10 else ''}")

        return True, "success" if pressure_valid else "no_pressure"
    else:
        logger.error(f"{date_str}: FAILED - no data extracted")
        return False, "failed"


def validate_existing_files(dates):
    """Check which existing NPZ files have valid pressure data."""
    logger.info(f"Validating {len(dates)} existing GFS files...")

    valid = []
    invalid = []
    missing = []

    for date_str in dates:
        output_file = GFS_OUTPUT_DIR / date_str / f"gfs_{date_str}_regional.npz"

        if not output_file.exists():
            missing.append(date_str)
            continue

        try:
            data = np.load(output_file)
            sp = data['sp']

            # Check if pressure is valid (not all zeros, has variation)
            if np.all(sp == 0):
                invalid.append((date_str, "all_zeros"))
            elif len(np.unique(sp)) < 10:
                invalid.append((date_str, f"only_{len(np.unique(sp))}_unique"))
            elif sp.mean() < 50000 or sp.mean() > 110000:
                invalid.append((date_str, f"bad_range_{sp.mean():.0f}"))
            else:
                valid.append(date_str)
        except Exception as e:
            invalid.append((date_str, f"read_error: {e}"))

    logger.info(f"\nValidation Results:")
    logger.info(f"  Valid: {len(valid)} dates")
    logger.info(f"  Invalid: {len(invalid)} dates")
    logger.info(f"  Missing: {len(missing)} dates")

    if invalid:
        logger.info(f"\nInvalid files (need reprocessing):")
        for date_str, reason in invalid[:20]:
            logger.info(f"  {date_str}: {reason}")
        if len(invalid) > 20:
            logger.info(f"  ... and {len(invalid) - 20} more")

    return valid, [d[0] for d in invalid], missing


def main():
    parser = argparse.ArgumentParser(
        description='FIXED GFS Download - properly extracts surface pressure',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('dates', nargs='*', help='Dates to process (YYYYMMDD)')
    parser.add_argument('--all', action='store_true', help='Process all CWL dates (requires STOFS data)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='Process date range (YYYYMMDD YYYYMMDD) - no STOFS data needed')
    parser.add_argument('--date-file', type=str, help='File with dates (one per line)')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--workers', type=int, default=4, help='Parallel downloads')
    parser.add_argument('--force', action='store_true', help='Reprocess even if file exists')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--diagnose', type=str, help='Diagnose GRIB file for a date')
    parser.add_argument('--validate', action='store_true', help='Validate existing files')
    parser.add_argument('--fix-invalid', action='store_true', help='Reprocess only invalid files')

    args = parser.parse_args()

    # Override output directory if specified
    global GFS_OUTPUT_DIR
    if args.output_dir:
        GFS_OUTPUT_DIR = Path(args.output_dir)
        logger.info(f"Output directory: {GFS_OUTPUT_DIR}")

    # Diagnostic mode
    if args.diagnose:
        date_str = args.diagnose
        grib_dir = GFS_OUTPUT_DIR / date_str

        # Try to find a GRIB file or download one
        grib_files = list(grib_dir.glob('*.grib2'))
        if grib_files:
            diagnose_grib_file(grib_files[0])
        else:
            # Download one file for diagnosis
            logger.info(f"Downloading sample GRIB file for {date_str}...")
            s3_client = get_s3_client()
            grib_dir.mkdir(parents=True, exist_ok=True)
            tmp_file = grib_dir / f"gfs.{date_str}.f000.grib2"

            if download_file(s3_client, date_str, 0, tmp_file):
                diagnose_grib_file(tmp_file)
            else:
                logger.error(f"Could not download GRIB file for {date_str}")
        return

    # Get dates
    if args.date_range:
        # Generate dates from range (no STOFS data needed)
        dates = generate_date_range(args.date_range[0], args.date_range[1])
        logger.info(f"Generated {len(dates)} dates from range {args.date_range[0]} to {args.date_range[1]}")
    elif args.date_file:
        # Read dates from file
        with open(args.date_file, 'r') as f:
            dates = [line.strip() for line in f if line.strip().isdigit() and len(line.strip()) == 8]
        logger.info(f"Read {len(dates)} dates from {args.date_file}")
    elif args.all:
        # Get dates from STOFS data directory
        if not STOFS_DATA_DIR.exists():
            logger.error(f"STOFS data directory not found: {STOFS_DATA_DIR}")
            logger.error("Use --date-range or --date-file instead, or specify dates directly")
            return
        dates = get_cwl_dates()
    elif args.dates:
        # Use provided dates directly (no validation against CWL)
        dates = [d for d in args.dates if d.isdigit() and len(d) == 8]
    else:
        # Default: show help
        parser.print_help()
        print("\nExamples:")
        print("  # Download specific dates:")
        print("  python download_gfs_fixed.py 20230108 20230109 20230110")
        print("")
        print("  # Download date range (no STOFS data needed):")
        print("  python download_gfs_fixed.py --date-range 20230108 20251217 --workers 8")
        print("")
        print("  # Download from date file:")
        print("  python download_gfs_fixed.py --date-file dates.txt --workers 8")
        print("")
        print("  # Validate and fix existing files:")
        print("  python download_gfs_fixed.py --date-range 20230108 20251217 --fix-invalid")
        return

    # Validation mode
    if args.validate:
        validate_existing_files(dates)
        return

    # Fix invalid files mode
    if args.fix_invalid:
        valid, invalid, missing = validate_existing_files(dates)
        dates = invalid + missing
        if not dates:
            logger.info("No files need fixing!")
            return
        logger.info(f"\nWill reprocess {len(dates)} dates...")

    logger.info("=" * 70)
    logger.info("FIXED GFS DOWNLOAD - Proper Surface Pressure Extraction")
    logger.info("=" * 70)
    logger.info(f"Dates to process: {len(dates)}")
    logger.info(f"Output: {GFS_OUTPUT_DIR}")
    logger.info(f"Forecast hours: {len(GFS_HOURS)} (3-hourly)")
    logger.info(f"Parallel workers: {args.workers}")
    logger.info(f"Force reprocess: {args.force}")
    logger.info("=" * 70)

    results = {'success': 0, 'no_pressure': 0, 'failed': 0, 'skipped': 0}
    start_time = time.time()

    for i, date_str in enumerate(dates):
        logger.info(f"\n[{i+1}/{len(dates)}] Processing {date_str}")

        success, status = process_date_fixed(
            date_str,
            force=args.force,
            verbose=args.verbose,
            workers=args.workers
        )

        results[status] = results.get(status, 0) + 1

    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Results:")
    for status, count in results.items():
        logger.info(f"  {status}: {count}")

    # Final validation
    logger.info("\nValidating output files...")
    validate_existing_files(dates)


if __name__ == '__main__':
    main()
