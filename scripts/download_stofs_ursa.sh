#!/bin/bash
#SBATCH --job-name=stofs_download
#SBATCH --account=gpu-nos-surge
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/download_stofs_%j.log
#SBATCH --error=outputs/download_stofs_%j.err

# ============================================================
# STOFS-2D Global Data Download Script for URSA
# Downloads raw NetCDF files from NOAA AWS S3 bucket
# Same approach as laptop download script
# ============================================================

set -e

echo "============================================================"
echo "STOFS-2D Global Data Download (AWS S3)"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

# Directories
PROJECT_DIR="${STOFS_PROJECT_DIR:-$(dirname $(dirname $(readlink -f $0)))}"
RAW_DIR="$PROJECT_DIR/data/stofs_raw"
mkdir -p "$RAW_DIR"

cd "$PROJECT_DIR"

# Activate environment
source ~/venv_stofs/bin/activate

# AWS S3 bucket base URL (same as your laptop script)
S3_BASE="https://noaa-gestofs-pds.s3.amazonaws.com"

# Date range - matching your 25k processed dates
START_DATE="20230108"
END_DATE="20260124"

echo "Download directory: $RAW_DIR"
echo "S3 bucket: $S3_BASE"
echo "Date range: $START_DATE to $END_DATE"
echo ""

# Function to download a single date
download_date() {
    local DATE=$1
    local OUTDIR="$RAW_DIR/$DATE"

    # We need the fields.cwl.nc file which contains water elevation timeseries
    # File pattern in S3: stofs_2d_glo.YYYYMMDD/00/rerun/stofs_2d_glo.t00z.fields.cwl.nc
    local OUTFILE="$OUTDIR/stofs_2d_glo.t00z.fields.cwl.nc"

    # Skip if already downloaded and file is not empty
    if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
        echo "  $DATE: Already exists, skipping"
        return 0
    fi

    mkdir -p "$OUTDIR"

    # S3 URL pattern (from your original download_stofs.py)
    # Try both _para and prod paths
    local S3_URL_PARA="${S3_BASE}/_para/stofs_2d_glo.${DATE}/00/rerun/stofs_2d_glo.t00z.fields.cwl.nc"
    local S3_URL_PROD="${S3_BASE}/stofs.${DATE}/stofs_2d_glo.t00z.fields.cwl.nc"

    # Also try the surf.63.nc file which contains elevation timeseries
    local S3_URL_SURF="${S3_BASE}/_para/stofs_2d_glo.${DATE}/00/rerun/stofs_2d_glo_surf.63.nc"

    # Try _para path first (what your laptop script used)
    if curl -s --head --connect-timeout 10 "$S3_URL_PARA" 2>/dev/null | head -n 1 | grep -q "200"; then
        echo "  $DATE: Downloading from S3 (_para)..."
        curl -s --connect-timeout 30 --max-time 3600 -o "$OUTFILE" "$S3_URL_PARA"
        if [ -s "$OUTFILE" ]; then
            echo "  $DATE: Success (fields.cwl.nc)"
            return 0
        fi
    fi

    # Try surf.63.nc (contains same elevation data)
    if curl -s --head --connect-timeout 10 "$S3_URL_SURF" 2>/dev/null | head -n 1 | grep -q "200"; then
        echo "  $DATE: Downloading surf.63.nc from S3..."
        curl -s --connect-timeout 30 --max-time 3600 -o "$OUTDIR/stofs_2d_glo_surf.63.nc" "$S3_URL_SURF"
        if [ -s "$OUTDIR/stofs_2d_glo_surf.63.nc" ]; then
            echo "  $DATE: Success (surf.63.nc)"
            return 0
        fi
    fi

    echo "  $DATE: Not found on S3"
    return 1
}

export -f download_date
export RAW_DIR
export S3_BASE

# Generate date list
python3 << 'EOF'
from datetime import datetime, timedelta

start = datetime.strptime("20230108", "%Y%m%d")
end = datetime.strptime("20260124", "%Y%m%d")

dates = []
current = start
while current <= end:
    dates.append(current.strftime("%Y%m%d"))
    current += timedelta(days=1)

with open("/tmp/dates_to_download.txt", "w") as f:
    for d in dates:
        f.write(d + "\n")

print(f"Generated {len(dates)} dates to download")
EOF

# Download in parallel (8 concurrent - don't overload S3)
echo ""
echo "Starting parallel download (8 concurrent)..."
echo "This may take several hours..."
echo ""

cat /tmp/dates_to_download.txt | xargs -P 8 -I {} bash -c 'download_date "$@"' _ {}

echo ""
echo "============================================================"
echo "Download complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Files downloaded:"
find "$RAW_DIR" -name "*.nc" | wc -l
echo ""
echo "Sample files:"
find "$RAW_DIR" -name "*.nc" | head -5
echo "..."

# Check total size
echo ""
echo "Total size:"
du -sh "$RAW_DIR"
