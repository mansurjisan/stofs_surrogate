#!/bin/bash
#SBATCH --job-name=preprocess_100k
#SBATCH --account=gpu-nos-surge
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/preprocess_100k_%j.log
#SBATCH --error=outputs/preprocess_100k_%j.err

# ============================================================
# STOFS 100k Mesh Preprocessing Pipeline
# 1. Create 100k mesh (if not exists)
# 2. Preprocess all dates
# ============================================================

set -e

echo "============================================================"
echo "STOFS 100k Mesh Preprocessing"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

cd /scratch5/purged/Mansur.Jisan/stofs_surrogate
source ~/venv_stofs/bin/activate

# Directories
STOFS_RAW="/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/stofs_raw"
GFS_DIR="/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/gfs_forcing_v2"
MESH_DIR="/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_100k"
OUTPUT_DIR="/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_100k"

mkdir -p "$MESH_DIR"
mkdir -p outputs

echo "Directories:"
echo "  STOFS raw: $STOFS_RAW"
echo "  GFS: $GFS_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Create mesh if not exists
MESH_FILE="$MESH_DIR/mesh.npz"

if [ ! -f "$MESH_FILE" ]; then
    echo "============================================================"
    echo "Step 1: Creating 100k mesh"
    echo "============================================================"

    # Find a STOFS file to use for coordinates
    STOFS_SAMPLE=$(find "$STOFS_RAW" -name "stofs_2d_glo.t00z.fields.cwl.nc" | head -1)

    if [ -z "$STOFS_SAMPLE" ]; then
        echo "ERROR: No STOFS files found in $STOFS_RAW"
        echo "Please run download_stofs_ursa.sh first"
        exit 1
    fi

    echo "Using STOFS file: $STOFS_SAMPLE"

    python3 scripts/create_100k_mesh.py \
        --stofs-file "$STOFS_SAMPLE" \
        --output-dir "$MESH_DIR" \
        --target-nodes 100000 \
        --max-edge-km 50

    echo ""
else
    echo "Mesh already exists: $MESH_FILE"
fi

# Step 2: Preprocess data
echo "============================================================"
echo "Step 2: Preprocessing STOFS data"
echo "============================================================"

python3 scripts/preprocess_100k_ursa.py \
    --mesh-file "$MESH_FILE" \
    --stofs-dir "$STOFS_RAW" \
    --gfs-dir "$GFS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --start-date 20230108 \
    --end-date 20260124 \
    --workers 32

echo ""
echo "============================================================"
echo "Preprocessing Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.npz 2>/dev/null | head -10
echo "..."
echo "Total files: $(ls $OUTPUT_DIR/processed_*.npz 2>/dev/null | wc -l)"
echo "Total size: $(du -sh $OUTPUT_DIR)"
