#!/bin/bash
#SBATCH --job-name=stofs_25k_v2
#SBATCH --account=gpu-nos-surge
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output=outputs/training_25k_v2_%j.log
#SBATCH --error=outputs/training_25k_v2_%j.err

# Enable pipefail to capture Python exit code
set -o pipefail

# ============================================================
# STOFS-GNN 25K V2 - URSA H100 TRAINING
# Enhanced Physics Features + Extended Rollout
#
# V2 Features:
#   - 8 forcing features (u10, v10, wind_speed, wind_speed_sq, wind_dir, pressure, dP_dx, dP_dy)
#   - 6 tidal constituents (M2, S2, N2, K1, O1, M4)
#   - Extended rollout (1->2->3->6->12 steps)
#   - Dynamic batch sizing (16->16->8->4->2)
#
# Expected: ~3-5 hours for 100 epochs
# ============================================================

echo "============================================================"
echo "STOFS-GNN 25K V2 - ENHANCED PHYSICS FEATURES"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

# Change to project directory
PROJECT_DIR="${STOFS_PROJECT_DIR:-$(dirname $(dirname $(dirname $(readlink -f $0))))}"
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd to project dir: $PROJECT_DIR"; exit 1; }

# Create output directories
mkdir -p outputs/checkpoints_25k_v2 outputs/figures_25k_v2

# Setup environment - use same venv as previous training
source ~/venv_stofs/bin/activate

echo "Environment check:"
echo "  Python: $(which python3)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU: None')"
echo ""

# Check data - V2 preprocessed data with 8 forcing features
DATA_DIR="$PROJECT_DIR/data/processed_25k_v2"
NUM_FILES=$(ls -1 $DATA_DIR/processed_*.npz 2>/dev/null | wc -l)
echo "Data files: $NUM_FILES in $DATA_DIR"

if [ "$NUM_FILES" -lt 10 ]; then
    echo "ERROR: Not enough data files. Make sure to upload processed_25k_v2 data."
    exit 1
fi

# Verify mesh file exists
if [ ! -f "$DATA_DIR/mesh.npz" ]; then
    echo "ERROR: mesh.npz not found in $DATA_DIR"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export STOFS_DATA_DIR="$DATA_DIR"
export STOFS_OUTPUT_DIR="$PROJECT_DIR"

echo "============================================================"
echo "STARTING 25K V2 TRAINING ON H100"
echo "============================================================"
echo "  - V2 Features: 8 forcing + 6 tidal constituents"
echo "  - Dataset: ~360 dates"
echo "  - TRUE batched model (memory scales with batch size)"
echo "  - Rollout schedule (smaller batches for true batching):"
echo "      Epochs 1-15:   1-step,  batch=4, grad_accum=16 (eff=64)"
echo "      Epochs 16-30:  2-step,  batch=4, grad_accum=16 (eff=64)"
echo "      Epochs 31-50:  3-step,  batch=2, grad_accum=16 (eff=32)"
echo "      Epochs 51-75:  6-step,  batch=2, grad_accum=16 (eff=32)"
echo "      Epochs 76-100: 12-step, batch=1, grad_accum=16 (eff=16)"
echo "  - Model: Hidden=128, Layers=6"
echo "  - Physics loss: MSE + mass + smoothness"
echo "============================================================"
echo ""

# Set CUDA_LAUNCH_BLOCKING for better error messages if issues occur
# export CUDA_LAUNCH_BLOCKING=1  # Uncomment if debugging CUDA errors

python3 scripts/train_25k_ursa_h100_v2.py 2>&1 | tee outputs/training_25k_v2_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh outputs/checkpoints_25k_v2/* 2>/dev/null || echo "  No checkpoints"
ls -lh outputs/figures_25k_v2/* 2>/dev/null || echo "  No figures"

exit $EXIT_CODE
