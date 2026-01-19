#!/bin/bash
#SBATCH --job-name=stofs_25k
#SBATCH --account=gpu-nos-surge
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=outputs/training_25k_h100_%j.log
#SBATCH --error=outputs/training_25k_h100_%j.err

# Enable pipefail to capture Python exit code
set -o pipefail

# ============================================================
# STOFS-GNN 25K - URSA H100 TRAINING
# Temporal Memory GNN with Tidal Harmonics
# Expected: ~2-4 hours for 100 epochs
# ============================================================

echo "============================================================"
echo "STOFS-GNN 25K URSA H100 TRAINING"
echo "Temporal Memory GNN with Tidal Harmonics"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

# Change to project directory
cd /scratch5/purged/Mansur.Jisan/stofs_surrogate || { echo "ERROR: Could not cd to project dir"; exit 1; }

# Create output directories
mkdir -p outputs/checkpoints_25k_h100 outputs/figures_25k_h100

# Setup environment - use same venv as 80k training
source ~/venv_stofs/bin/activate

echo "Environment check:"
echo "  Python: $(which python3)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU: None')"
echo ""

# Check data - use full Mid-Atlantic dataset
DATA_DIR="/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_25k_midatl"
NUM_FILES=$(ls -1 $DATA_DIR/processed_*.npz 2>/dev/null | wc -l)
echo "Data files: $NUM_FILES in $DATA_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export STOFS_DATA_DIR="$DATA_DIR"
export STOFS_OUTPUT_DIR="/scratch5/purged/Mansur.Jisan/stofs_surrogate"

echo "============================================================"
echo "STARTING 25K TRAINING ON H100"
echo "  - Full year dataset (~360 dates)"
echo "  - Batch size: 16 (H100 optimized)"
echo "  - Hidden dim: 128, 6 layers"
echo "  - Max rollout: 3 steps"
echo "  - Physics-informed loss (MSE + mass + smoothness)"
echo "============================================================"
echo ""

python3 scripts/train_25k_ursa_h100.py 2>&1 | tee outputs/training_25k_h100_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh outputs/checkpoints_25k_h100/* 2>/dev/null || echo "  No checkpoints"
ls -lh outputs/figures_25k_h100/* 2>/dev/null || echo "  No figures"

exit $EXIT_CODE
