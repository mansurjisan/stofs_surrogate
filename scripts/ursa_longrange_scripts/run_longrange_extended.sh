#!/bin/bash
#SBATCH --job-name=stofs_lr_ext
#SBATCH --account=gpu-nos-surge
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00
#SBATCH --output=outputs/longrange_ext_%j.log
#SBATCH --error=outputs/longrange_ext_%j.err

# Enable pipefail to capture Python exit code
set -o pipefail

# ============================================================
# STOFS-GNN EXTENDED LONG-RANGE TRAINING
# 36-step and 48-step rollout training
#
# Resumes from epoch 50 checkpoint (after 24-step training)
# Continues with:
#   - Epochs 51-65: 36-step rollout (36h)
#   - Epochs 66-80: 48-step rollout (48h)
#
# Walltime: 7 days (168 hours)
# ============================================================

echo "============================================================"
echo "STOFS-GNN EXTENDED LONG-RANGE TRAINING"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

# Change to project directory
cd /scratch5/purged/Mansur.Jisan/stofs_surrogate || { echo "ERROR: Could not cd to project dir"; exit 1; }

# Create output directories
mkdir -p outputs/checkpoints_25k_longrange

# Setup environment
source ~/venv_stofs/bin/activate

echo "Environment check:"
echo "  Python: $(which python3)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU: None')"
echo ""

# Check for epoch 50 checkpoint
CKPT_DIR="outputs/checkpoints_25k_longrange"
if [ ! -f "$CKPT_DIR/checkpoint_longrange_epoch_50.pt" ]; then
    echo "WARNING: epoch 50 checkpoint not found"
    echo "Looking for latest checkpoint..."
    ls -la $CKPT_DIR/*.pt 2>/dev/null || echo "  No checkpoints found"
    echo ""
    echo "Script will resume from latest available checkpoint"
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "============================================================"
echo "STARTING EXTENDED LONG-RANGE TRAINING"
echo "============================================================"
echo "  - Resume from: epoch 50 (or latest)"
echo "  - Schedule:"
echo "      Epochs 51-65: 36-step rollout (36h)"
echo "      Epochs 66-80: 48-step rollout (48h)"
echo "  - Batch size: 1 (due to 447k edges)"
echo "  - Learning rate: 1e-5 (extended fine-tuning)"
echo "  - Checkpoint interval: every 2 epochs"
echo "  - Walltime: 7 days (168 hours)"
echo "============================================================"
echo ""

python3 scripts/ursa_longrange_scripts/train_25k_longrange_extended.py 2>&1 | tee outputs/longrange_ext_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
echo "Checkpoints:"
ls -lh outputs/checkpoints_25k_longrange/*.pt 2>/dev/null || echo "  No checkpoints"

exit $EXIT_CODE
