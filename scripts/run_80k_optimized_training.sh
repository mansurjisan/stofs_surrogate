#!/bin/bash
#SBATCH --job-name=stofs_80k_opt
#SBATCH --output=outputs/training_80k_optimized_%j.log
#SBATCH --error=outputs/training_80k_optimized_%j.err
#SBATCH --time=336:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu

# Enable pipefail to capture Python exit code, not tee's
set -o pipefail

# ============================================================
# STOFS-GNN 80K - OPTIMIZED TRAINING
# Uses 100 dates per epoch with random resampling
# Expected: ~2-3 hours per epoch, ~10-12 days total
# ============================================================

echo "============================================================"
echo "STOFS-GNN 80K OPTIMIZED TRAINING"
echo "100 dates per epoch, gradient accumulation"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

cd ~/stofs_surrogate || { echo "ERROR: Could not cd to ~/stofs_surrogate"; exit 1; }

mkdir -p outputs/checkpoints_80k_optimized outputs/figures_80k_optimized

# Setup environment
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base 2>/dev/null || true
fi

echo "Environment check:"
echo "  Python: $(which python3)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU: None')"
echo ""

# Check data
DATA_DIR="data/processed_80k_option_a"
NUM_FILES=$(ls -1 $DATA_DIR/processed_*.npz 2>/dev/null | wc -l)
echo "Data files: $NUM_FILES"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export STOFS_DATA_DIR="$HOME/stofs_surrogate/data/processed_80k_option_a"
export STOFS_OUTPUT_DIR="$HOME/stofs_surrogate"

echo "============================================================"
echo "STARTING OPTIMIZED TRAINING"
echo "  - 50 dates per epoch (random sample from 253)"
echo "  - 150 epochs (more epochs to see all data)"
echo "  - Gradient accumulation: 4 steps"
echo "  - Expected: ~1.3 hours per epoch, ~8 days total"
echo "============================================================"
echo ""

python3 scripts/train_80k_optimized.py 2>&1 | tee outputs/training_80k_optimized_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh outputs/checkpoints_80k_optimized/* 2>/dev/null || echo "  No checkpoints"
ls -lh outputs/figures_80k_optimized/* 2>/dev/null || echo "  No figures"

exit $EXIT_CODE
