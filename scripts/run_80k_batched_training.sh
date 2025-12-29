#!/bin/bash
#SBATCH --job-name=stofs_80k_batched
#SBATCH --output=outputs/training_80k_batched_%j.log
#SBATCH --error=outputs/training_80k_batched_%j.err
#SBATCH --time=336:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu

# ============================================================
# STOFS-GNN 80K Node Training - TRUE BATCHED VERSION
# 5-10x faster than per-sample processing
# ============================================================
#
# Usage (interactive):
#   srun --pty --partition=gpu --cpus-per-task=2 --time=8:00:00 bash
#   ./scripts/run_80k_batched_training.sh
#
# Usage (batch):
#   sbatch scripts/run_80k_batched_training.sh
#
# ============================================================

echo "============================================================"
echo "STOFS-GNN 80K NODE TRAINING - TRUE BATCHED"
echo "One forward/backward pass per batch (5-10x faster)"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

# Set working directory
cd ~/stofs_surrogate || { echo "ERROR: Could not cd to ~/stofs_surrogate"; exit 1; }

# Create output directories
mkdir -p outputs/checkpoints_80k_batched outputs/figures_80k_batched

# Setup Python environment
echo "Setting up Python environment..."
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base 2>/dev/null || true
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate base 2>/dev/null || true
fi

# Verify environment
echo ""
echo "Environment check:"
echo "  Python: $(which python3)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU: None')"
echo ""

# Check data files
echo "Checking data files..."
DATA_DIR="data/processed_80k_option_a"

if [ ! -f "$DATA_DIR/mesh.npz" ]; then
    echo "ERROR: Mesh file not found: $DATA_DIR/mesh.npz"
    exit 1
fi

NUM_FILES=$(ls -1 $DATA_DIR/processed_*.npz 2>/dev/null | wc -l)
echo "  Mesh: $DATA_DIR/mesh.npz"
echo "  Data files: $NUM_FILES"

# Show GPU memory
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export CUDA_LAUNCH_BLOCKING=0
export STOFS_DATA_DIR="$HOME/stofs_surrogate/data/processed_80k_option_a"
export STOFS_OUTPUT_DIR="$HOME/stofs_surrogate"

# Run training
echo "============================================================"
echo "STARTING TRUE BATCHED TRAINING"
echo "Expected: ~1-2 hours per epoch (vs 8 hours with per-sample)"
echo "Total: ~5-8 days for 100 epochs"
echo "============================================================"
echo ""

python3 scripts/train_80k_batched.py 2>&1 | tee outputs/training_80k_batched_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Show output files
echo ""
echo "Output files:"
ls -lh outputs/checkpoints_80k_batched/* 2>/dev/null || echo "  No checkpoint files found"
ls -lh outputs/figures_80k_batched/* 2>/dev/null || echo "  No figure files found"

exit $EXIT_CODE
