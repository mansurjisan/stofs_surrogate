#!/bin/bash
#SBATCH --job-name=stofs_25k_gnn
#SBATCH --output=outputs/training_25k_%j.log
#SBATCH --error=outputs/training_25k_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#
# NOTE: This cluster has no GRES configured (GRES=null)
# GPU access is via partition only - no --gres or --gpus needed

# ============================================================
# STOFS-GNN 25K Node Training Script for ParallelWorks
# ============================================================
#
# Usage (interactive):
#   srun --pty --gres=gpu:1 --cpus-per-task=4 --time=6:00:00 bash
#   ./scripts/run_25k_training.sh
#
# Usage (batch):
#   sbatch scripts/run_25k_training.sh
#
# ============================================================

echo "============================================================"
echo "STOFS-GNN 25K NODE TRAINING"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

# Set working directory
cd ~/stofs_surrogate || { echo "ERROR: Could not cd to ~/stofs_surrogate"; exit 1; }

# Create output directories
mkdir -p outputs/checkpoints outputs/figures

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_geo

# Verify environment
echo ""
echo "Environment check:"
echo "  Python: $(which python3)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Check data files
echo "Checking data files..."
DATA_DIR="data/processed_25k"

if [ ! -f "$DATA_DIR/mesh_25k.npz" ]; then
    echo "ERROR: Mesh file not found: $DATA_DIR/mesh_25k.npz"
    exit 1
fi

NUM_FILES=$(ls -1 $DATA_DIR/processed_*.npz 2>/dev/null | wc -l)
echo "  Mesh: $DATA_DIR/mesh_25k.npz"
echo "  Data files: $NUM_FILES"

if [ "$NUM_FILES" -lt 10 ]; then
    echo "WARNING: Only $NUM_FILES data files found (expected 15)"
fi

# Show GPU memory
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0

# Run training
echo "============================================================"
echo "STARTING TRAINING"
echo "============================================================"
echo ""

python3 scripts/train_25k_15day.py 2>&1 | tee outputs/training_25k_$(date +%Y%m%d_%H%M%S).log

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
ls -lh outputs/checkpoints/*25k* 2>/dev/null || echo "  No checkpoint files found"
ls -lh outputs/figures/*25k* 2>/dev/null || echo "  No figure files found"

exit $EXIT_CODE
