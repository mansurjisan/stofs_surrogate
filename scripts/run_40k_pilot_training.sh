#!/bin/bash
#SBATCH --job-name=stofs_40k_pilot
#SBATCH --output=outputs/training_40k_pilot_%j.log
#SBATCH --error=outputs/training_40k_pilot_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#
# NOTE: This cluster has no GRES configured (GRES=null)
# GPU access is via partition only - no --gres or --gpus needed

# ============================================================
# STOFS-GNN 40K Node Pilot Training Script for ParallelWorks
# Mid-Atlantic + New England Domain (Norfolk VA to Portland ME)
# ============================================================
#
# Usage (interactive):
#   srun --pty --partition=gpu --cpus-per-task=4 --time=6:00:00 bash
#   ./scripts/run_40k_pilot_training.sh
#
# Usage (batch):
#   sbatch scripts/run_40k_pilot_training.sh
#
# ============================================================

echo "============================================================"
echo "STOFS-GNN 40K NODE PILOT TRAINING"
echo "Mid-Atlantic + New England (37-45N, 77-66W)"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

# Set working directory
cd ~/stofs_surrogate || { echo "ERROR: Could not cd to ~/stofs_surrogate"; exit 1; }

# Create output directories
mkdir -p outputs/checkpoints_40k_pilot outputs/figures_40k_pilot

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
DATA_DIR="data/processed_40k"

if [ ! -f "$DATA_DIR/mesh.npz" ]; then
    echo "ERROR: Mesh file not found: $DATA_DIR/mesh.npz"
    echo "Please upload the processed data first."
    exit 1
fi

NUM_FILES=$(ls -1 $DATA_DIR/processed_*.npz 2>/dev/null | wc -l)
echo "  Mesh: $DATA_DIR/mesh.npz"
echo "  Data files: $NUM_FILES"

if [ "$NUM_FILES" -lt 30 ]; then
    echo "WARNING: Only $NUM_FILES data files found (expected 30 for pilot)"
fi

# Show GPU memory
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export CUDA_LAUNCH_BLOCKING=0

# Set data paths for the training script
export STOFS_DATA_DIR="$HOME/stofs_surrogate/data/processed_40k"
export STOFS_OUTPUT_DIR="$HOME/stofs_surrogate"

# Run training
echo "============================================================"
echo "STARTING PILOT TRAINING (30 days, 40k nodes)"
echo "============================================================"
echo ""

python3 scripts/train_midatlantic_40k_pilot.py 2>&1 | tee outputs/training_40k_pilot_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "PILOT TRAINING COMPLETE"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Show output files
echo ""
echo "Output files:"
ls -lh outputs/checkpoints_40k_pilot/* 2>/dev/null || echo "  No checkpoint files found"
ls -lh outputs/figures_40k_pilot/* 2>/dev/null || echo "  No figure files found"

exit $EXIT_CODE
