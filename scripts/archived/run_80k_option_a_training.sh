#!/bin/bash
#SBATCH --job-name=stofs_80k_opt_a
#SBATCH --output=outputs/training_80k_option_a_%j.log
#SBATCH --error=outputs/training_80k_option_a_%j.err
#SBATCH --time=336:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#
# NOTE: This cluster has no GRES configured (GRES=null)
# GPU access is via partition only - no --gres or --gpus needed

# ============================================================
# STOFS-GNN 80K Node Training - Option A
# Long Island Sound to Southern Maine (40-44°N, 74-69°W)
# ============================================================
#
# Domain: NY, CT, RI, MA, Southern ME
# Nodes: 80,000
# Edges: ~580,000 (KNN k=6)
# Resolution: 1.5 km
#
# Usage (interactive):
#   srun --pty --partition=gpu --cpus-per-task=4 --time=8:00:00 bash
#   ./scripts/run_80k_option_a_training.sh
#
# Usage (batch):
#   sbatch scripts/run_80k_option_a_training.sh
#
# ============================================================

echo "============================================================"
echo "STOFS-GNN 80K NODE TRAINING - OPTION A"
echo "Long Island Sound to Southern Maine (40-44°N, 74-69°W)"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

# Set working directory
cd ~/stofs_surrogate || { echo "ERROR: Could not cd to ~/stofs_surrogate"; exit 1; }

# Create output directories
mkdir -p outputs/checkpoints_80k_option_a outputs/figures_80k_option_a

# Activate conda environment (use base or existing env with PyTorch)
echo "Setting up Python environment..."

# Try to source conda if available
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    # Use base environment (should have pytorch installed)
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
    echo "Please upload the processed data first."
    exit 1
fi

NUM_FILES=$(ls -1 $DATA_DIR/processed_*.npz 2>/dev/null | wc -l)
echo "  Mesh: $DATA_DIR/mesh.npz"
echo "  Data files: $NUM_FILES"

# Verify mesh configuration
echo ""
echo "Mesh configuration:"
python3 -c "
import numpy as np
mesh = np.load('$DATA_DIR/mesh.npz')
print(f'  Nodes: {len(mesh[\"lon\"]):,}')
print(f'  Edges: {mesh[\"edge_index\"].shape[1]:,}')
print(f'  Edges/node: {mesh[\"edge_index\"].shape[1] / len(mesh[\"lon\"]):.1f}')
print(f'  Lon: {mesh[\"lon\"].min():.2f} to {mesh[\"lon\"].max():.2f}')
print(f'  Lat: {mesh[\"lat\"].min():.2f} to {mesh[\"lat\"].max():.2f}')
"

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
export STOFS_DATA_DIR="$HOME/stofs_surrogate/data/processed_80k_option_a"
export STOFS_OUTPUT_DIR="$HOME/stofs_surrogate"

# Run training
echo "============================================================"
echo "STARTING TRAINING - 80k Option A (Full Dataset ~300 days)"
echo "Expected duration: ~45 days"
echo "Checkpoints saved every 10 epochs for resume capability"
echo "============================================================"
echo ""

python3 scripts/train_80k_option_a.py 2>&1 | tee outputs/training_80k_option_a_$(date +%Y%m%d_%H%M%S).log

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
ls -lh outputs/checkpoints_80k_option_a/* 2>/dev/null || echo "  No checkpoint files found"
ls -lh outputs/figures_80k_option_a/* 2>/dev/null || echo "  No figure files found"

exit $EXIT_CODE
