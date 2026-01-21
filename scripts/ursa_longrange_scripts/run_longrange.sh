#!/bin/bash
#SBATCH --job-name=stofs_longrange
#SBATCH --account=gpu-nos-surge
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00
#SBATCH --output=outputs/longrange_%j.log
#SBATCH --error=outputs/longrange_%j.err

# Enable pipefail to capture Python exit code
set -o pipefail

# ============================================================
# STOFS-GNN 25K LONG-RANGE FINE-TUNING
# Fine-tuning with enhanced long-range edge connectivity
#
# Walltime: 7 days (168 hours)
# Expected: ~20 epochs per submission (~8.4 hours/epoch)
# ============================================================

echo "============================================================"
echo "STOFS-GNN 25K LONG-RANGE FINE-TUNING"
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

# Check data
DATA_DIR="/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_25k_v2"
MESH_DIR="/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_25k_v2_longrange"

echo "Data directory: $DATA_DIR"
echo "Mesh directory (long-range): $MESH_DIR"

if [ ! -f "$MESH_DIR/mesh.npz" ]; then
    echo "ERROR: Long-range mesh.npz not found in $MESH_DIR"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "============================================================"
echo "STARTING LONG-RANGE FINE-TUNING"
echo "============================================================"
echo "  - Base model: checkpoint_epoch_60.pt (25K V2)"
echo "  - Long-range edges: 447k (vs 185k original)"
echo "  - Batch size: 1 (due to 2.4x more edges)"
echo "  - Learning rate: 2e-5 (fine-tuning)"
echo "  - Checkpoint interval: every 2 epochs"
echo "  - Walltime: 7 days (168 hours)"
echo "  - Expected epochs: ~20 per submission"
echo "============================================================"
echo ""

python3 scripts/train_25k_longrange.py 2>&1 | tee outputs/longrange_$(date +%Y%m%d_%H%M%S).log

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
