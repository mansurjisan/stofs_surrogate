#!/bin/bash
# Multi-date rollout comparison for 80k model

export STOFS_DATA_DIR=/mnt/f/STOFS_TRAINING_DATA/processed_80k_option_a
export STOFS_MODEL_PATH=/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_80k_h100/best_model.pt
export STOFS_OUTPUT_DIR=/mnt/d/AI_4_STOFS/stofs_surrogate

DATES="20240115 20231201 20241115 20240301 20230215"

echo "=============================================="
echo "Multi-Date 48h Rollout Comparison - 80k Model"
echo "=============================================="
echo ""

for DATE in $DATES; do
    echo "--- Date: $DATE ---"
    python scripts/rollout_80k_model.py --date $DATE --hours 48 2>&1 | grep -E "(Overall RMSE|Max error|Boston:|Portland_ME:|New_London:|Newport:|Bridgeport:|Kings_Point:)"
    echo ""
done

echo "=============================================="
echo "Done!"
