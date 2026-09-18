#!/usr/bin/env bash
# ============================================================
# IC-MoDTI MoA module training (activating/inhibitory binary classification + fine-tuning + contrastive learning)
# Dependencies: Step1 environment; run 1_preprocessing first to generate encoder_save/
# Usage:
#   bash run_moa.sh                          # all defaults (P51787 experimental protein)
#   bash run_moa.sh --exp P51787             # fine-tune for the specified experimental protein
#   bash run_moa.sh --contrastive_weight 0.1 # enable contrastive learning (FT3)
# Outputs: my_model.h5 / preprocessor.joblib / feature_columns.json
#          optimal_threshold.json / training_metrics.json + evaluation figures
# ============================================================
set -e
cd "$(dirname "$0")"

# 1. Configure environment (automatically sets LD_PRELOAD)
if [ -f "../env_config/activate_step1.sh" ]; then
    source "../env_config/activate_step1.sh"
else
    echo "[WARN] env_config script not found, skipping LD_PRELOAD setup"
fi

# 2. Train the MoA model (5-fold CV + 3-stage fine-tuning + threshold optimization)
python fix_train_model_AUC_with_ROC_v254_activity_DSDP_v10_5X4_contrastive.py \
    -m main_exp.csv \
    -p ../1_preprocessing/data/protein_features_unique_260414.csv \
    -l ../1_preprocessing/data/ligand_feature_exp.csv \
    -d data \
    "$@"

echo "[OK] MoA module training complete"
