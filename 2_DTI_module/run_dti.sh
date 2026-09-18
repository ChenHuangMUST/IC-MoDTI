#!/usr/bin/env bash
# ============================================================
# IC-MoDTI DTI module training (binding/non-binding binary classification)
# Dependencies: Step1 environment; run 1_preprocessing first to generate encoder_save/
# Inputs: positive/negative sample group files under data/ + protein/ligand feature files
# Outputs: my_model.h5 / preprocessor.joblib / feature_columns.json
#          training_metrics.json + evaluation plots (pdf)
# ============================================================
set -e
cd "$(dirname "$0")"

# 1. Configure environment
if [ -f "../env_config/activate_step1.sh" ]; then
    source "../env_config/activate_step1.sh"
else
    echo "[WARN] env_config script not found, skipping LD_PRELOAD setup"
fi

# 2. Positive sample group file: extracted from the main file where DSDP_Score < -3
echo ">>> Generating DTI positive sample group file (DSDP_Score < -3)..."
python - <<'EOF'
import pandas as pd
main = pd.read_csv("data/main_columns_True25_total_True_with_DSDP_26070501.csv")
pos = main[main["DSDP_Score"] < -3][["protein", "ligand", "DSDP_Score"]]
pos.to_csv("data/dti_positive_group.csv", index=False)
print(f"Positive sample pairs: {len(pos)}")
EOF

# 3. Train the DTI model (5-fold cross-validation)
python positive_negative_double_csv_with_AUC_v8.py \
    --pg data/dti_positive_group.csv \
    --pp ../1_preprocessing/data/protein_features_unique_260414.csv \
    --pl ../1_preprocessing/data/ligand_feature_260703.csv \
    --ng data/new_negative_all_10625.csv \
    --np ../1_preprocessing/data/protein_features_unique_260414.csv \
    --nl ../1_preprocessing/data/ligand_feature_260703.csv

echo "[OK] DTI module training complete"
