#!/usr/bin/env bash
# ============================================================
# IC-MoDTI preprocessing (Step 1): protein/ligand autoencoder pretraining
# Dependencies: Step1 environment (see env_config/activate_step1.sh)
# Inputs: data/protein_features_unique_260414.csv
#         data/ligand_feature_260703.csv
# Outputs: 6 files under encoder_save/ (encoders + scalers + lookup tables)
# ============================================================
set -e
cd "$(dirname "$0")"

# 1. Configure the environment (sets LD_PRELOAD automatically)
if [ -f "../env_config/activate_step1.sh" ]; then
    source "../env_config/activate_step1.sh"
else
    echo "[WARN] env_config script not found, skipping LD_PRELOAD setup"
fi

# 2. Run pretraining (defaults: protein latent dim 256, ligand latent dim 64, 100 epochs)
python pretrain_encoder.py \
    --protein_csv data/protein_features_unique_260414.csv \
    --ligand_csv data/ligand_feature_260703.csv \
    --protein_latent_dim 256 \
    --ligand_latent_dim 64 \
    --epoch 100 \
    --batch 256

echo "[OK] Preprocessing done; pretrained encoder files generated under encoder_save/"
