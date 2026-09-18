#!/usr/bin/env bash
# ============================================================
# IC-MoDTI one-click full pipeline (for end users)
# Input: one CSV table (UniProt ID + SMILES/InChI) or three separate tables
# Output: DTI/MoA prediction results (or trained models)
#
# Usage:
#   bash run_all.sh input_table.csv                    # full pipeline: prepare → descriptors → predict
#   bash run_all.sh input_table.csv --train            # full pipeline and retrain models
#   bash run_all.sh input_table.csv --skip-descriptor  # protein/ligand already in library, skip descriptor calculation
#
# Input table format (see README.md):
#   protein,uniprot_id,ligand,smiles,inchi
#   KCNQ1,P51787,ligand_A,CC(=O)Nc1ccc(N)cc1,
# ============================================================
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

INPUT="$1"
[ -z "$INPUT" ] && echo "Usage: bash run_all.sh input_table.csv [--train|--skip-descriptor]" && exit 1
shift || true

MODE="predict"
for a in "$@"; do
    case "$a" in
        --train) MODE="train" ;;
        --skip-descriptor) SKIP_DESC=1 ;;
    esac
done

WORK="$(cd "$(dirname "$INPUT")" && pwd)/icmotdi_output"
mkdir -p "$WORK"
echo ">>> Output directory: $WORK"

# 0. Environment setup (LD_PRELOAD)
source "$ROOT/env_config/activate_step1.sh" 2>/dev/null || true

# 1. Prepare input: download fasta/PDB, convert SMILES/InChI -> InChIKey
echo ""
echo "════════════════════════════════════════════"
echo " Step 1/5: Input data preparation (UniProt + AlphaFold + rdkit)"
echo "════════════════════════════════════════════"
cd "$ROOT/0_data_preparation"
python prepare_inputs.py --input "$INPUT" --out "$WORK" 2>&1 | tail -30

# 2. Descriptor calculation (only needed for new molecules)
if [ -z "$SKIP_DESC" ]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo " Step 2/5: Descriptor calculation (automatic environment switch)"
    echo "════════════════════════════════════════════"
    bash "$ROOT/run_pipeline.sh" all 2>&1 | tail -20 || echo "[WARN] Descriptor calculation partly failed, please check the output"
else
    echo "[SKIP] Descriptor calculation (--skip-descriptor)"
fi

# 3. Feature assembly
echo ""
echo "════════════════════════════════════════════"
echo " Step 3/5: Feature assembly (aligned with feature_columns.json)"
echo "════════════════════════════════════════════"
mkdir -p "$WORK/features"
python "$ROOT/0_descriptor_calculation/assemble_features.py" \
    --protein-dir "$WORK" \
    --ligand-dir "$WORK" \
    --out-dir "$WORK/features" \
    --ref-dir "$ROOT/2_DTI_module" 2>&1 | tail -15 || echo "[WARN] Assembly failed, please check the descriptor output"

# 4. Latent vector completion (add new molecules to latent table)
echo ""
echo "════════════════════════════════════════════"
echo " Step 4/5: Latent vector completion (new protein/ligand -> encoder)"
echo "════════════════════════════════════════════"
python "$ROOT/0_data_preparation/infer_latent.py" \
    --protein-features "$WORK/features/protein_features_assembled.csv" \
    --ligand-features "$WORK/features/ligand_feature_assembled.csv" \
    2>&1 | tail -10 || echo "[WARN] Latent vector completion failed (does not affect MoA prediction)"

# 5. Predict / train
echo ""
echo "════════════════════════════════════════════"
echo " Step 5/5: Model prediction"
echo "════════════════════════════════════════════"
COMBOS="$WORK/combinations.csv"
PROT_FEAT="$WORK/features/protein_features_assembled.csv"
LIG_FEAT="$WORK/features/ligand_feature_assembled.csv"

if [ ! -f "$PROT_FEAT" ] || [ ! -f "$LIG_FEAT" ]; then
    echo "[WARN] New features not generated; falling back to existing features in the library (library molecules only)"
    PROT_FEAT="$ROOT/1_preprocessing/data/protein_features_unique_260414.csv"
    LIG_FEAT="$ROOT/1_preprocessing/data/ligand_feature_260703.csv"
fi

if [ "$MODE" = "train" ]; then
    echo ">>> Training mode: retrain DTI + MoA models"
    cd "$ROOT/1_preprocessing" && bash run_pretrain.sh 2>&1 | tail -5
    cd "$ROOT/2_DTI_module" && python positive_negative_double_csv_with_AUC_v8.py \
        --pg "$WORK/combinations.csv" --pp "$PROT_FEAT" --pl "$LIG_FEAT" \
        --ng "$ROOT/2_DTI_module/data/3_negative.csv" --np "$PROT_FEAT" --nl "$LIG_FEAT" 2>&1 | tail -10
    cd "$ROOT/3_MoA_module" && python fix_train_model_AUC_with_ROC_v254_activity_DSDP_v10_5X4_contrastive.py \
        -m "$ROOT/3_MoA_module/data/main_columns_True25_total_True_with_DSDP_26070501.csv" \
        -p "$PROT_FEAT" -l "$LIG_FEAT" -d data 2>&1 | tail -10
else
    echo ">>> Prediction mode: use trained models"
    # DTI
    cd "$ROOT/2_DTI_module"
    python positive_negative_external.py -p "$PROT_FEAT" -l "$LIG_FEAT" \
        -g "$COMBOS" -o "$WORK/dti_predictions.csv" 2>&1 | tail -8
    # MoA
    cd "$ROOT/3_MoA_module"
    python external_data_verification_v3_to_v23.py -p "$PROT_FEAT" -l "$LIG_FEAT" \
        -g "$COMBOS" -o "$WORK/moa_predictions.csv" 2>&1 | tail -8
fi

echo ""
echo "════════════════════════════════════════════"
echo " Done! Result files:"
echo "   DTI: $WORK/dti_predictions.csv"
echo "   MoA: $WORK/moa_predictions.csv"
echo "   (training mode: models saved in 2_DTI_module/ and 3_MoA_module/)"
echo "════════════════════════════════════════════"
