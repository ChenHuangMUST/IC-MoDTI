#!/usr/bin/env bash
# ============================================================
# IC-MoDTI full one-click pipeline (descriptor calculation → feature assembly → model usage)
# Automatically switches to the corresponding conda environment for each descriptor.
#
# Usage:
#   bash run_pipeline.sh all                 # calculate all descriptors
#   bash run_pipeline.sh ligand              # ligand descriptors only (E-state, MACCS)
#   bash run_pipeline.sh protein             # protein descriptors only (all)
#   bash run_pipeline.sh profeatx            # run a single designated descriptor
#   bash run_pipeline.sh zernike / deepfri / protparam / peptides / estate / maccs
#
# Prerequisites: conda environments rdkit / FP-Zernike / PPDOCK3 / Step1 etc. already created
#       (see README.md; the Peptides descriptor is a pure Python implementation).
# ============================================================
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/env_config/activate_env.sh" 2>/dev/null || true

TASK="${1:-all}"

run_in_env() {
    local env="$1"
    shift
    echo ""
    echo "=========================================="
    echo ">>> Switching to environment [$env] to run: $*"
    echo "=========================================="
    source "$ROOT/env_config/activate_env.sh" "$env"
    bash -c "$*"
}

case "$TASK" in
    all)
        bash "$0" ligand
        bash "$0" protein
        ;;
    ligand)
        bash "$0" estate
        bash "$0" maccs
        ;;
    protein)
        bash "$0" profeatx
        bash "$0" zernike
        bash "$0" deepfri
        bash "$0" protparam
        bash "$0" peptides
        ;;
    estate)
        run_in_env rdkit "cd $ROOT/0_descriptor_calculation/ligand && python Estate.py --help"
        ;;
    maccs)
        run_in_env rdkit "cd $ROOT/0_descriptor_calculation/ligand && python MACCS.py --help"
        ;;
    profeatx)
        run_in_env PPDOCK3 "cd $ROOT/0_descriptor_calculation/protein && python profeatx_calculator.py --help"
        ;;
    zernike)
        run_in_env FP-Zernike "cd $ROOT/0_descriptor_calculation/protein && python Zernike_calculate.py --help"
        ;;
    deepfri)
        run_in_env DeepFRI "cd $ROOT/0_descriptor_calculation/protein && python DeepFRI.py --help"
        ;;
    protparam)
        run_in_env FP-Zernike "cd $ROOT/0_descriptor_calculation/protein && python 2protparam_group_v3.py --help"
        ;;
    peptides)
        # Pure Python implementation (no R required), argument: input fasta directory
        if [ -z "$PEPTIDES_INPUT" ]; then
            echo ">>> Peptides descriptor (pure Python implementation, no R required)"
            echo "    Usage: PEPTIDES_INPUT=fasta_dir bash run_pipeline.sh peptides"
            echo "    Or:    python 0_descriptor_calculation/protein/peptides_descriptors.py -i fasta_dir -o output.csv"
        else
            run_in_env Step1 "cd $ROOT/0_descriptor_calculation/protein && python peptides_descriptors.py -i $PEPTIDES_INPUT -o ${PEPTIDES_OUT:-Peptides_all.csv}"
        fi
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available: all / ligand / protein / estate / maccs / profeatx / zernike / deepfri / protparam / peptides"
        exit 1
        ;;
esac

echo ""
echo "[OK] Task [$TASK] done"
