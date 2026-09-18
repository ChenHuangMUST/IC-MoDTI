#!/usr/bin/env bash
# ============================================================
# Unified multi-environment configuration for descriptor calculation
# The descriptors used in this paper depend on different conda environments:
#   - ligand E-state / MACCS : rdkit
#   - protein profeatx       : PPDOCK3 or Step1
#   - protein Zernike        : FP-Zernike
#   - protein DeepFRI        : DeepFRI
#   - protein Protparam      : FP-Zernike (pure Python parsing)
#   - protein Peptides       : Step1 (pure Python implementation)
# This script automatically switches environments based on the DESCRIPTOR_ENV environment variable, and automatically
# handles LD_PRELOAD (libstdc++) issues.
# ============================================================
set -e

# Target environment name (can be overridden by a command-line argument)
DESCRIPTOR_ENV="${DESCRIPTOR_ENV:-$1}"

# Auto-detect the conda installation directory
CONDA_BASE="${CONDA_BASE:-}"
if [ -z "$CONDA_BASE" ]; then
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/opt/anaconda3"; do
        if [ -x "$base/bin/conda" ]; then
            CONDA_BASE="$base"
            break
        fi
    done
fi

if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda installation directory not found, please set the CONDA_BASE environment variable"
    exit 1
fi

echo "[OK] conda directory: $CONDA_BASE"
echo "[OK] target env : $DESCRIPTOR_ENV"

# 1. Activate conda
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 2. Activate the target environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$DESCRIPTOR_ENV" ]; then
    conda activate "$DESCRIPTOR_ENV"
    echo "[OK] Activated environment: $CONDA_DEFAULT_ENV"
fi

# 3. LD_PRELOAD adaptation (libstdc++ version conflict)
ENV_LIB="$CONDA_PREFIX/lib"
if [ -e "$ENV_LIB/libstdc++.so.6" ]; then
    export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
    if [ -z "$LD_PRELOAD" ] || [[ "$LD_PRELOAD" != *"libstdc++.so.6"* ]]; then
        export LD_PRELOAD="$ENV_LIB/libstdc++.so.6 $LD_PRELOAD"
    fi
    echo "[OK] LD_PRELOAD: $LD_PRELOAD"
fi

echo "[OK] env_config done: $CONDA_DEFAULT_ENV"
