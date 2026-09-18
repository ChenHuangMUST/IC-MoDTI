#!/usr/bin/env bash
# ============================================================
# IC-MoDTI automatic environment setup script (fixes LD_PRELOAD libstdc++ issues)
# Usage: source activate_step1.sh  or  bash activate_step1.sh
# Note: just run this script after conda activate Step1
# ============================================================
set -e

ENV_NAME="Step1"

# 1. Auto-locate the conda environment (use CONDA_PREFIX when conda is already activated)
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    ENV_LIB="$CONDA_PREFIX/lib"
else
    # Try auto-detection
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/opt/anaconda3"; do
        if [ -d "$base/envs/$ENV_NAME/lib" ]; then
            ENV_LIB="$base/envs/$ENV_NAME/lib"
            break
        fi
    done
fi

if [ -z "$ENV_LIB" ] || [ ! -d "$ENV_LIB" ]; then
    echo "[WARN] lib directory for $ENV_NAME environment not found, please run first: conda activate $ENV_NAME"
    return 1 2>/dev/null || exit 1
fi

# 2. Set LD_LIBRARY_PATH (prioritize the environment lib containing libstdc++)
export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"

# 3. Set LD_PRELOAD (fixes GLIBCXX version conflicts)
STDCPP="$ENV_LIB/libstdc++.so.6"
if [ -e "$STDCPP" ]; then
    if [ -z "$LD_PRELOAD" ] || [[ "$LD_PRELOAD" != *"libstdc++.so.6"* ]]; then
        export LD_PRELOAD="$STDCPP $LD_PRELOAD"
    fi
fi

# 4. Print current python information
echo "[OK] Step1 environment configured"
echo "     Python   : $(which python)"
echo "     LD_PRELOAD: $LD_PRELOAD"
