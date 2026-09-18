# Non-official packages

This directory contains the only components that **cannot** be installed by name
and version from public package indexes (PyPI / conda-forge / official channels).
Everything else is covered by `environment.yml` and `requirements.txt` at the
repository root.

## Contents

| Archive | Description |
|---------|-------------|
| `profeatx_linux_x86_64.tar.gz` | Custom-compiled `profeatx` executable (Linux x86_64, dynamically linked). It is the feature-calculation program used by `0_descriptor_calculation/protein/profeatx_calculator.py` to generate the `protein_profeatx_*` descriptors. |

### Extracting profeatx

```bash
tar -xzf env_packages/profeatx_linux_x86_64.tar.gz -C /opt/tools/profeatx
chmod +x /opt/tools/profeatx/profeatx
```

Then run `profeatx_calculator.py` with the directory containing the `profeatx`
binary available on `PATH` (or configure the script accordingly).

## External tools (not bundled, publicly available)

| Tool | Used for | Where to obtain |
|------|----------|-----------------|
| DeepFRI (source + trained models, ~1.2 GB of weights) | `protein_DeepFRI_*` GO features | Public project: https://github.com/flatironinstitute/DeepFRI (trained models are published with the project / on Zenodo) |
| FP-Zernike | `protein_Zernike_*` descriptors | The original FP-Zernike distribution used in the paper (contact the authors if you need the exact snapshot) |
| PyMOL, APBS, HDF5 and other conda packages | descriptor calculation environments | Public conda channels (`conda-forge`, `schrodinger`, ...); see `0_descriptor_calculation/environment_record.md` |

## Note

All packages in the Step1 model environment (tensorflow, torch, scikit-learn,
pandas, xgboost, lightgbm, catboost, rdkit, ...) were verified to come from
public indexes, so no conda environment snapshot is needed: recreate it with
`environment.yml` + `requirements.txt` (total download size is dominated by the
official CUDA-enabled TensorFlow/PyTorch wheels).
