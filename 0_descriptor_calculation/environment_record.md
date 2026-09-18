# Descriptor Calculation Environment Record

Source: original environment usage notes from the authors' server workspace (machine-specific path omitted).

## Model Computation and Validation (DTI / MoA Training)

- Scripts: train_model_AUC_with_ROC_v2.py / positive_negative_double_csv_with_AUC.py / positive_priority_test.py
- **Environment: Step1**
- Note: LD_PRELOAD must be set for each run (already handled automatically by `env_config/activate_step1.sh`):

```bash
export LD_PRELOAD=/home/idddgroup/miniconda3/envs/Step1/lib/libstdc++.so.6
```

## Ligand Descriptors

| Descriptor | Script | Environment |
|--------|------|------|
| E-state | Estate.py | **rdkit** |
| MACCS | MACCS.py | **rdkit** |

## Protein Descriptors

| Descriptor | Script | Environment |
|--------|------|------|
| Profeatx | profeatx_calculator.py (depends on the compiled profeatx program, execute permission required) | **PPDOCK3 or Step1** |
| Zernike | Zernike_calculate.py → mainMakeZernikeDataBase.py (batch) | **FP-Zernike** |
| DeepFRI | DeepFRI.py (depends on the DeepFRI-master directory) | **DeepFRI** |
| Protparam | 2protparam_v3.py + 2protparam_group_v3.py | **FP-Zernike** |
| Peptides | peptides_descriptors.py (pure Python implementation) | **Step1** |

## Server conda Environment Overview

`/home/idddgroup/miniconda3/envs/` contains: DSDP, DeepFRI, FP-Zernike, PPDOCK3, Step1, Step10, Step2, TraCPI, TraCPI2, aidti, drugban, dtiam, fastqc, psichic_fp, rdkit, rdkit2

Environments related to this paper:

| Environment | Purpose |
|------|------|
| Step1 | Model training/inference (tensorflow 2.13.1, Python 3.8.20) |
| rdkit | Ligand E-state / MACCS |
| PPDOCK3 | Profeatx (Step1 also works) |
| FP-Zernike | Zernike / Protparam |
| DeepFRI | DeepFRI GO features |
| Step1 | Peptides (peptides_descriptors.py) |
