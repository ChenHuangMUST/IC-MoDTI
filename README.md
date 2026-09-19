# IC-MoDTI

**Ion Channel Model for Drug–Target Interaction prediction**

Author: **Zelin Yi**
License: GPL-2.0 (see [LICENSE](LICENSE))

IC-MoDTI predicts whether a compound binds to an ion-channel protein (DTI) and,
if it binds, whether it acts as an activator or inhibitor (MoA). The model uses
seven molecular descriptor families together with auto-encoder latent vectors,
and is trained with a multi-stage schedule (pre-training → weighted fine-tuning →
optional contrastive learning).

| Module | Task | Output |
|--------|------|--------|
| **DTI** | Does the compound bind the ion channel? | `BINDING` / `NON_BINDING` + probability |
| **MoA** | If bound, activation or inhibition? | `ACTIVATING` / `INHIBITORY` + probability |

## Repository layout

```
IC-MoDTI/
├── run_all.sh                        # one-click end-to-end pipeline for end users
├── run_pipeline.sh                   # descriptor calculation pipeline (auto env switching)
├── environment.yml                   # conda-level dependencies (Python 3.8, Linux)
├── requirements.txt                  # pip dependencies (official packages only)
├── env_config/                       # environment activation helpers / LD_PRELOAD fix
├── env_packages/                     # non-official custom packages (profeatx binary)
├── 0_descriptor_calculation/         # descriptor calculation methods
│   ├── ligand/                       #   E-state, MACCS (rdkit)
│   ├── protein/                      #   Profeatx, DeepFRI, Zernike, Protparam, Peptides
│   ├── templates/                    #   input/output format samples
│   └── assemble_features.py          #   align descriptors to model feature columns
├── 0_data_preparation/               # download FASTA/PDB, convert SMILES/InChI, latent inference
├── 1_preprocessing/                  # auto-encoder pre-training (Step 1)
│   ├── data/                         #   protein/ligand feature tables (277 proteins, 1313 ligands)
│   └── encoder_save/                 #   pre-trained encoder artifacts
├── 2_DTI_module/                     # binding / non-binding model
└── 3_MoA_module/                     # activating / inhibitory model
```

## Environment

The model environment (**Step1**) is Python 3.8.20 with TensorFlow 2.13.1,
PyTorch 2.4.1, scikit-learn 1.3.2, pandas 2.0.3, xgboost / lightgbm / catboost,
rdkit, and the usual scientific stack. Every package was verified to come from
public indexes, so the environment is fully reproducible:

```bash
conda env create -f environment.yml      # conda-level dependencies
conda activate Step1
pip install -r requirements.txt          # official pip packages (CUDA wheels are large)
```

Notes

* `requirements.txt` contains the exact pip versions used in the paper. Five
  PyTorch-Geometric extension wheels come from the official PyG index
  (`data.pyg.org`), and `descriptastorus` is pinned to a public GitHub commit —
  both are listed with their URLs.
* The only non-official component is the custom-compiled `profeatx` executable,
  provided in `env_packages/` (see `env_packages/README.md`).
* The descriptor calculations use additional conda environments
  (`rdkit`, `FP-Zernike`, `PPDOCK3`, `DeepFRI`) — see
  `0_descriptor_calculation/environment_record.md`. `env_config/activate_env.sh`
  switches between them automatically.
* On Linux servers you may need the libstdc++ workaround, handled by
  `source env_config/activate_step1.sh` (also called automatically by the
  `run_*.sh` scripts).

## Pretrained models

Small artifacts are included in this repository (`1_preprocessing/encoder_save/`,
`2_DTI_module/preprocessor.joblib`, `training_metrics.json`,
`3_MoA_module/preprocessor.joblib`, `optimal_threshold.json`, ...).

The large weight files are distributed as GitHub Release assets:

| Release asset | Place it at |
|---------------|-------------|
| `DTI_my_model.h5` | `2_DTI_module/my_model.h5` |
| `MoA_my_model.h5` | `3_MoA_module/my_model.h5` |
| `protein_encoder.h5` | `1_preprocessing/encoder_save/protein_encoder.h5` |

Download page: <https://github.com/ChenHuangMUST/IC-MoDTI/releases/tag/models-v1>

Included model versions: DTI = full model `positive_negative_v601_retest_new_negative`
(Youden threshold 0.2999); MoA = `KCNH2_exp_adjusted_v2/moa_r2` (threshold 0.3813).

## Quick start (prediction, no training)

### 1. Input table

Create a CSV with the following columns (column names are fixed, values may be empty):

| Column | Required | Description | Example |
|--------|:---:|-------------|---------|
| `protein` | recommended | protein label (defaults to the UniProt ID) | `KCNQ1` |
| `uniprot_id` | required for protein rows | UniProt accession | `P51787` |
| `ligand` | recommended | ligand label (defaults to the first 14 chars of the InChIKey) | `ibuprofen` |
| `smiles` | one of smiles/inchi for ligand rows | ligand SMILES | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |
| `inchi` | one of smiles/inchi for ligand rows | ligand InChI | `InChI=1S/C13H18O2/...` |

```csv
protein,uniprot_id,ligand,smiles,inchi
KCNQ1,P51787,ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,
VEGFR2,P35968,,,
,,aspirin,,InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)
```

`prepare_inputs.py` automatically downloads FASTA sequences from UniProt and
PDB structures from AlphaFold DB, and converts SMILES/InChI to InChIKey.

### 2. Run

```bash
source env_config/activate_step1.sh

# full pipeline: input preparation → descriptors → features → prediction
bash run_all.sh my_inputs.csv

# skip descriptor calculation if all proteins/ligands are already in
# 1_preprocessing/data/*.csv
bash run_all.sh my_inputs.csv --skip-descriptor
```

Results are written next to the input table in `icmotdi_output/`:

| File | Content |
|------|---------|
| `dti_predictions.csv` | `protein, ligand, Binding_Probability, Binding_Prediction, Binding_Label` |
| `moa_predictions.csv` | `protein, ligand, Predicted_Probability, Predicted_Class, Predicted_Priority` |
| `combinations.csv`, `ligand_info.csv`, `prepare_report.txt` | parsed combinations, ligand identity table, download/conversion report |
| `fasta/`, `pdb/` | downloaded sequences and structures |

Existing feature tables in `1_preprocessing/data/` already cover **234 proteins
and 5,297 ligands** (the extended ligand table used by MoA contains 5,298,
including the target compound); if your molecules are in those tables you can
skip descriptor calculation entirely.

### 3. Direct module prediction

```bash
# DTI
cd 2_DTI_module
python positive_negative_external.py \
    -p ../1_preprocessing/data/protein_features_unique_260414.csv \
    -l ../1_preprocessing/data/ligand_feature_260703.csv \
    -g combinations.csv -o binding_predictions.csv

# MoA
cd ../3_MoA_module
python external_data_verification_v3_to_v23.py \
    -p ../1_preprocessing/data/protein_features_unique_260414.csv \
    -l ../1_preprocessing/data/ligand_feature_exp.csv \
    -g combinations.csv -o moa_predictions.csv
```

## Training (reproducing the paper)

```bash
# Step 1 — pre-train the protein/ligand auto-encoders (optional; artifacts included)
cd 1_preprocessing && bash run_pretrain.sh

# Step 2 — DTI (binding / non-binding)
cd 2_DTI_module && bash run_dti.sh

# Step 3 — MoA (activating / inhibitory)
cd 3_MoA_module && bash run_moa.sh
```

The MoA training script implements the three-stage fine-tuning described in the
paper (activity-weighted FT1, DSDP-weighted FT2, optional target-protein FT3 with
supervised contrastive loss) plus Youden-threshold optimization. See
`bash run_moa.sh --help` for all options (`--exp`, `--contrastive_weight`, ...).

### Descriptor calculation

```bash
bash run_pipeline.sh all        # all descriptors (environment switching is automatic)
bash run_pipeline.sh ligand     # E-state + MACCS
bash run_pipeline.sh protein    # Profeatx + DeepFRI + Zernike + Protparam + Peptides
```

| Descriptor | Target | Dim. | Script | Environment |
|------------|--------|-----:|--------|-------------|
| Profeatx | protein | 14338 | `protein/profeatx_calculator.py` (+ custom `profeatx` binary) | PPDOCK3 / Step1 |
| DeepFRI | protein | 489 | `protein/DeepFRI.py` | DeepFRI |
| Zernike | protein | 484 | `protein/Zernike_calculate.py` + `Zernike_merge.py` | FP-Zernike |
| Protparam | protein | 19 | `protein/2protparam_v3.py` / `2protparam_group_v3.py` | FP-Zernike |
| Peptides | protein | 64 | `protein/peptides_descriptors.py` (pure Python, validated against the R `Peptides` package) | Step1 |
| E-state | ligand | 158 | `ligand/Estate.py` | rdkit |
| MACCS | ligand | 166 | `ligand/MACCS.py` | rdkit |

After descriptor calculation, align everything to the model feature columns with
`0_descriptor_calculation/assemble_features.py` (see `--help`); column names match
`feature_columns.json` in the DTI/MoA modules.

## Data

All counts below were re-verified directly against the shipped files (the same
files used to train the bundled models):

* `1_preprocessing/data/protein_features_unique_260414.csv` — protein descriptors (**234 proteins**, 15,394 features)
* `1_preprocessing/data/ligand_feature_260703.csv` — ligand descriptors (**5,297 ligands**, 324 features)
* `1_preprocessing/data/ligand_feature_exp.csv` — extended ligand table (**5,298 ligands**, including the target compound used for MoA)
* `2_DTI_module/data/main_columns_True25_total_True_with_DSDP_26070501.csv` — main combination table (**4,415 pairs**; 4,413 of them with `DSDP_Score < -3`)
* `3_MoA_module/data/main_exp.csv` — MoA training file (**4,745 samples**)
* `2_DTI_module/data/new_negative_all_10625.csv` — full negative set (**10,625 pairs**, `-3 < DSDP_Score < 0`)
* Positive pairs are defined by `DSDP_Score < -3`.

## Reproduced metrics

All bundled-model values are taken directly from the shipped
`training_metrics.json` files and were verified to be identical to the original
training outputs on the authors' server. Both modules are reported with the same
metric set for the same evaluation protocol.

**5-fold cross-validation (both modules)**

| Module | AUC | Accuracy | AP |
|--------|----:|---------:|---:|
| DTI | 0.9982 ± 0.0011 | 0.9907 ± 0.0037 | 0.9742 ± 0.0102 |
| MoA | 0.9328 ± 0.0060 | 0.8725 ± 0.0109 | 0.8924 ± 0.0086 |

**Held-out / overall evaluation (both modules)**

| Module | ROC AUC | AP | Youden threshold |
|--------|--------:|---:|-----------------:|
| DTI | 0.9982 (overall) | 0.9960 (overall) | 0.2999 |
| MoA | 0.9652 (test) | 0.9435 (test) | 0.3813 |

The original project notes record paper values only for DTI 5-fold CV
(AUC 0.9893 ± 0.0062, Accuracy 0.9639 ± 0.0124, AP 0.9689 ± 0.0090) and MoA
test-set evaluation (AUC 0.9617, AP 0.9471); the remaining protocol/metric
combinations were not recorded there.

## FAQ

**`GLIBCXX_3.4.x not found` on Linux?**
Run `source env_config/activate_step1.sh` first (the `run_*.sh` scripts do this
automatically).

**A protein/ligand is not in the feature tables.**
Compute its descriptors (`run_pipeline.sh`), align them with
`assemble_features.py`, then either re-run pre-training or transform with the
existing scaler/encoder. New molecules need the encoder artifacts.

**Column mismatch after adding descriptors?**
All descriptors must match the dimensions listed above; use
`assemble_features.py --ref-dir 2_DTI_module` to force alignment.

**GPU out of memory during MoA training?**
Reduce `batch_size` in the training script or pin a single GPU with
`CUDA_VISIBLE_DEVICES=0`.

## Citation

If you use this code, please cite:

> **IC-MoDTI: A deep learning model for ion channel drug–target interaction prediction** (under review)

## License

GNU General Public License v2.0 — Copyright (C) 2026 Zelin Yi. See [LICENSE](LICENSE).
