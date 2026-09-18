#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC-MoDTI latent vector supplement tool (new protein/ligand -> autoencoder latent vector)

Background: DTI model features = raw descriptors + autoencoder latent vectors (z_prot_256 + z_lig_64).
encoder_utils.py looks up a pre-generated latent table; new proteins/ligands are not in the table,
so predictions return NaN. This tool runs inference on new molecules with the pretrained encoder
to generate/update the latent table.

Usage:
  # Mode A: pass assembled feature files, automatically detect new molecules and append them to the latent table
  python infer_latent.py --protein-features new_protein_features.csv \\
      --ligand-features new_ligand_features.csv \\
      --latent-dir 1_preprocessing/encoder_save        # auto-located by default

  # Mode B: supplement a single protein only
  python infer_latent.py --protein-features new_protein_features.csv

Notes:
  - Reads the latent table (prot_latent_table.pkl / lig_latent_table.pkl)
  - Uses encoder_utils.infer_protein_z / infer_ligand_z to infer IDs missing from the table
  - The updated table is saved in place (the original table is backed up as .bak.pkl)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2_DTI_module"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_preprocessing"))


def locate_latent_dir():
    """Locate encoder_save (1_preprocessing/encoder_save)"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("IC_MOTDI_ENCODER_DIR"),
        os.path.join(here, "..", "1_preprocessing", "encoder_save"),
        os.path.join(here, "encoder_save"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError("encoder_save directory not found; please set IC_MOTDI_ENCODER_DIR")


def infer_protein_latent(df, latent_dir):
    """Infer latent vectors for proteins missing from the latent table; return the DataFrame to append"""
    from encoder_utils import PROT_LATENT_TABLE, PROT_LATENT_DIM, infer_protein_z
    known = set(PROT_LATENT_TABLE["protein"])
    new_ids = [p for p in df["protein"].unique() if p not in known]
    if not new_ids:
        return None
    sub = df[df["protein"].isin(new_ids)].drop_duplicates("protein")
    z = infer_protein_z(sub)
    out = pd.DataFrame({"protein": new_ids, "z_protein": [v for v in z]})
    print(f"[OK] {len(new_ids)} new proteins, latent dim {PROT_LATENT_DIM}")
    return out


def infer_ligand_latent(df, latent_dir):
    """Infer latent vectors for ligands missing from the latent table"""
    from encoder_utils import LIG_LATENT_TABLE, LIG_LATENT_DIM, infer_ligand_z
    known = set(LIG_LATENT_TABLE["ligand"])
    new_ids = [l for l in df["ligand"].unique() if l not in known]
    if not new_ids:
        return None
    sub = df[df["ligand"].isin(new_ids)].drop_duplicates("ligand")
    z = infer_ligand_z(sub)
    out = pd.DataFrame({"ligand": new_ids, "z_ligand": [v for v in z]})
    print(f"[OK] {len(new_ids)} new ligands, latent dim {LIG_LATENT_DIM}")
    return out


def main():
    ap = argparse.ArgumentParser(description="latent vector supplement")
    ap.add_argument("--protein-features", help="protein feature file (protein + protein_* columns)")
    ap.add_argument("--ligand-features", help="ligand feature file (ligand + ligand_* columns)")
    ap.add_argument("--latent-dir", default=None, help="encoder_save directory (auto-located by default)")
    ap.add_argument("--no-backup", action="store_true", help="do not back up the original table")
    args = ap.parse_args()

    if not (args.protein_features or args.ligand_features):
        ap.error("provide at least --protein-features or --ligand-features")

    latent_dir = args.latent_dir or locate_latent_dir()
    print(f"encoder_save directory: {latent_dir}")

    # Append and save
    def save_table(kind, new_df):
        path = os.path.join(latent_dir, f"{kind}_latent_table.pkl")
        table = pd.read_pickle(path)
        if not args.no_backup:
            bak = path.replace(".pkl", ".bak.pkl")
            if not os.path.exists(bak):
                table.to_pickle(bak)
                print(f"  original table backed up: {bak}")
        table = pd.concat([table, new_df], ignore_index=True)
        table.to_pickle(path)
        print(f"  latent table updated: {path} ({len(table)} rows)")

    if args.protein_features:
        df = pd.read_csv(args.protein_features)
        new = infer_protein_latent(df, latent_dir)
        if new is not None:
            save_table("prot", new)
        else:
            print("[INFO] all proteins already in the latent table; nothing to add")

    if args.ligand_features:
        df = pd.read_csv(args.ligand_features)
        new = infer_ligand_latent(df, latent_dir)
        if new is not None:
            save_table("lig", new)
        else:
            print("[INFO] all ligands already in the latent table; nothing to add")

    print("\n[OK] Done. DTI/MoA training and prediction will automatically use the updated latent table.")


if __name__ == "__main__":
    main()
