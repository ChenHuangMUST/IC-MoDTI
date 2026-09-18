#!/usr/bin/env python3
"""
IC-MoDTI feature assembly tool
Assembles the CSVs produced by the descriptor calculation modules into the
protein_features_*.csv / ligand_feature_*.csv needed by the model (aligned with feature_columns.json).

Usage:
  python assemble_features.py --protein-dir descriptor_output_dir/protein \
                              --ligand-dir descriptor_output_dir/ligand \
                              --out-dir output_dir \
                              --ref-dir model_dir (containing feature_columns.json)

Column alignment rules (refer to feature_columns.json / already computed feature files):
  protein: protein_Profeatx_*  protein_DeepFRI_*  protein_Zernike_*
           protein_Peptides_*  protein_Protparam_*
  ligand: ligand_Estate_atom_*  ligand_Estate_sum_*  ligand_MACCS_*
"""

import os
import sys
import json
import argparse
import glob
import pandas as pd
import numpy as np

PREFIXES_PROTEIN = {
    "profeatx": "protein_profeatx",
    "deepfri": "protein_DeepFRI",
    "zernike": "protein_Zernike",
    "peptides": "protein_Peptides",
    "protparam": "protein_Protparam",
}
PREFIXES_LIGAND = {
    "estate_atom": "ligand_Estate_atom",
    "estate_sum": "ligand_Estate_sum",
    "maccs": "ligand_MACCS",
}


def load_any_csv(path):
    """Read a CSV possibly containing a BOM, trying multiple encodings"""
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin1")


def find_files(directory, patterns):
    """Find matching files under a directory (recursive)"""
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(directory, "**", pat), recursive=True))
    return hits


def load_protein_descriptors(protein_dir, ref_cols):
    """Load and assemble all protein descriptors"""
    need = [c for c in ref_cols if c.startswith("protein_") and not c.startswith("protein_profeatx")]
    # ---- File discovery rules for each descriptor ----
    candidates = {
        "deepfri": ["260313_features.csv", "*features*.csv", "*DeepFRI*.csv", "*deepfri*.csv"],
        "zernike": ["Zernike_all_features.csv", "*Zernike_all*.csv"],
        "peptides": ["Peptides_all.csv", "*peptides*.csv", "*Peptides*.csv"],
        "protparam": ["2protparam_all.csv", "*protparam*.csv", "*Protparam*.csv"],
    }
    frames = {}
    for key, pats in candidates.items():
        files = find_files(protein_dir, pats)
        if not files:
            print(f"[WARN] Protein descriptor [{key}]: no files found: {pats}")
            continue
        # Prefer the one with the most columns (most complete)
        best = None
        best_n = 0
        for f in files:
            try:
                df = load_any_csv(f)
                n = len(df.columns)
                if n > best_n:
                    best, best_n = df, n
            except Exception:
                continue
        if best is not None:
            frames[key] = best
            print(f"[OK] Protein descriptor [{key}]: {best_n} columns <- {os.path.basename(files[0])}")

    # profeatx is large, handle it separately (supports *_all.csv collections)
    profeatx_files = find_files(protein_dir, ["260318_profeatx_all.csv", "profeatx_all_combined.csv", "*profeatx*all*.csv"])
    if profeatx_files:
        best, best_n = None, 0
        for f in profeatx_files:
            try:
                df = load_any_csv(f)
                if len(df.columns) > best_n:
                    best, best_n = df, len(df.columns)
            except Exception:
                continue
        if best is not None:
            frames["profeatx"] = best
            print(f"[OK] Protein descriptor [profeatx]: {best_n} columns")

    # ---- Merge ----
    if not frames:
        print("[ERROR] No protein descriptor files found")
        return None

    merged = None
    for key, df in frames.items():
        df = df.copy()
        if "protein" not in df.columns:
            print(f"[WARN] {key} is missing the protein column, skipping")
            continue
        df = df.drop_duplicates(subset=["protein"])
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="protein", how="outer")

    if merged is None:
        return None
    print(f"[OK] Protein descriptors merged: {len(merged)} rows x {len(merged.columns)} columns")
    return merged


def load_ligand_descriptors(ligand_dir, ref_cols):
    """Load and assemble all ligand descriptors"""
    candidates = {
        "estate": ["*estate*.csv", "Estate*.csv", "*Estate*.csv"],
        "maccs": ["*maccs*.csv", "*MACCS*.csv", "*MACCS*online*.csv"],
    }
    frames = {}
    for key, pats in candidates.items():
        files = find_files(ligand_dir, pats)
        if not files:
            print(f"[WARN] Ligand descriptor [{key}]: no files found: {pats}")
            continue
        best, best_n = None, 0
        for f in files:
            try:
                df = load_any_csv(f)
                if len(df.columns) > best_n:
                    best, best_n = df, len(df.columns)
            except Exception:
                continue
        if best is not None:
            frames[key] = best
            print(f"[OK] Ligand descriptor [{key}]: {best_n} columns")

    if not frames:
        print("[ERROR] No ligand descriptor files found")
        return None

    merged = None
    for key, df in frames.items():
        df = df.copy()
        id_col = "ligand"
        if id_col not in df.columns:
            # Try UNII or another identifier
            for c in ["ligand", "UNII", "InChI.Key", "InChIKey"]:
                if c in df.columns:
                    id_col = c
                    break
        if id_col not in df.columns:
            print(f"[WARN] {key} is missing the identifier column, skipping")
            continue
        df = df.rename(columns={id_col: "ligand"})
        df = df.drop_duplicates(subset=["ligand"])
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="ligand", how="outer")

    print(f"[OK] Ligand descriptors merged: {len(merged)} rows x {len(merged.columns)} columns")
    return merged


def ensure_feature_columns(df, ref_cols, prefix_map, kind):
    """Ensure output columns align with ref_cols: fill missing columns with NaN, drop extra columns"""
    kept = [c for c in ref_cols if c.startswith(tuple(prefix_map.values()))]
    for c in kept:
        if c not in df.columns:
            df[c] = np.nan
    df = df[["protein" if kind == "protein" else "ligand"] + kept]
    return df


def main():
    ap = argparse.ArgumentParser(description="Descriptor feature assembly")
    ap.add_argument("--protein-dir", required=True, help="Protein descriptor output directory")
    ap.add_argument("--ligand-dir", required=True, help="Ligand descriptor output directory")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--ref-dir", required=True, help="Model directory (containing feature_columns.json)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read the reference feature columns (either DTI or MoA)
    ref_json = None
    for d in [args.ref_dir]:
        for f in glob.glob(os.path.join(d, "feature_columns.json")):
            ref_json = f
    if not ref_json:
        print("[ERROR] feature_columns.json not found")
        sys.exit(1)
    with open(ref_json, encoding="utf-8") as f:
        ref_cols = json.load(f)
    print(f"[OK] Reference feature columns: {len(ref_cols)}")

    # Assemble protein
    prot = load_protein_descriptors(args.protein_dir, ref_cols)
    if prot is not None:
        prot = ensure_feature_columns(prot, ref_cols, PREFIXES_PROTEIN, "protein")
        out = os.path.join(args.out_dir, "protein_features_assembled.csv")
        prot.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] Protein feature output: {out} ({len(prot)} x {len(prot.columns)})")

    # Assemble ligand
    lig = load_ligand_descriptors(args.ligand_dir, ref_cols)
    if lig is not None:
        lig = ensure_feature_columns(lig, ref_cols, PREFIXES_LIGAND, "ligand")
        out = os.path.join(args.out_dir, "ligand_feature_assembled.csv")
        lig.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] Ligand feature output: {out} ({len(lig)} x {len(lig.columns)})")

    print("\n[OK] Assembly complete. Use 1_preprocessing/encoder_save in the model directory to connect with the training script.")


if __name__ == "__main__":
    main()
