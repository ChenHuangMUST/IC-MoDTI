#!/usr/bin/env python3
"""
Pure Python implementation of Peptides descriptors (replacement for the R Peptides package)

Principle: all sequence descriptors in the Peptides R package (tScales/stScales/mswhimScores/
kideraFactors/fasgaiVectors/blosumIndices/protFP/vhseScales/zScales)
are "sequence means of amino acid coefficient tables":
    descriptor = sum(scale[aa] for aa in seq) / len(seq)
instaIndex is a weighted sum over the dipeptide coefficient table (Guruprasad 1990).

The coefficient tables come from the Peptides 2.4.6 package data/AAdata.RData (exported to CSV,
see peptides_tables/ in this directory), and correspond exactly to the 64
protein_Peptides_* columns in feature_columns.json of this thesis.

Usage:
  python peptides_descriptors.py -i fasta_dir -o output.csv
  python peptides_descriptors.py -f sequence_file.fasta -o output.csv
  python peptides_descriptors.py --check          # self-check: compare with R example values
"""

import os
import sys
import glob
import argparse
import pandas as pd

TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peptides_tables")

# ============ Coefficient table loading ============
_tables = {}

def _load_table(name):
    """Load a 20AA x N scales coefficient table, returning {scale: {aa: value}}"""
    if name in _tables:
        return _tables[name]
    path = os.path.join(TABLES_DIR, f"AAdata_{name}.csv")
    df = pd.read_csv(path)
    aa_col = df.columns[-1]
    df = df.set_index(aa_col)
    scales = {}
    for col in df.columns:
        scales[col] = df[col].to_dict()
    _tables[name] = scales
    return scales

def _load_all():
    return {
        "tScales": _load_table("tScales"),
        "stScales": _load_table("stScales"),
        "mswhimScores": _load_table("MSWHIM"),
        "kideraFactors": _load_table("kideraFactors"),
        "fasgaiVectors": _load_table("FASGAI"),
        "blosumIndices": _load_table("BLOSUM"),
        "protFP": _load_table("ProtFP"),
        "vhseScales": _load_table("VHSE"),
        "zScales": _load_table("zScales"),
    }

def _clean_seq(seq):
    """Remove whitespace/newlines, convert to uppercase, keep the 20 standard amino acids + '-'"""
    seq = "".join(seq.split()).upper()
    return seq

# ============ Individual descriptor calculations ============
def _scale_mean(seq, scales):
    """Average the coefficient values over each amino acid in the sequence (matches the R implementation; na.rm=TRUE is equivalent to skipping unknown residues)"""
    seq = _clean_seq(seq)
    n = len(seq)
    if n == 0:
        return [float("nan")] * len(scales)
    results = []
    for scale in scales:
        vals = [scales[scale].get(aa, 0.0) for aa in seq]
        results.append(sum(vals) / n)
    return results

def tScales(seq):
    return _scale_mean(seq, _load_all()["tScales"])

def stScales(seq):
    return _scale_mean(seq, _load_all()["stScales"])

def mswhimScores(seq):
    return _scale_mean(seq, _load_all()["mswhimScores"])

def kideraFactors(seq):
    return _scale_mean(seq, _load_all()["kideraFactors"])

def fasgaiVectors(seq):
    return _scale_mean(seq, _load_all()["fasgaiVectors"])

def blosumIndices(seq):
    return _scale_mean(seq, _load_all()["blosumIndices"])

def protFP(seq):
    return _scale_mean(seq, _load_all()["protFP"])

def vhseScales(seq):
    return _scale_mean(seq, _load_all()["vhseScales"])

def zScales(seq):
    return _scale_mean(seq, _load_all()["zScales"])

# Guruprasad dipeptide instability coefficient table (instaIndex, matching the R package, extracted from the R source)
import json as _json
_GURU_FILE = os.path.join(TABLES_DIR, "guruprasad.json")
if os.path.exists(_GURU_FILE):
    with open(_GURU_FILE, "r", encoding="utf-8") as _f:
        GURUPRASAD = _json.load(_f)
else:
    raise FileNotFoundError(f"Missing dipeptide coefficient table: {_GURU_FILE}")

def instaIndex(seq):
    """Instability index: Guruprasad (1990), (10/L) * sum(dipeptide weights), matching the R package"""
    seq = _clean_seq(seq)
    n = len(seq)
    if n < 2:
        return float("nan")
    total = 0.0
    for i in range(n - 1):
        total += GURUPRASAD.get(seq[i:i+2], 1.0)
    return (10.0 / n) * total

# ============ Full 64-column output ============
def compute_all(seq):
    """Return a dict consistent with protein_Peptides_* in feature_columns.json"""
    out = {}
    # tScales: T1-T5
    vals = tScales(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_tScales_T{i}"] = v
    # stScales: ST1-ST8
    vals = stScales(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_stScales_ST{i}"] = v
    # mswhimScores: MSWHIM1-3
    vals = mswhimScores(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_mswhimScores_MSWHIM{i}"] = v
    # kideraFactors: KF1-KF10
    vals = kideraFactors(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_kideraFactors_KF{i}"] = v
    # fasgaiVectors: F1-F6
    vals = fasgaiVectors(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_fasgaiVectors_F{i}"] = v
    # blosumIndices: BLOSUM1-10
    vals = blosumIndices(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_blosumIndices_BLOSUM{i}"] = v
    # protFP: ProtFP1-8
    vals = protFP(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_protFP_ProtFP{i}"] = v
    # vhseScales: VHSE1-8
    vals = vhseScales(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_vhseScales_VHSE{i}"] = v
    # zScales: Z1-Z5
    vals = zScales(seq)
    for i, v in enumerate(vals, 1):
        out[f"protein_Peptides_zScales_Z{i}"] = v
    # instaIndex
    out["protein_Peptides_instaIndex"] = instaIndex(seq)
    return out

# ============ FASTA parsing ============
def parse_fasta(path):
    """Parse a FASTA file, returning [(id, seq)]
    Note: the protein ID is taken from the file name first (consistent with the custom names in prepare_inputs),
    avoiding downstream merge failures caused by UniProt headers (>sp|P51787|KCNQ1_HUMAN...).
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        cur_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_seq:
                    records.append((cur_seq, "".join(cur_seq)))
                    cur_seq = []
                # Only record the sequence; the ID will be taken from the file name later
                continue
            cur_seq.append(line)
        if cur_seq:
            records.append((None, "".join(cur_seq)))
    # Use the file name as the protein ID
    fname = os.path.splitext(os.path.basename(path))[0]
    return [(fname, seq) for _, seq in records]


def main():
    ap = argparse.ArgumentParser(description="Pure Python implementation of Peptides descriptors (replacement for R)")
    ap.add_argument("-i", "--input", help="FASTA file or directory containing fasta files")
    ap.add_argument("-o", "--output", default="Peptides_all.csv", help="Output CSV")
    ap.add_argument("--seq", help="Input a sequence directly for testing")
    ap.add_argument("--check", action="store_true", help="Self-check: compare with example values from the R package documentation")
    args = ap.parse_args()

    if args.check:
        # R documentation example: QWGRRCCGWGPGRRYCVRWC
        seq = "QWGRRCCGWGPGRRYCVRWC"
        print("=== Self-check: compare with examples in the R Peptides package documentation ===")
        print(f"Sequence: {seq}")
        exp = {
            "tScales": [-3.2700, -0.0035, -0.3855, -0.1475, 0.7585],
            "stScales": [-0.63760, 0.07965, 0.05150, 0.07135, -0.27905, -0.80995, 0.58020, 0.54400],
            "zScales": [0.6200, 0.0865, 0.0665, 0.7280, -0.8740],
            "instaIndex": 83.68,
        }
        vals_t = tScales(seq)
        print("tScales    :", [round(v, 5) for v in vals_t], "Expected:", exp["tScales"])
        vals_s = stScales(seq)
        print("stScales   :", [round(v, 5) for v in vals_s], "Expected:", exp["stScales"])
        vals_z = zScales(seq)
        print("zScales    :", [round(v, 5) for v in vals_z], "Expected:", exp["zScales"])
        ii = instaIndex(seq)
        print("instaIndex :", round(ii, 2), "Expected:", exp["instaIndex"])
        return

    if args.seq:
        row = compute_all(args.seq)
        df = pd.DataFrame([row])
        df.insert(0, "protein", "test")
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Saved: {args.output} ({len(df.columns)} columns)")
        return

    if not args.input:
        ap.error("requires -i input, --seq, or --check")

    # Collect all fasta files
    if os.path.isdir(args.input):
        fastas = sorted(glob.glob(os.path.join(args.input, "*.fasta")))
    else:
        fastas = [args.input]

    if not fastas:
        print("[ERROR] No fasta files found")
        sys.exit(1)

    rows = []
    for fp in fastas:
        for pid, seq in parse_fasta(fp):
            row = compute_all(seq)
            row["protein"] = pid
            rows.append(row)
        print(f"[OK] {os.path.basename(fp)}: {len(parse_fasta(fp))} sequences")

    df = pd.DataFrame(rows)
    cols = ["protein"] + [c for c in df.columns if c != "protein"]
    df = df[cols]
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} sequences in total, {len(df.columns) - 1} descriptors, output: {args.output}")


if __name__ == "__main__":
    main()
