#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC-MoDTI input data preparation tool
====================================
The user only needs to provide:
  1. Protein: UniProt ID (optional custom name)
  2. Ligand: SMILES or InChI (optional custom name)
  3. (Optional) combination relationships

This tool automatically:
  - Protein: download FASTA sequences from UniProt
  - Protein: download PDB structures from AlphaFold (auto-adapts version v1~v6+, based on the API response)
  - Ligand: SMILES/InChI -> InChI + InChIKey (computed with rdkit; InChIKey is used as the model identifier)
  - Generate the combination file (protein, ligand columns, model input format)

Usage:
  python prepare_inputs.py --input input_table.csv --out output_dir
  python prepare_inputs.py --proteins protein_table.csv --ligands ligand_table.csv --combinations combination_table.csv --out output_dir

Input table format (--input; the three kinds of information may be in one table or split across tables):

  protein,uniprot_id,ligand,smiles,inchi
  KCNQ1,P51787,ligandA,CC(=O)Nc1ccc(N)cc1,
  ,,ligandB,,InChI=1S/C5H6N2/c6-5-1-3-7-4-2-5/h1-4H,(H2,6,7)

  - protein:   custom protein name (uses uniprot_id if empty)
  - uniprot_id: UniProt protein ID (e.g. P51787)
  - ligand:    custom ligand name (uses the first 14 characters of the generated InChIKey if empty)
  - smiles:    ligand SMILES (provide at least one of smiles/inchi)
  - inchi:     ligand InChI

Output:
  <out>/fasta/<protein>.fasta          # sequences downloaded from UniProt
  <out>/pdb/<protein>.pdb              # structures downloaded from AlphaFold (version auto-adapted)
  <out>/ligand_info.csv                # ligand, smiles, inchi, inchikey lookup table
  <out>/combinations.csv               # protein, ligand combinations (model input)
  <out>/prepare_report.txt             # download/conversion result summary
"""

import os
import sys
import csv
import json
import time
import argparse
import urllib.request
import urllib.error
import ssl

# ------------------------------------------------------------
# Network download (with SSL compatibility)
# ------------------------------------------------------------
_CTX = ssl.create_default_context()
_CTX_INSECURE = ssl._create_unverified_context()


def http_get(url, timeout=60, retries=3, insecure_fallback=True):
    """Download text/bytes with retry + SSL fallback"""
    last_err = None
    for attempt in range(retries):
        for ctx in ([_CTX, _CTX_INSECURE] if insecure_fallback else [_CTX]):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "IC-MoDTI/1.0"})
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    return resp.read()
            except Exception as e:
                last_err = e
                continue
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
    raise last_err


# ------------------------------------------------------------
# Protein: UniProt FASTA download
# ------------------------------------------------------------
def _validate_uniprot_id(uid):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", uid or ""):
        raise ValueError(f"invalid UniProt ID: {uid!r}")


def download_uniprot_fasta(uniprot_id, out_path, timeout=60):
    """Download UniProt FASTA (https://rest.uniprot.org/uniprotkb/<ID>.fasta)"""
    _validate_uniprot_id(uniprot_id)
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    data = http_get(url, timeout=timeout)
    with open(out_path, "wb") as f:
        f.write(data)
    # Verify the content is FASTA (guards against 404 pages)
    head = data[:500].decode("utf-8", errors="replace").strip()
    if not head.startswith(">"):
        os.remove(out_path)
        raise ValueError(f"UniProt returned non-FASTA content: {head[:80]}")
    return out_path


# ------------------------------------------------------------
# Protein: AlphaFold PDB download (version auto-adapt)
# ------------------------------------------------------------
def _af_api_pdb_url(uniprot_id):
    """Method 1: query the AlphaFold API for the currently valid PDB download link"""
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    data = http_get(url, timeout=60)
    entries = json.loads(data.decode("utf-8", errors="replace"))
    if not entries:
        return None
    pdb_url = entries[0].get("pdbUrl")
    return pdb_url


def _af_version_fallback(uniprot_id):
    """Method 2: try known/guessed versions in order (v6 -> v1), take the first downloadable one"""
    url_template = "https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_{ver}.pdb"
    # Try common versions first; extend this list if the server is updated
    for ver in ("v6", "v5", "v4", "v3", "v2", "v1"):
        url = url_template.format(id=uniprot_id, ver=ver)
        try:
            data = http_get(url, timeout=30, retries=1)
            head = data[:200].decode("utf-8", errors="replace")
            if head.startswith(("ATOM", "HEADER", "REMARK", "MODEL")):
                return url, data
        except Exception:
            continue
    return None, None


def download_alphafold_pdb(uniprot_id, out_path, timeout=60):
    """
    Download an AlphaFold PDB structure and return the URL actually used.
    Strategy: prefer the API (automatically gets the server's current version); fall back to version probing on failure.
    """
    _validate_uniprot_id(uniprot_id)
    url_used = None
    # 1) API approach
    try:
        pdb_url = _af_api_pdb_url(uniprot_id)
        if pdb_url:
            data = http_get(pdb_url, timeout=timeout)
            with open(out_path, "wb") as f:
                f.write(data)
            url_used = pdb_url
            print(f"    [AlphaFold API] download succeeded: {pdb_url}")
            return url_used
    except Exception as e:
        print(f"    [AlphaFold API] failed ({e}), trying version fallback...")

    # 2) Version fallback
    url, data = _af_version_fallback(uniprot_id)
    if data is None:
        raise ValueError(f"could not download the PDB for {uniprot_id} from AlphaFold (both API and v1~v6 failed)")
    with open(out_path, "wb") as f:
        f.write(data)
    url_used = url
    print(f"    [AlphaFold version fallback] download succeeded: {url}")
    return url_used


# ------------------------------------------------------------
# Ligand: SMILES/InChI -> InChI + InChIKey (rdkit)
# ------------------------------------------------------------
def _rdkit_convert(smiles=None, inchi=None):
    """Convert SMILES or InChI to (inchi, inchikey) with rdkit"""
    from rdkit import Chem

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"cannot parse SMILES: {smiles}")
        inchi = Chem.MolToInchi(mol)
    elif inchi:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            raise ValueError(f"cannot parse InChI: {inchi}")
    else:
        raise ValueError("at least one of smiles or inchi must be provided")

    inchikey = Chem.MolToInchiKey(mol)
    return inchi, inchikey


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------
def read_input_table(args):
    """Read the input table and normalize it into a unified structure"""
    rows = []  # dict: protein, uniprot_id, ligand, smiles, inchi

    if args.input:
        with open(args.input, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "protein": (r.get("protein") or "").strip(),
                    "uniprot_id": (r.get("uniprot_id") or r.get("uniprot") or "").strip(),
                    "ligand": (r.get("ligand") or "").strip(),
                    "smiles": (r.get("smiles") or "").strip(),
                    "inchi": (r.get("inchi") or "").strip(),
                })
    else:
        def load(path):
            out = []
            with open(path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    out.append(r)
            return out

        if args.proteins:
            for r in load(args.proteins):
                rows.append({
                    "protein": (r.get("protein") or "").strip(),
                    "uniprot_id": (r.get("uniprot_id") or r.get("uniprot") or "").strip(),
                    "ligand": "", "smiles": "", "inchi": "",
                })
        if args.ligands:
            for r in load(args.ligands):
                rows.append({
                    "protein": "", "uniprot_id": "",
                    "ligand": (r.get("ligand") or "").strip(),
                    "smiles": (r.get("smiles") or "").strip(),
                    "inchi": (r.get("inchi") or "").strip(),
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description="IC-MoDTI input data preparation")
    ap.add_argument("--input", help="mixed input table (protein/uniprot_id/ligand/smiles/inchi)")
    ap.add_argument("--proteins", help="protein table (protein, uniprot_id)")
    ap.add_argument("--ligands", help="ligand table (ligand, smiles or inchi)")
    ap.add_argument("--combinations", help="combination table (protein, ligand)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-pdb", action="store_true", help="skip PDB download (sequences only)")
    args = ap.parse_args()

    if not (args.input or (args.proteins or args.ligands)):
        ap.error("provide --input or --proteins/--ligands")
    if args.input and (args.proteins or args.ligands):
        ap.error("use either --input or --proteins/--ligands, not both")

    os.makedirs(os.path.join(args.out, "fasta"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "pdb"), exist_ok=True)

    rows = read_input_table(args)
    report = []
    proteins = {}   # protein_name -> uniprot_id
    ligands = {}    # ligand_name -> dict(smiles, inchi, inchikey)
    combos = []     # (protein, ligand)

    # ===== Protein processing =====
    print("=" * 60)
    print("Step 1: protein FASTA/PDB download (UniProt + AlphaFold version auto-adapt)")
    print("=" * 60)
    for r in rows:
        uid = r["uniprot_id"]
        if not uid:
            continue
        pname = r["protein"] or uid
        if pname in proteins:
            continue
        proteins[pname] = uid
        print(f"  [{pname}] <- UniProt {uid}")
        # FASTA
        try:
            fa = os.path.join(args.out, "fasta", f"{pname}.fasta")
            download_uniprot_fasta(uid, fa)
            print(f"    [OK] FASTA: {fa}")
            report.append(f"FASTA OK   {pname}\t{uid}")
        except Exception as e:
            print(f"    [FAIL] FASTA: {e}")
            report.append(f"FASTA FAIL {pname}\t{uid}\t{e}")
        # PDB
        if not args.no_pdb:
            try:
                pd = os.path.join(args.out, "pdb", f"{pname}.pdb")
                url = download_alphafold_pdb(uid, pd)
                report.append(f"PDB OK     {pname}\t{uid}\t{url}")
            except Exception as e:
                print(f"    [FAIL] PDB: {e}")
                report.append(f"PDB FAIL   {pname}\t{uid}\t{e}")
        time.sleep(0.3)  # polite interval

    # ===== Ligand processing =====
    print("=" * 60)
    print("Step 2: ligand SMILES/InChI -> InChI + InChIKey (rdkit)")
    print("=" * 60)
    try:
        from rdkit import Chem
        has_rdkit = True
    except ImportError:
        has_rdkit = False
        print("  [WARN] rdkit not found; cannot convert to InChIKey; run conda activate rdkit first")

    for r in rows:
        if not (r["smiles"] or r["inchi"]):
            continue
        lname = r["ligand"]
        try:
            inchi, inchikey = _rdkit_convert(r["smiles"], r["inchi"])
            if not lname:
                lname = inchikey[:14]
            ligands[lname] = {"smiles": r["smiles"], "inchi": inchi, "inchikey": inchikey}
            print(f"  [{lname}] InChIKey = {inchikey}")
            report.append(f"LIGAND OK  {lname}\t{inchikey}")
        except Exception as e:
            print(f"  [FAIL] ligand {lname or r['smiles'][:20]}: {e}")
            report.append(f"LIGAND FAIL {lname or '?'}\t{e}")

    # ===== Combination file =====
    print("=" * 60)
    print("Step 3: generate the combination file")
    print("=" * 60)
    if args.combinations:
        with open(args.combinations, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                combos.append((row["protein"].strip(), row["ligand"].strip()))
    else:
        # Infer from protein+ligand on the same row of the input table; if none, generate all combinations
        for r in rows:
            if r["protein"] and r["ligand"]:
                combos.append((r["protein"], r["ligand"]))
        if not combos and proteins and ligands:
            combos = [(p, l) for p in proteins for l in ligands]
            print(f"  automatically generated all combinations: {len(combos)}")

    combo_path = os.path.join(args.out, "combinations.csv")
    with open(combo_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["protein", "ligand"])
        for p, l in combos:
            w.writerow([p, l])
    print(f"  combination file: {combo_path} ({len(combos)} rows)")

    # ===== Ligand info table =====
    lig_path = os.path.join(args.out, "ligand_info.csv")
    with open(lig_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ligand", "smiles", "inchi", "inchikey"])
        for lname, info in ligands.items():
            w.writerow([lname, info["smiles"], info["inchi"], info["inchikey"]])
    print(f"  ligand info table: {lig_path} ({len(ligands)} rows)")

    # ===== Summary report =====
    rep_path = os.path.join(args.out, "prepare_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("IC-MoDTI Input Preparation Report\n" + "=" * 50 + "\n")
        f.write("\n".join(report) + "\n")
        f.write(f"\nproteins: {len(proteins)} | ligands: {len(ligands)} | combinations: {len(combos)}\n")
    print(f"\nSummary report: {rep_path}")
    print("=" * 60)
    print(f"proteins: {len(proteins)} | ligands: {len(ligands)} | combinations: {len(combos)}")
    print("Done! Next: bash run_pipeline.sh all or predict directly (see README.md)")


if __name__ == "__main__":
    main()
