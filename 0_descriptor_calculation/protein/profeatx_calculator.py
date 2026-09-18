# profeatx_calculator.py
import os
import subprocess
import pandas as pd
from tqdm import tqdm
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run profeatx to compute protein feature encodings (calculation stage)')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the input FASTA folder.')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to the output folder for intermediate result files.')
    parser.add_argument('-t', '--types', nargs='+',
                        help='List of feature types to compute, separated by spaces. Example: -t AAC PAAC DPC')
    parser.add_argument('--id-source', choices=['filename', 'fasta_header'], default='filename',
                        help='Protein ID source: filename (default, uses the fasta file name) or fasta_header (uses the UniProt ID in the > header)')
    return parser.parse_args()

def main():
    args = parse_arguments()

    fasta_folder = args.input
    output_folder = args.output

    # Locate the compiled profeatx program automatically: env var first, then current/parent directory
    import shutil
    profeatx_bin = os.environ.get("PROFEATX_BIN", "")
    if not profeatx_bin:
        for cand in ["./profeatx", "profeatx",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "profeatx")]:
            if os.path.exists(cand):
                profeatx_bin = cand
                break
    if not profeatx_bin or not os.path.exists(profeatx_bin):
        print("[ERROR] Compiled profeatx program not found. Set the PROFEATX_BIN environment variable to the profeatx executable.")
        sys.exit(1)
    print(f"Using profeatx: {profeatx_bin}")

    all_types = [
        "AAC", "AAPAS", "APAAC", "CKSAAGP", "CKSAAP", "CMV", "CT", "CTDC", "CTDT", "CTDD",
        "DDE", "DPC", "EBGW", "GAAC", "GDPC", "Geary", "GTPC", "KSCT", "Moran", "NMB",
        "PAAC", "QSO", "SOCN", "TPC"
    ]

    if args.types:
        invalid_types = []
        for t in args.types:
            if t not in all_types:
                invalid_types.append(t)
        if invalid_types:
            print(f"Error: invalid types: {', '.join(invalid_types)}")
            print(f"Valid types are: {', '.join(all_types)}")
            sys.exit(1)
        types_to_calculate = args.types
        print(f"User specified {len(types_to_calculate)} types: {', '.join(types_to_calculate)}")
    else:
        types_to_calculate = all_types
        print(f"No types specified; computing all {len(all_types)} types by default")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fasta_files = [f for f in os.listdir(fasta_folder) if f.endswith(".fasta")]
    total_fasta_files = len(fasta_files)
    print(f"{total_fasta_files} FASTA files in total to process")
    print(f"Will compute {len(types_to_calculate)} encoding types\n")

    for encoding_type in types_to_calculate:
        print(f"Processing type: {encoding_type}")
        all_results = None
        processed_count = 0
        error_count = 0

        pbar = tqdm(fasta_files, desc=f"Processing {encoding_type}", unit="files")
        for fasta_file in pbar:
            fasta_path = os.path.join(fasta_folder, fasta_file)
            protein_id = os.path.splitext(fasta_file)[0]
            output_file = os.path.join(output_folder, f"{protein_id}_{encoding_type}.csv")

            command = f"{profeatx_bin} -i {fasta_path} -o {output_file} -e {encoding_type}"
            try:
                subprocess.run(command, shell=True, check=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processed_count += 1
            except subprocess.CalledProcessError:
                error_count += 1
                continue
            except Exception:
                error_count += 1
                continue

            if not os.path.exists(output_file):
                error_count += 1
                continue
            try:
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                if len(lines) < 2:
                    error_count += 1
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    continue
            except IOError:
                error_count += 1
                if os.path.exists(output_file):
                    os.remove(output_file)
                continue

            if all_results is None:
                colnames = lines[0].strip().split("\t")
                all_results = pd.DataFrame(columns=colnames)
                data_line = lines[1].strip().split("\t")
            else:
                data_line = lines[1].strip().split("\t")

            name_parts = data_line[0].split("|")
            # Support fasta files downloaded from UniProt (headers like >sp|P35968|VGFR2_HUMAN...):
            # By default use the fasta file name as the protein ID (consistent with prepare_inputs custom names),
            # so that protein names stay consistent when merging downstream.
            if args.id_source == "filename":
                name = protein_id
            elif len(name_parts) >= 2:
                name = name_parts[1]
            else:
                name = data_line[0]

            data_row = [name] + data_line[1:]
            new_row = pd.DataFrame([data_row], columns=colnames)
            all_results = pd.concat([all_results, new_row], ignore_index=True)

            if os.path.exists(output_file):
                os.remove(output_file)

        pbar.close()
        success_count = processed_count - error_count
        print(f"Type {encoding_type} done: {success_count} succeeded, {error_count} failed")

        if all_results is not None and not all_results.empty:
            column_mapping = {}
            if len(all_results.columns) > 0:
                column_mapping[all_results.columns[0]] = 'protein'
            for i, col in enumerate(all_results.columns[1:], 1):
                column_mapping[col] = f'protein_profeatx_{encoding_type}_{col}'
            all_results.rename(columns=column_mapping, inplace=True)

            type_output_file = os.path.join(output_folder, f"{encoding_type}_all.csv")
            all_results.to_csv(type_output_file, index=False)
            print(f"  Individual file saved: {type_output_file}")
        else:
            print(f"  Warning: type {encoding_type} produced no valid results; skipping save.")
        print()

    print("All feature calculations complete. Results for each type were saved as separate *_all.csv files.")
    print(f"Use the merge module to process the folder: {output_folder}")

if __name__ == '__main__':
    main()