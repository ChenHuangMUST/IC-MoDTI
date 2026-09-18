import os
import subprocess
import csv
import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Batch process FASTA files using 2protparam_v3.py.')
    parser.add_argument('-i', '--input_dir', required=True, help='Directory containing FASTA files')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file to combine all results')
    args = parser.parse_args()

    # Retrieve the user-provided arguments
    input_dir = args.input_dir
    output_file = args.output

    # Get all files with the .fasta extension in the input directory
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith('.fasta')]

    # Check whether the directory contains any FASTA files
    if not fasta_files:
        print(f"No .fasta files found in directory '{input_dir}'.")
        return

    print(f"Found {len(fasta_files)} FASTA files in directory '{input_dir}'. Starting batch processing...")

    # Create the combined output CSV file for merging all results
    with open(output_file, 'w', newline='') as combined_csv:
        # Define the CSV header; it must exactly match the output columns of 2protparam_v3.py
        fieldnames = [
            'protein',
            'protein_Protparam_Molecular_weight',
            'protein_Protparam_Atomic_composition_C',
            'protein_Protparam_Atomic_composition_H',
            'protein_Protparam_Atomic_composition_N',
            'protein_Protparam_Atomic_composition_O',
            'protein_Protparam_Atomic_composition_S',
            'protein_Protparam_Atomic_composition_total',
            'protein_Protparam_OC_Ext_coefficient',
            'protein_Protparam_OC_Abs',
            'protein_Protparam_RC_Ext_coefficient',
            'protein_Protparam_RC_Abs',
            'protein_Protparam_Instability_index',
            'protein_Protparam_Aliphatic_index',
            'protein_Protparam_GRAVY',
            'protein_Protparam_Analysis_Aromaticity',
            'protein_Protparam_Analysis_SSF_1',
            'protein_Protparam_Analysis_SSF_2',
            'protein_Protparam_Analysis_SSF_3',
            'protein_Protparam_Analysis_Isoelectric_point'
        ]
        writer = csv.DictWriter(combined_csv, fieldnames=fieldnames)
        writer.writeheader()  # Write the header

        # Iterate over and process each FASTA file
        for index, fasta_file in enumerate(fasta_files, 1):
            fasta_path = os.path.join(input_dir, fasta_file)  # Build the full file path
            temp_output = os.path.join(input_dir, f"temp_{fasta_file}.csv")  # Create a temporary output file for each file

            print(f"Processing file ({index}/{len(fasta_files)}): {fasta_file}")

            try:
                # Call the modified single-analysis script 2protparam_v3.py
                # This assumes 2protparam_v3.py is in the same directory as the current script
                subprocess.run([
                    'python', '2protparam_v3.py',
                    '-i', fasta_path,
                    '-o', temp_output
                ], check=True)  # check=True ensures the subprocess succeeds, otherwise an exception is raised

                # Read the analysis results of the single file (temporary CSV file)
                with open(temp_output, 'r') as temp_csv:
                    reader = csv.DictReader(temp_csv)
                    for row in reader:
                        # Write each row of results to the combined output file
                        writer.writerow(row)

                # Delete the temporary file to keep the directory clean
                os.remove(temp_output)
                print(f"  Done: {fasta_file}")

            except subprocess.CalledProcessError as e:
                # If the external script call fails, print the error but continue processing other files
                print(f"  Error processing file {fasta_file}: {e}")
                # If the temporary file was created, try to delete it
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            except FileNotFoundError as e:
                # If the 2protparam_v3.py script cannot be found
                print(f"  Error: analysis script '2protparam_v3.py' not found. Make sure it is in the same directory as this script.")
                break
            except Exception as e:
                # Catch other unknown errors
                print(f"  Unknown error while processing file {fasta_file}: {e}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)

    print(f"\nBatch processing complete! All results merged and saved to: {output_file}")

# Program entry point
if __name__ == '__main__':
    main()