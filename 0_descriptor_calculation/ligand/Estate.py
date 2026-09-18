import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter
import requests
import argparse
from typing import Optional
import sys
import time

def get_inchi_from_inchikey(inchikey: str, max_retries: int = 3) -> Optional[str]:
    """
    Retrieve an InChI string from an InChI Key via the PubChem API
    
    Parameters:
    inchikey: InChI Key string
    max_retries: maximum number of retries
    
    Returns:
    InChI string, or None if retrieval fails
    """
    if pd.isna(inchikey) or not isinstance(inchikey, str) or not inchikey.strip():
        return None
    
    inchikey = inchikey.strip()
    
    for attempt in range(max_retries):
        try:
            # PubChem REST API: get the CID from the InChI Key
            pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
            response = requests.get(pubchem_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                cid = data.get('IdentifierList', {}).get('CID', [None])[0]
                
                if cid:
                    # Get the InChI from the CID
                    inchi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChI/JSON"
                    inchi_response = requests.get(inchi_url, timeout=10)
                    
                    if inchi_response.status_code == 200:
                        inchi_data = inchi_response.json()
                        inchi = inchi_data.get('PropertyTable', {}).get('Properties', [{}])[0].get('InChI')
                        return inchi
                    
            elif response.status_code == 503:  # service temporarily unavailable
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    print(f"API request failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            else:
                print(f"Error retrieving InChI from PubChem (InChI Key: {inchikey}): HTTP status code {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Timeout retrieving InChI from PubChem (InChI Key: {inchikey})")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
        except Exception as e:
            print(f"Error retrieving InChI from PubChem (InChI Key: {inchikey}): {e}")
            return None
    
    return None

def process_molecules(input_path, output_path, input_type, column_name):
    """
    Main function for processing molecule data
    
    Parameters:
    input_path: input CSV file path
    output_path: output CSV file path
    input_type: input type, 'inchi' or 'inchikey'
    column_name: name of the column containing InChI or InChIKey in the input file
    """
    # Read the CSV file
    try:
        df = pd.read_csv(input_path, encoding='windows-1252')
    except UnicodeDecodeError:
        # If windows-1252 encoding fails, try other encodings
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except:
            df = pd.read_csv(input_path, encoding='latin1')
    
    # Check whether the required columns exist
    if column_name not in df.columns:
        print(f"Missing required column in CSV file: {column_name}")
        return
    
    # List of atom type names (79, in the same order as returned by Fingerprinter.FingerprintMol)
    atom_type_names = [
        "sLi", "ssBe", "ssssBe", "ssBH", "sssB", "ssssB", "sCH3", "dCH2", "ssCH2", 
        "tCH", "dsCH", "aaCH", "sssCH", "ddC", "tsC", "dssC", "aasC", "aaaC", 
        "ssssC", "sNH3", "sNH2", "ssNH2", "dNH", "ssNH", "aaNH", "tN", "sssNH", 
        "dsN", "aaN", "sssN", "ddsN", "aasN", "ssssN", "sOH", "dO", "ssO", "aaO", 
        "sF", "sSiH3", "ssSiH2", "sssSiH", "ssssSi", "sPH2", "ssPH", "sssP", 
        "dsssP", "sssssP", "sSH", "dS", "ssS", "aaS", "dssS", "ddssS", "sCl", 
        "sGeH3", "ssGeH2", "sssGeH", "ssssGe", "sAsH2", "ssAsH", "sssAs", 
        "sssdAs", "sssssAs", "sSeH", "dSe", "ssSe", "aaSe", "dssSe", "ddssSe", 
        "sBr", "sSnH3", "ssSnH2", "sssSnH", "ssssSn", "sI", "sPbH3", "ssPbH2", 
        "sssPbH", "ssssPb"
    ]
    
    # Create column names for atom counts and E-state sums
    atom_count_columns = [f'ligand_Estate_atom_{name}' for name in atom_type_names]
    es_sum_columns = [f'ligand_Estate_sum_{name}' for name in atom_type_names]
    
    # Create the result DataFrame
    results = []
    
    # Statistics
    total_molecules = len(df)
    successful_molecules = 0
    failed_molecules = 0
    conversion_failures = 0
    
    print(f"Starting to process {total_molecules} molecules...")
    print(f"Input type: {input_type}")
    print(f"Column used: {column_name}")
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        input_str = row[column_name]
        
        # If the input type is inchikey, it needs to be converted to inchi
        if input_type == 'inchikey':
            inchi_str = get_inchi_from_inchikey(input_str)
            if inchi_str is None:
                print(f"Could not obtain InChI from InChI Key (row {index+1}): {input_str}")
                conversion_failures += 1
                continue
        else:  # input_type == 'inchi'
            inchi_str = input_str
        
        # Use the original input as the identifier
        ligand_id = row.get('ligand')
        
        # Parse the molecule from the InChI string
        mol = Chem.MolFromInchi(inchi_str)
        if mol is None:
            print(f"Could not parse molecule from InChI string (row {index+1}): {input_str}")
            # Add a result row filled with NaN
            atom_counts = [np.nan] * len(atom_type_names)
            es_sums = [np.nan] * len(atom_type_names)
            failed_molecules += 1
        else:
            # Generate the EState fingerprint
            try:
                atom_counts, es_sums = Fingerprinter.FingerprintMol(mol)
                successful_molecules += 1
            except Exception as e:
                print(f"Error generating EState fingerprint (row {index+1}): {e}")
                atom_counts = [np.nan] * len(atom_type_names)
                es_sums = [np.nan] * len(atom_type_names)
                failed_molecules += 1
        
        # Check the length of the returned list
        if isinstance(atom_counts, list) and len(atom_counts) != len(atom_type_names):
            print(f"Warning: atom type count mismatch (row {index+1}). Expected {len(atom_type_names)}, got {len(atom_counts)}")
            # Adjust the list length
            if len(atom_counts) < len(atom_type_names):
                atom_counts = list(atom_counts) + [0] * (len(atom_type_names) - len(atom_counts))
            else:
                atom_counts = atom_counts[:len(atom_type_names)]
        
        if isinstance(es_sums, list) and len(es_sums) != len(atom_type_names):
            print(f"Warning: E-state sums count mismatch (row {index+1}). Expected {len(atom_type_names)}, got {len(es_sums)}")
            # Adjust the list length
            if len(es_sums) < len(atom_type_names):
                es_sums = list(es_sums) + [0.0] * (len(atom_type_names) - len(es_sums))
            else:
                es_sums = es_sums[:len(atom_type_names)]
        
        # Create the result row
        result_row = {'ligand': ligand_id}
        
        # Add atom counts
        for i, col_name in enumerate(atom_count_columns):
            result_row[col_name] = atom_counts[i] if i < len(atom_counts) else 0
        
        # Add E-state sums
        for i, col_name in enumerate(es_sum_columns):
            result_row[col_name] = es_sums[i] if i < len(es_sums) else 0.0
        
        results.append(result_row)
        
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1} molecules")
    
    # Convert to a DataFrame
    if not results:
        print("No molecules were processed successfully; cannot generate a result file")
        return
    
    result_df = pd.DataFrame(results)
    
    # Reorder the columns: ligand first, then atom counts columns, and E-state sums columns last
    column_order = ['ligand'] + atom_count_columns + es_sum_columns
    result_df = result_df.reindex(columns=column_order)
    
    # Save to a CSV file
    result_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Total molecules processed: {len(result_df)}")
    print(f"Successfully processed molecules: {successful_molecules}")
    print(f"Failed molecules: {failed_molecules}")
    if input_type == 'inchikey':
        print(f"Molecules with failed InChI conversion: {conversion_failures}")
    
    # Optional: also save separate files for atom counts and E-state sums
    if output_path.endswith('.csv'):
        base_path = output_path[:-4]
        
        # Save atom counts to a separate file
        atom_counts_df = result_df[['ligand'] + atom_count_columns]
        atom_counts_df.to_csv(f"{base_path}_atom_counts.csv", index=False)
        print(f"Atom counts saved separately to {base_path}_atom_counts.csv")
        
        # Save E-state sums to a separate file
        es_sums_df = result_df[['ligand'] + es_sum_columns]
        es_sums_df.to_csv(f"{base_path}_es_sums.csv", index=False)
        print(f"E-state sums saved separately to {base_path}_es_sums.csv")
    
    # Show the first few rows of data
    print("\nPreview of the first 5 rows:")
    print(result_df.head())
    
    # Show column names
    print("\nColumn name list:")
    for i, col in enumerate(column_order[:20]):  # only show the first 20 columns
        print(f"{i+1}. {col}")
    if len(column_order) > 20:
        print(f"... {len(column_order)} columns in total")

def main():
    # Create the command-line argument parser
    parser = argparse.ArgumentParser(description='Generate E-state fingerprint descriptors')
    
    # Add arguments
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-t', '--type', choices=['inchi', 'inchikey'], default='inchi',
                       help='Input type: inchi or inchikey (default: inchi)')
    parser.add_argument('-l', '--column', default=None,
                       help='Name of the column containing InChI or InChIKey in the input file (default: selected automatically based on type)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine the column name
    column_name = args.column
    if column_name is None:
        column_name = 'InChI' if args.type == 'inchi' else 'InChI.Key'
    
    # Process molecules
    process_molecules(args.input, args.output, args.type, column_name)

if __name__ == "__main__":
    main()