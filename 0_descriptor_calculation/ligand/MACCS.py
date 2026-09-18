import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import requests
import time
import argparse
import sys
from typing import Optional, Dict, List

def get_inchi_from_inchikey(inchikey: str) -> Optional[str]:
    """
    Retrieve an InChI string from an InChI Key via the PubChem API
    
    Parameters:
    inchikey: InChI Key string
    
    Returns:
    InChI string, or None if retrieval fails
    """
    if pd.isna(inchikey) or not isinstance(inchikey, str) or not inchikey.strip():
        return None
    
    inchikey = inchikey.strip()
    
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
        
        return None
    
    except Exception as e:
        print(f"Error retrieving InChI from PubChem (InChI Key: {inchikey}): {e}")
        return None

def compute_maccs_fingerprint(inchi_str: Optional[str]) -> Optional[str]:
    """
    Compute the MACCS fingerprint of an InChI string
    
    Parameters:
    inchi_str: InChI string
    
    Returns:
    166-bit binary string, or None if it cannot be computed
    """
    if inchi_str is None or pd.isna(inchi_str) or not inchi_str.strip():
        return None
    
    try:
        mol = Chem.MolFromInchi(inchi_str.strip())
        if mol is None:
            return None
        
        # Compute the MACCS fingerprint
        fp = MACCSkeys.GenMACCSKeys(mol)
        # Convert to a full 166-bit binary string
        binary_str = ''.join('1' if fp[i] else '0' for i in range(166))
        return binary_str
    
    except Exception as e:
        print(f"Error computing MACCS fingerprint (InChI: {inchi_str[:50]}...): {e}")
        return None

def process_csv_with_maccs(
    input_csv_path: str, 
    output_csv_path: str,
    input_type: str = 'inchikey',
    column_name: str = 'ligand',
    batch_size: int = 100,
    delay: float = 0.1
) -> pd.DataFrame:
    """
    Compute MACCS fingerprints from a CSV file and add them as new columns
    
    Parameters:
    input_csv_path: input CSV file path
    output_csv_path: output CSV file path
    input_type: input type, 'inchikey' or 'inchi'
    column_name: name of the column containing InChI Key or InChI
    batch_size: number of molecules per batch (used to display progress)
    delay: delay between API requests (seconds) to avoid requesting too quickly
    
    Returns:
    DataFrame containing the MACCS fingerprint columns
    """
    # Validate the input type
    if input_type not in ['inchikey', 'inchi']:
        raise ValueError("input_type must be 'inchikey' or 'inchi'")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_csv_path, encoding='windows-1252')
    except UnicodeDecodeError:
        # If windows-1252 encoding fails, try utf-8
        df = pd.read_csv(input_csv_path, encoding='utf-8')
    
    # Check whether the column exists
    if column_name not in df.columns:
        available_columns = ', '.join(df.columns.tolist())
        raise ValueError(f"Missing column in CSV file: '{column_name}'. Available columns: {available_columns}")
    
    print(f"Starting to process {len(df)} molecules...")
    print(f"Input type: {input_type}")
    print(f"Input column name: '{column_name}'")
    
    # Store InChI strings
    inchi_list = []
    
    if input_type == 'inchikey':
        # Obtain InChI strings from InChI Keys
        print("Retrieving InChI strings from InChI Keys...")
        failed_inchikeys = []
        
        for i, inchikey in enumerate(df[column_name]):
            if pd.isna(inchikey) or not str(inchikey).strip():
                inchi_list.append(None)
                continue
            
            inchikey_str = str(inchikey).strip()
            inchi = get_inchi_from_inchikey(inchikey_str)
            
            if inchi is None:
                inchi_list.append(None)
                failed_inchikeys.append((i, inchikey_str))
                print(f"Warning: could not retrieve InChI (row {i+1}, InChI Key: {inchikey_str})")
            else:
                inchi_list.append(inchi)
            
            # Show progress
            if (i + 1) % batch_size == 0:
                print(f"Retrieved InChI for {i+1}/{len(df)} molecules")
            
            # Avoid requesting too quickly
            time.sleep(delay)
        
        # Show statistics
        success_count = sum(1 for inchi in inchi_list if inchi is not None)
        print(f"\nInChI retrieval complete:")
        print(f"  Success: {success_count}/{len(df)}")
        print(f"  Failed: {len(failed_inchikeys)}")
        
        if failed_inchikeys:
            print("Failed InChI Keys (first 10):")
            for idx, (row_idx, inchikey) in enumerate(failed_inchikeys[:10]):
                print(f"  Row {row_idx+1}: {inchikey}")
    
    elif input_type == 'inchi':
        # Use the InChI strings directly
        print("Using InChI strings directly...")
        inchi_list = df[column_name].tolist()
        print(f"Read {len(inchi_list)} InChI strings")
    
    # Add the InChI column to the DataFrame
    df['InChI'] = inchi_list
    
    # Store MACCS fingerprints
    maccs_columns = []
    failed_maccs = []
    
    # Compute the MACCS fingerprint for each molecule
    print("\nComputing MACCS fingerprints...")
    for i, inchi in enumerate(df['InChI']):
        if inchi is None:
            maccs_columns.append([np.nan] * 166)
            failed_maccs.append(i)
        else:
            binary_str = compute_maccs_fingerprint(inchi)
            
            if binary_str is None:
                maccs_columns.append([np.nan] * 166)
                failed_maccs.append(i)
                if i < 10:  # only show details for the first 10 failures
                    print(f"Could not compute MACCS fingerprint for molecule {i+1} (InChI: {str(inchi)[:50]}...)")
            else:
                # Convert the binary string to a list
                bit_list = list(binary_str)
                maccs_columns.append(bit_list)
        
        # Show progress
        if (i + 1) % batch_size == 0:
            print(f"Computed MACCS fingerprints for {i+1}/{len(df)} molecules")
    
    # Create the MACCS fingerprint DataFrame
    maccs_df = pd.DataFrame(maccs_columns)
    
    # Set column names: ligand_MACCS_Col1 to ligand_MACCS_Col166
    maccs_column_names = [f'ligand_MACCS_Col{i+1}' for i in range(166)]
    maccs_df.columns = maccs_column_names
    
    # Merge the MACCS fingerprint columns into the original DataFrame
    result_df = pd.concat([df, maccs_df], axis=1)
    
    # Save to a new CSV file
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    # Show final statistics
    maccs_success = len(df) - len(failed_maccs)
    print(f"\nProcessing complete!")
    print(f"Results saved to: {output_csv_path}")
    print(f"Original column count: {len(df.columns) - 1}, new column count: {len(result_df.columns)}")
    print(f"MACCS fingerprint computation statistics:")
    print(f"  Success: {maccs_success}/{len(df)}")
    print(f"  Failed: {len(failed_maccs)}")
    
    return result_df

def main():
    """Main function: parse command-line arguments and process files"""
    parser = argparse.ArgumentParser(
        description='Compute MACCS fingerprints from a CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python MACCS.py -i input.csv -o output.csv -t inchikey -l ligand
  python MACCS.py -i input.csv -o output.csv -t inchi -l InChI
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-t', '--type', choices=['inchikey', 'inchi'], default='inchikey',
                       help='Input type: inchikey (InChI Key) or inchi (InChI string) (default: inchikey)')
    parser.add_argument('-l', '--column', required=True, help='Name of the column containing InChI Key or InChI')
    parser.add_argument('-b', '--batch', type=int, default=50,
                       help='Number of molecules per batch, used to display progress (default: 50)')
    parser.add_argument('-d', '--delay', type=float, default=0.2,
                       help='Delay between API requests (seconds) to avoid requesting too quickly (default: 0.2)')
    
    args = parser.parse_args()
    
    # Check whether the input file exists
    import os
    if not os.path.exists(args.input):
        print(f"Error: input file does not exist: {args.input}")
        sys.exit(1)
    
    # Create the output directory (if it does not exist)
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # First check the CSV file columns
    print("Checking CSV file column names...")
    try:
        df_sample = pd.read_csv(args.input, nrows=1, encoding='windows-1252')
        print(f"Columns in CSV file: {', '.join(df_sample.columns.tolist())}")
    except UnicodeDecodeError:
        try:
            df_sample = pd.read_csv(args.input, nrows=1, encoding='utf-8')
            print(f"Columns in CSV file: {', '.join(df_sample.columns.tolist())}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    try:
        # Process the file
        result = process_csv_with_maccs(
            input_csv_path=args.input,
            output_csv_path=args.output,
            input_type=args.type,
            column_name=args.column,
            batch_size=args.batch,
            delay=args.delay
        )
        
        # Show the first few rows of the result
        print("\nResult preview:")
        print(result.head())
        
        # Show the newly added columns
        print("\nNewly added MACCS columns:")
        maccs_cols = [col for col in result.columns if 'ligand_MACCS_' in col]
        print(f"Number of MACCS columns: {len(maccs_cols)}")
        print(f"First 5 MACCS column names: {maccs_cols[:5]}")
        
        # Count successes and failures
        total_rows = len(result)
        successful_maccs = result['ligand_MACCS_Col1'].notna().sum()
        print(f"\nFinal statistics:")
        print(f"Total molecules: {total_rows}")
        print(f"Successfully computed MACCS fingerprints: {successful_maccs}")
        print(f"Failed: {total_rows - successful_maccs}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()