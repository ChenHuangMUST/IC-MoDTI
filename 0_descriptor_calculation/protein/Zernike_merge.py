#!/usr/bin/env python3
"""
Zernike descriptor merge script
Features: read the four output folders and merge them into a unified CSV file
"""

import os
import pickle
import pandas as pd
import glob
import argparse
from pathlib import Path

def merge_type_pkl(pkl_dir, type_name, output_csv=None):
    """
    Merge pkl files of a specific type
    
    Args:
        pkl_dir: directory path of the pkl files
        type_name: type name (ATOM, PM, PS, GM)
        output_csv: output CSV file path (optional)
    
    Returns:
        pandas DataFrame
    """
    
    # Check whether the directory exists
    if not os.path.exists(pkl_dir):
        print(f"✗ Error: directory does not exist: {pkl_dir}")
        return None
    
    # Find all pkl files
    pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    if not pkl_files:
        print(f"⚠ Warning: no .pkl files found in {pkl_dir}")
        return None
    
    print(f"✓ Found {len(pkl_files)} pkl files of type {type_name}")
    
    all_data = pd.DataFrame()
    
    for pkl_file in pkl_files:
        try:
            # Read the pkl file
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Get key and values
            if isinstance(data, dict):
                key = list(data.keys())[0]
                values = data[key]
                
                # Create DataFrame
                feature_columns = [f'protein_Zernike_{type_name}_{i+1}' 
                                 for i in range(len(values))]
                df = pd.DataFrame([values], columns=feature_columns)
                
                # Set the index (protein name)
                file_name = Path(pkl_file).stem
                # Remove the type suffix (e.g. _ATOM)
                if file_name.endswith(f"_{type_name}"):
                    protein_name = file_name[:-(len(type_name)+1)]
                else:
                    protein_name = file_name
                
                df.index = [protein_name]
                
                # Append to the combined data
                all_data = pd.concat([all_data, df])
                
            else:
                print(f"⚠ Warning: {pkl_file} is not a dict; skipping")
                
        except Exception as e:
            print(f"✗ Error reading {pkl_file}: {e}")
            continue
    
    # Add the protein column
    all_data.insert(0, 'protein', all_data.index)
    
    # Reset the index
    all_data.reset_index(drop=True, inplace=True)
    
    # Save as a CSV file
    if output_csv:
        all_data.to_csv(output_csv, index=False)
        print(f"✓ Saved {type_name} data to: {output_csv}")
    
    return all_data

def merge_all_types(base_name, output_dir="."):
    """
    Merge Zernike descriptors of all four types
    
    Args:
        base_name: base name of the output folders
        output_dir: output directory
    
    Returns:
        merged DataFrame
    """
    
    # Type configuration
    type_configs = {
        'ATOM': {
            'folder': f'{base_name}AtomZernikeDescriptor',
            'suffix': '_ATOM'
        },
        'PM': {
            'folder': f'{base_name}PymolMeshZernikeDescriptor',
            'suffix': '_PM'
        },
        'PS': {
            'folder': f'{base_name}PymolSurfaceZernikeDescriptor',
            'suffix': '_PS'
        },
        'GM': {
            'folder': f'{base_name}gmconvertZernikeDescriptor',
            'suffix': '_GM'
        }
    }
    
    all_dfs = {}
    
    print("="*60)
    print("Starting Zernike descriptor merge")
    print("="*60)
    
    # Process each type separately
    for type_name, config in type_configs.items():
        folder = config['folder']
        
        if not os.path.exists(folder):
            print(f"⚠ Warning: folder {folder} does not exist; skipping {type_name}")
            continue
        
        print(f"\nProcessing {type_name} type...")
        output_csv = os.path.join(output_dir, f"{type_name}.csv")
        
        df = merge_type_pkl(folder, type_name, output_csv)
        if df is not None and not df.empty:
            all_dfs[type_name] = df
            print(f"✓ {type_name}: {df.shape[0]} proteins, {df.shape[1]-1} features")
    
    if not all_dfs:
        print("✗ Error: no valid data found")
        return None
    
    # Merge all types
    print(f"\n{'='*60}")
    print("Merging data of all types...")
    
    # Start from the first DataFrame
    merged_df = all_dfs[list(all_dfs.keys())[0]].copy()
    
    # Merge the other types
    for type_name, df in all_dfs.items():
        if type_name == list(all_dfs.keys())[0]:
            continue
        
        # Make sure both DataFrames have the protein column
        if 'protein' in merged_df.columns and 'protein' in df.columns:
            # Merge on the protein column
            merged_df = pd.merge(
                merged_df, 
                df, 
                on='protein',
                how='outer',
                suffixes=('', f'_{type_name}')
            )
    
    # Reorder columns: protein first
    if 'protein' in merged_df.columns:
        cols = ['protein'] + [col for col in merged_df.columns if col != 'protein']
        merged_df = merged_df[cols]
    
    # Save the merged result
    merged_csv = os.path.join(output_dir, "Zernike_all_features.csv")
    merged_df.to_csv(merged_csv, index=False)
    
    print(f"\n{'='*60}")
    print("Merge complete!")
    print(f"Total proteins: {merged_df.shape[0]}")
    print(f"Total features: {merged_df.shape[1]-1}")
    print(f"Merged file: {merged_csv}")
    
    # Save summary information
    summary_file = os.path.join(output_dir, "merge_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Zernike descriptor merge summary\n")
        f.write("="*50 + "\n")
        f.write(f"Total proteins: {merged_df.shape[0]}\n")
        f.write(f"Total features: {merged_df.shape[1]-1}\n")
        f.write("\nIncluded types:\n")
        for type_name in all_dfs.keys():
            f.write(f"  - {type_name}\n")
        f.write(f"\nOutput files:\n")
        f.write(f"  - Merged file: {merged_csv}\n")
        for type_name in all_dfs.keys():
            f.write(f"  - {type_name}: {type_name}.csv\n")
    
    print(f"Summary information: {summary_file}")
    
    return merged_df

def main():
    parser = argparse.ArgumentParser(description='Merge Zernike descriptor results')
    parser.add_argument('-b', '--base_name', required=True, # New required argument
                       help='Base name of the output folders. For example, if the pdb folder is `./data/pdb/`, the base name might be `pdb`')
    parser.add_argument('-o', '--output', default=".",
                       help='Output directory path (default: current directory)')
    parser.add_argument('-t', '--type', choices=['ATOM', 'PM', 'PS', 'GM', 'ALL'],
                       default='ALL', help='Type to process: ATOM, PM, PS, GM, ALL (default: ALL)')
    
    args = parser.parse_args()
    
    # Create the output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.type == 'ALL':
        merged_df = merge_all_types(args.base_name, args.output)
        if merged_df is not None:
            # Show the first few rows
            print(f"\nData preview:")
            print(merged_df.head())
            
            # Show column information
            print(f"\nColumn names:")
            for i, col in enumerate(merged_df.columns[:10]):  # Show only the first 10 columns
                print(f"  {i+1:3d}. {col}")
            if len(merged_df.columns) > 10:
                print(f"  ... {len(merged_df.columns)} columns in total")
    else:
        # Process a single type
        type_folders = {
            'ATOM': f'{args.base_name}AtomZernikeDescriptor',  # Dynamically generated
            'PM': f'{args.base_name}PymolMeshZernikeDescriptor',  # Dynamically generated
            'PS': f'{args.base_name}PymolSurfaceZernikeDescriptor',  # Dynamically generated
            'GM': f'{args.base_name}gmconvertZernikeDescriptor'  # Dynamically generated
        }
        
        folder = type_folders.get(args.type)
        if folder and os.path.exists(folder):
            output_csv = os.path.join(args.output, f"{args.type}.csv")
            df = merge_type_pkl(folder, args.type, output_csv)
            if df is not None:
                print(f"\n{args.type} type data:")
                print(f"Shape: {df.shape}")
                print(f"First 5 rows:")
                print(df.head())
        else:
            print(f"✗ Error: folder {folder} does not exist")

if __name__ == "__main__":
    main()