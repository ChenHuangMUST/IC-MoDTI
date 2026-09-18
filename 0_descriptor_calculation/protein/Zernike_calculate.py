#!/usr/bin/env python3
"""
Zernike descriptor batch calculation script
Features: read a PDB folder and compute four types of Zernike descriptors
"""

import os
import sys
import argparse
import subprocess
import glob
from pathlib import Path

def run_zernike_calculation(pdb_dir, mode, threads=4):
    """Run Zernike calculation for a specific mode"""
    
    # Check whether mainMakeZernikeDataBase.py exists
    script_paths = [
        "mainMakeZernikeDataBase.py",
        "FP-Zernike/mainMakeZernikeDataBase.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                    "FP-Zernike", "mainMakeZernikeDataBase.py")
    ]
    
    main_script = None
    for path in script_paths:
        if os.path.exists(path):
            main_script = path
            break
    
    if not main_script:
        print("✗ Error: mainMakeZernikeDataBase.py not found")
        print("Make sure env_create.py has been run to create the environment")
        return False
    
    # Check the PDB directory
    if not os.path.exists(pdb_dir):
        print(f"✗ Error: PDB directory does not exist: {pdb_dir}")
        return False
    
    # Check PDB files
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    if not pdb_files:
        print(f"✗ Warning: no .pdb files found in {pdb_dir}")
        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb.gz"))
        if pdb_files:
            print(f"✓ Found {len(pdb_files)} .pdb.gz files")
        else:
            print("Make sure PDB files have a .pdb or .pdb.gz extension")
            return False
    
    print(f"✓ Found {len(pdb_files)} PDB files")
    
    # Run the calculation command
    cmd = [
        "python", main_script,
        "-i", pdb_dir,
        "-t", str(threads),
        "-mode", mode
    ]
    
    print(f"\nComputing Zernike descriptors for mode {mode}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {mode} calculation complete")
        if result.stdout:
            print("Output:", result.stdout[:500])
    except subprocess.CalledProcessError as e:
        print(f"✗ {mode} calculation failed: {e}")
        print("Error output:", e.stderr)
        return False
    
    return True

def calculate_all_modes(pdb_dir, threads=4):
    """Compute Zernike descriptors for all four modes"""
    
    modes = ["ATOM", "PM", "PS", "GM"]

    # 1. Get the base name of the input folder (e.g. pdb)
    input_base = os.path.basename(os.path.normpath(pdb_dir))

    # 2. Dynamically build the output folder names of mainMakeZernikeDataBase.py from the base name and mode
    # Mapping: ATOM -> Atom, PM -> PymolMesh, PS -> PymolSurface, GM -> gmconvert
    mode_suffix_map = {
        "ATOM": "Atom",
        "PM": "PymolMesh", 
        "PS": "PymolSurface",
        "GM": "gmconvert"
    }
    
    # Build the correct output folder list
    output_folders = [f"{input_base}{mode_suffix_map[mode]}ZernikeDescriptor" for mode in modes]
    
    # Check whether the environment is activated
    env_dir = "FP-Zernike"
    if not os.path.exists(env_dir):
        print("⚠ Warning: FP-Zernike directory does not exist")
        print("Run first: python env_create.py")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Compute each mode
    for i, mode in enumerate(modes):
        print(f"\n{'='*60}")
        print(f"Mode {i+1}/4: {mode}")
        print(f"Output folder: {output_folders[i]}")
        print('='*60)
        
        success = run_zernike_calculation(pdb_dir, mode, threads)
        if not success:
            print(f"✗ {mode} mode calculation interrupted")
            response = input("Continue with the next mode? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Check output files
    print(f"\n{'='*60}")
    print("Calculation result check:")
    print('='*60)
    
    all_good = True
    for folder in output_folders:
        if os.path.exists(folder):
            pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
            print(f"✓ {folder}: found {len(pkl_files)} .pkl files")
        else:
            print(f"✗ {folder}: folder does not exist")
            all_good = False
    
    if all_good:
        print(f"\n✅ All calculations complete!")
        print(f"\nNext step: run python Zernike_merge.py to merge the results")
    else:
        print(f"\n⚠ Some calculations may not be complete")
    
    return all_good

def main():
    parser = argparse.ArgumentParser(description='Compute Zernike descriptors')
    parser.add_argument('-i', '--input', required=True, 
                       help='Path to the directory containing PDB files')
    parser.add_argument('-t', '--threads', type=int, default=4,
                       help='Number of threads to use (default: 4)')
    parser.add_argument('-m', '--mode', choices=['ATOM', 'PM', 'PS', 'GM', 'ALL'],
                       default='ALL', help='Calculation mode: ATOM, PM, PS, GM, ALL (default: ALL)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Zernike descriptor batch calculation tool")
    print("="*60)
    
    if args.mode == 'ALL':
        calculate_all_modes(args.input, args.threads)
    else:
        run_zernike_calculation(args.input, args.mode, args.threads)

if __name__ == "__main__":
    main()