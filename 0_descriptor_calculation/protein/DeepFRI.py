#!/usr/bin/env python3
"""
DeepFRI full-pipeline processing tool
Features: 1. Batch-run DeepFRI predictions 2. Extract features from JSON results
"""

import os
import json
import pandas as pd
import argparse
import glob
import sys
import subprocess
from pathlib import Path
import time

class DeepFRIProcessor:
    """DeepFRI full-pipeline processor"""
    
    def __init__(self, pdb_dir=None, json_dir=None, output_csv=None, 
                 model_config='./trained_models/model_config.json',
                 ontology='mf'):
        self.pdb_dir = pdb_dir
        self.json_dir = json_dir
        self.output_csv = output_csv
        self.model_config = model_config
        self.ontology = ontology
        
    def run_predictions(self, threads=1):
        """Run DeepFRI predictions (batch process PDB files)"""
        if not self.pdb_dir or not os.path.exists(self.pdb_dir):
            print(f"Error: PDB directory does not exist: {self.pdb_dir}")
            return False
        
        if not self.json_dir:
            self.json_dir = os.path.join(os.path.dirname(self.pdb_dir), "DeepFRI_JSON")
        
        # Ensure the JSON output directory exists
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Get all PDB files
        pdb_files = glob.glob(os.path.join(self.pdb_dir, "*.pdb"))
        if not pdb_files:
            print(f"Error: no PDB files found in {self.pdb_dir}")
            return False
        
        print(f"Found {len(pdb_files)} PDB files")
        print(f"JSON output directory: {self.json_dir}")
        print(f"Starting DeepFRI prediction...")
        
        success_count = 0
        failed_files = []
        
        for i, pdb_file in enumerate(pdb_files, 1):
            filename = os.path.basename(pdb_file)
            base_name = os.path.splitext(filename)[0]
            output_prefix = os.path.join(self.json_dir, base_name)
            
            # Build the command line
            cmd = [
                'python', 'predict.py',
                '--pdb_fn', pdb_file,
                '--model_config', self.model_config,
                '--ontology', self.ontology,
                '--output_fn_prefix', output_prefix,
            ]
            
            print(f"[{i}/{len(pdb_files)}] Processing: {filename}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"  ✓ Success: output to {output_prefix}_{self.ontology.upper()}_*")
                success_count += 1
            except subprocess.CalledProcessError as e:
                error_msg = f"  ✗ Failed: {e}"
                if e.stderr:
                    error_msg += f"\n    Error message: {e.stderr[:200]}"
                print(error_msg)
                failed_files.append(filename)
            except FileNotFoundError:
                print(f"  ✗ Failed: predict.py not found; make sure it is in the current directory or PATH")
                failed_files.append(filename)
        
        print(f"\nPrediction complete:")
        print(f"  ✓ Success: {success_count}/{len(pdb_files)}")
        if failed_files:
            print(f"  ✗ Failed: {len(failed_files)}")
            for f in failed_files[:5]:  # Show only the first 5 failed files
                print(f"    - {f}")
            if len(failed_files) > 5:
                print(f"    ... {len(failed_files)-5} more")
        
        return success_count > 0
    
    def extract_features(self, mode='group'):
        """Extract features from JSON files"""
        if not self.json_dir or not os.path.exists(self.json_dir):
            print(f"Error: JSON directory does not exist: {self.json_dir}")
            return False
        
        if not self.output_csv:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_csv = os.path.join(os.path.dirname(self.json_dir), 
                                          f"DeepFRI_features_{timestamp}.csv")
        
        # Get JSON files
        if mode == 'group':
            if os.path.isdir(self.json_dir):
                json_files = glob.glob(os.path.join(self.json_dir, "*.json"))
            else:
                json_files = glob.glob(self.json_dir)
        elif mode == 'single':
            if os.path.isfile(self.json_dir) and self.json_dir.endswith('.json'):
                json_files = [self.json_dir]
            else:
                print(f"Error: not a valid JSON file: {self.json_dir}")
                return False
        else:
            print(f"Error: unsupported mode: {mode}")
            return False
        
        if not json_files:
            print(f"Error: no JSON files found: {self.json_dir}")
            return False
        
        print(f"Found {len(json_files)} JSON files")
        
        # Store data
        all_data = []
        row_names = []
        goterms = None
        
        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                y_hat = json_data.get("Y_hat", [])
                goterms = json_data.get("goterms", [])
                
                # Support two Y_hat structures:
                #  1) Single-sequence prediction: Y_hat = [[...489 values...]] (current DeepFRI predict.py output)
                #  2) Multi-sequence:             Y_hat = [[...942...], [...], ...]
                if y_hat and isinstance(y_hat[0], (list, tuple)):
                    y_hat = y_hat[0]  # Take the first chain
                if goterms and isinstance(goterms[0], list):
                    goterms = goterms[0]
                
                n_feats = len(y_hat)
                # Feature count must follow goterms (the old script's 942 was the full GO set; the current mf model uses 489)
                expected = len(goterms) if goterms else 942
                if n_feats != expected:
                    print(f"Warning [{i}/{len(json_files)}]: Y_hat length of {os.path.basename(json_file)}={n_feats}, expected={expected}")
                    continue
                
                # Extract row name: prefer the file name body; also support custom names from prepare_inputs
                file_base = os.path.splitext(os.path.basename(json_file))[0]
                # If the file name has the AF-<ID>-F1-model_vX form, take the middle UniProt ID; otherwise use the full file name
                for marker in ("_MF_pred_scores", "_MF_predictions", ".MF"):
                    file_base = file_base.replace(marker, "")
                if file_base.startswith("AF-"):
                    parts = file_base.split("-")
                    if len(parts) >= 2:
                        file_base = parts[1]
                row_name = file_base
                
                all_data.append(y_hat)
                row_names.append(row_name)
                
                # Get goterms
                if goterms is None:
                    goterms = json_data["goterms"]
                if goterms and len(goterms) != n_feats:
                    print(f"Warning: inconsistent goterms length: {len(goterms)} vs {n_feats}")
                    goterms = None
                
                print(f"[{i}/{len(json_files)}] ✓ Processed: {os.path.basename(json_file)}")
                
            except Exception as e:
                print(f"Error [{i}/{len(json_files)}]: failed to process {json_file}: {e}")
                continue
        
        if not all_data:
            print("Error: no valid data")
            return False
        
        # Create DataFrame
        df = pd.DataFrame(all_data, index=row_names)
        
        # Set column names (matching the actual number of goterms; names carry the protein_DeepFRI prefix for direct downstream use)
        n_feats_out = len(df.columns)
        if goterms and len(goterms) == n_feats_out:
            df.columns = [f"protein_DeepFRI_Col{i+1}_{goterms[i]}" for i in range(n_feats_out)]
        else:
            df.columns = [f"protein_DeepFRI_Col{i+1}" for i in range(n_feats_out)]
        df.index.name = "protein"
        
        # Save
        output_dir = os.path.dirname(self.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df.to_csv(self.output_csv)
        
        print(f"\nExtraction complete:")
        print(f"  ✓ Samples: {len(df)}")
        print(f"  ✓ Features: {len(df.columns)}")
        print(f"  ✓ Output file: {self.output_csv}")
        
        return True
    
    def run_full_pipeline(self, threads=1, mode='group'):
        """Run the full pipeline: prediction + extraction"""
        print("="*60)
        print("DeepFRI full-pipeline processing")
        print("="*60)
        
        # Step 1: run predictions
        if self.pdb_dir:
            print("\n[Step 1] Run DeepFRI predictions")
            print("-"*40)
            if not self.run_predictions(threads):
                print("Prediction step failed; aborting pipeline")
                return False
        else:
            print("Skipping prediction step (no PDB directory provided)")
        
        # Step 2: extract features
        print("\n[Step 2] Extract features to CSV")
        print("-"*40)
        if not self.extract_features(mode):
            print("Feature extraction failed")
            return False
        
        print("\n" + "="*60)
        print("✓ Full pipeline complete!")
        print("="*60)
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='DeepFRI full-pipeline processing tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Full pipeline (predict + extract)
  python deepfri_pipeline.py --predict -i ./pdb_files/ -o ./output/features.csv
  
  # Extract features only (from JSON files)
  python deepfri_pipeline.py --extract -j ./json_results/ -o ./output/features.csv -t group
  
  # Run predictions only
  python deepfri_pipeline.py --predict -i ./pdb_files/ --json_dir ./json_output/
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--predict', action='store_true',
                          help='Run DeepFRI predictions (requires PDB files)')
    mode_group.add_argument('--extract', action='store_true',
                          help='Extract features from JSON files')
    mode_group.add_argument('--full', action='store_true',
                          help='Full pipeline: prediction + extraction')
    
    # Input/output arguments
    parser.add_argument('-i', '--input', 
                       help='Input PDB file or directory (for predict mode)')
    parser.add_argument('-j', '--json_dir',
                       help='JSON directory or a single JSON file (for extract mode)')
    parser.add_argument('-o', '--output', 
                       help='Output CSV file path')
    
    # Other arguments
    parser.add_argument('-t', '--type', choices=['group', 'single'], default='group',
                       help='Processing type: group (batch), single (single file) [default: group]')
    parser.add_argument('--model_config', default='./trained_models/model_config.json',
                       help='Model config file path [default: ./trained_models/model_config.json]')
    parser.add_argument('--ontology', default='mf',
                       choices=['mf', 'bp', 'cc', 'ec'],
                       help='GO ontology: mf, bp, cc, ec [default: mf]')
    parser.add_argument('--threads', type=int, default=1,
                       help='Number of threads (for prediction) [default: 1]')
    
    args = parser.parse_args()
    
    # Create processor
    processor = DeepFRIProcessor(
        pdb_dir=args.input if (args.predict or args.full) else None,
        json_dir=args.json_dir if args.extract else (args.output if args.full else None),
        output_csv=args.output,
        model_config=args.model_config,
        ontology=args.ontology
    )
    
    # Execute according to mode
    if args.full:
        # Full pipeline
        if not args.input:
            print("Error: full pipeline mode requires an input PDB directory (-i)")
            sys.exit(1)
        processor.run_full_pipeline(args.threads, args.type)
    elif args.predict:
        # Predict only
        if not args.input:
            print("Error: predict mode requires an input PDB directory (-i)")
            sys.exit(1)
        processor.run_predictions(args.threads)
    elif args.extract:
        # Extract only
        if not args.json_dir:
            print("Error: extract mode requires a JSON directory or file (-j)")
            sys.exit(1)
        processor.extract_features(args.type)

if __name__ == "__main__":
    main()