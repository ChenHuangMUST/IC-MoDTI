import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
from tensorflow.keras.models import load_model
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from encoder_utils import merge_latent_embedding, get_latent_feature_names


def parse_arguments():
    parser = argparse.ArgumentParser(description='Protein-ligand binding/non-binding external data prediction')
    parser.add_argument('-p', '--protein_file', required=True, help='Protein feature file')
    parser.add_argument('-l', '--ligand_file', required=True, help='Ligand feature file')
    parser.add_argument('-g', '--combination_file', required=True, help='Combination file')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('-o', '--output', default='binding_predictions.csv', help='Output file')
    args = parser.parse_args()
    return args


def load_binding_model():
    """Load the binding/non-binding model"""
    print("=== Loading binding/non-binding model ===")
    model_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(model_dir, "my_model.h5")
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    feature_path = os.path.join(model_dir, "feature_columns.json")
    threshold_path = os.path.join(model_dir, "training_metrics.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature columns not found: {feature_path}")

    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(feature_path, 'r') as f:
        required_features = json.load(f)

    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            metrics = json.load(f)
            if "best_threshold" in metrics:
                threshold = metrics["best_threshold"]
            elif "youden_threshold" in metrics:
                threshold = metrics["youden_threshold"]

    print(f"Model loaded, using threshold: {threshold:.4f}")
    return model, preprocessor, required_features, threshold


def load_and_merge_data(args):
    combo = pd.read_csv(args.combination_file)
    protein = pd.read_csv(args.protein_file).drop_duplicates("protein")
    ligand = pd.read_csv(args.ligand_file).drop_duplicates("ligand")

    df = pd.merge(combo, protein, on="protein", how="inner")
    df = pd.merge(df, ligand, on="ligand", how="inner")
    
    # Added: concatenate autoencoder latent vectors to match the training pipeline
    df, _, _ = merge_latent_embedding(df)

    print(f"Number of valid prediction combinations: {len(df)}")
    return df


def predict_binding(args):
    model, preprocessor, required_cols, threshold = load_binding_model()
    df = load_and_merge_data(args)

    # Feature completeness check
    missing_cols = [c for c in required_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"External data missing feature columns: {missing_cols[:10]} ... {len(missing_cols)} columns in total")
    # Strictly follow the training column order to avoid ColumnTransformer mismatch
    X = df[required_cols].copy()

    X_trans = preprocessor.transform(X)
    prob = model.predict(X_trans, verbose=0).ravel()
    pred_class = (prob >= threshold).astype(int)

    result = df[["protein", "ligand"]].copy()
    result["Binding_Probability"] = prob
    result["Binding_Prediction"] = pred_class
    result["Binding_Label"] = result["Binding_Prediction"].map({1: "BINDING", 0: "NON_BINDING"})

    result.to_csv(args.output, index=False)
    print(f"\nPrediction complete! Results saved to: {args.output}")
    print(f"Total count: {len(result)}")
    print(f"BINDING: {(result.Binding_Label == 'BINDING').sum()}")
    print(f"NON_BINDING: {(result.Binding_Label == 'NON_BINDING').sum()}")

    return args.output


if __name__ == "__main__":
    args = parse_arguments()
    predict_binding(args)