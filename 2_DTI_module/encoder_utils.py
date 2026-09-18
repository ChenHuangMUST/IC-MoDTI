import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# ========= Auto-locate the encoder_save directory (compatibility adaptation) =========
# Priority: environment variable IC_MOTDI_ENCODER_DIR > encoder_save in parent directory > current directory
def _locate_encoder_dir():
    env_dir = os.environ.get("IC_MOTDI_ENCODER_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_preprocessing", "encoder_save"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoder_save"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            return c
    return candidates[0]

BASE_ENCODER_DIR = _locate_encoder_dir()

PROT_ENCODER_PATH = os.path.join(BASE_ENCODER_DIR, "protein_encoder.h5")
LIG_ENCODER_PATH = os.path.join(BASE_ENCODER_DIR, "ligand_encoder.h5")
PROT_SCALER_PATH = os.path.join(BASE_ENCODER_DIR, "prot_scaler.joblib")
LIG_SCALER_PATH = os.path.join(BASE_ENCODER_DIR, "lig_scaler.joblib")
PROT_LATENT_TABLE_PATH = os.path.join(BASE_ENCODER_DIR, "prot_latent_table.pkl")
LIG_LATENT_TABLE_PATH = os.path.join(BASE_ENCODER_DIR, "lig_latent_table.pkl")

# Globally load pretrained encoders and lookup tables (loaded only once)
PROT_ENCODER = tf.keras.models.load_model(PROT_ENCODER_PATH, compile=False)
LIG_ENCODER = tf.keras.models.load_model(LIG_ENCODER_PATH, compile=False)
PROT_SCALER = joblib.load(PROT_SCALER_PATH)
LIG_SCALER = joblib.load(LIG_SCALER_PATH)
PROT_LATENT_TABLE = pd.read_pickle(PROT_LATENT_TABLE_PATH)
LIG_LATENT_TABLE = pd.read_pickle(LIG_LATENT_TABLE_PATH)

PROT_LATENT_DIM = PROT_ENCODER.output_shape[-1]
LIG_LATENT_DIM = LIG_ENCODER.output_shape[-1]


def merge_latent_embedding(df):
    """
    Input: merged df containing raw protein and ligand features (includes protein and ligand columns)
    Output: df with new z_protein_xx and z_ligand_xx latent vector columns
    """
    # Merge protein latent vectors
    df = df.merge(PROT_LATENT_TABLE, on="protein", how="left")
    # Merge ligand latent vectors
    df = df.merge(LIG_LATENT_TABLE, on="ligand", how="left")

    # Split list latent vectors into separate numeric columns
    z_prot_cols = [f"z_prot_{i}" for i in range(PROT_LATENT_DIM)]
    z_lig_cols = [f"z_lig_{i}" for i in range(LIG_LATENT_DIM)]

    prot_z_arr = np.vstack(df["z_protein"].values)
    lig_z_arr = np.vstack(df["z_ligand"].values)

    # Approach: build a latent-feature sub-DataFrame first, then concat once to avoid fragmentation
    df_prot_z = pd.DataFrame(prot_z_arr, columns=z_prot_cols, index=df.index)
    df_lig_z = pd.DataFrame(lig_z_arr, columns=z_lig_cols, index=df.index)
    df_latent = pd.concat([df_prot_z, df_lig_z], axis=1)

    # Merge all latent features at once instead of inserting column by column
    df = pd.concat([df, df_latent], axis=1)

    # Drop intermediate list columns, keep only numeric features
    df.drop(columns=["z_protein", "z_ligand"], inplace=True)
    return df, z_prot_cols, z_lig_cols

def get_latent_feature_names():
    """Return the list of new latent feature column names for the preprocessing pipeline"""
    z_prot_cols = [f"z_prot_{i}" for i in range(PROT_LATENT_DIM)]
    z_lig_cols = [f"z_lig_{i}" for i in range(LIG_LATENT_DIM)]
    all_latent = z_prot_cols + z_lig_cols
    return all_latent, z_prot_cols, z_lig_cols

# Extract representations on the fly for new proteins/ligands (used by the external prediction script)
def infer_protein_z(raw_protein_feature_df):
    feat_cols = [c for c in raw_protein_feature_df.columns if c.startswith("protein_")]
    X = PROT_SCALER.transform(raw_protein_feature_df[feat_cols])
    z = PROT_ENCODER.predict(X, verbose=0)
    return z

def infer_ligand_z(raw_ligand_feature_df):
    feat_cols = [c for c in raw_ligand_feature_df.columns if c.startswith("ligand_")]
    X = LIG_SCALER.transform(raw_ligand_feature_df[feat_cols])
    z = LIG_ENCODER.predict(X, verbose=0)
    return z