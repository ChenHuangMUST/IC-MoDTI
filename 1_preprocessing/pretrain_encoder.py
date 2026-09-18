import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_csv", type=str, default="protein_features_unique_260414.csv")
    parser.add_argument("--ligand_csv", type=str, default="ligand_feature_260703.csv")
    parser.add_argument("--protein_latent_dim", type=int, default=256)
    parser.add_argument("--ligand_latent_dim", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    return parser.parse_args()

# -------------------------- Protein autoencoder --------------------------
def build_protein_ae(input_dim, latent_dim):
    # Encoder
    inp = Input(shape=(input_dim,))
    x = Dense(2048, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    z = Dense(latent_dim, activation="linear", name="protein_z")(x)

    # Decoder
    x = Dense(512, activation="relu")(z)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(2048, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)

    ae = Model(inp, out)
    encoder = Model(inp, z)
    ae.compile(optimizer="adam", loss="mse")
    return ae, encoder

# -------------------------- Ligand autoencoder --------------------------
def build_ligand_ae(input_dim, latent_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(256, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    z = Dense(latent_dim, activation="linear", name="ligand_z")(x)

    x = Dense(128, activation="relu")(z)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)

    ae = Model(inp, out)
    encoder = Model(inp, z)
    ae.compile(optimizer="adam", loss="mse")
    return ae, encoder

def main():
    args = parse_args()
    os.makedirs("encoder_save", exist_ok=True)

    # 1. Train the protein encoder
    df_prot = pd.read_csv(args.protein_csv).drop_duplicates("protein")
    prot_feat_cols = [c for c in df_prot.columns if c.startswith("protein_")]
    X_prot_raw = df_prot[prot_feat_cols].values
    scaler_prot = StandardScaler()
    X_prot = scaler_prot.fit_transform(X_prot_raw)
    prot_input_dim = X_prot.shape[1]

    prot_ae, prot_encoder = build_protein_ae(prot_input_dim, args.protein_latent_dim)
    prot_ae.fit(
        X_prot, X_prot,
        epochs=args.epoch, batch_size=args.batch,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    prot_encoder.save("encoder_save/protein_encoder.h5")
    joblib.dump(scaler_prot, "encoder_save/prot_scaler.joblib")
    print(f"✅ Protein encoder saved, latent dim = {args.protein_latent_dim}")

    # 2. Train the ligand encoder
    df_lig = pd.read_csv(args.ligand_csv).drop_duplicates("ligand")
    lig_feat_cols = [c for c in df_lig.columns if c.startswith("ligand_")]
    X_lig_raw = df_lig[lig_feat_cols].values
    scaler_lig = StandardScaler()
    X_lig = scaler_lig.fit_transform(X_lig_raw)
    lig_input_dim = X_lig.shape[1]

    lig_ae, lig_encoder = build_ligand_ae(lig_input_dim, args.ligand_latent_dim)
    lig_ae.fit(
        X_lig, X_lig,
        epochs=args.epoch, batch_size=args.batch,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    lig_encoder.save("encoder_save/ligand_encoder.h5")
    joblib.dump(scaler_lig, "encoder_save/lig_scaler.joblib")
    print(f"✅ Ligand encoder saved, latent dim = {args.ligand_latent_dim}")

    # 3. Precompute all protein/ligand latent vectors and save lookup tables (downstream joins directly, speeding up training)
    df_prot["z_protein"] = list(prot_encoder.predict(X_prot, verbose=0))
    df_lig["z_ligand"] = list(lig_encoder.predict(X_lig, verbose=0))
    df_prot[["protein", "z_protein"]].to_pickle("encoder_save/prot_latent_table.pkl")
    df_lig[["ligand", "z_ligand"]].to_pickle("encoder_save/lig_latent_table.pkl")
    print("✅ Latent lookup table saved for downstream merge")

if __name__ == "__main__":
    main()