import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import json
import argparse
import os
# ========= Environment adaptation: automatically set LD_LIBRARY_PATH / LD_PRELOAD =========
def _setup_runtime_env():
    env_lib = os.environ.get("IC_MOTDI_ENV_LIB")
    if not env_lib:
        # Auto-detect the lib directory of the current Python environment
        for cand in [os.path.join(os.path.dirname(sys.executable), "..", "lib")]:
            if os.path.isdir(cand):
                env_lib = os.path.abspath(cand)
                break
    if env_lib and os.path.isdir(env_lib):
        os.environ['LD_LIBRARY_PATH'] = env_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        # libstdc++.so.6 version adaptation (avoid GLIBCXX version conflicts)
        stdcpp = os.path.join(env_lib, "libstdc++.so.6")
        if os.path.exists(stdcpp):
            os.environ['LD_PRELOAD'] = stdcpp + " " + os.environ.get("LD_PRELOAD", "")

_setup_runtime_env()
import subprocess
import tempfile
from encoder_utils import merge_latent_embedding, get_latent_feature_names
SCRIPT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
EXTERNAL_VERIFY_SCRIPT = os.path.join(SCRIPT_ROOT, "external_data_verification_v3_to_v23.py")

training_metrics = {
    "weight_scale": None,
    "dsdp_weight_factor": None,
    "pretrain_epochs": None,
    "pretrain_train_loss": None,
    "pretrain_train_acc": None,
    "pretrain_train_auc": None,
    "pretrain_val_loss": None,
    "pretrain_val_acc": None,
    "pretrain_val_auc": None,
    "ft1_valid_samples": None, "ft1_total_samples": None,
    "ft1_weight_min": None, "ft1_weight_max": None, "ft1_real_epochs": None,
    "ft1_pre_auc": None, "ft1_pre_loss": None, "ft1_pre_acc": None,
    "ft1_post_auc": None, "ft1_post_loss": None, "ft1_post_acc": None,
    "ft1_val_auc": None, "ft1_val_loss": None, "ft1_val_acc": None,
    "ft2_valid_samples": None, "ft2_total_samples": None,
    "ft2_dsdp_low_count": None, "ft2_weight_min": None, "ft2_weight_max": None, "ft2_dsdp_factor": None,
    "ft2_real_epochs": None,
    "ft2_pre_auc": None, "ft2_pre_loss": None, "ft2_pre_acc": None,
    "ft2_post_auc": None, "ft2_post_loss": None, "ft2_post_acc": None,
    "ft2_val_auc": None, "ft2_val_loss": None, "ft2_val_acc": None,
    "cv_mean_auc": None, "cv_std_auc": None,
    "cv_mean_acc": None, "cv_std_acc": None,
    "cv_mean_ap": None, "cv_std_ap": None,
    "youden_threshold": None, "best_threshold": None,
    "external_best_accuracy": None,
    "youden_sensitivity": None, "youden_specificity": None,
    "final_train_loss": None,
    "final_train_acc": None,
    "final_train_auc": None,
    "final_val_loss": None,
    "final_val_acc": None,
    "final_val_auc": None,
    "test_roc_auc": None,
    "test_avg_precision": None,
    "exp_protein_id": None,
    "ft3_valid_samples": None, "ft3_total_samples": None,
    "ft3_weight_min": None, "ft3_weight_max": None, "ft3_real_epochs": None,
    "ft3_pre_auc": None, "ft3_pre_loss": None, "ft3_pre_acc": None,
    "ft3_post_auc": None, "ft3_post_loss": None, "ft3_post_acc": None,
    "ft3_val_auc": None, "ft3_val_loss": None, "ft3_val_acc": None
}

def save_metrics(path="training_metrics.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2, ensure_ascii=False)
    print(f"All metrics saved to: {path}")

mpl.use('PDF')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train model with 5-fold CV')
    parser.add_argument('-m', '--main_file', required=True)
    parser.add_argument('-p', '--protein_file', required=True)
    parser.add_argument('-l', '--ligand_file', required=True)
    parser.add_argument('-d', '--base_path', default='.')
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5)
    parser.add_argument('--fine_tune_epochs', type=int, default=500)
    parser.add_argument('--weight_scale', type=float, default=1.0)
    parser.add_argument('--dsdp_weight_factor', type=float, default=2.0)
    parser.add_argument('-exp', '--exp', type=str, default=None)
    # ===== Contrastive learning parameters (new) =====
    parser.add_argument('--contrastive_weight', type=float, default=0.0,
                        help='Contrastive loss weight for FT3 (0=disabled, try 0.05~0.3)')
    parser.add_argument('--contrastive_temp', type=float, default=0.1,
                        help='Temperature for supervised contrastive loss')
    return parser.parse_args()

# ===== Supervised Contrastive Loss =====
class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    """
    SupCon Loss: pull embeddings of the same class together within a batch and push different classes apart.
    Used as a regularization term in FT3 to refine the embedding space of P51787.
    """
    def __init__(self, temperature=0.1, name='supcon_loss'):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, y_true, embeddings):
        y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
        pos_mask = tf.cast(tf.equal(y_true, tf.transpose(y_true)), tf.float32)
        batch_size = tf.shape(y_true)[0]
        pos_mask = pos_mask - tf.eye(batch_size)
        sim = tf.matmul(embeddings, embeddings, transpose_b=True) / self.temperature
        exp_sim = tf.exp(sim)
        sum_exp = tf.reduce_sum(exp_sim, axis=1, keepdims=True)
        pos_sum = tf.reduce_sum(exp_sim * pos_mask, axis=1)
        pos_count = tf.reduce_sum(pos_mask, axis=1)
        pos_sum = tf.where(pos_count > 0, pos_sum, tf.ones_like(pos_sum) * 1e-10)
        frac = (pos_sum / tf.maximum(pos_count, 1e-10)) / (sum_exp[:, 0] + 1e-10)
        loss = -tf.math.log(frac + 1e-10)
        return tf.reduce_mean(loss)

# ===== Other functions (load_data, generate_simulated_external_data, etc.) unchanged =====
def load_data(args):
    main_path = os.path.join(args.base_path, args.main_file)
    protein_path = os.path.join(args.base_path, args.protein_file)
    ligand_path = os.path.join(args.base_path, args.ligand_file)
    df_main = pd.read_csv(main_path)
    df_protein = pd.read_csv(protein_path)
    df_ligand = pd.read_csv(ligand_path)
    if 'DSDP_Score' not in df_main.columns:
        raise ValueError("Missing DSDP_Score column in main file")
    df = pd.merge(df_main, df_protein, on='protein', how='inner')
    df = pd.merge(df, df_ligand, on='ligand', how='inner')
    return df, df_main, df_protein, df_ligand

def generate_simulated_external_data(df_main, output_path):
    simulated_external = df_main[['protein', 'ligand']].drop_duplicates().reset_index(drop=True)
    simulated_external.to_csv(output_path, index=False)
    print(f"Simulated external data saved: {output_path}")
    return output_path

def run_external_prediction(args, protein_file, ligand_file, combination_file, threshold,
                            external_script_path=EXTERNAL_VERIFY_SCRIPT,
                            output_pred_path="temp_pred.csv"):
    # Check dependency files
    for f in ['my_model.h5', 'preprocessor.joblib', external_script_path]:
        if not os.path.exists(f):
            print(f"Missing required file: {f}")
            return None

    my_env = os.environ.copy()
    my_env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    my_env['TF_ENABLE_ONEDNN_OPTS'] = '0'

    cmd = ['python', external_script_path, '-p', protein_file,
           '-l', ligand_file, '-g', combination_file,
           '-t', str(threshold), '-o', output_pred_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=my_env)
    except subprocess.CalledProcessError as e:
        print(f"Prediction failed (rc={e.returncode})")
        if e.stdout:
            print(f"stdout tail: {e.stdout[-500:]}")
        if e.stderr:
            print(f"stderr tail: {e.stderr[-500:]}")
        return None
    
    pred_df = pd.read_csv(output_pred_path)
    main_full_path = os.path.join(args.base_path, args.main_file)
    main_df = pd.read_csv(main_full_path)
    true_labels = main_df.set_index(['protein', 'ligand'])['Priority'].to_dict()
    pred_df['True_Priority'] = pred_df.apply(lambda row: true_labels.get((row['protein'], row['ligand']), np.nan), axis=1)
    pred_df['True_Class'] = pred_df['True_Priority'].apply(lambda x: 1 if x == 'ACTIVATING' else (0 if x == 'INHIBITORY' else np.nan))
    valid_mask = ~np.isnan(pred_df['True_Class'])
    if not valid_mask.any(): return None
    accuracy = accuracy_score(pred_df.loc[valid_mask, 'True_Class'], pred_df.loc[valid_mask, 'Predicted_Class'])
    if os.path.exists(output_pred_path): os.remove(output_pred_path)
    return accuracy

def optimize_threshold_by_external(args, start_threshold, step_size, protein_file, ligand_file, combination_file, max_iter=20, external_script_path=EXTERNAL_VERIFY_SCRIPT):
    threshold_accuracy_map = {}
    test_thresholds = [max(0.0, min(1.0, start_threshold-step_size)), start_threshold, max(0.0, min(1.0, start_threshold+step_size))]
    for t in test_thresholds:
        acc = run_external_prediction(args, protein_file, ligand_file, combination_file, t, external_script_path)
        if acc is not None: threshold_accuracy_map[t] = acc
    if not threshold_accuracy_map: return start_threshold, 0.0, threshold_accuracy_map
    best_t_initial = max(threshold_accuracy_map.keys(), key=lambda x: threshold_accuracy_map[x])
    best_acc_initial = threshold_accuracy_map[best_t_initial]
    direction = 0
    if best_t_initial == start_threshold + step_size: direction = 1
    elif best_t_initial == start_threshold - step_size: direction = -1
    else: return start_threshold, best_acc_initial, threshold_accuracy_map
    current_t, current_acc, iter_count = best_t_initial, best_acc_initial, 0
    while iter_count < max_iter:
        next_t = max(0.0, min(1.0, current_t + direction * step_size))
        if next_t == current_t: break
        next_acc = run_external_prediction(args, protein_file, ligand_file, combination_file, next_t, external_script_path)
        if next_acc is None: break
        threshold_accuracy_map[next_t] = next_acc
        if next_acc > current_acc: current_t, current_acc = next_t, next_acc
        else: break
        iter_count +=1
    return current_t, current_acc, threshold_accuracy_map

def weighted_binary_crossentropy(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# ===== Fine-tune 1: Activity (unchanged) =====
def fine_tune_activity_only(model, preprocessor, df, args, X_val, y_val):
    fine_tune_mask = (~np.isnan(df['Activity_Value']) & ~np.isnan(df['Priority'].apply(lambda x: 1 if x=='ACTIVATING' else 0 if x=='INHIBITORY' else np.nan)))
    df_fine = df[fine_tune_mask].copy()
    print(f"\n===== FINE-TUNE 1: Activity_Value =====")
    print(f"Valid samples: {len(df_fine)} / Total: {len(df)}")
    if len(df_fine) ==0: return model
    y_fine = df_fine['Priority'].apply(lambda x:1 if x=='ACTIVATING' else 0)
    X_fine = pd.concat([df_fine.filter(like='protein_'), df_fine.filter(like='ligand_'), df_fine[['protein','ligand']]], axis=1)
    X_fine_processed = preprocessor.transform(X_fine)
    X_val_processed = preprocessor.transform(X_val)
    activity_vals = df_fine['Activity_Value'].values
    min_val, max_val = np.min(activity_vals), np.max(activity_vals)
    weights = 0.5 + 1.5*(activity_vals-min_val)/(max_val-min_val) if max_val>min_val else np.ones_like(activity_vals)
    weights *= args.weight_scale
    print(f"Activity weight range: {np.min(weights):.2f} ~ {np.max(weights):.2f}")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')], weighted_metrics=['accuracy', tf.keras.metrics.AUC(name='weighted_auc')])
    pre_fine_metrics = model.evaluate(X_fine_processed, y_fine, sample_weight=weights, verbose=0)
    pre_fine_loss, pre_fine_acc, pre_fine_auc = pre_fine_metrics[0], pre_fine_metrics[1], pre_fine_metrics[2]
    for layer in model.layers[:-4]: layer.trainable=False
    for layer in model.layers[-4:]: layer.trainable=True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')], weighted_metrics=['accuracy', tf.keras.metrics.AUC(name='weighted_auc')])
    early_stopping_fine = EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-6, verbose=1, restore_best_weights=True)
    history_fine = model.fit(X_fine_processed, y_fine, sample_weight=weights, epochs=args.fine_tune_epochs, batch_size=16, validation_data=(X_val_processed, y_val), callbacks=[early_stopping_fine], verbose=1)
    for layer in model.layers: layer.trainable=True
    post_fine_metrics = model.evaluate(X_fine_processed, y_fine, sample_weight=weights, verbose=0)
    post_fine_loss, post_fine_acc, post_fine_auc = post_fine_metrics[0], post_fine_metrics[1], post_fine_metrics[2]
    val_metrics = model.evaluate(preprocessor.transform(X_val), y_val, verbose=0)
    val_loss_post, val_acc_post, val_auc_post = val_metrics[0], val_metrics[1], val_metrics[2]
    print(f"Val AUC: {val_auc_post:.4f}")
    print(f"Fine-tune 1 done, real epochs: {len(history_fine.history['loss'])}")
    training_metrics["ft1_valid_samples"]=int(len(df_fine))
    training_metrics["ft1_total_samples"]=int(len(df))
    training_metrics["ft1_weight_min"]=float(np.min(weights))
    training_metrics["ft1_weight_max"]=float(np.max(weights))
    training_metrics["ft1_real_epochs"]=int(len(history_fine.history['loss']))
    training_metrics["ft1_pre_auc"]=float(pre_fine_auc)
    training_metrics["ft1_pre_loss"]=float(pre_fine_loss)
    training_metrics["ft1_pre_acc"]=float(pre_fine_acc)
    training_metrics["ft1_post_auc"]=float(post_fine_auc)
    training_metrics["ft1_post_loss"]=float(post_fine_loss)
    training_metrics["ft1_post_acc"]=float(post_fine_acc)
    training_metrics["ft1_val_auc"]=float(val_auc_post)
    training_metrics["ft1_val_loss"]=float(val_loss_post)
    training_metrics["ft1_val_acc"]=float(val_acc_post)
    save_metrics()
    return model

# ===== Fine-tune 2: DSDP (unchanged) =====
def fine_tune_dsdp_only(model, preprocessor, df, args, X_val, y_val):
    fine_tune_mask = (~np.isnan(df['DSDP_Score']) & ~np.isnan(df['Priority'].apply(lambda x:1 if x=='ACTIVATING' else 0 if x=='INHIBITORY' else np.nan)))
    df_fine = df[fine_tune_mask].copy()
    print(f"\n===== FINE-TUNE 2: DSDP_Score =====")
    print(f"Valid samples: {len(df_fine)} / Total: {len(df)}")
    if len(df_fine)==0: return model
    y_fine = df_fine['Priority'].apply(lambda x:1 if x=='ACTIVATING' else 0)
    X_fine = pd.concat([df_fine.filter(like='protein_'), df_fine.filter(like='ligand_'), df_fine[['protein','ligand']]], axis=1)
    X_fine_processed = preprocessor.transform(X_fine)
    X_val_processed = preprocessor.transform(X_val)
    dsdp_scores = df_fine['DSDP_Score'].values
    k = args.dsdp_weight_factor
    weights = 1.0 + k * np.exp((-3.0 - dsdp_scores) / 4.0)
    dsdp_weight_mask = dsdp_scores < -7
    print(f"DSDP < -7 count: {np.sum(dsdp_weight_mask)} / {len(df_fine)}")
    print(f"DSDP weight range: {np.min(weights):.2f} ~ {np.max(weights)}")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')], weighted_metrics=['accuracy', tf.keras.metrics.AUC(name='weighted_auc')])
    pre_fine_metrics = model.evaluate(X_fine_processed, y_fine, sample_weight=weights, verbose=0)
    pre_fine_loss, pre_fine_acc, pre_fine_auc = pre_fine_metrics[0], pre_fine_metrics[1], pre_fine_metrics[2]
    for layer in model.layers[:-4]: layer.trainable=False
    for layer in model.layers[-4:]: layer.trainable=True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr*0.2), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')], weighted_metrics=['accuracy', tf.keras.metrics.AUC(name='weighted_auc')])
    early_stopping_fine = EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-6, verbose=1, restore_best_weights=True)
    history_fine = model.fit(X_fine_processed, y_fine, sample_weight=weights, epochs=args.fine_tune_epochs, batch_size=16, validation_data=(X_val_processed, y_val), callbacks=[early_stopping_fine], verbose=1)
    for layer in model.layers: layer.trainable=True
    post_fine_metrics = model.evaluate(X_fine_processed, y_fine, sample_weight=weights, verbose=0)
    post_fine_loss, post_fine_acc, post_fine_auc = post_fine_metrics[0], post_fine_metrics[1], post_fine_metrics[2]
    val_metrics = model.evaluate(preprocessor.transform(X_val), y_val, verbose=0)
    val_loss_post, val_acc_post, val_auc_post = val_metrics[0], val_metrics[1], val_metrics[2]
    print(f"Val AUC: {val_auc_post:.4f}")
    print(f"Fine-tune 2 done, real epochs: {len(history_fine.history['loss'])}")
    training_metrics["ft2_valid_samples"]=int(len(df_fine))
    training_metrics["ft2_total_samples"]=int(len(df))
    training_metrics["ft2_dsdp_low_count"]=int(np.sum(dsdp_weight_mask))
    training_metrics["ft2_weight_min"]=float(np.min(weights))
    training_metrics["ft2_weight_max"]=float(np.max(weights))
    training_metrics["ft2_dsdp_factor"]=float(args.dsdp_weight_factor)
    training_metrics["ft2_real_epochs"]=int(len(history_fine.history['loss']))
    training_metrics["ft2_pre_auc"]=float(pre_fine_auc)
    training_metrics["ft2_pre_loss"]=float(pre_fine_loss)
    training_metrics["ft2_pre_acc"]=float(pre_fine_acc)
    training_metrics["ft2_post_auc"]=float(post_fine_auc)
    training_metrics["ft2_post_loss"]=float(post_fine_loss)
    training_metrics["ft2_post_acc"]=float(post_fine_acc)
    training_metrics["ft2_val_auc"]=float(val_auc_post)
    training_metrics["ft2_val_loss"]=float(val_loss_post)
    training_metrics["ft2_val_acc"]=float(val_acc_post)
    save_metrics()
    return model

# ===== Fine-tune 3: contrastive-learning enhanced version (modified) =====
def fine_tune_target_protein_only(model, preprocessor, df, args, target_prot_id, X_val, y_val, metrics_store=None, save_func=save_metrics):
    fine_tune_mask = (df['protein'] == target_prot_id) & ~np.isnan(df['Priority'].apply(lambda x:1 if x=='ACTIVATING' else 0 if x=='INHIBITORY' else np.nan))
    df_fine = df[fine_tune_mask].copy()
    print(f"\n===== FINE-TUNE 3: Target Exp Protein {target_prot_id} =====")
    print(f"Valid exp protein samples: {len(df_fine)} / Total full dataset: {len(df)}")
    if len(df_fine) == 0:
        print(f"No valid labeled data for target protein {target_prot_id}, skip FT3")
        return model

    y_fine = df_fine['Priority'].apply(lambda x:1 if x=='ACTIVATING' else 0)
    X_fine = pd.concat([df_fine.filter(like='protein_'), df_fine.filter(like='ligand_'), df_fine[['protein','ligand']]], axis=1)
    X_fine_processed = preprocessor.transform(X_fine)
    X_val_processed = preprocessor.transform(X_val)

    base_weight = 1.0
    exp_boost = 3.0
    weights = np.full(len(df_fine), base_weight * exp_boost)
    print(f"Exp protein weight fixed boost: {exp_boost}, weight range: {np.min(weights):.2f} ~ {np.max(weights):.2f}")

    # ===== Freeze/unfreeze (same as original logic) =====
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True

    ft3_lr = args.fine_tune_lr * 0.2 * 0.2

    # ===== Decide whether to enable contrastive learning =====
    use_contrastive = (args.contrastive_weight > 0) and (len(df_fine) >= 4)

    if use_contrastive:
        print(f">>> Contrastive learning ENABLED (weight={args.contrastive_weight}, temp={args.contrastive_temp})")
        return _fine_tune_contrastive(
            model, X_fine_processed, y_fine.values, weights,
            X_val_processed, y_val.values,
            args, ft3_lr, metrics_store, save_func, df, len(df_fine), len(df),
            X_val, preprocessor
        )
    else:
        print(">>> Contrastive learning DISABLED (using original FT3)")
        if args.contrastive_weight <= 0:
            print("    Reason: --contrastive_weight = 0")
        if len(df_fine) < 4:
            print("    Reason: too few samples (<4) for contrastive loss")
        return _fine_tune_original(
            model, X_fine_processed, y_fine.values, weights,
            X_val_processed, y_val.values,
            args, ft3_lr, metrics_store, save_func, df, len(df_fine), len(df)
        )


def _fine_tune_original(model, X_fine, y_fine, weights, X_val, y_val, args, ft3_lr, metrics_store, save_func, df, n_valid, n_total):
    """Original FT3 logic (no contrastive learning)"""
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
                  weighted_metrics=['accuracy', tf.keras.metrics.AUC(name='weighted_auc')])
    pre_fine_metrics = model.evaluate(X_fine, y_fine, sample_weight=weights, verbose=0)
    pre_fine_loss, pre_fine_acc, pre_fine_auc = pre_fine_metrics[0], pre_fine_metrics[1], pre_fine_metrics[2]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ft3_lr), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
                  weighted_metrics=['accuracy', tf.keras.metrics.AUC(name='weighted_auc')])
    early_stopping_fine = EarlyStopping(monitor='val_dense_3_loss', patience=3, min_delta=1e-6, verbose=1, restore_best_weights=True)
    history_fine = model.fit(X_fine, y_fine, sample_weight=weights, epochs=args.fine_tune_epochs, batch_size=16,
                             validation_data=(X_val, y_val), callbacks=[early_stopping_fine], verbose=1)
    for layer in model.layers: layer.trainable = True
    post_fine_metrics = model.evaluate(X_fine, y_fine, sample_weight=weights, verbose=0)
    post_fine_loss, post_fine_acc, post_fine_auc = post_fine_metrics[0], post_fine_metrics[1], post_fine_metrics[2]
    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    val_loss_post, val_acc_post, val_auc_post = val_metrics[0], val_metrics[1], val_metrics[2]
    print(f"FT3 Val AUC: {val_auc_post:.4f}")
    print(f"Fine-tune 3 done, real epochs: {len(history_fine.history['loss'])}")

    if metrics_store is not None:
        metrics_store["ft3_valid_samples"] = int(n_valid)
        metrics_store["ft3_total_samples"] = int(n_total)
        metrics_store["ft3_weight_min"] = float(np.min(weights))
        metrics_store["ft3_weight_max"] = float(np.max(weights))
        metrics_store["ft3_real_epochs"] = int(len(history_fine.history['loss']))
        metrics_store["ft3_pre_auc"] = float(pre_fine_auc)
        metrics_store["ft3_pre_loss"] = float(pre_fine_loss)
        metrics_store["ft3_pre_acc"] = float(pre_fine_acc)
        metrics_store["ft3_post_auc"] = float(post_fine_auc)
        metrics_store["ft3_post_loss"] = float(post_fine_loss)
        metrics_store["ft3_post_acc"] = float(post_fine_acc)
        metrics_store["ft3_val_auc"] = float(val_auc_post)
        metrics_store["ft3_val_loss"] = float(val_loss_post)
        metrics_store["ft3_val_acc"] = float(val_acc_post)
        save_func()
    return model


def _fine_tune_contrastive(model, X_fine, y_fine, weights, X_val, y_val, args, ft3_lr, metrics_store, save_func, df, n_valid, n_total, X_val_raw, preprocessor):
    """Contrastive FT3: joint training with BCE + SupCon"""
    # 1. Build the two-head model (shared layers)
    # Model structure: ... → Dense(1024,relu) [layer -4] → BN [ -3] → Dropout [ -2] → Dense(1,sigmoid) [ -1]
    embed_source = model.layers[-4].output  # Dense(1024,relu) -> 1024-dim embedding

    # L2-normalize the embeddings (keeps norms stable)
    embedding = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='embedding_output')(embed_source)

    two_head = Model(inputs=model.input, outputs=[model.output, embedding], name='ft3_contrastive')

    # 2. Freeze/unfreeze (same as original FT3)
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True

    # 3. Compile: BCE for the classification head, SupCon for the embedding head
    classifier_name = model.layers[-1].name
    two_head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ft3_lr),
        loss={
            classifier_name: 'binary_crossentropy',
            'embedding_output': SupervisedContrastiveLoss(temperature=args.contrastive_temp)
        },
        loss_weights={
            classifier_name: 1.0,
            'embedding_output': args.contrastive_weight
        }
    )

    # 4. Prepare data in two-head format
    train_y = {classifier_name: y_fine, 'embedding_output': y_fine}
    train_w = {classifier_name: weights, 'embedding_output': weights}
    # ====== Change 1: pass only classification labels for validation ======
    val_y = {classifier_name: y_val}

    # 5. Evaluate pre-fine-tune metrics (single-head model is enough)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
                  weighted_metrics=[])
    pre_fine_metrics = model.evaluate(X_fine, y_fine, sample_weight=weights, verbose=0)
    if len(pre_fine_metrics) >= 3:
        pre_fine_loss, pre_fine_acc, pre_fine_auc = pre_fine_metrics[0], pre_fine_metrics[1], pre_fine_metrics[2]
    else:
        pre_fine_loss, pre_fine_acc, pre_fine_auc = pre_fine_metrics[0], 0.0, 0.0

    # ====== Change 2: EarlyStopping monitors the classification head loss ======
    es_monitor = f'val_{classifier_name}_loss'
    early_stopping_fine = EarlyStopping(monitor=es_monitor, patience=3, min_delta=1e-6, verbose=1, restore_best_weights=True)

    history_fine = two_head.fit(
        X_fine, train_y,
        sample_weight=train_w,
        epochs=args.fine_tune_epochs,
        batch_size=min(32, len(y_fine)),
        validation_data=(X_val, val_y),
        callbacks=[early_stopping_fine],
        verbose=1
    )

    # 7. Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    # 8. Evaluate post-fine-tune metrics (weights are shared, so model can be evaluated directly)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
                  weighted_metrics=[])
    post_fine_metrics = model.evaluate(X_fine, y_fine, sample_weight=weights, verbose=0)
    if len(post_fine_metrics) >= 3:
        post_fine_loss, post_fine_acc, post_fine_auc = post_fine_metrics[0], post_fine_metrics[1], post_fine_metrics[2]
    else:
        post_fine_loss, post_fine_acc, post_fine_auc = post_fine_metrics[0], 0.0, 0.0

    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    if len(val_metrics) >= 3:
        val_loss_post, val_acc_post, val_auc_post = val_metrics[0], val_metrics[1], val_metrics[2]
    else:
        val_loss_post, val_acc_post, val_auc_post = val_metrics[0], 0.0, 0.0

    print(f"FT3 Contrastive - Val AUC: {val_auc_post:.4f}")
    print(f"Fine-tune 3 (contrastive) done, real epochs: {len(history_fine.history['loss'])}")

    # 9. Save metrics
    if metrics_store is not None:
        metrics_store["ft3_valid_samples"] = int(n_valid)
        metrics_store["ft3_total_samples"] = int(n_total)
        metrics_store["ft3_weight_min"] = float(np.min(weights))
        metrics_store["ft3_weight_max"] = float(np.max(weights))
        metrics_store["ft3_real_epochs"] = int(len(history_fine.history['loss']))
        metrics_store["ft3_pre_auc"] = float(pre_fine_auc)
        metrics_store["ft3_pre_loss"] = float(pre_fine_loss)
        metrics_store["ft3_pre_acc"] = float(pre_fine_acc)
        metrics_store["ft3_post_auc"] = float(post_fine_auc)
        metrics_store["ft3_post_loss"] = float(post_fine_loss)
        metrics_store["ft3_post_acc"] = float(post_fine_acc)
        metrics_store["ft3_val_auc"] = float(val_auc_post)
        metrics_store["ft3_val_loss"] = float(val_loss_post)
        metrics_store["ft3_val_acc"] = float(val_acc_post)
        save_func()

    return model


# ===================== MAIN 5-FOLD CV =====================
if __name__ == "__main__":
    args = parse_args()
    args.base_path = os.path.abspath(args.base_path)
    df, df_main, df_protein, df_ligand = load_data(args)
    df, z_prot_cols, z_lig_cols = merge_latent_embedding(df)

    training_metrics["weight_scale"] = float(args.weight_scale)
    training_metrics["dsdp_weight_factor"] = float(args.dsdp_weight_factor)
    training_metrics["exp_protein_id"] = args.exp
    save_metrics()

    y = df['Priority'].apply(lambda x: 1 if x == 'ACTIVATING' else (0 if x == 'INHIBITORY' else np.nan))
    features = pd.concat([df.filter(like='protein_'), df.filter(like='ligand_'), df[['protein','ligand']]], axis=1)

    valid_mask = ~np.isnan(y)
    features_valid = features[valid_mask].reset_index(drop=True)
    y_valid = y[valid_mask].reset_index(drop=True)

    numeric_prefixes = ['protein_DeepFRI','protein_Peptides','protein_Protparam','protein_Zernike','protein_profeatx','ligand_Estate_atom','ligand_Estate_sum']
    binary_prefixes = ['ligand_MACCS']
    numeric_features = features_valid.columns[features_valid.columns.str.startswith(tuple(numeric_prefixes))]
    binary_features = features_valid.columns[features_valid.columns.str.startswith(tuple(binary_prefixes))]

    all_latent, _, _ = get_latent_feature_names()
    latent_features = [col for col in features_valid.columns if col in all_latent]
    full_numeric_features = list(numeric_features) + latent_features

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    binary_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, full_numeric_features),
        ('binary', binary_transformer, binary_features)
    ])
    preprocessor.fit(features_valid)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {"fold_auc":[],"fold_acc":[],"fold_ap":[],"fold_y_true":[],"fold_y_pred":[],"fold_pred_pos":[],"fold_pred_neg":[]}

    for fold, (train_idx, val_idx) in enumerate(skf.split(features_valid, y_valid)):
        print(f"\n{'='*50}\n                FOLD {fold+1}/{n_splits}\n{'='*50}")
        X_train_fold, X_val_fold = features_valid.iloc[train_idx], features_valid.iloc[val_idx]
        y_train_fold, y_val_fold = y_valid.iloc[train_idx], y_valid.iloc[val_idx]

        tf.keras.backend.clear_session()
        input_dim = preprocessor.transform(X_train_fold).shape[1]
        X_input = Input(shape=(input_dim,), dtype='float32')
        FC1 = Dense(4096, activation='relu')(X_input)
        FC1 = BatchNormalization()(FC1)
        FC1 = Dropout(0.2)(FC1)
        FC2 = Dense(2048, activation='relu')(FC1)
        FC2 = BatchNormalization()(FC2)
        FC2 = Dropout(0.2)(FC2)
        FC3 = Dense(1024, activation='relu')(FC2)
        FC3 = BatchNormalization()(FC3)
        FC3 = Dropout(0.2)(FC3)
        predictions = Dense(1, activation='sigmoid')(FC3)
        model = Model(inputs=X_input, outputs=predictions)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        history = model.fit(preprocessor.transform(X_train_fold), y_train_fold, epochs=500, batch_size=32, validation_data=(preprocessor.transform(X_val_fold), y_val_fold), callbacks=[early_stopping], verbose=1)

        final_epoch = early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else 100
        training_metrics["pretrain_epochs"] = int(final_epoch)
        training_metrics["pretrain_train_loss"] = float(history.history['loss'][-1])
        training_metrics["pretrain_train_acc"] = float(history.history['accuracy'][-1])
        training_metrics["pretrain_train_auc"] = float(history.history['auc'][-1])
        training_metrics["pretrain_val_loss"] = float(history.history['val_loss'][-1])
        training_metrics["pretrain_val_acc"] = float(history.history['val_accuracy'][-1])
        training_metrics["pretrain_val_auc"] = float(history.history['val_auc'][-1])
        save_metrics()

        model = fine_tune_activity_only(model, preprocessor, df, args, X_val_fold, y_val_fold)
        model = fine_tune_dsdp_only(model, preprocessor, df, args, X_val_fold, y_val_fold)
        if args.exp is not None:
            target_prot = args.exp.strip()
            model = fine_tune_target_protein_only(
                model, preprocessor, df, args, target_prot,
                X_val_fold, y_val_fold, metrics_store=training_metrics
            )

        y_pred_proba = model.predict(preprocessor.transform(X_val_fold)).ravel()
        y_true = y_val_fold.values
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_f = auc(fpr, tpr)
        ap_f = average_precision_score(y_true, y_pred_proba)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        acc_f = accuracy_score(y_true, y_pred)
        cv_results["fold_auc"].append(auc_f)
        cv_results["fold_acc"].append(acc_f)
        cv_results["fold_ap"].append(ap_f)
        cv_results["fold_y_true"].extend(y_true.tolist())
        cv_results["fold_y_pred"].extend(y_pred_proba.tolist())
        cv_results["fold_pred_pos"].extend(y_pred_proba[y_true==1].tolist())
        cv_results["fold_pred_neg"].extend(y_pred_proba[y_true==0].tolist())
        print(f"Fold {fold+1} Done | AUC={auc_f:.4f} | Acc={acc_f:.4f} | AP={ap_f:.4f}")

    cv_detail_df = pd.DataFrame({
        "fold": list(range(1, n_splits + 1)),
        "fold_auc": cv_results["fold_auc"],
        "fold_acc": cv_results["fold_acc"],
        "fold_ap": cv_results["fold_ap"]
    })
    cv_detail_df.to_csv("cv_fold_details.csv", index=False, encoding="utf-8-sig")
    print("cv_fold_details.csv saved")

    mean_auc, std_auc = np.mean(cv_results["fold_auc"]), np.std(cv_results["fold_auc"])
    mean_acc, std_acc = np.mean(cv_results["fold_acc"]), np.std(cv_results["fold_acc"])
    mean_ap, std_ap = np.mean(cv_results["fold_ap"]), np.std(cv_results["fold_ap"])
    print("\n"+"="*60+f"\n                5-FOLD CV RESULT\nAUC: {mean_auc:.4f}±{std_auc:.4f}\nACC: {mean_acc:.4f}±{std_acc:.4f}\nAP:  {mean_ap:.4f}±{std_ap:.4f}\n"+"="*60)

    training_metrics["cv_mean_auc"]=float(mean_auc)
    training_metrics["cv_std_auc"]=float(std_auc)
    training_metrics["cv_mean_acc"]=float(mean_acc)
    training_metrics["cv_std_acc"]=float(std_acc)
    training_metrics["cv_mean_ap"]=float(mean_ap)
    training_metrics["cv_std_ap"]=float(std_ap)
    save_metrics()

    print("\nTraining final model on full dataset...")
    tf.keras.backend.clear_session()
    input_dim = preprocessor.transform(features_valid).shape[1]
    X_input = Input(shape=(input_dim,))
    FC1 = Dense(4096, activation='relu')(X_input)
    FC1 = BatchNormalization()(FC1)
    FC1 = Dropout(0.2)(FC1)
    FC2 = Dense(2048, activation='relu')(FC1)
    FC2 = BatchNormalization()(FC2)
    FC2 = Dropout(0.2)(FC2)
    FC3 = Dense(1024, activation='relu')(FC2)
    FC3 = BatchNormalization()(FC3)
    FC3 = Dropout(0.2)(FC3)
    predictions = Dense(1, activation='sigmoid')(FC3)
    final_model = Model(inputs=X_input, outputs=predictions)
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    from sklearn.model_selection import train_test_split
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(features_valid, y_valid, test_size=0.1, stratify=y_valid, random_state=42)

    early_stopping_final = EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-6, verbose=1, restore_best_weights=True)
    final_history = final_model.fit(preprocessor.transform(X_train_final), y_train_final, epochs=500, batch_size=32, validation_data=(preprocessor.transform(X_val_final), y_val_final), callbacks=[early_stopping_final], verbose=1)

    training_metrics["final_train_loss"] = float(final_history.history['loss'][-1])
    training_metrics["final_train_acc"] = float(final_history.history['accuracy'][-1])
    training_metrics["final_train_auc"] = float(final_history.history['auc'][-1])
    training_metrics["final_val_loss"] = float(final_history.history['val_loss'][-1])
    training_metrics["final_val_acc"] = float(final_history.history['val_accuracy'][-1])
    training_metrics["final_val_auc"] = float(final_history.history['val_auc'][-1])
    save_metrics()

    final_model = fine_tune_activity_only(final_model, preprocessor, df, args, features_valid, y_valid)
    final_model = fine_tune_dsdp_only(final_model, preprocessor, df, args, features_valid, y_valid)
    if args.exp is not None:
        target_prot_id = args.exp.strip()
        final_model = fine_tune_target_protein_only(
            final_model, preprocessor, df, args, target_prot_id,
            features_valid, y_valid, metrics_store=training_metrics
        )

    final_model.save('my_model.h5')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    all_latent, _, _ = get_latent_feature_names()
    all_features = full_numeric_features + binary_features.tolist()
    with open('feature_columns.json','w') as f:
        json.dump(all_features, f)
    print("Final model saved")

    y_pred_final = final_model.predict(preprocessor.transform(features_valid)).ravel()
    fpr, tpr, thresholds_j = roc_curve(y_valid, y_pred_final)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_j = thresholds_j[optimal_idx]
    training_metrics["youden_sensitivity"] = float(tpr[optimal_idx])
    training_metrics["youden_specificity"] = float(1 - fpr[optimal_idx])
    save_metrics()

    simulated_external_path = "simulated_external_data.csv"
    generate_simulated_external_data(df_main, simulated_external_path)
    best_threshold, best_accuracy, threshold_accuracy_map = optimize_threshold_by_external(
        args, optimal_threshold_j, args.step_size,
        os.path.join(args.base_path, args.protein_file),
        os.path.join(args.base_path, args.ligand_file),
        simulated_external_path,
        external_script_path=EXTERNAL_VERIFY_SCRIPT
    )

    optimal_threshold_dict = {
        "youden_threshold": float(optimal_threshold_j),
        "optimal_threshold": float(best_threshold),
        "external_best_acc": float(best_accuracy)
    }
    with open("optimal_threshold.json", "w", encoding="utf-8") as f:
        json.dump(optimal_threshold_dict, f, indent=2)
    print(f"Optimal threshold saved: {best_threshold:.4f}")
    training_metrics["youden_threshold"] = float(optimal_threshold_j)
    training_metrics["best_threshold"] = float(best_threshold)
    training_metrics["external_best_accuracy"] = float(best_accuracy)
    save_metrics()

    # ===================== Plotting (same as original code) =====================
    plt.figure(figsize=(8,5))
    plt.bar(["AUC","Accuracy","AP"], [mean_auc,mean_acc,mean_ap], yerr=[std_auc,std_acc,std_ap], capsize=10, color=['#ff9999','#66b3ff','#99ff99'])
    plt.ylim(0,1)
    plt.title("5-Fold CV Performance (Mean±Std)")
    plt.grid(alpha=0.3)
    plt.savefig("cv_5fold_metrics.pdf", bbox_inches='tight')
    plt.close()

    fpr_all, tpr_all, _ = roc_curve(cv_results["fold_y_true"], cv_results["fold_y_pred"])
    plt.figure(figsize=(8,6))
    plt.plot(fpr_all, tpr_all, lw=3, color='red', label=f'Mean ROC (AUC={mean_auc:.2f}±{std_auc:.2f})')
    plt.plot([0,1],[0,1],'--',c='navy')
    plt.xlabel('FPR'), plt.ylabel('TPR'), plt.title('5-Fold CV ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig("cv_5fold_roc.pdf", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    plt.boxplot([cv_results["fold_pred_neg"], cv_results["fold_pred_pos"]], labels=["INHIBITORY","ACTIVATING"])
    plt.title("5-Fold CV Prediction Distribution")
    plt.ylabel("Probability")
    plt.grid(alpha=0.3)
    plt.savefig("cv_5fold_boxplot.pdf", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,5))
    x_ticks = list(range(1, n_splits + 1))
    plt.plot(x_ticks, cv_results["fold_auc"], 'o-', label='AUC', lw=2)
    plt.plot(x_ticks, cv_results["fold_acc"], 's-', label='Accuracy', lw=2)
    plt.plot(x_ticks, cv_results["fold_ap"], 'd-', label='AP', lw=2)
    plt.xticks(x_ticks)
    plt.ylim(0.5,1)
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('5-Fold CV Metric Trend')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("cv_5fold_trend.pdf", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    thresholds_list = sorted(threshold_accuracy_map.keys())
    accuracies_list = [threshold_accuracy_map[t] for t in thresholds_list]
    plt.plot(thresholds_list, accuracies_list, 'b-o', lw=2)
    plt.scatter(best_threshold, best_accuracy, c='red', s=150, label=f'Best {best_threshold:.2f}')
    plt.scatter(optimal_threshold_j, threshold_accuracy_map.get(optimal_threshold_j,0), c='orange', s=100, label=f'Youden {optimal_threshold_j:.2f}')
    plt.xlabel('Threshold'), plt.ylabel('Accuracy'), plt.title('Threshold vs Accuracy')
    plt.grid(alpha=0.3), plt.legend()
    plt.savefig('threshold_accuracy_iterative.pdf', bbox_inches='tight')
    plt.close()

    fpr_final, tpr_final, _ = roc_curve(y_valid, y_pred_final)
    roc_auc = auc(fpr_final, tpr_final)
    plt.figure(figsize=(8,6))
    plt.plot(fpr_final, tpr_final, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'--',c='navy',lw=2)
    idx_y = np.argmin(np.abs(thresholds_j - optimal_threshold_j))
    idx_b = np.argmin(np.abs(thresholds_j - best_threshold))
    plt.scatter(fpr_final[idx_y], tpr_final[idx_y], c='orange', s=120, label=f'Youden {optimal_threshold_j:.2f}')
    plt.scatter(fpr_final[idx_b], tpr_final[idx_b], c='red', s=120, label=f'Best {best_threshold:.2f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.plot(final_history.history['loss'], label='Train loss')
    plt.plot(final_history.history['val_loss'], label='Val loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(132)
    plt.plot(final_history.history['accuracy'], label='Train acc')
    plt.plot(final_history.history['val_accuracy'], label='Val acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(133)
    plt.plot(final_history.history['auc'], label='Train AUC')
    plt.plot(final_history.history['val_auc'], label='Val AUC')
    plt.title('AUC')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.pdf', bbox_inches='tight')
    plt.close()

    y_pred_pos = y_pred_final[y_valid == 1]
    y_pred_neg = y_pred_final[y_valid == 0]
    plt.figure(figsize=(10,6))
    plt.hist(y_pred_neg, bins=50, alpha=0.7, label='INHIBITORY', color='blue')
    plt.hist(y_pred_pos, bins=50, alpha=0.7, label='ACTIVATING', color='red')
    plt.axvline(optimal_threshold_j, c='orange', ls='--', lw=2, label=f'Youden {optimal_threshold_j:.2f}')
    plt.axvline(best_threshold, c='green', ls='--', lw=2, label=f'Best {best_threshold:.2f}')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('probability_distribution.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    y_pred_y = (y_pred_final >= optimal_threshold_j).astype(int)
    cm_y = confusion_matrix(y_valid, y_pred_y)
    ConfusionMatrixDisplay(cm_y, display_labels=['INHIBITORY','ACTIVATING']).plot(cmap='Blues', ax=plt.gca())
    plt.title(f'Youden {optimal_threshold_j:.2f}')
    plt.subplot(122)
    y_pred_b = (y_pred_final >= best_threshold).astype(int)
    cm_b = confusion_matrix(y_valid, y_pred_b)
    ConfusionMatrixDisplay(cm_b, display_labels=['INHIBITORY','ACTIVATING']).plot(cmap='Blues', ax=plt.gca())
    plt.title(f'Best {best_threshold:.2f}')
    plt.tight_layout()
    plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
    plt.close()

    precision, recall, thresholds_pr = precision_recall_curve(y_valid, y_pred_final)
    avg_precision = average_precision_score(y_valid, y_pred_final)
    training_metrics["test_avg_precision"] = float(avg_precision)
    training_metrics["test_roc_auc"] = float(roc_auc)
    save_metrics()

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, color='darkgreen', lw=2, label=f'AP = {avg_precision:.2f}')
    idx_y_pr = np.argmin(np.abs(thresholds_pr - optimal_threshold_j))
    idx_b_pr = np.argmin(np.abs(thresholds_pr - best_threshold))
    plt.scatter(recall[idx_y_pr], precision[idx_y_pr], c='orange', s=120, label=f'Youden {optimal_threshold_j:.2f}')
    plt.scatter(recall[idx_b_pr], precision[idx_b_pr], c='red', s=120, label=f'Best {best_threshold:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.savefig('precision_recall_curve.pdf', bbox_inches='tight')
    plt.close()

    # ===================== --exp target protein dedicated output folder =====================
    if args.exp is not None:
        import shutil
        target_protein_id = args.exp.strip()
        ori_cwd = os.getcwd()
        out_dir = f"{target_protein_id}_exp_model"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        shutil.copy(os.path.join(ori_cwd, "my_model.h5"), os.path.join(out_dir, "my_model.h5"))
        shutil.copy(os.path.join(ori_cwd, "preprocessor.joblib"), os.path.join(out_dir, "preprocessor.joblib"))
        shutil.copy(os.path.join(ori_cwd, "feature_columns.json"), os.path.join(out_dir, "feature_columns.json"))
        shutil.copy(os.path.join(ori_cwd, "training_metrics.json"), os.path.join(out_dir, "training_metrics.json"))
        shutil.copy(os.path.join(ori_cwd, "external_data_verification_v3_to_v23.py"), os.path.join(out_dir, "external_data_verification_v3_to_v23.py"))

        os.chdir(out_dir)
        print(f"\n==================== EXPORT EXP TARGET PROTEIN {target_protein_id} FULL RESULTS ====================")

        df_exp = df[df["protein"] == target_protein_id].copy()
        df_exp_main = df_main[df_main["protein"] == target_protein_id].copy()
        sample_count = len(df_exp)
        print(f"Exp target protein total samples: {sample_count}")

        y_exp = df_exp['Priority'].apply(lambda x: 1 if x == 'ACTIVATING' else (0 if x == 'INHIBITORY' else np.nan))
        feat_exp = pd.concat([df_exp.filter(like='protein_'), df_exp.filter(like='ligand_'), df_exp[['protein','ligand']]], axis=1)
        mask_exp_valid = ~np.isnan(y_exp)
        feat_exp_valid = feat_exp[mask_exp_valid].reset_index(drop=True)
        y_exp_valid = y_exp[mask_exp_valid].reset_index(drop=True)

        y_pred_exp = final_model.predict(preprocessor.transform(feat_exp_valid)).ravel()
        fpr_exp, tpr_exp, th_j_exp = roc_curve(y_exp_valid, y_pred_exp)
        opt_idx_j_exp = np.argmax(tpr_exp - fpr_exp)
        youden_t_exp = th_j_exp[opt_idx_j_exp]

        sim_ext_path = "simulated_external_data.csv"
        generate_simulated_external_data(df_main, sim_ext_path)
        best_t_exp, best_acc_exp, th_acc_map_exp = optimize_threshold_by_external(
            args, youden_t_exp, args.step_size,
            os.path.join(args.base_path, args.protein_file),
            os.path.join(args.base_path, args.ligand_file),
            sim_ext_path,
            external_script_path=EXTERNAL_VERIFY_SCRIPT
        )
        opt_th_dict_ext = {
            "youden_threshold": float(youden_t_exp),
            "optimal_threshold": float(best_t_exp),
            "external_best_acc": float(best_acc_exp)
        }
        with open("optimal_threshold.json", "w", encoding="utf-8") as f:
            json.dump(opt_th_dict_ext, f, indent=2)

        prec_ext, rec_ext, _ = precision_recall_curve(y_exp_valid, y_pred_exp)
        ap_ext = average_precision_score(y_exp_valid, y_pred_exp)
        roc_auc_ext = auc(fpr_exp, tpr_exp)


        # Plot
        plt.figure(figsize=(10,6))
        ths_sorted = sorted(th_acc_map_exp.keys())
        accs_sorted = [th_acc_map_exp[t] for t in ths_sorted]
        plt.plot(ths_sorted, accs_sorted, 'b-o', lw=2)
        plt.scatter(best_t_exp, best_acc_exp, c='red', s=150, label=f'Best {best_t_exp:.2f}')
        plt.scatter(youden_t_exp, th_acc_map_exp.get(youden_t_exp,0), c='orange', s=100, label=f'Youden {youden_t_exp:.2f}')
        plt.xlabel('Threshold'), plt.ylabel('Accuracy'), plt.title(f'Threshold vs Accuracy | Exp Protein {target_protein_id}')
        plt.grid(alpha=0.3), plt.legend()
        plt.savefig('threshold_accuracy_iterative.pdf', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,6))
        plt.plot(fpr_exp, tpr_exp, color='darkorange', lw=2, label=f'AUC = {roc_auc_ext:.2f}')
        plt.plot([0,1],[0,1],'--',c='navy',lw=2)
        idx_y = np.argmin(np.abs(th_j_exp - youden_t_exp))
        idx_b = np.argmin(np.abs(th_j_exp - best_t_exp))
        plt.scatter(fpr_exp[idx_y], tpr_exp[idx_y], c='orange', s=120, label=f'Youden {youden_t_exp:.2f}')
        plt.scatter(fpr_exp[idx_b], tpr_exp[idx_b], c='red', s=120, label=f'Best {best_t_exp:.2f}')
        plt.xlabel('FPR'), plt.ylabel('TPR'), plt.title(f'ROC Curve | Exp Protein {target_protein_id}')
        plt.legend(loc='lower right'), plt.grid(alpha=0.3)
        plt.savefig('roc_curve.pdf', bbox_inches='tight')
        plt.close()

        y_pred_p_ext = y_pred_exp[y_exp_valid == 1]
        y_pred_n_ext = y_pred_exp[y_exp_valid == 0]
        plt.figure(figsize=(10,6))
        plt.hist(y_pred_n_ext, bins=50, alpha=0.7, label='INHIBITORY', color='blue')
        plt.hist(y_pred_p_ext, bins=50, alpha=0.7, label='ACTIVATING', color='red')
        plt.axvline(youden_t_exp, c='orange', ls='--', lw=2, label=f'Youden {youden_t_exp:.2f}')
        plt.axvline(best_t_exp, c='green', ls='--', lw=2, label=f'Best {best_t_exp:.2f}')
        plt.xlabel('Probability'), plt.ylabel('Frequency')
        plt.title(f'Prediction Distribution | Exp Protein {target_protein_id}')
        plt.legend(), plt.grid(alpha=0.3)
        plt.savefig('probability_distribution.pdf', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12,5))
        plt.subplot(121)
        y_pred_y_ext = (y_pred_exp >= youden_t_exp).astype(int)
        cm_y_ext = confusion_matrix(y_exp_valid, y_pred_y_ext)
        ConfusionMatrixDisplay(cm_y_ext, display_labels=['INHIBITORY','ACTIVATING']).plot(cmap='Blues', ax=plt.gca())
        plt.title(f'Youden {youden_t_exp:.2f}')
        plt.subplot(122)
        y_pred_b_ext = (y_pred_exp >= best_t_exp).astype(int)
        cm_b_ext = confusion_matrix(y_exp_valid, y_pred_b_ext)
        ConfusionMatrixDisplay(cm_b_ext, display_labels=['INHIBITORY','ACTIVATING']).plot(cmap='Blues', ax=plt.gca())
        plt.title(f'Best {best_t_exp:.2f}')
        plt.suptitle(f"Confusion Matrix | Exp Protein {target_protein_id}")
        plt.tight_layout()
        plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,6))
        plt.plot(rec_ext, prec_ext, color='darkgreen', lw=2, label=f'AP = {ap_ext:.2f}')
        idx_y_pr = np.argmin(np.abs(th_j_exp - youden_t_exp))
        idx_b_pr = np.argmin(np.abs(th_j_exp - best_t_exp))
        plt.scatter(rec_ext[idx_y_pr], prec_ext[idx_y_pr], c='orange', s=120, label=f'Youden {youden_t_exp:.2f}')
        plt.scatter(rec_ext[idx_b_pr], prec_ext[idx_b_pr], c='red', s=120, label=f'Best {best_t_exp:.2f}')
        plt.xlabel('Recall'), plt.ylabel('Precision')
        plt.title(f'PR Curve | Exp Protein {target_protein_id}')
        plt.legend(loc='lower left'), plt.grid(alpha=0.3)
        plt.savefig('precision_recall_curve.pdf', bbox_inches='tight')
        plt.close()

        if os.path.exists(sim_ext_path):
            os.remove(sim_ext_path)
        os.chdir(ori_cwd)
        print(f"==================== EXP PIPELINE FOR {target_protein_id} FINISHED, OUTPUT DIR: {out_dir} ====================\n")

    print("\n=== ALL DONE ===")
