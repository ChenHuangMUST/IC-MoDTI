import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score, ConfusionMatrixDisplay,
    accuracy_score
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import argparse
import os
from encoder_utils import merge_latent_embedding, get_latent_feature_names


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train model with 5-Fold CV, deduplication and optimized EarlyStop')
    # Positive file arguments
    parser.add_argument('--pg', required=True, help='Positive group file path (protein, ligand, DSDP_Score)')
    parser.add_argument('--pp', required=True, help='Positive protein descriptors file path (protein + protein_* columns)')
    parser.add_argument('--pl', required=True, help='Positive ligand descriptors file path (ligand + ligand_* columns)')
    # Negative file arguments
    parser.add_argument('--ng', required=True, help='Negative group file path (protein, ligand, DSDP_Score)')
    parser.add_argument('--np', required=True, help='Negative protein descriptors file path (protein + protein_* columns)')
    parser.add_argument('--nl', required=True, help='Negative ligand descriptors file path (ligand + ligand_* columns)')
    return parser.parse_args()


# Load and merge datasets
def load_and_merge_data(group_path, protein_path, ligand_path, is_positive=True):
    group_df = pd.read_csv(group_path)
    protein_df = pd.read_csv(protein_path).drop_duplicates(subset=['protein'])
    ligand_df = pd.read_csv(ligand_path).drop_duplicates(subset=['ligand'])

    merged_df = pd.merge(group_df, protein_df, on='protein', how='inner')
    merged_df = pd.merge(merged_df, ligand_df, on='ligand', how='inner')

    if is_positive:
        # Positive labels & filtering
        merged_df['label'] = merged_df['DSDP_Score'].apply(lambda x: 1 if x < -3 else 0)
        merged_df = merged_df[merged_df['DSDP_Score'] < -3]
    else:
        # Negative labels & filtering
        merged_df['label'] = merged_df['DSDP_Score'].apply(lambda x: 0 if (-3 < x < 0) else np.nan)

    return merged_df


# Compute Youden index and optimal threshold
def get_optimal_threshold(fpr, tpr, thresholds):
    youden = tpr - fpr
    idx_youden = np.argmax(youden)
    optimal_threshold_j = thresholds[idx_youden]
    fpr_j, tpr_j = fpr[idx_youden], tpr[idx_youden]

    idx_best = idx_youden
    best_threshold = thresholds[idx_best]
    fpr_b, tpr_b = fpr[idx_best], tpr[idx_best]

    return (optimal_threshold_j, best_threshold,
            fpr_j, tpr_j, fpr_b, tpr_b,
            youden, thresholds)


def main():
    args = parse_args()

    # 1. Load positive and negative samples (labels stored in a column to keep them in sync)
    positive_df = load_and_merge_data(
        group_path=args.pg,
        protein_path=args.pp,
        ligand_path=args.pl,
        is_positive=True
    )
    negative_df = load_and_merge_data(
        group_path=args.ng,
        protein_path=args.np,
        ligand_path=args.nl,
        is_positive=False
    )

    # Merge all data (features + labels in one table)
    total_df = pd.concat([positive_df, negative_df], axis=0)

    # 2. Global deduplication: deduplicate by the protein + ligand composite key (features and labels stay in sync)
    total_df = total_df.drop_duplicates(subset=["protein", "ligand"], keep="first")

    # 3. Filter out invalid samples with empty labels
    total_df = total_df.dropna(subset=["label"])
    total_df = total_df.reset_index(drop=True)

    print(f"After deduplication & filter, total samples: {len(total_df)}")
    print(f"Positive samples: {(total_df['label'] == 1).sum()}")
    print(f"Negative samples: {(total_df['label'] == 0).sum()}")

    # ========== Added: concatenate protein and ligand autoencoder latent representations ==========
    total_df, z_prot_cols, z_lig_cols = merge_latent_embedding(total_df)
    # ======================================================

    # Split features and labels
    y = total_df["label"].astype(int)
    features = total_df.drop(columns=["label", "DSDP_Score"])

    # Feature column definitions
    numeric_prefixes = [
        'protein_DeepFRI',
        'protein_Peptides',
        'protein_Protparam',
        'protein_Zernike',
        'protein_profeatx',
        'ligand_Estate_atom',
        'ligand_Estate_sum'
    ]
    binary_prefixes = ['ligand_MACCS']

    numeric_features = features.columns[features.columns.str.startswith(tuple(numeric_prefixes))]
    # Fix: add missing .columns
    binary_features = features.columns[features.columns.str.startswith(tuple(binary_prefixes))]

    # ========== Added code ==========
    all_latent, _, _ = get_latent_feature_names()
    latent_features = [c for c in features.columns if c in all_latent]
    # Latent vectors are continuous numeric features, add them to the numeric preprocessing branch
    full_numeric_features = list(numeric_features) + latent_features
    # ==============================

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    binary_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, full_numeric_features),
            ('binary', binary_transformer, binary_features)
        ]
    )


    # ===================== 5-fold stratified cross-validation =====================
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_auc_list = []
    cv_acc_list = []
    cv_ap_list = []
    cv_y_true_all = []
    cv_y_pred_prob_all = []
    last_history = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, y)):
        print(f"\n========== Fold {fold_idx + 1} / {n_splits} ==========")
        X_train_fold = features.iloc[train_idx]
        X_val_fold = features.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # -------- Initialize and fit the preprocessor separately inside each fold --------
        numeric_prefixes = [
            'protein_DeepFRI','protein_Peptides','protein_Protparam',
            'protein_Zernike','protein_profeatx','ligand_Estate_atom','ligand_Estate_sum'
        ]
        binary_prefixes = ['ligand_MACCS']
        numeric_features = X_train_fold.columns[X_train_fold.columns.str.startswith(tuple(numeric_prefixes))]
        binary_features = X_train_fold.columns[X_train_fold.columns.str.startswith(tuple(binary_prefixes))]

        # ========== Added: append latent features ==========
        all_latent, _, _ = get_latent_feature_names()
        latent_features = [col for col in X_train_fold.columns if col in all_latent]
        full_numeric_features = list(numeric_features) + latent_features
        # ======================================

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        # MACCS binary features switched to passthrough, removing the useless OneHot
        binary_transformer = 'passthrough'

        # Use the expanded full_numeric_features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, full_numeric_features),
                ('binary', binary_transformer, binary_features)
            ]
        )

        # Fit on the current fold's training set only
        X_train_trans = preprocessor.fit_transform(X_train_fold)
        X_val_trans = preprocessor.transform(X_val_fold)
        # ---------------------------------------------
        input_dim = X_train_trans.shape[1]

        # Build model
        tf.keras.backend.clear_session()
        X_input = Input(shape=(input_dim,), dtype='float32')
        FC1 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(X_input)
        FC1 = BatchNormalization()(FC1)
        FC1 = Dropout(0.35)(FC1)

        FC2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(FC1)
        FC2 = BatchNormalization()(FC2)
        FC2 = Dropout(0.35)(FC2)

        FC3 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(FC2)
        FC3 = BatchNormalization()(FC3)
        FC3 = Dropout(0.35)(FC3)

        predictions = Dense(1, activation='sigmoid')(FC3)
        model = Model(inputs=X_input, outputs=predictions)

        # Also lower the learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        # Early stopping: patience=5 + restore best weights
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            min_delta=1e-4,
            verbose=1,
            restore_best_weights=True
        )

        # Train the single-fold model
        history = model.fit(
            X_train_trans,
            y_train_fold,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_trans, y_val_fold),
            callbacks=[early_stopping],
            verbose=1
        )
        last_history = history  # Save the last fold's history

        # Predict
        y_val_pred_prob = model.predict(X_val_trans, verbose=0).ravel()
        y_val_true = y_val_fold.values

        # Compute metrics
        fold_auc = roc_auc_score(y_val_true, y_val_pred_prob)
        fpr, tpr, thresholds = roc_curve(y_val_true, y_val_pred_prob)
        youden_j = tpr - fpr
        best_thr = thresholds[np.argmax(youden_j)]
        y_val_pred = (y_val_pred_prob >= best_thr).astype(int)
        fold_acc = accuracy_score(y_val_true, y_val_pred)
        fold_ap = average_precision_score(y_val_true, y_val_pred)

        # Save this fold's results
        cv_auc_list.append(fold_auc)
        cv_acc_list.append(fold_acc)
        cv_ap_list.append(fold_ap)
        cv_y_true_all.extend(y_val_true.tolist())
        cv_y_pred_prob_all.extend(y_val_pred_prob.tolist())

        print(f"Fold {fold_idx+1} -> AUC: {fold_auc:.4f} | Acc: {fold_acc:.4f} | AP: {fold_ap:.4f}")

    # ========== Fix core error: cv_acc → cv_acc_list ==========
    mean_auc = np.mean(cv_auc_list)
    std_auc = np.std(cv_auc_list)
    mean_acc = np.mean(cv_acc_list)
    std_acc = np.std(cv_acc_list)
    mean_ap = np.mean(cv_ap_list)
    std_ap = np.std(cv_ap_list)

    print("\n" + "="*60)
    print(f"5-Fold Cross Validation Final Result")
    print(f"AUC   : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Acc   : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"AP    : {mean_ap:.4f} ± {std_ap:.4f}")
    print("="*60)

    # ===================== Added: plots dedicated to 5-fold cross-validation =====================
    # 1. cv_5fold_metrics.pdf mean ± std bar chart
    plt.figure(figsize=(8,5))
    plt.bar(["AUC","Accuracy","AP"], [mean_auc,mean_acc,mean_ap], 
            yerr=[std_auc,std_acc,std_ap], capsize=10, 
            color=['#ff9999','#66b3ff','#99ff99'])
    plt.ylim(0,1)
    plt.title("5-Fold CV Performance (Mean±Std)")
    plt.grid(alpha=0.3)
    plt.savefig("cv_5fold_metrics.pdf", bbox_inches='tight')
    plt.close()

    # 2. cv_5fold_trend.pdf per-fold metric trends
    plt.figure(figsize=(10,5))
    x_ticks = list(range(1, n_splits + 1))
    plt.plot(x_ticks, cv_auc_list, 'o-', label='AUC', lw=2)
    plt.plot(x_ticks, cv_acc_list, 's-', label='Accuracy', lw=2)
    plt.plot(x_ticks, cv_ap_list, 'd-', label='AP', lw=2)
    plt.xticks(x_ticks)
    plt.ylim(0.5,1)
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('5-Fold CV Metric Trend')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("cv_5fold_trend.pdf", bbox_inches='tight')
    plt.close()

    # 3. cv_5fold_boxplot.pdf predicted probability boxplot for positive and negative samples
    cv_y_arr = np.array(cv_y_true_all)
    cv_pred_arr = np.array(cv_y_pred_prob_all)
    pos_pred = cv_pred_arr[cv_y_arr == 1]
    neg_pred = cv_pred_arr[cv_y_arr == 0]
    plt.figure(figsize=(10,6))
    plt.boxplot([neg_pred, pos_pred], labels=["Negative(No Bind)","Positive(Bind)"])
    plt.title("5-Fold CV Prediction Distribution Boxplot")
    plt.ylabel("Predicted Probability")
    plt.grid(alpha=0.3)
    plt.savefig("cv_5fold_boxplot.pdf", bbox_inches='tight')
    plt.close()

    # 4. cv_5fold_roc.pdf merged ROC curve across all folds
    fpr_all, tpr_all, _ = roc_curve(cv_y_true_all, cv_y_pred_prob_all)
    roc_auc_all = auc(fpr_all, tpr_all)
    plt.figure(figsize=(8,6))
    plt.plot(fpr_all, tpr_all, lw=3, color='red', 
            label=f'Mean ROC (AUC={mean_auc:.2f}±{std_auc:.2f})')
    plt.plot([0,1],[0,1],'--',c='navy')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('5-Fold CV Merged ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig("cv_5fold_roc.pdf", bbox_inches='tight')
    plt.close()
    # ======================================================================

    # Save detailed per-fold metrics to csv
    cv_detail_df = pd.DataFrame({
        "fold": list(range(1, n_splits + 1)),
        "fold_auc": cv_auc_list,
        "fold_acc": cv_acc_list,
        "fold_ap": cv_ap_list
    })
    cv_detail_df.to_csv("cv_fold_details.csv", index=False, encoding="utf-8-sig")
    print("✅ Per-fold detailed metrics saved to cv_fold_details.csv")    

    # Global ROC / PR computation
    fpr_total, tpr_total, thresholds_total = roc_curve(cv_y_true_all, cv_y_pred_prob_all)
    roc_auc_total = auc(fpr_total, tpr_total)
    optimal_threshold_j, best_threshold, fpr_j, tpr_j, fpr_b, tpr_b, youden, thresholds_all = \
        get_optimal_threshold(fpr_total, tpr_total, thresholds_total)

    precision, recall, thresholds_pr = precision_recall_curve(cv_y_true_all, cv_y_pred_prob_all)
    avg_precision_total = average_precision_score(cv_y_true_all, cv_y_pred_prob_all)


    # Save metrics JSON
    training_metrics = {
        "cv_mean_auc": float(mean_auc),
        "cv_std_auc": float(std_auc),
        "cv_mean_acc": float(mean_acc),
        "cv_std_acc": float(std_acc),
        "cv_mean_ap": float(mean_ap),
        "cv_std_ap": float(std_ap),
        "overall_roc_auc": float(roc_auc_total),
        "overall_avg_precision": float(avg_precision_total),
        "youden_threshold": float(optimal_threshold_j),
        "best_threshold": float(best_threshold)
    }
    with open("training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=4)

    # 1. ROC Curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr_total, tpr_total, color='darkorange', lw=2, label=f'AUC = {roc_auc_total:.2f}')
    plt.plot([0,1],[0,1],'--',c='navy',lw=2)
    plt.scatter(fpr_j, tpr_j, c='orange', s=120, label=f'Youden {optimal_threshold_j:.2f}')
    plt.scatter(fpr_b, tpr_b, c='red', s=120, label=f'Best {best_threshold:.2f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.pdf', bbox_inches='tight')
    plt.close()

    # 2. Training curves (last fold used as the example curve, scope decoupled)
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.plot(last_history.history['loss'], label='Train loss')
    plt.plot(last_history.history['val_loss'], label='Val loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(132)
    plt.plot(last_history.history['accuracy'], label='Train acc')
    plt.plot(last_history.history['val_accuracy'], label='Val acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(133)
    plt.plot(last_history.history['auc'], label='Train AUC')
    plt.plot(last_history.history['val_auc'], label='Val AUC')
    plt.title('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.pdf', bbox_inches='tight')
    plt.close()

    # 3. Prediction probability distribution histogram
    cv_y_arr = np.array(cv_y_true_all)
    cv_pred_arr = np.array(cv_y_pred_prob_all)
    y_pred_pos = cv_pred_arr[cv_y_arr == 1]
    y_pred_neg = cv_pred_arr[cv_y_arr == 0]

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

    # 4. Dual-threshold confusion matrices (fix: add the second plot)
    plt.figure(figsize=(12,5))
    y_pred_y = (cv_pred_arr >= optimal_threshold_j).astype(int)
    cm_y = confusion_matrix(cv_y_arr, y_pred_y)
    disp_y = ConfusionMatrixDisplay(cm_y, display_labels=['INHIBITORY','ACTIVATING'])
    disp_y.plot(cmap='Blues', ax=plt.subplot(121))
    plt.title(f'Youden {optimal_threshold_j:.2f}')

    y_pred_b = (cv_pred_arr >= best_threshold).astype(int)
    cm_b = confusion_matrix(cv_y_arr, y_pred_b)
    disp_b = ConfusionMatrixDisplay(cm_b, display_labels=['INHIBITORY','ACTIVATING'])
    disp_b.plot(cmap='Blues', ax=plt.subplot(122))  # Fix: actually call plot
    plt.title(f'Best {best_threshold:.2f}')

    plt.tight_layout()
    plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
    plt.close()

    # 5. PR curve
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, color='darkgreen', lw=2, label=f'AP = {avg_precision_total:.4f}')
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

    # Release all plotting resources
    plt.close('all')

    # Save the preprocessor and feature list
    joblib.dump(preprocessor, 'preprocessor.joblib')
    # Export all features (raw numeric + latent representations + binary MACCS)
    all_latent, _, _ = get_latent_feature_names()
    latent_features_list = [c for c in features.columns if c in all_latent]
    numeric_features_list = full_numeric_features
    binary_features_list = binary_features.tolist()
    feature_columns = numeric_features_list + binary_features_list
    with open('feature_columns.json', 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f)

    # ============ Added model-saving code ============
    model.save("my_model.h5")
    print("✅ Binding DNN model saved to my_model.h5")
    # ==========================================

    print("\n✅ All tasks finished, files saved.")



if __name__ == '__main__':
    main()