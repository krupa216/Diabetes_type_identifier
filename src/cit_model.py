"""
src/cit_model.py
Conditional Inference Tree — Diabetes Multiclass Classifier
Dataset: Kaggle Multiclass Diabetes (yasserhessein)
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, cohen_kappa_score
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns


# ── CONFIG ──────────────────────────────────────────────────────────────────
import os
DATA_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Multiclass Diabetes Dataset.csv")
MODEL_PATH  = "models/cit_model.pkl"
OUTPUT_DIR  = "outputs"
TARGET_COL  = "CLASS"
DROP_COLS   = ["ID", "No_Pation"]      # Non-feature columns
CLASS_NAMES = ["No Diabetes", "Pre-Diabetes", "Diabetes"]


# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[✓] Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"    Class distribution:\n{df[TARGET_COL].value_counts()}\n")
    return df


# ── 2. PREPROCESS ────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Drop ID columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Encode gender
    le = LabelEncoder()
    if "Gender" in df.columns:
        df["Gender"] = le.fit_transform(df["Gender"].astype(str))

    # Encode target
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = le.fit_transform(df[TARGET_COL].astype(str))

    # Drop rows with missing values
    df.dropna(inplace=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"[✓] Preprocessed: {X.shape[1]} features, {len(y)} samples")
    return X, y


# ── 3. BALANCE + NORMALIZE ───────────────────────────────────────────────────
def balance_and_normalize(X, y):
    # SMOTE for class imbalance
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[✓] After SMOTE: {len(y_res)} samples")
    print(f"    Balanced classes: {np.bincount(y_res)}\n")

    # Quantile transform for normalization
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    X_norm = pd.DataFrame(qt.fit_transform(X_res), columns=X.columns)

    return X_norm, y_res, qt


# ── 4. FEATURE SELECTION ─────────────────────────────────────────────────────
def select_features(X, y, k: int = 8):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_sel = selector.fit_transform(X, y)
    selected = list(X.columns[selector.get_support()])
    scores   = selector.scores_[selector.get_support()]

    print(f"[✓] Top {k} features selected:")
    for feat, sc in sorted(zip(selected, scores), key=lambda x: -x[1]):
        print(f"    {feat:<20} score={sc:.4f}")
    print()

    return X_sel, selected, selector


# ── 5. TRAIN CIT MODEL ───────────────────────────────────────────────────────
def train_cit_model(data_path: str = DATA_PATH) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Pipeline
    df          = load_data(data_path)
    X, y        = preprocess(df)
    X_bal, y_bal, qt = balance_and_normalize(X, y)
    X_sel, feats, selector = select_features(X_bal, y_bal, k=8)

    # Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_bal, test_size=0.3,
        stratify=y_bal, random_state=42
    )

    # CIT model (log_loss criterion ≈ conditional inference split)
    model = DecisionTreeClassifier(
        criterion        = "log_loss",
        max_depth        = 7,
        min_samples_split= 20,
        min_samples_leaf = 10,
        class_weight     = "balanced",
        random_state     = 42
    )

    # 10-Fold Stratified CV
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train,
                             cv=skf, scoring="f1_macro", n_jobs=-1)

    print(f"[✓] 10-Fold CV Macro F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Final fit
    model.fit(X_train, y_train)

    # Evaluation
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc     = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    kappa   = cohen_kappa_score(y_test, y_pred)
    auc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    print(f"\n{'='*50}")
    print(f"  Accuracy     : {acc*100:.2f}%")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Cohen Kappa  : {kappa:.4f}")
    print(f"  AUC (OvR)    : {auc_ovr:.4f}")
    print(f"{'='*50}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Save outputs
    _plot_confusion_matrix(y_test, y_pred)
    _plot_tree(model, feats)
    _save_report(acc, macro_f1, kappa, auc_ovr, cv_f1, y_test, y_pred)

    # Persist model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[✓] Model saved to {MODEL_PATH}")

    return {
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "kappa":    round(kappa, 4),
        "auc":      round(auc_ovr, 4),
        "cv_f1_mean": round(cv_f1.mean(), 4),
        "cv_f1_std":  round(cv_f1.std(), 4),
    }


# ── HELPERS ──────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Confusion Matrix — CIT Diabetes Classifier", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix saved")


def _plot_tree(model, feature_names):
    fig, ax = plt.subplots(figsize=(28, 12))
    plot_tree(model, feature_names=feature_names,
              class_names=CLASS_NAMES,
              filled=True, rounded=True, fontsize=7, ax=ax)
    ax.set_title("Conditional Inference Tree — Diabetes Type Classifier", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ctree_plot.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[✓] Tree plot saved")


def _save_report(acc, macro_f1, kappa, auc, cv_f1, y_test, y_pred):
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write("="*55 + "\n")
        f.write("  DIABETES CIT CLASSIFIER — EVALUATION REPORT\n")
        f.write("="*55 + "\n\n")
        f.write(f"  Accuracy     : {acc*100:.2f}%\n")
        f.write(f"  Macro F1     : {macro_f1:.4f}\n")
        f.write(f"  Cohen Kappa  : {kappa:.4f}\n")
        f.write(f"  AUC (OvR)    : {auc:.4f}\n")
        f.write(f"  CV F1 (mean) : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}\n\n")
        f.write(report)
    print(f"[✓] Report saved")


if __name__ == "__main__":
    metrics = train_cit_model()
    print("\n[DONE] Training complete:", metrics)
