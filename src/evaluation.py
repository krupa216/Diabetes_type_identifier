"""
src/evaluation.py
Evaluation utilities for imbalanced multiclass medical classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, cohen_kappa_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["No Diabetes", "Pre-Diabetes", "Diabetes"]


def full_evaluation(model, X_test, y_test, output_dir="outputs"):
    """Run complete evaluation suite."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Core metrics
    acc      = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    kappa    = cohen_kappa_score(y_test, y_pred)
    auc_ovr  = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    print("\n" + "="*55)
    print("  EVALUATION REPORT")
    print("="*55)
    print(f"  Accuracy       : {acc*100:.2f}%")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Micro F1       : {micro_f1:.4f}")
    print(f"  Cohen's Kappa  : {kappa:.4f}")
    print(f"  AUC (OvR)      : {auc_ovr:.4f}")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_roc_curves(y_test, y_proba, output_dir)

    return {
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "kappa":    round(kappa, 4),
        "auc":      round(auc_ovr, 4),
    }


def plot_confusion_matrix(y_true, y_pred, output_dir="outputs"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, linewidths=0.5)
    ax.set_title("Confusion Matrix — CIT Diabetes Classifier",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix → {output_dir}/confusion_matrix.png")


def plot_roc_curves(y_true, y_proba, output_dir="outputs"):
    """Plot OvR ROC curves for all 3 classes."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = ["#10b981", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — One-vs-Rest (Multiclass)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=150)
    plt.close()
    print(f"[✓] ROC curves → {output_dir}/roc_curves.png")
