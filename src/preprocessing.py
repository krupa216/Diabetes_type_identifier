"""
src/preprocessing.py
Data preprocessing utilities for Diabetes CIT Classifier
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV and perform basic cleaning."""
    df = pd.read_csv(path)

    # Strip whitespace from string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[Preprocessing] Removed {before - len(df)} duplicates")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Gender and CLASS columns."""
    df = df.copy()

    gender_map = {"M": 1, "F": 0, "Male": 1, "Female": 0, "m": 1, "f": 0}
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(gender_map).fillna(0).astype(int)

    class_map = {"N": 0, "P": 1, "Y": 2,
                 "No Diabetes": 0, "Pre-Diabetes": 1, "Diabetes": 2,
                 0: 0, 1: 1, 2: 2}
    if "CLASS" in df.columns:
        df["CLASS"] = df["CLASS"].map(class_map)
        df["CLASS"].fillna(df["CLASS"].mode()[0], inplace=True)
        df["CLASS"] = df["CLASS"].astype(int)

    return df


def apply_smote(X, y, random_state=42):
    """Apply SMOTE only on training data to fix class imbalance."""
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def normalize(X_train, X_test):
    """Apply Quantile Normalization."""
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_norm = qt.fit_transform(X_train)
    X_test_norm  = qt.transform(X_test)
    return X_train_norm, X_test_norm, qt


def full_pipeline(df: pd.DataFrame, test_size=0.3, random_state=42):
    """
    End-to-end preprocessing pipeline:
    1. Encode categoricals
    2. Train/test split (stratified)
    3. SMOTE on training set only
    4. Quantile normalize
    """
    df = encode_categoricals(df)
    df.dropna(inplace=True)

    drop_cols = [c for c in ["ID", "No_Pation"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    X = df.drop(columns=["CLASS"])
    y = df["CLASS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # SMOTE only on train
    X_train_bal, y_train_bal = apply_smote(X_train, y_train, random_state)

    # Normalize
    X_train_norm, X_test_norm, qt = normalize(X_train_bal, X_test)

    print(f"[Pipeline] Train: {X_train_norm.shape}, Test: {X_test_norm.shape}")
    print(f"[Pipeline] Class distribution (train after SMOTE): {np.bincount(y_train_bal)}")

    return X_train_norm, X_test_norm, y_train_bal, y_test, qt, list(X.columns)
