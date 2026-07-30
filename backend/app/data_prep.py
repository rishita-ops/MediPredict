"""
data_prep.py
============
Loads and cleans the four raw disease datasets, then hands them to
feature_engineering.py to add the custom engineered features.

Data sources (all real, public datasets — not synthetic):
  - Heart:    UCI Heart Disease (Cleveland), 303 patients, 13 clinical features
  - Diabetes: Pima Indians Diabetes dataset, 768 patients, 8 features
  - Liver:    Indian Liver Patient Dataset (ILPD), 583 patients, 10 features
  - Kidney:   UCI Chronic Kidney Disease dataset, 400 patients, 24 features
"""

import pandas as pd
import numpy as np
from pathlib import Path

from . import feature_engineering as fe

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_heart():
    df = pd.read_csv(DATA_DIR / "heart.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna()
    df = fe.heart_features(df)
    df = fe.add_age_bucket_risk(df, "age")
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])
    return X, y


def load_diabetes():
    cols = ["preg", "glucose", "bp", "skinthickness", "insulin", "bmi", "dpf", "age", "outcome"]
    df = pd.read_csv(DATA_DIR / "diabetes.csv", header=None, names=cols)
    df = df.dropna()
    df = fe.diabetes_features(df)
    df = fe.add_age_bucket_risk(df, "age")
    y = df["outcome"].astype(int)
    X = df.drop(columns=["outcome"])
    return X, y


def load_liver():
    df = pd.read_csv(DATA_DIR / "liver.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "male": 1, "female": 0})
    df = fe.add_missing_value_signal(df, df.columns.tolist())
    df = df.dropna()
    df = fe.liver_features(df)
    df = fe.add_age_bucket_risk(df, "age")
    # dataset col: 1 = liver disease, 2 = no disease -> convert to 1/0
    y = (df["dataset"] == 1).astype(int)
    X = df.drop(columns=["dataset"])
    return X, y


def load_kidney():
    df = pd.read_csv(DATA_DIR / "kidney.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # clean whitespace / stray characters that are common in this raw file
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace(
                {"nan": np.nan, "?": np.nan, "\t?": np.nan, "\tno": "no", "\tyes": "yes"}
            )

    numeric_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod",
                     "pot", "hemo", "pcv", "wc", "rc"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = fe.add_missing_value_signal(df, df.columns.tolist())

    # fill remaining numeric NaNs with column median, categorical with mode
    for c in df.columns:
        if df[c].dtype in (np.float64, np.int64):
            df[c] = df[c].fillna(df[c].median())
        else:
            if c not in ("classification",):
                df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "unknown")

    df = fe.kidney_features(df)
    df = fe.add_age_bucket_risk(df, "age")

    y = df["classification"].astype(str).str.strip().map(
        lambda v: 1 if v.startswith("ckd") else 0
    )
    X = df.drop(columns=["classification"])

    # one-hot encode remaining categorical (yes/no, normal/abnormal, etc.)
    # (pandas may infer either python `object` dtype or its newer StringDtype
    # for text columns depending on version, so we check for both)
    cat_cols = [
        c for c in X.columns
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c])
    ]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.astype({c: int for c in X.columns if X[c].dtype == bool})

    return X, y


LOADERS = {
    "heart": load_heart,
    "diabetes": load_diabetes,
    "liver": load_liver,
    "kidney": load_kidney,
}
