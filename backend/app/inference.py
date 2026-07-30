"""
inference.py
=============
Turns a raw patient-form submission (the kind of thing a UI would collect)
into the exact feature vector each trained model expects, by re-running the
SAME feature-engineering functions used during training, then reindexing to
the saved column order (any one-hot column not produced from a single row
input is safely filled with 0 by the reindex).
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from . import feature_engineering as fe

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_MODEL_CACHE = {}
_FEATURES_CACHE = {}
_PERCENTILE_CACHE = {}

# sensible "healthy" defaults for categorical fields, used only when a
# select-type field is left blank -- numeric fields are imputed from the
# real training-data median instead (see impute_missing below)
CATEGORICAL_DEFAULTS = {
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no",
    "gender": "Male", "sex": 1, "cp": 0, "fbs": 0, "restecg": 0, "exang": 0,
    "slope": 1, "thal": 2,
}


def _load(disease):
    if disease not in _MODEL_CACHE:
        _MODEL_CACHE[disease] = joblib.load(MODEL_DIR / f"{disease}_model.joblib")
        with open(MODEL_DIR / f"{disease}_features.json") as f:
            _FEATURES_CACHE[disease] = json.load(f)
    return _MODEL_CACHE[disease], _FEATURES_CACHE[disease]


def _load_percentiles(disease):
    if disease not in _PERCENTILE_CACHE:
        with open(MODEL_DIR / f"{disease}_percentiles.json") as f:
            _PERCENTILE_CACHE[disease] = json.load(f)
    return _PERCENTILE_CACHE[disease]


def impute_missing(disease, raw: dict):
    """Fills any blank/None/NaN field with the real training-data median
    (for numeric fields) or a healthy-default category (for select fields).
    Returns (filled_dict, list_of_imputed_field_names) -- the caller/UI
    should visibly flag which fields were imputed rather than pretend they
    were entered."""
    percentiles = _load_percentiles(disease)
    filled = dict(raw)
    imputed = []

    for field, value in list(raw.items()):
        is_blank = value is None or value == "" or (isinstance(value, float) and pd.isna(value))
        if not is_blank:
            continue
        if field in percentiles:
            filled[field] = percentiles[field]["median"]
            imputed.append(field)
        elif field in CATEGORICAL_DEFAULTS:
            filled[field] = CATEGORICAL_DEFAULTS[field]
            imputed.append(field)
        else:
            filled[field] = 0
            imputed.append(field)

    return filled, imputed


def _row(d):
    return pd.DataFrame([d])

def build_heart_vector(raw: dict) -> pd.DataFrame:
    d = dict(raw)

    # 1. Map Categorical / Select Options to Numeric Values
    sex_map = {"Male": 1, "Female": 0, "1": 1, "0": 0}
    cp_map = {"Asymptomatic": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Typical angina": 3, "0": 0, "1": 1, "2": 2, "3": 3}
    fbs_map = {"Yes": 1, "No": 0, "1": 1, "0": 0}
    restecg_map = {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2, "0": 0, "1": 1, "2": 2}
    exang_map = {"Yes": 1, "No": 0, "1": 1, "0": 0}
    slope_map = {"Downsloping": 0, "Flat": 1, "Upsloping": 2, "0": 0, "1": 1, "2": 2}
    thal_map = {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3, "1": 1, "2": 2, "3": 3}

    if "sex" in d and d["sex"] in sex_map:
        d["sex"] = sex_map[d["sex"]]
    if "cp" in d and d["cp"] in cp_map:
        d["cp"] = cp_map[d["cp"]]
    if "fbs" in d and d["fbs"] in fbs_map:
        d["fbs"] = fbs_map[d["fbs"]]
    if "restecg" in d and d["restecg"] in restecg_map:
        d["restecg"] = restecg_map[d["restecg"]]
    if "exang" in d and d["exang"] in exang_map:
        d["exang"] = exang_map[d["exang"]]
    if "slope" in d and d["slope"] in slope_map:
        d["slope"] = slope_map[d["slope"]]
    if "thal" in d and d["thal"] in thal_map:
        d["thal"] = thal_map[d["thal"]]

    # 2. Convert Numeric Fields to Floats/Ints
    for num_col in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]:
        if num_col in d and d[num_col] is not None and d[num_col] != "":
            d[num_col] = float(d[num_col])

    df = _row(d)
    df = fe.heart_features(df)
    df = fe.add_age_bucket_risk(df, "age")
    return df


def build_diabetes_vector(raw: dict) -> pd.DataFrame:
    df = _row(raw)
    df = fe.diabetes_features(df)
    df = fe.add_age_bucket_risk(df, "age")
    return df


def build_liver_vector(raw: dict) -> pd.DataFrame:
    d = dict(raw)
    d["gender"] = 1 if str(d.get("gender", "Male")).lower().startswith("m") else 0
    df = _row(d)
    df["missing_value_count"] = 0
    df = fe.liver_features(df)
    df = fe.add_age_bucket_risk(df, "age")
    return df


def build_kidney_vector(raw: dict) -> pd.DataFrame:
    d = dict(raw)
    df = _row(d)
    df["missing_value_count"] = 0
    df = fe.kidney_features(df)
    df = fe.add_age_bucket_risk(df, "age")

    cat_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


BUILDERS = {
    "heart": build_heart_vector,
    "diabetes": build_diabetes_vector,
    "liver": build_liver_vector,
    "kidney": build_kidney_vector,
}


def predict(disease: str, raw: dict):
    if disease not in BUILDERS:
        raise ValueError(f"Unknown disease '{disease}'")

    filled_raw, imputed_fields = impute_missing(disease, raw)

    model, feature_order = _load(disease)
    df = BUILDERS[disease](filled_raw)
    df = df.reindex(columns=feature_order, fill_value=0)

    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "disease": disease,
        "prediction": pred,
        "probability": round(proba, 4),
        "risk_label": "High risk" if pred == 1 else "Low risk",
        "imputed_fields": imputed_fields,
        "filled_inputs": filled_raw,
    }
