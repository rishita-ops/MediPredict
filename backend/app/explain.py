"""
explain.py
==========
Everything that turns a bare probability into something a clinician could
actually reason about:

  - real_shap_values()     per-feature contribution to THIS prediction,
                            computed with SHAP's KernelExplainer against
                            the actual fitted pipeline (not a canned list
                            of notes — a live calculation).
  - model_uncertainty()    how much the 4 base learners inside the stack
                            disagree with each other on this specific
                            patient. Wide disagreement = less trustworthy
                            prediction, even if the final probability
                            looks confident.
  - ood_check()             flags any input feature that falls outside the
                            1st-99th percentile band the model was actually
                            trained on.
  - framingham_10yr_risk()  the real, published D'Agostino et al. 2008
                            general CVD risk formula, computed
                            independently of the ML model, as a second
                            opinion for the heart disease page.
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from . import inference as inf

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_BG_CACHE = {}
_PCTL_CACHE = {}
_EXPLAINER_CACHE = {}


def _load_background(disease):
    if disease not in _BG_CACHE:
        _BG_CACHE[disease] = pd.read_json(MODEL_DIR / f"{disease}_background.json")
    return _BG_CACHE[disease]


def _load_percentiles(disease):
    if disease not in _PCTL_CACHE:
        with open(MODEL_DIR / f"{disease}_percentiles.json") as f:
            _PCTL_CACHE[disease] = json.load(f)
    return _PCTL_CACHE[disease]


def _get_explainer(disease, model, feature_order):
    if disease not in _EXPLAINER_CACHE:
        bg = _load_background(disease).reindex(columns=feature_order, fill_value=0)
        f = lambda X: model.predict_proba(pd.DataFrame(X, columns=feature_order))[:, 1]
        _EXPLAINER_CACHE[disease] = shap.KernelExplainer(f, bg)
    return _EXPLAINER_CACHE[disease]


def real_shap_values(disease, raw, top_n=8):
    """Returns the top_n features that pushed this specific prediction up
    or down, with their real SHAP contribution values (not static notes)."""
    model, feature_order = inf._load(disease)
    filled_raw, _ = inf.impute_missing(disease, raw)
    df = inf.BUILDERS[disease](filled_raw).reindex(columns=feature_order, fill_value=0)

    explainer = _get_explainer(disease, model, feature_order)
    # nsamples kept modest -- this runs per-request, so we trade a little
    # precision for speed. Good enough to rank which features mattered.
    shap_vals = explainer.shap_values(df, nsamples=120, silent=True)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[-1]
    shap_vals = np.array(shap_vals).reshape(-1)

    contributions = [
        {"feature": feat, "value": float(df.iloc[0][feat]), "impact": float(val)}
        for feat, val in zip(feature_order, shap_vals)
    ]
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return contributions[:top_n]


def model_uncertainty(disease, raw):
    """Standard deviation of the 4 base learners' predicted probabilities
    for this one patient. A stacked model can look confident overall while
    its base learners are quietly split -- this surfaces that."""
    model, feature_order = inf._load(disease)
    filled_raw, _ = inf.impute_missing(disease, raw)
    df = inf.BUILDERS[disease](filled_raw).reindex(columns=feature_order, fill_value=0)

    scaler = model.named_steps["scaler"]
    stack = model.named_steps["stack"]
    X_scaled = scaler.transform(df)

    base_probs = {}
    for name, est in stack.named_estimators_.items():
        base_probs[name] = float(est.predict_proba(X_scaled)[0, 1])

    values = list(base_probs.values())
    mean = sum(values) / len(values)
    std = float(np.std(values))

    return {
        "base_model_probabilities": base_probs,
        "mean_probability": round(mean, 4),
        "std_dev": round(std, 4),
        "confidence_band": [round(max(0, mean - std), 4), round(min(1, mean + std), 4)],
        "agreement": "high" if std < 0.08 else ("moderate" if std < 0.18 else "low"),
    }


def ood_check(disease, raw):
    """Flags which raw input values fall outside the 1st-99th percentile
    range the model actually saw during training -- i.e. where the model
    is extrapolating rather than recognizing a familiar pattern."""
    percentiles = _load_percentiles(disease)
    flags = []
    for field, value in raw.items():
        if field not in percentiles or not isinstance(value, (int, float)):
            continue
        stats = percentiles[field]
        if value < stats["p1"] or value > stats["p99"]:
            z = (value - stats["mean"]) / stats["std"]
            flags.append({
                "feature": field,
                "value": value,
                "expected_range": [round(stats["p1"], 2), round(stats["p99"], 2)],
                "z_score": round(float(z), 2),
            })
    return flags


# ---------------------------------------------------------------------------
# Framingham 2008 General CVD Risk Score (D'Agostino et al., Circulation 2008)
# A real, independently-published formula -- not derived from our ML model,
# used here purely as a second opinion for heart disease.
# ---------------------------------------------------------------------------

def framingham_10yr_risk(sex, age, total_chol, hdl, sbp, bp_treated, smoker, diabetic):
    """
    sex: 'male' or 'female'
    age, total_chol, hdl, sbp: numeric, mg/dL for cholesterol, mmHg for sbp
    bp_treated, smoker, diabetic: bool
    Returns estimated 10-year CVD risk as a percentage (0-100).
    """
    ln = math.log
    sbp_term = ln(sbp)

    if sex == "male":
        s0 = 0.88936
        baseline = 23.9802
        total = (
            3.06117 * ln(age)
            + 1.12370 * ln(total_chol)
            - 0.93263 * ln(hdl)
            + (1.99881 * sbp_term if bp_treated else 1.93303 * sbp_term)
            + 0.65451 * (1 if smoker else 0)
            + 0.57367 * (1 if diabetic else 0)
        )
    else:
        s0 = 0.95012
        baseline = 26.1931
        total = (
            2.32888 * ln(age)
            + 1.20904 * ln(total_chol)
            - 0.70833 * ln(hdl)
            + (2.82263 * sbp_term if bp_treated else 2.76157 * sbp_term)
            + 0.52873 * (1 if smoker else 0)
            + 0.69154 * (1 if diabetic else 0)
        )

    risk = 1 - (s0 ** math.exp(total - baseline))
    return round(max(0.0, min(1.0, risk)) * 100, 2)
