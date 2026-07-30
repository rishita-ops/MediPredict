"""
train.py
========
Trains a stacked ensemble model for each of the four diseases:
  base learners  -> Logistic Regression, Random Forest, XGBoost, SVM
  meta learner   -> Logistic Regression (combines the 4 base predictions)

Stacking beats any single model here because each base learner has a
different bias: Logistic Regression is great on linear-ish relationships,
Random Forest and XGBoost pick up non-linear interactions and feature
importance, and SVM is strong on margin-based separation in the smaller,
cleaner datasets (heart, liver). The meta-learner then learns which base
model to trust more for which kind of patient.

Saves, per disease: the fitted pipeline (scaler + stack), feature column
order, and evaluation metrics (accuracy, precision, recall, F1, ROC-AUC via
5-fold cross-validation) to backend/models/.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import data_prep as dp

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_stack():
    base_learners = [
        ("logreg", LogisticRegression(max_iter=2000, C=0.8)),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
        ("xgb", XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, verbosity=0,
        )),
        ("svm", SVC(probability=True, kernel="rbf", C=1.5, gamma="scale")),
    ]
    meta = LogisticRegression(max_iter=2000)
    stack = StackingClassifier(
        estimators=base_learners, final_estimator=meta,
        cv=5, passthrough=False, n_jobs=-1,
    )
    return Pipeline([("scaler", StandardScaler()), ("stack", stack)])


def evaluate(pipeline, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=skf, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
    }


def train_all():
    summary = {}
    for disease, loader in dp.LOADERS.items():
        print(f"\n=== Training {disease} ===")
        X, y = loader()
        feature_order = list(X.columns)

        pipeline = build_stack()
        metrics = evaluate(pipeline, X, y)
        print(disease, metrics)

        pipeline.fit(X, y)  # final fit on all data for the saved model

        joblib.dump(pipeline, MODEL_DIR / f"{disease}_model.joblib")
        with open(MODEL_DIR / f"{disease}_features.json", "w") as f:
            json.dump(feature_order, f)

        # background sample for SHAP's KernelExplainer (a representative
        # subset of real training rows, not the whole dataset -> keeps
        # per-prediction SHAP computation fast)
        bg_sample = X.sample(min(60, len(X)), random_state=42)
        bg_sample.to_json(MODEL_DIR / f"{disease}_background.json", orient="records")

        # per-feature percentiles, used for out-of-distribution detection:
        # if a new patient's value falls outside the 1st-99th percentile
        # band seen during training, we flag it as "outside what this
        # model has actually learned from"
        percentiles = {
            col: {
                "p1": float(X[col].quantile(0.01)),
                "p99": float(X[col].quantile(0.99)),
                "mean": float(X[col].mean()),
                "median": float(X[col].median()),
                "std": float(X[col].std() or 1.0),
            }
            for col in X.columns
        }
        with open(MODEL_DIR / f"{disease}_percentiles.json", "w") as f:
            json.dump(percentiles, f, indent=2)

        summary[disease] = {
            "metrics": metrics,
            "n_samples": len(y),
            "n_features": len(feature_order),
        }

    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== ALL DONE ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_all()
