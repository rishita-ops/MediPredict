"""
main.py
========
MediPredict backend API. Run with:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /api/diseases                  -> list supported diseases + their input schema
    POST /api/predict/{disease}         -> run a prediction, saves it to history
    GET  /api/history?disease=&limit=   -> past predictions
    DELETE /api/history/{record_id}     -> delete one history record
    GET  /api/metrics                   -> training metrics for every model
    GET  /api/features/{disease}        -> engineered-feature explanations for that disease
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from . import inference
from . import explain
from . import briefing
from . import feature_engineering as fe
from . import reference_ranges
from . import glossary
from .database import init_db, get_session, PredictionRecord

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(title="MediPredict API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

DISEASE_SCHEMAS = {
    "heart": {
        "label": "Heart Disease",
        "fields": [
            {"name": "age", "type": "number", "label": "Age (years)"},
            {"name": "sex", "type": "select", "label": "Sex", "options": [["1", "Male"], ["0", "Female"]]},
            {"name": "cp", "type": "select", "label": "Chest pain type",
             "options": [["0", "Asymptomatic"], ["1", "Atypical angina"], ["2", "Non-anginal pain"], ["3", "Typical angina"]]},
            {"name": "trestbps", "type": "number", "label": "Resting blood pressure (mm Hg)"},
            {"name": "chol", "type": "number", "label": "Serum cholesterol (mg/dl)"},
            {"name": "fbs", "type": "select", "label": "Fasting blood sugar > 120 mg/dl", "options": [["1", "Yes"], ["0", "No"]]},
            {"name": "restecg", "type": "select", "label": "Resting ECG result", "options": [["0", "Normal"], ["1", "ST-T abnormality"], ["2", "LV hypertrophy"]]},
            {"name": "thalach", "type": "number", "label": "Max heart rate achieved"},
            {"name": "exang", "type": "select", "label": "Exercise-induced angina", "options": [["1", "Yes"], ["0", "No"]]},
            {"name": "oldpeak", "type": "number", "label": "ST depression induced by exercise"},
            {"name": "slope", "type": "select", "label": "Slope of peak exercise ST segment", "options": [["0", "Downsloping"], ["1", "Flat"], ["2", "Upsloping"]]},
            {"name": "ca", "type": "number", "label": "Number of major vessels colored (0-3)"},
            {"name": "thal", "type": "select", "label": "Thalassemia", "options": [["1", "Normal"], ["2", "Fixed defect"], ["3", "Reversible defect"]]},
        ],
    },
    "diabetes": {
        "label": "Diabetes",
        "fields": [
            {"name": "preg", "type": "number", "label": "Number of pregnancies"},
            {"name": "glucose", "type": "number", "label": "Plasma glucose concentration"},
            {"name": "bp", "type": "number", "label": "Diastolic blood pressure (mm Hg)"},
            {"name": "skinthickness", "type": "number", "label": "Triceps skin-fold thickness (mm)"},
            {"name": "insulin", "type": "number", "label": "2-hour serum insulin (mu U/ml)"},
            {"name": "bmi", "type": "number", "label": "Body mass index"},
            {"name": "dpf", "type": "number", "label": "Diabetes pedigree function"},
            {"name": "age", "type": "number", "label": "Age (years)"},
        ],
    },
    "liver": {
        "label": "Liver Disease",
        "fields": [
            {"name": "age", "type": "number", "label": "Age (years)"},
            {"name": "gender", "type": "select", "label": "Gender", "options": [["Male", "Male"], ["Female", "Female"]]},
            {"name": "total_bilirubin", "type": "number", "label": "Total bilirubin"},
            {"name": "direct_bilirubin", "type": "number", "label": "Direct bilirubin"},
            {"name": "alkaline_phosphotase", "type": "number", "label": "Alkaline phosphotase"},
            {"name": "alamine_aminotransferase", "type": "number", "label": "Alamine aminotransferase (ALT)"},
            {"name": "aspartate_aminotransferase", "type": "number", "label": "Aspartate aminotransferase (AST)"},
            {"name": "total_protiens", "type": "number", "label": "Total proteins"},
            {"name": "albumin", "type": "number", "label": "Albumin"},
            {"name": "albumin_and_globulin_ratio", "type": "number", "label": "Albumin/globulin ratio"},
        ],
    },
    "kidney": {
        "label": "Kidney Disease",
        "fields": [
            {"name": "age", "type": "number", "label": "Age (years)"},
            {"name": "bp", "type": "number", "label": "Blood pressure (mm Hg)"},
            {"name": "sg", "type": "number", "label": "Urine specific gravity"},
            {"name": "al", "type": "number", "label": "Albumin (urine, 0-5)"},
            {"name": "su", "type": "number", "label": "Sugar (urine, 0-5)"},
            {"name": "rbc", "type": "select", "label": "Red blood cells (urine)", "options": [["normal", "Normal"], ["abnormal", "Abnormal"]]},
            {"name": "pc", "type": "select", "label": "Pus cell", "options": [["normal", "Normal"], ["abnormal", "Abnormal"]]},
            {"name": "pcc", "type": "select", "label": "Pus cell clumps", "options": [["notpresent", "Not present"], ["present", "Present"]]},
            {"name": "ba", "type": "select", "label": "Bacteria", "options": [["notpresent", "Not present"], ["present", "Present"]]},
            {"name": "bgr", "type": "number", "label": "Blood glucose random"},
            {"name": "bu", "type": "number", "label": "Blood urea"},
            {"name": "sc", "type": "number", "label": "Serum creatinine"},
            {"name": "sod", "type": "number", "label": "Sodium"},
            {"name": "pot", "type": "number", "label": "Potassium"},
            {"name": "hemo", "type": "number", "label": "Hemoglobin"},
            {"name": "pcv", "type": "number", "label": "Packed cell volume"},
            {"name": "wc", "type": "number", "label": "White blood cell count"},
            {"name": "rc", "type": "number", "label": "Red blood cell count"},
            {"name": "htn", "type": "select", "label": "Hypertension", "options": [["yes", "Yes"], ["no", "No"]]},
            {"name": "dm", "type": "select", "label": "Diabetes mellitus", "options": [["yes", "Yes"], ["no", "No"]]},
            {"name": "cad", "type": "select", "label": "Coronary artery disease", "options": [["yes", "Yes"], ["no", "No"]]},
            {"name": "appet", "type": "select", "label": "Appetite", "options": [["good", "Good"], ["poor", "Poor"]]},
            {"name": "pe", "type": "select", "label": "Pedal edema", "options": [["yes", "Yes"], ["no", "No"]]},
            {"name": "ane", "type": "select", "label": "Anemia", "options": [["yes", "Yes"], ["no", "No"]]},
        ],
    },
}


@app.get("/api/diseases")
def list_diseases():
    return DISEASE_SCHEMAS


@app.get("/api/metrics")
def get_metrics():
    path = MODEL_DIR / "training_summary.json"
    if not path.exists():
        raise HTTPException(404, "Models not trained yet. Run `python -m app.train` first.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/features/{disease}")
def get_feature_notes(disease: str):
    if disease not in DISEASE_SCHEMAS:
        raise HTTPException(404, f"Unknown disease '{disease}'")
    all_notes = fe.get_all_feature_notes()
    with open(MODEL_DIR / f"{disease}_features.json") as f:
        feature_order = json.load(f)
    engineered = [f for f in feature_order if f in all_notes]
    return {name: all_notes[name] for name in engineered}


@app.get("/api/reference-ranges/{disease}")
def get_reference_ranges(disease: str):
    if disease not in DISEASE_SCHEMAS:
        raise HTTPException(404, f"Unknown disease '{disease}'")
    ranges = reference_ranges.get_ranges(disease)
    return {
        field: {"low": low, "high": high, "unit": unit, "note": note}
        for field, (low, high, unit, note) in ranges.items()
    }


@app.get("/api/glossary")
def get_glossary_terms():
    return glossary.get_glossary()


@app.post("/api/predict/{disease}")
def predict(disease: str, payload: dict, db: Session = Depends(get_session)):
    if disease not in DISEASE_SCHEMAS:
        raise HTTPException(404, f"Unknown disease '{disease}'")

    patient_label = payload.pop("patient_label", "Unnamed patient")
    try:
        result = inference.predict(disease, payload)
        result["uncertainty"] = explain.model_uncertainty(disease, payload)
        result["ood_flags"] = explain.ood_check(disease, payload)
    except FileNotFoundError:
        raise HTTPException(500, "Model not trained yet. Run `python -m app.train` first.")
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")

    record = PredictionRecord(
        disease=disease,
        patient_label=patient_label,
        input_json=json.dumps(payload),
        probability=result["probability"],
        prediction=result["prediction"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    result["record_id"] = record.id
    result["created_at"] = record.created_at.isoformat()
    return result


@app.post("/api/whatif/{disease}")
def whatif(disease: str, payload: dict):
    """Lightweight prediction with no history save and no SHAP -- built for
    a UI slider that recalculates on every drag. Still returns the real
    model-disagreement uncertainty, since that's cheap to compute."""
    if disease not in DISEASE_SCHEMAS:
        raise HTTPException(404, f"Unknown disease '{disease}'")
    payload.pop("patient_label", None)
    try:
        result = inference.predict(disease, payload)
        result["uncertainty"] = explain.model_uncertainty(disease, payload)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")
    return result


@app.post("/api/explain/{disease}")
def explain_prediction(disease: str, payload: dict):
    """Full explainability breakdown for one patient: real SHAP feature
    contributions + out-of-distribution flags. Not saved to history --
    call this after /api/predict when the user wants the 'why'."""
    if disease not in DISEASE_SCHEMAS:
        raise HTTPException(404, f"Unknown disease '{disease}'")
    payload.pop("patient_label", None)
    try:
        shap_contributions = explain.real_shap_values(disease, payload)
        ood_flags = explain.ood_check(disease, payload)
    except FileNotFoundError:
        raise HTTPException(500, "Model not trained yet. Run `python -m app.train` first.")
    except Exception as e:
        raise HTTPException(400, f"Explanation failed: {e}")
    return {"disease": disease, "shap_contributions": shap_contributions, "ood_flags": ood_flags}


class FraminghamInput(dict):
    pass


@app.post("/api/framingham")
def framingham(payload: dict):
    """Real, independently-published D'Agostino et al. (2008) 10-year CVD
    risk score -- a second opinion for heart disease, computed with a
    formula, not the ML model."""
    required = ["sex", "age", "total_chol", "hdl", "sbp", "bp_treated", "smoker", "diabetic"]
    missing = [f for f in required if f not in payload]
    if missing:
        raise HTTPException(400, f"Missing fields: {missing}")
    try:
        risk_pct = explain.framingham_10yr_risk(
            sex=payload["sex"], age=float(payload["age"]),
            total_chol=float(payload["total_chol"]), hdl=float(payload["hdl"]),
            sbp=float(payload["sbp"]), bp_treated=bool(payload["bp_treated"]),
            smoker=bool(payload["smoker"]), diabetic=bool(payload["diabetic"]),
        )
    except Exception as e:
        raise HTTPException(400, f"Framingham calculation failed: {e}")
    return {"ten_year_cvd_risk_percent": risk_pct, "source": "D'Agostino et al., Circulation 2008"}


@app.get("/api/history")
def get_history(disease: Optional[str] = None, limit: int = 50, db: Session = Depends(get_session)):
    q = db.query(PredictionRecord)
    if disease:
        q = q.filter(PredictionRecord.disease == disease)
    q = q.order_by(PredictionRecord.created_at.desc()).limit(limit)
    return [
        {
            "id": r.id,
            "disease": r.disease,
            "patient_label": r.patient_label,
            "input": json.loads(r.input_json),
            "probability": r.probability,
            "prediction": r.prediction,
            "created_at": r.created_at.isoformat(),
        }
        for r in q.all()
    ]


@app.delete("/api/history/{record_id}")
def delete_history(record_id: int, db: Session = Depends(get_session)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(404, "Record not found")
    db.delete(record)
    db.commit()
    return {"deleted": record_id}


@app.get("/api/history/trend")
def get_history_trend(patient_label: str, disease: Optional[str] = None, db: Session = Depends(get_session)):
    """All past predictions for one named patient, oldest first -- for a
    risk-over-time sparkline. Matching is by patient_label string only."""
    q = db.query(PredictionRecord).filter(PredictionRecord.patient_label == patient_label)
    if disease:
        q = q.filter(PredictionRecord.disease == disease)
    q = q.order_by(PredictionRecord.created_at.asc())
    return [
        {"id": r.id, "disease": r.disease, "probability": r.probability,
         "prediction": r.prediction, "created_at": r.created_at.isoformat()}
        for r in q.all()
    ]


# Fields directly comparable across at least two of the four disease
# datasets -- used to build the cross-disease differential matrix. Every
# other field gets filled with that disease's own training median, and
# the response says so explicitly.
SHARED_FIELD_MAP = {
    "age": {"heart": "age", "diabetes": "age", "liver": "age", "kidney": "age"},
    "systolic_or_resting_bp": {"heart": "trestbps", "kidney": "bp"},
    "diastolic_bp": {"diabetes": "bp"},
    "glucose": {"diabetes": "glucose", "kidney": "bgr"},
    "bmi": {"diabetes": "bmi"},
}


@app.post("/api/differential")
def differential_matrix(payload: dict):
    """Runs all 4 disease models from shared fields only (age, blood
    pressure, glucose, bmi), filling everything else from that disease's
    training median. A real computation, but explicitly lower-confidence
    for diseases that got only a fraction of their normal inputs."""
    results = {}
    for disease in DISEASE_SCHEMAS:
        # start every field for this disease as "not provided" so the
        # imputation step (which only fills blanks on EXISTING keys) fills
        # in every field the shared inputs didn't cover
        partial_input = {f["name"]: None for f in DISEASE_SCHEMAS[disease]["fields"]}
        provided_fields = []
        for shared_key, disease_map in SHARED_FIELD_MAP.items():
            if disease in disease_map and payload.get(shared_key) is not None:
                field_name = disease_map[disease]
                partial_input[field_name] = payload[shared_key]
                provided_fields.append(field_name)

        try:
            result = inference.predict(disease, partial_input)
            results[disease] = {
                "probability": result["probability"],
                "risk_label": result["risk_label"],
                "fields_provided": provided_fields,
                "fields_imputed": result["imputed_fields"],
                "confidence": "indicative only -- most fields imputed from training medians"
                              if len(result["imputed_fields"]) > len(provided_fields)
                              else "partial data",
            }
        except Exception as e:
            results[disease] = {"error": str(e)}

    return results


@app.post("/api/briefing/{disease}")
def generate_briefing(disease: str, payload: dict):
    """Generates a printable PDF briefing packet for one patient: result,
    base-model agreement, SHAP contributions, and OOD flags."""
    if disease not in DISEASE_SCHEMAS:
        raise HTTPException(404, f"Unknown disease '{disease}'")

    patient_label = payload.pop("patient_label", "Unnamed patient")
    try:
        result = inference.predict(disease, payload)
        uncertainty = explain.model_uncertainty(disease, payload)
        shap_contributions = explain.real_shap_values(disease, payload)
        ood_flags = explain.ood_check(disease, payload)
    except Exception as e:
        raise HTTPException(400, f"Briefing generation failed: {e}")

    pdf_buffer = briefing.build_briefing_pdf(
        disease_label=DISEASE_SCHEMAS[disease]["label"],
        patient_label=patient_label,
        raw_inputs=payload,
        result=result,
        uncertainty=uncertainty,
        shap_contributions=shap_contributions,
        ood_flags=ood_flags,
    )
    filename = f"medipredict_{disease}_briefing.pdf"
    return StreamingResponse(
        pdf_buffer, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
