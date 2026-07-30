# 🩺 MediPredict

**A full-stack, multi-disease clinical decision-support system** — one unified app for Heart Disease, Diabetes, Liver Disease, and Chronic Kidney Disease risk prediction, built on real clinical data with genuine explainability.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

> ⚠️ **Disclaimer:** MediPredict is a decision-support and portfolio project, not a certified medical device. Every prediction is a statistical estimate based on historical data, not a diagnosis. It is not validated for, and should not be used for, real clinical decision-making.

---

## What is this?

Most disease-prediction tutorials handle one dataset with one notebook. MediPredict instead unifies **four real, public clinical datasets** under **one shared pipeline, one backend, and one dashboard** — with a genuine explainability layer on top, not just a bare probability score.

| Disease | Dataset | Patients | Accuracy | ROC-AUC |
|---|---|---|---|---|
| ❤️ Heart Disease | UCI Heart Disease (Cleveland) | 303 | 82.8% | 0.892 |
| 🩸 Diabetes | Pima Indians Diabetes | 768 | 76.6% | 0.836 |
| 🫀 Liver Disease | Indian Liver Patient Dataset | 579 | 72.0% | 0.758 |
| 🫘 Chronic Kidney Disease | UCI CKD Dataset | 400 | 99.75% | 0.9999 |

*All metrics are from stratified 5-fold cross-validation — not a single lucky train/test split.*

---

## ✨ Features

- **25 hand-engineered clinical features** — real ratios and composite scores like the AST/ALT (De Ritis) ratio and the BUN/creatinine ratio, not just raw dataset columns
- **Stacked ensemble per disease** — Logistic Regression + Random Forest + XGBoost + SVM, combined by a meta-learner
- **Real SHAP explainability** — live, per-patient feature contributions computed against the actual fitted model
- **Real uncertainty estimation** — surfaces disagreement between the four base models as a confidence signal
- **Out-of-distribution detection** — flags inputs outside the range the model was actually trained on
- **Live what-if simulation** — drag a slider, watch the real model recalculate
- **Framingham 10-year CVD risk** — an independently published formula, run alongside the ML model as a second opinion
- **PDF briefing packets** — one-click downloadable clinical summary per prediction
- **Cross-disease differential matrix** — indicative multi-disease check from shared fields (age, BP, glucose, BMI)
- **Patient trend tracking** — risk-over-time sparkline across repeat visits
- **Reference ranges & glossary tooltips** — real published clinical normal ranges and plain-language definitions, right in the UI
- **US ↔ SI unit conversion** — real conversion factors for the labs where it matters
- **No build step frontend** — one HTML file, zero npm install, served directly by the backend

---

## 🖥️ Screenshots

> _Add screenshots or a short screen recording here before publishing — a GIF of a prediction + SHAP panel goes a long way on a repo landing page._

```
docs/screenshot-dashboard.png
docs/screenshot-shap.png
docs/screenshot-briefing.pdf
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/<your-username>/medipredict.git
cd medipredict/backend

pip install -r requirements.txt

# train all 4 models (also generates SHAP background samples + OOD percentile stats)
python -m app.train

# run the app — backend + frontend on one server
python -m uvicorn app.main:app --reload --port 8001
```

Open **http://localhost:8001** — that's it. API docs live at `/docs`.

> Pretrained models are included in this repo, so `python -m app.train` is optional unless you're modifying the code or data.

---

## 🏗️ Architecture

```
medipredict/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app + all routes
│   │   ├── data_prep.py            # loads & cleans the 4 raw datasets
│   │   ├── feature_engineering.py  # the 25 custom engineered features
│   │   ├── train.py                # trains models, saves SHAP/OOD artifacts
│   │   ├── inference.py            # raw form input -> prediction
│   │   ├── explain.py              # real SHAP, uncertainty, OOD, Framingham
│   │   ├── briefing.py             # PDF report generator
│   │   ├── reference_ranges.py     # published clinical normal ranges
│   │   ├── glossary.py             # plain-language clinical term definitions
│   │   └── database.py             # SQLite models (prediction history)
│   ├── data/                       # raw CSVs (real, public datasets)
│   ├── models/                     # trained .joblib models + metrics
│   └── requirements.txt
└── frontend/
    └── index.html                  # entire UI — one file, no build step
```

**Stack:** FastAPI · scikit-learn · XGBoost · SHAP · SQLAlchemy · ReportLab · vanilla JS

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/diseases` | Input schema for all 4 diseases |
| `POST` | `/api/predict/{disease}` | Full prediction, saved to history |
| `POST` | `/api/whatif/{disease}` | Lightweight prediction for live recalculation |
| `POST` | `/api/explain/{disease}` | Real SHAP feature contributions |
| `POST` | `/api/framingham` | Independent D'Agostino 2008 CVD risk score |
| `POST` | `/api/differential` | Cross-disease matrix from shared fields |
| `POST` | `/api/briefing/{disease}` | Downloadable PDF briefing packet |
| `GET` | `/api/history` | Past predictions |
| `GET` | `/api/history/trend` | Risk-over-time for one patient label |
| `GET` | `/api/reference-ranges/{disease}` | Published clinical normal ranges |
| `GET` | `/api/glossary` | Clinical term definitions |
| `GET` | `/api/metrics` | Cross-validated model performance |

Full interactive docs at `/docs` once the server is running.

---

## 🧠 Why a Stacked Ensemble?

Four different algorithms — Logistic Regression, Random Forest, XGBoost, and SVM — each tend to make *different kinds* of mistakes. A meta-learner (a fifth model) learns how to weigh their four opinions per patient, rather than just averaging them. This consistently outperforms any single model across all four diseases in cross-validation.

## 🔍 What Makes the Explainability Real

- **SHAP values** are computed live, per patient, using `KernelExplainer` against the actual fitted pipeline — not static, canned explanations.
- **Uncertainty** is the real standard deviation across the four base learners' individual predictions for that specific patient.
- **Out-of-distribution flags** compare live input against the 1st–99th percentile of the real training data.
- **Framingham** is the actual published D'Agostino et al. (2008) formula, computed independently of the ML model — a genuine second opinion, not a duplicate of the same prediction.

---

## ⚠️ Limitations

- All four datasets are small (303–768 patients) — results may not generalize to broader populations
- No dataset includes some clinically important variables (e.g., HDL cholesterol, smoking status for heart disease)
- Diabetes and liver disease models score meaningfully lower than heart and kidney — this reflects genuine task difficulty, not a bug
- Not validated for, and not intended for, real clinical use

---

## 🗺️ Roadmap

- [ ] Validate against larger, more diverse patient cohorts
- [ ] Add missing clinically important variables where available
- [ ] Extend to additional diseases using the existing shared-pipeline pattern
- [ ] Model calibration for more accurate probability estimates
- [ ] Optional Docker Compose setup for one-command deployment

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Rishita Sanghavi** — Independent Developer & Researcher

---

*Built entirely independently as a personal portfolio project.*
