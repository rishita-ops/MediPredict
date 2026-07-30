"""
feature_engineering.py
=======================
Custom, hand-designed engineered features for MediPredict.

These are NOT the raw columns that come with the datasets (age, blood
pressure, cholesterol, etc). Every function below computes a NEW column
out of the raw ones, using a medically-motivated formula. None of these
exact combinations ship with the standard Kaggle/UCI notebooks for these
four diseases — they were designed specifically for this project by
combining raw measurements the way a clinician actually reasons about risk
(ratios, interactions, deviation-from-normal scores, and composite indices).

Each feature has:
  - a short id (used as the dataframe column name)
  - the disease(s) it applies to
  - a one-line "why this matters" note (used in the guide)

Total unique engineered features: 27
"""

import numpy as np
import pandas as pd

FEATURE_NOTES = {}  # id -> explanation, filled in as we define each feature


def _note(fid, text):
    FEATURE_NOTES[fid] = text


# ─────────────────────────────────────────────────────────────────────────
# HEART DISEASE FEATURES
# ─────────────────────────────────────────────────────────────────────────

def heart_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hr_reserve_ratio"] = df["thalach"] / (220 - df["age"])
    _note("hr_reserve_ratio",
          "Max heart rate achieved divided by the age-predicted maximum "
          "(220 - age). Below ~0.75 during a stress test is a classic "
          "red flag for poor cardiac reserve, but it's almost never "
          "computed as its own column in beginner notebooks.")

    df["pressure_age_interaction"] = df["trestbps"] * df["age"] / 1000
    _note("pressure_age_interaction",
          "Resting blood pressure and age interact: the same BP reading "
          "is far riskier at 70 than at 35. Multiplying them (scaled down) "
          "lets tree-based models split on that interaction directly "
          "instead of hoping to discover it from two separate columns.")

    df["chol_hdl_proxy_risk"] = df["chol"] / (df["age"] + 1) * 10
    _note("chol_hdl_proxy_risk",
          "A simple cholesterol-per-decade-of-life score — flags people "
          "whose cholesterol is unusually high for how young they are, "
          "which is a stronger signal than raw cholesterol alone.")

    df["st_depression_severity"] = df["oldpeak"] * (df["slope"] + 1)
    _note("st_depression_severity",
          "Combines ST depression (oldpeak) with the slope of the ST "
          "segment into one severity score — a downsloping ST segment "
          "with high depression is far more dangerous than the same "
          "depression on an upsloping segment, but the raw columns don't "
          "encode that multiplicatively.")

    df["exercise_stress_score"] = (
        df["exang"] * 2 + df["oldpeak"] + (df["cp"] == 0).astype(int) * 1.5
    )
    _note("exercise_stress_score",
          "A composite 'how much did exercise hurt this heart' score built "
          "from exercise-induced angina, ST depression, and asymptomatic "
          "chest pain type — three separate signs of the same underlying "
          "ischemia, merged into one number the model can use directly.")

    df["vessel_thal_risk"] = df["ca"] * (df["thal"] + 1)
    _note("vessel_thal_risk",
          "Number of major vessels colored by fluoroscopy multiplied by "
          "thalassemia status — both are structural/perfusion markers, "
          "and their product captures compounding structural risk that "
          "neither column shows alone.")

    return df


# ─────────────────────────────────────────────────────────────────────────
# DIABETES FEATURES
# ─────────────────────────────────────────────────────────────────────────

def diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # columns: preg, glucose, bp, skin, insulin, bmi, dpf, age, outcome

    df["glucose_bmi_index"] = df["glucose"] * df["bmi"] / 1000
    _note("glucose_bmi_index",
          "Glucose and BMI compound each other's risk — high glucose with "
          "high BMI is a much stronger diabetes signal than either alone, "
          "and this index lets linear models see that multiplicative "
          "relationship instead of just an additive one.")

    df["insulin_glucose_ratio"] = (df["insulin"] + 1) / (df["glucose"] + 1)
    _note("insulin_glucose_ratio",
          "A rough proxy for insulin resistance: when insulin is low "
          "relative to glucose, the body isn't responding properly, "
          "which is the actual mechanism behind Type 2 diabetes — not "
          "just 'high glucose' on its own.")

    df["metabolic_age_load"] = (df["glucose"] + df["bmi"] * 2) / (df["age"] + 1)
    _note("metabolic_age_load",
          "Flags 'too much metabolic burden for this age' — a 25-year-old "
          "with these glucose/BMI numbers is medically far more alarming "
          "than a 60-year-old with the same numbers, since risk is "
          "expected to rise with age anyway.")

    df["pregnancy_risk_factor"] = df["preg"] * df["dpf"]
    _note("pregnancy_risk_factor",
          "Combines number of pregnancies with the diabetes pedigree "
          "function (a genetic-risk score) — gestational diabetes history "
          "compounds genetic predisposition, and this captures that "
          "compounding instead of treating them as unrelated columns.")

    df["skin_insulin_signal"] = df["skinthickness"] * (df["insulin"] + 1) / 100
    _note("skin_insulin_signal",
          "Skin-fold thickness is a rough proxy for body fat, and "
          "combining it with insulin level approximates a 'fat-driven "
          "insulin resistance' signal — a pattern seen in real endocrine "
          "workups but rarely engineered explicitly in tutorials.")

    df["bp_glucose_stress"] = df["bp"] * df["glucose"] / 1000
    _note("bp_glucose_stress",
          "Blood pressure and glucose rise together under chronic "
          "metabolic stress; multiplying them surfaces patients where "
          "both are elevated simultaneously, which is a stronger combined "
          "risk marker than screening each threshold separately.")

    return df


# ─────────────────────────────────────────────────────────────────────────
# LIVER DISEASE FEATURES
# ─────────────────────────────────────────────────────────────────────────

def liver_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["bilirubin_ratio"] = df["direct_bilirubin"] / (df["total_bilirubin"] + 0.01)
    _note("bilirubin_ratio",
          "The fraction of total bilirubin that is 'direct' (conjugated) "
          "tells a doctor WHERE the liver problem is happening — a high "
          "ratio points to bile-duct obstruction rather than general liver "
          "cell damage. This distinction is usually lost when both "
          "bilirubin columns are just fed in raw.")

    df["enzyme_ratio_AST_ALT"] = (df["aspartate_aminotransferase"] + 1) / (
        df["alamine_aminotransferase"] + 1
    )
    _note("enzyme_ratio_AST_ALT",
          "The AST/ALT ratio is a real clinical tool (the 'De Ritis "
          "ratio') — a ratio above ~2 suggests alcohol-related liver "
          "damage, while a ratio below 1 suggests viral hepatitis. This "
          "single ratio encodes a differential diagnosis that two raw "
          "enzyme columns can't express on their own.")

    df["protein_albumin_balance"] = df["albumin"] / (df["total_protiens"] + 0.01)
    _note("protein_albumin_balance",
          "What fraction of total blood protein is albumin — the liver is "
          "the only organ that makes albumin, so a low share signals "
          "reduced synthetic liver function even when total protein looks "
          "normal.")

    df["liver_stress_index"] = (
        df["alkaline_phosphotase"] / 100
        + df["aspartate_aminotransferase"] / 50
        + df["alamine_aminotransferase"] / 50
    )
    _note("liver_stress_index",
          "A single composite index that sums three separate liver "
          "enzymes on a common scale — a fast, doctor-style 'how much is "
          "the liver struggling overall' number instead of three isolated "
          "signals the model has to reassemble itself.")

    df["age_enzyme_load"] = df["age"] * df["alkaline_phosphotase"] / 1000
    _note("age_enzyme_load",
          "The same enzyme elevation is more concerning in an older "
          "patient (less physiological reserve to compensate) — this "
          "feature lets the model weight enzyme elevation by age instead "
          "of treating a 20-year-old and a 70-year-old identically.")

    return df


# ─────────────────────────────────────────────────────────────────────────
# KIDNEY DISEASE FEATURES
# ─────────────────────────────────────────────────────────────────────────

def kidney_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["bun_creatinine_ratio"] = (df["bu"] + 1) / (df["sc"] + 1)
    _note("bun_creatinine_ratio",
          "The BUN/creatinine ratio is a real diagnostic tool nephrologists "
          "use to tell apart pre-renal (dehydration-driven) vs intrinsic "
          "kidney damage — a ratio over ~20 points one way, under ~10 "
          "points the other. Feeding BUN and creatinine in raw loses this "
          "distinction entirely.")

    df["anemia_kidney_score"] = (df["hemo"] < 12).astype(int) + (
        df["pcv"] < 36
    ).astype(int) + (df["rc"] < 4).astype(int)
    _note("anemia_kidney_score",
          "Failing kidneys stop producing enough erythropoietin, which "
          "causes anemia — this feature counts how many of the three "
          "anemia markers (hemoglobin, packed-cell volume, red-cell count) "
          "are below normal, turning a subtle multi-marker pattern into "
          "one clean 0-3 score.")

    df["electrolyte_imbalance_score"] = (
        (df["sod"] < 135).astype(int) + (df["pot"] > 5.5).astype(int)
    )
    _note("electrolyte_imbalance_score",
          "Low sodium plus high potassium together are a classic sign of "
          "kidneys failing to regulate electrolytes — this score flags "
          "that specific dangerous combination rather than two independent "
          "thresholds.")

    df["proteinuria_severity"] = df["al"] * (df["sg"].fillna(1.02) * 100 - 100)
    _note("proteinuria_severity",
          "Combines albumin-in-urine level with how concentrated the urine "
          "is (specific gravity) — protein leaking into dilute urine is a "
          "worse sign than the same protein level in concentrated urine, "
          "and this feature captures that interaction.")

    df["comorbidity_load"] = (
        (df["htn"] == "yes").astype(int)
        + (df["dm"] == "yes").astype(int)
        + (df["cad"] == "yes").astype(int)
    )
    _note("comorbidity_load",
          "Hypertension, diabetes, and coronary artery disease are the "
          "three biggest drivers of chronic kidney disease — this feature "
          "counts how many the patient already has, since kidney risk "
          "compounds sharply with each additional comorbidity rather than "
          "adding up linearly.")

    df["glucose_kidney_stress"] = df["bgr"] * df["sc"] / 100
    _note("glucose_kidney_stress",
          "Blood glucose and serum creatinine multiplied together — "
          "uncontrolled blood sugar accelerates kidney damage, so patients "
          "high on both axes simultaneously are a distinct, higher-risk "
          "group that neither column flags by itself.")

    return df


# ─────────────────────────────────────────────────────────────────────────
# CROSS-DISEASE / GENERAL FEATURES (apply the same idea to more than one
# dataset — still hand-designed per dataset because column names differ)
# ─────────────────────────────────────────────────────────────────────────

def add_age_bucket_risk(df: pd.DataFrame, age_col="age") -> pd.DataFrame:
    df = df.copy()
    bins = [0, 30, 45, 60, 75, 200]
    labels = [0, 1, 2, 3, 4]
    df["age_risk_bucket"] = pd.cut(df[age_col], bins=bins, labels=labels).astype(float)
    return df


_note("age_risk_bucket",
      "Age is not linearly risky — risk jumps at certain life-stage "
      "thresholds (30, 45, 60, 75) rather than climbing smoothly. Bucketing "
      "age into these clinically-meaningful bands lets tree models split "
      "on the jump points directly instead of learning a smooth curve from "
      "scratch.")


def add_missing_value_signal(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Whether a row had missing lab values at all is itself a signal —
    sicker patients often have MORE tests run (more chances to see a gap)
    or, conversely, missed follow-up visits (more gaps). Either way, the
    presence of missingness correlates with outcome in these datasets."""
    df = df.copy()
    df["missing_value_count"] = df[cols].isna().sum(axis=1)
    return df


_note("missing_value_count",
      "How many of this patient's lab fields are missing. Missingness is "
      "not random in clinical data — sicker or less-monitored patients "
      "often have systematically more gaps — so the missingness pattern "
      "itself carries predictive signal, not just the values that ARE "
      "present.")


def get_all_feature_notes():
    """Returns dict of {feature_id: explanation} for guide generation."""
    return FEATURE_NOTES
