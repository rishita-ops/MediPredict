"""
reference_ranges.py
====================
Published clinical "normal" reference ranges for the lab/exam fields used
across the four diseases. These are standard textbook ranges (Harrison's /
common lab reference intervals) -- used to show the person where their
input sits relative to normal BEFORE they even run a prediction, and to
provide a sane fallback value for imputing a field the user left blank.

Format per field: (low, high, unit, midpoint_for_imputation)
midpoint_for_imputation is only used if a training-data median isn't
available for some reason -- the primary imputation source is the
training-data median saved by train.py.
"""

REFERENCE_RANGES = {
    "heart": {
        "trestbps": (90, 120, "mmHg", "Resting systolic blood pressure. 120-129 is elevated, 130+ is high."),
        "chol": (125, 200, "mg/dL", "Total cholesterol. 200-239 is borderline, 240+ is high."),
        "thalach": (100, 190, "bpm", "Max heart rate achieved during exercise. Expected max is roughly 220 minus age."),
        "oldpeak": (0, 1.0, "mm", "ST depression induced by exercise. Above 1mm is considered clinically significant."),
    },
    "diabetes": {
        "glucose": (70, 99, "mg/dL", "Fasting-range plasma glucose. 100-125 is prediabetic, 126+ is diabetic range."),
        "bp": (60, 80, "mmHg", "Diastolic blood pressure. Above 90 is considered high."),
        "bmi": (18.5, 24.9, "kg/m²", "Body mass index. 25-29.9 is overweight, 30+ is obese."),
        "insulin": (16, 166, "mu U/mL", "2-hour serum insulin. Wide normal range; very high values suggest insulin resistance."),
        "skinthickness": (10, 25, "mm", "Triceps skin-fold thickness, a rough body-fat proxy."),
    },
    "liver": {
        "total_bilirubin": (0.1, 1.2, "mg/dL", "Total bilirubin. Elevated levels suggest liver dysfunction or bile duct issues."),
        "direct_bilirubin": (0.1, 0.3, "mg/dL", "Direct (conjugated) bilirubin. High values point toward bile-duct obstruction."),
        "alkaline_phosphotase": (44, 147, "IU/L", "Alkaline phosphatase. Elevated in bile-duct obstruction or bone disease."),
        "alamine_aminotransferase": (7, 56, "U/L", "ALT. Elevated specifically signals liver cell damage."),
        "aspartate_aminotransferase": (10, 40, "U/L", "AST. Elevated in liver damage, but also muscle/heart damage."),
        "total_protiens": (6.0, 8.3, "g/dL", "Total blood protein."),
        "albumin": (3.4, 5.4, "g/dL", "Albumin, made only by the liver. Low levels signal reduced synthetic function."),
        "albumin_and_globulin_ratio": (1.0, 2.5, "ratio", "Albumin-to-globulin ratio."),
    },
    "kidney": {
        "bp": (60, 90, "mmHg", "Blood pressure. Sustained high readings accelerate kidney damage."),
        "sg": (1.005, 1.030, "", "Urine specific gravity. Very low or high values suggest concentration problems."),
        "bgr": (70, 140, "mg/dL", "Random blood glucose. Sustained highs damage kidney filtration over time."),
        "bu": (7, 20, "mg/dL", "Blood urea. Elevated when kidneys aren't filtering waste properly."),
        "sc": (0.6, 1.3, "mg/dL", "Serum creatinine. The single most direct marker of kidney filtration rate."),
        "sod": (135, 145, "mEq/L", "Sodium. Kidneys regulate this tightly; abnormal levels signal dysfunction."),
        "pot": (3.5, 5.0, "mEq/L", "Potassium. Failing kidneys often can't clear excess potassium -- a dangerous combination."),
        "hemo": (12.0, 17.0, "g/dL", "Hemoglobin. Low levels are common in CKD since kidneys make less erythropoietin."),
        "pcv": (36, 50, "%", "Packed cell volume (hematocrit)."),
        "wc": (4000, 11000, "cells/cumm", "White blood cell count."),
        "rc": (4.2, 5.9, "million/cumm", "Red blood cell count."),
    },
}


def get_ranges(disease):
    return REFERENCE_RANGES.get(disease, {})
