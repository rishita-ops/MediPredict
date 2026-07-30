"""
glossary.py
===========
Plain-language clinical definitions for terms used across the app, served
to the frontend for tooltip popups.
"""

GLOSSARY = {
    "cp": "Chest pain type. 'Typical angina' is classic exercise-related chest pain from reduced blood flow to the heart; 'asymptomatic' means no chest pain despite underlying disease, which is actually a more concerning pattern in this dataset.",
    "thal": "Thalassemia status from a nuclear stress test. 'Fixed defect' suggests a permanent area of reduced blood flow (old damage); 'reversible defect' suggests an area that's short on blood flow only during stress (active ischemia).",
    "ca": "Number of major coronary vessels visibly narrowed on a fluoroscopy (X-ray with contrast dye). More narrowed vessels means more extensive coronary artery disease.",
    "oldpeak": "ST depression: how much the ST segment of the heart's electrical trace dips during exercise compared to rest. A bigger dip suggests the heart muscle is short on oxygen under exertion.",
    "slope": "The slope of the ST segment during peak exercise. A downsloping pattern is the most concerning of the three; upsloping is the least concerning.",
    "dpf": "Diabetes Pedigree Function -- a score summarizing family history of diabetes, weighted by how closely related the affected relatives are and their age of onset.",
    "bilirubin_ratio_note": "De Ritis-style bilirubin split: direct (conjugated) vs total bilirubin. The split tells you whether a liver problem is more about bile-duct drainage or general liver-cell damage.",
    "ast_alt": "AST/ALT ratio, sometimes called the De Ritis ratio. Above ~2 suggests alcohol-related liver damage; below ~1 suggests viral hepatitis. It's a real differential-diagnosis tool, not just two numbers side by side.",
    "sg": "Urine specific gravity: how concentrated the urine is relative to pure water. Kidneys that can't concentrate urine properly (a sign of tubule damage) produce urine with specific gravity close to 1.000.",
    "al": "Albumin detected in urine, scored 0-5. Healthy kidneys shouldn't let much protein through their filters, so any albumin in urine is a sign of filter damage.",
    "bun_creatinine": "BUN/creatinine ratio. Nephrologists use this specific ratio to tell apart dehydration-driven kidney stress (high ratio) from actual structural kidney damage (lower ratio, both values elevated together).",
    "rbc_urine": "Red blood cells found on urine microscopy. 'Abnormal' can mean anything from a urinary tract infection to kidney inflammation.",
    "shap": "SHAP (SHapley Additive exPlanations): a method from game theory that fairly splits credit for a prediction among all the input features, based on how much each one actually changed the outcome for this specific patient.",
    "ood": "Out-of-distribution: when a patient's values fall outside what the model saw during training. The model has to extrapolate rather than recognize a familiar pattern, so predictions in this zone deserve more skepticism.",
    "framingham": "The Framingham 10-year cardiovascular risk score: a formula published in 2008 from decades of the Framingham Heart Study, estimating the chance of a cardiovascular event in the next 10 years from age, cholesterol, blood pressure, smoking, and diabetes status.",
    "stacking": "Stacked ensemble: several different models each make their own prediction, then a final 'meta' model learns how to best combine those predictions -- often more accurate than any single model alone.",
}


def get_glossary():
    return GLOSSARY
