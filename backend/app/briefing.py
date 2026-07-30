"""
briefing.py
===========
Builds a clean, printable PDF summary of one prediction: patient inputs,
the model's result, the model-disagreement uncertainty band, the top SHAP
feature contributions, and any out-of-distribution flags. Meant to be
handed to a clinician alongside the raw numbers, not in place of them.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

ACCENT = colors.HexColor("#1F6F5C")
WARN = colors.HexColor("#9A3B26")
MUTED = colors.HexColor("#5A5A5A")


def build_briefing_pdf(disease_label, patient_label, raw_inputs, result, uncertainty, shap_contributions, ood_flags):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=ACCENT, fontSize=20)
    h2 = ParagraphStyle("H2X", parent=styles["Heading2"], textColor=ACCENT, spaceBefore=14, spaceAfter=6)
    normal = ParagraphStyle("NormalX", parent=styles["Normal"], fontSize=10, leading=14)
    muted = ParagraphStyle("MutedX", parent=styles["Normal"], fontSize=8.5, textColor=MUTED, leading=12)

    story = []
    story.append(Paragraph("MediPredict — Risk Prediction Summary", title_style))
    story.append(Paragraph(f"{disease_label} · Patient: {patient_label}", normal))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}", muted))
    story.append(HRFlowable(width="100%", color=ACCENT, thickness=1, spaceBefore=8, spaceAfter=14))

    # --- result block ---
    risk_color = WARN if result["prediction"] == 1 else ACCENT
    story.append(Paragraph("Model Result", h2))
    result_table = Table([
        ["Predicted risk", f"{round(result['probability']*100,1)}%"],
        ["Risk category", result["risk_label"]],
        ["Model agreement", uncertainty["agreement"].capitalize()],
        ["Confidence band", f"{round(uncertainty['confidence_band'][0]*100,1)}% – {round(uncertainty['confidence_band'][1]*100,1)}%"],
    ], colWidths=[2.2 * inch, 3.5 * inch])
    result_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (1, 0), (1, 0), risk_color),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#E0E0E0")),
    ]))
    story.append(result_table)

    # --- base model breakdown ---
    story.append(Paragraph("Base Model Agreement", h2))
    story.append(Paragraph(
        "The final risk score is produced by combining four independently-trained models. "
        "Wide disagreement between them (a low 'agreement' rating above) means the prediction "
        "should be treated with more caution even if the headline number looks confident.",
        normal))
    bm_rows = [["Model", "Predicted probability"]] + [
        [name.upper(), f"{round(p*100,1)}%"] for name, p in uncertainty["base_model_probabilities"].items()
    ]
    bm_table = Table(bm_rows, colWidths=[2.2 * inch, 3.5 * inch])
    bm_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0EEE7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#E0E0E0")),
    ]))
    story.append(bm_table)

    # --- SHAP contributions ---
    if shap_contributions:
        story.append(Paragraph("What Drove This Prediction", h2))
        story.append(Paragraph(
            "Each row below is a real SHAP contribution — how much this specific feature value "
            "pushed the prediction up (+) or down (-) for this patient, computed against the actual model.",
            normal))
        shap_rows = [["Feature", "Patient value", "Contribution"]]
        for c in shap_contributions:
            arrow = "▲ raises risk" if c["impact"] > 0 else "▼ lowers risk"
            shap_rows.append([c["feature"], f"{c['value']:.3g}", f"{c['impact']:+.3f}  ({arrow})"])
        shap_table = Table(shap_rows, colWidths=[2.3 * inch, 1.3 * inch, 2.1 * inch])
        shap_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0EEE7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#E9E9E9")),
        ]))
        story.append(shap_table)

    # --- OOD flags ---
    if ood_flags:
        story.append(Paragraph("Out-of-Distribution Warnings", h2))
        story.append(Paragraph(
            "The values below fall outside the range this model actually saw during training. "
            "The prediction is extrapolating for these fields rather than recognizing a familiar pattern.",
            normal))
        ood_rows = [["Feature", "Value", "Expected range", "Z-score"]] + [
            [f["feature"], str(f["value"]), f"{f['expected_range'][0]} – {f['expected_range'][1]}", str(f["z_score"])]
            for f in ood_flags
        ]
        ood_table = Table(ood_rows, colWidths=[1.8 * inch, 1.1 * inch, 1.8 * inch, 1 * inch])
        ood_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FBE9E4")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#E9E9E9")),
        ]))
        story.append(ood_table)
    else:
        story.append(Paragraph("Out-of-Distribution Check", h2))
        story.append(Paragraph("All input values fall within the range this model was trained on.", normal))

    # --- raw inputs appendix ---
    story.append(Paragraph("Raw Inputs Used", h2))
    input_rows = [["Field", "Value"]] + [[k, str(v)] for k, v in raw_inputs.items()]
    input_table = Table(input_rows, colWidths=[2.5 * inch, 3.2 * inch])
    input_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), MUTED),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(input_table)

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#CCCCCC"), thickness=0.5))
    story.append(Paragraph(
        "This document is generated by a statistical model trained on historical patient data. "
        "It is a decision-support estimate, not a medical diagnosis, and is not a substitute for "
        "clinical judgment or laboratory-confirmed testing. Generated by MediPredict — built by Rishita Sanghavi.",
        muted))

    doc.build(story)
    buf.seek(0)
    return buf
