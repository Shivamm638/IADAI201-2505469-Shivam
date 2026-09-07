"""
Build the submission PDF required by the assignment's checklist of evidence.

Edit ``submission/student_details.json`` first, then run:

    python scripts/05_make_submission.py

Produces ``submission/SA1_Submission_ParkVision_AI.pdf`` containing the GitHub
link, the live Streamlit link, the student's details, a project summary and the
measured results.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import config as C

DETAILS_PATH = C.PROJECT_ROOT / "submission" / "student_details.json"
OUT_PATH = C.PROJECT_ROOT / "submission" / "SA1_Submission_ParkVision_AI.pdf"

NAVY = colors.HexColor("#0F2C4C")
BLUE = colors.HexColor("#1E63A8")
GREY = colors.HexColor("#64748B")
LIGHT = colors.HexColor("#F3F6FB")
BORDER = colors.HexColor("#D5DCE7")
AMBER = colors.HexColor("#B45309")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=21, leading=25,
                                textColor=NAVY, spaceAfter=2),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=11.5, leading=15,
                                   textColor=BLUE, alignment=TA_CENTER, spaceAfter=3),
        "scenario": ParagraphStyle("sc", parent=base["Normal"], fontSize=9.5, leading=13,
                                   textColor=GREY, alignment=TA_CENTER, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12.5, leading=16,
                             textColor=NAVY, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=13.5),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=9, leading=12),
        "small": ParagraphStyle("s", parent=base["Normal"], fontSize=8, leading=11,
                                textColor=GREY),
        "warn": ParagraphStyle("w", parent=base["Normal"], fontSize=9, leading=12.5,
                               textColor=AMBER),
    }


def link(url: str, label: str | None = None) -> str:
    if not url or url.startswith("FILL_IN"):
        return f'<font color="#B45309">{url or "(not provided)"}</font>'
    return f'<link href="{url}"><font color="#1E63A8"><u>{label or url}</u></font></link>'


def value_or_placeholder(value: str) -> str:
    if not value or value.startswith("FILL_IN"):
        return f'<font color="#B45309"><b>{value or "(not provided)"}</b></font>'
    return value


def kv_table(rows, s, col_widths=(52 * mm, 116 * mm)):
    data = [
        [Paragraph(f"<b>{k}</b>", s["cell"]), Paragraph(v, s["cell"])] for k, v in rows
    ]
    table = Table(data, colWidths=list(col_widths))
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    return table


def data_table(header, rows, s, col_widths):
    data = [[Paragraph(f"<b>{h}</b>", s["cell"]) for h in header]]
    data += [[Paragraph(str(c), s["cell"]) for c in row] for row in rows]
    table = Table(data, colWidths=list(col_widths), repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    return table


def load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def build() -> None:
    if not DETAILS_PATH.exists():
        raise SystemExit(f"Missing {DETAILS_PATH}")

    details = load_json(DETAILS_PATH)
    metrics = load_json(C.METRICS_PATH)
    dataset = load_json(C.RESULTS_DIR / "dataset_report.json")
    training = load_json(C.RESULTS_DIR / "training_summary.json")
    robust = load_json(C.RESULTS_DIR / "robustness.json")

    s = styles()
    story = []

    repo_url = ""
    if not str(details.get("github_username", "")).startswith("FILL_IN") and \
       not str(details.get("github_repository", "")).startswith("FILL_IN"):
        repo_url = f"https://github.com/{details['github_username']}/{details['github_repository']}"
    else:
        repo_url = "FILL_IN — https://github.com/&lt;username&gt;/&lt;repository&gt;"

    app_url = details.get("streamlit_app_url", "")

    # ------------------------------------------------------------- header
    story.append(Paragraph("Summative Assessment — Submission Document", s["title"]))
    story.append(Paragraph(details.get("assignment_title", ""), s["subtitle"]))
    story.append(Paragraph(details.get("scenario", ""), s["scenario"]))
    story.append(HRFlowable(width="100%", color=BORDER, thickness=1, spaceAfter=10))

    # ------------------------------------------------------- 1. the links
    story.append(Paragraph("1 · Project links", s["h2"]))
    story.append(kv_table([
        ("GitHub repository", link(repo_url)),
        ("Live Streamlit app", link(app_url)),
        ("Assessor access granted to", details.get("assessor_access_granted_to", "")),
    ], s))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Both links are clickable. The repository contains all code, the trained model, "
        "the processed dataset, the evaluation results and a detailed README.", s["small"]))

    # -------------------------------------------------- 2. candidate details
    story.append(Paragraph("2 · Candidate details", s["h2"]))
    story.append(kv_table([
        ("Student's full name", value_or_placeholder(details.get("student_full_name", ""))),
        ("Candidate registration number",
         value_or_placeholder(details.get("candidate_registration_number", ""))),
        ("CRS name", details.get("crs_name", "Artificial Intelligence")),
        ("Course name", details.get("course_name", "")),
        ("School name", value_or_placeholder(details.get("school_name", ""))),
        ("Type of assessment", details.get("assessment_type", "")),
        ("Date of submission", date.today().strftime("%d %B %Y")),
    ], s))

    # ------------------------------------------------------- 3. the project
    story.append(Paragraph("3 · Project summary", s["h2"]))
    story.append(Paragraph(
        "<b>ParkVision AI</b> analyses a photograph of a parking lot and reports, bay by bay, "
        "which spaces are occupied and which are free. Every bay is perspective-corrected out "
        "of the photograph and classified by a MobileNetV2 network fine-tuned on the PKLot "
        "dataset. The per-bay results are aggregated into availability counts, an occupancy "
        "percentage, a congestion level and a recommendation, then presented through a "
        "Streamlit dashboard with colour-coded overlays — green for empty, red for occupied, "
        "amber where the model was unsure.", s["body"]))
    story.append(Spacer(1, 3 * mm))
    story.append(kv_table([
        ("Approach", "Slot-level image classification over surveyed bay geometry"),
        ("Model", "MobileNetV2 (ImageNet), two-phase transfer learning"),
        ("Dataset", "PKLot — Almeida et al., Expert Systems with Applications, 2015"),
        ("Interface", "Streamlit dashboard, deployed on Streamlit Community Cloud"),
    ], s))

    # ------------------------------------------------------- 4. the dataset
    if dataset:
        dataset_block = [Paragraph("4 · Dataset", s["h2"])]
        crops = dataset.get("crops_per_split", {})
        rows = [
            [split.title(),
             f"{crops.get(split, {}).get('empty', 0):,}",
             f"{crops.get(split, {}).get('occupied', 0):,}",
             f"{sum(crops.get(split, {}).values()):,}"]
            for split in ("train", "val", "test")
        ]
        rows.append(["<b>Total</b>",
                     f"<b>{sum(crops.get(sp, {}).get('empty', 0) for sp in crops):,}</b>",
                     f"<b>{sum(crops.get(sp, {}).get('occupied', 0) for sp in crops):,}</b>",
                     f"<b>{dataset.get('total_crops', 0):,}</b>"])
        dataset_block.append(data_table(["Split", "Empty", "Occupied", "Total"], rows, s,
                                        (42 * mm, 42 * mm, 42 * mm, 42 * mm)))
        dataset_block.append(Spacer(1, 3 * mm))
        dataset_block.append(Paragraph(
            f"Built from {dataset.get('photographs_indexed', 0):,} PKLot photographs across "
            f"{len(dataset.get('lots', {}))} parking lots and "
            f"{len(dataset.get('weather', {}))} weather conditions, resized to "
            f"{'x'.join(map(str, dataset.get('image_size', [224, 224])))} pixels. "
            "<b>Split strategy:</b> whole (lot, calendar date) groups are assigned to a single "
            "split, so no day appears in more than one split and the reported accuracy is not "
            "inflated by near-identical frames taken five minutes apart.", s["small"]))
        story.append(KeepTogether(dataset_block))

    # ------------------------------------------------------- 5. the results
    if metrics:
        test = metrics.get("test", {})
        story.append(Paragraph("5 · Results", s["h2"]))
        rows = [
            ["Test accuracy", f"{test.get('accuracy', 0) * 100:.2f}%"],
            ["Precision (occupied)", f"{test.get('precision_occupied', 0) * 100:.2f}%"],
            ["Recall (occupied)", f"{test.get('recall_occupied', 0) * 100:.2f}%"],
            ["F1 score (occupied)", f"{test.get('f1_occupied', 0) * 100:.2f}%"],
            ["ROC AUC", f"{test.get('roc_auc', 0):.4f}"],
            ["Errors on the held-out split",
             f"{test.get('errors', 0)} of {test.get('n_samples', 0)} bays"],
        ]
        if training:
            rows.append(["Training", f"{training.get('epochs_run', 0)} epochs "
                                     f"({training.get('epochs_head', 0)} head + "
                                     f"{training.get('epochs_finetune', 0)} fine-tune)"])
        story.append(data_table(["Metric", "Value"], rows, s, (100 * mm, 68 * mm)))

        if metrics.get("by_weather"):
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph("<b>Accuracy by weather condition</b>", s["cell"]))
            story.append(Spacer(1, 1.5 * mm))
            story.append(data_table(
                ["Weather", "Bays tested", "Accuracy"],
                [[k.title(), f"{v['n']:,}", f"{v['accuracy'] * 100:.2f}%"]
                 for k, v in metrics["by_weather"].items()],
                s, (56 * mm, 56 * mm, 56 * mm)))

        scene_rows = (robust or {}).get("scene_test", [])
        if scene_rows:
            correct = sum(1 for r in scene_rows if r.get("congestion_correct"))
            mean_err = sum(r.get("available_error", 0) for r in scene_rows) / len(scene_rows)
            mean_acc = sum(r.get("slot_accuracy") or 0 for r in scene_rows) / len(scene_rows)
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph(
                f"<b>End-to-end testing on {len(scene_rows)} held-out whole-lot photographs:</b> "
                f"mean slot accuracy {mean_acc * 100:.2f}%, mean error in the free-bay count "
                f"{mean_err:.2f} bays, congestion level called correctly on "
                f"{correct} of {len(scene_rows)} scenes.", s["small"]))

    # ----------------------------------------------------- 6. the checklist
    story.append(Paragraph("6 · Repository checklist", s["h2"]))
    checklist = [
        "All project code and script files (<b>.py</b> and <b>.ipynb</b>)",
        "Functional link to the deployed Streamlit web application",
        "The dataset used in the project (<b>data/processed/</b>, 6,000 labelled slot crops)",
        "The trained model (<b>models/parkvision_mobilenetv2.keras</b>)",
        "Detailed <b>README.md</b> covering research findings, references, data preparation, "
        "models and training parameters, accuracy metrics, and screenshots of the demo",
        "Repository accessible to <b>ai.assignments@wacpinternational.org</b>",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(item, s["cell"]), leftIndent=10) for item in checklist],
        bulletType="bullet", bulletFontSize=7, leftIndent=12, bulletColor=BLUE,
    ))

    unresolved = [k for k, v in details.items()
                  if isinstance(v, str) and v.startswith("FILL_IN") and not k.startswith("_")]
    if unresolved:
        story.append(Spacer(1, 5 * mm))
        story.append(KeepTogether([
            HRFlowable(width="100%", color=AMBER, thickness=1, spaceAfter=4),
            Paragraph(
                "<b>Before submitting:</b> the fields shown in amber above are still "
                "placeholders. Fill them in <b>submission/student_details.json</b> and re-run "
                "<b>python scripts/05_make_submission.py</b> to regenerate this PDF.<br/>"
                "Outstanding: " + ", ".join(unresolved), s["warn"]),
        ]))

    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", color=BORDER, thickness=0.5, spaceAfter=4))
    story.append(Paragraph(
        f"ParkVision AI · {details.get('crs_name', '')} · {details.get('course_name', '')} · "
        f"generated {date.today().isoformat()}", s["small"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(OUT_PATH), pagesize=A4,
        leftMargin=21 * mm, rightMargin=21 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="ParkVision AI — Summative Assessment Submission",
        author=details.get("student_full_name", ""),
    ).build(story)

    print(f"wrote {OUT_PATH}")
    if unresolved:
        print("\n  Placeholders still to fill in submission/student_details.json:")
        for key in unresolved:
            print(f"    - {key}")
        print("  Then re-run this script.")


if __name__ == "__main__":
    build()
