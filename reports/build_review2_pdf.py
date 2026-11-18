"""Build the consolidated PathSnap Review 2 Technical Report PDF from its Markdown source.

Reads `reports/PATHSNAP_REVIEW2_TECHNICAL_REPORT.md` and embeds actual generated
architecture diagrams and experimental plots into a styled multi-page PDF at:
    reports/PATHSNAP_REVIEW2_TECHNICAL_REPORT.pdf
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
MD_SOURCE = REPORTS_DIR / "PATHSNAP_REVIEW2_TECHNICAL_REPORT.md"
PDF_OUTPUT = REPORTS_DIR / "PATHSNAP_REVIEW2_TECHNICAL_REPORT.pdf"


def _escape_inline_md(text: str) -> str:
    """Convert minimal Markdown inline syntax (**bold**, *italic*, `code`) to ReportLab XML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Inline code
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.5" color="#1a365d">\1</font>', text)
    # Bold
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    # LaTeX simple symbols cleanup for crisp PDF rendering (after HTML escaping)
    text = text.replace("$P(y_{t,c}=1 \\mid \\mathbf{x}_{t,c})$", "<i>P</i>(<i>y</i><sub>t,c</sub> = 1 | <b>x</b><sub>t,c</sub>)")
    text = text.replace("$R_\\oplus = 6{,}371{,}000\\text{ m}$", "R = 6,371,000 m")
    text = text.replace("$\\le 25\\text{ m}$", "&lt;= 25 m")
    text = text.replace("$O(T \\cdot R)$", "<i>O</i>(<i>T</i> &middot; <i>R</i>)")
    text = text.replace("$&gt;150\\text{ m}$", "&gt;150 m")
    text = text.replace("$K$", "<i>K</i>")
    text = text.replace("$o_t$", "<i>o</i><sub>t</sub>")
    return text


def _scaled_image(img_path: Path, max_width_cm: float = 16.5, max_height_cm: float = 10.2) -> Image:
    with PILImage.open(img_path) as im:
        w_px, h_px = im.size
    aspect = h_px / max(w_px, 1)
    max_w = max_width_cm * cm
    max_h = max_height_cm * cm
    w = max_w
    h = w * aspect
    if h > max_h:
        h = max_h
        w = h / aspect
    return Image(str(img_path), width=w, height=h)


def _add_page_chrome(canvas, doc) -> None:
    canvas.saveState()
    page_num = canvas.getPageNumber()
    width, height = A4
    if page_num > 1:
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.6)
        canvas.line(1.8 * cm, height - 1.45 * cm, width - 1.8 * cm, height - 1.45 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawString(1.8 * cm, height - 1.25 * cm, "PathSnap — Distributed Map-Matching & Route Reconstruction (Team 13 — Review 2)")
        canvas.drawRightString(width - 1.8 * cm, height - 1.25 * cm, "PES University | UE24CS352A")

    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.6)
    canvas.line(1.8 * cm, 1.35 * cm, width - 1.8 * cm, 1.35 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(1.8 * cm, 0.9 * cm, "Person 4 Integration Report — Kruthik Reddy (feat/kruthik-reddy-mlp-baseline)")
    canvas.drawRightString(width - 1.8 * cm, 0.9 * cm, f"Page {page_num}")
    canvas.restoreState()


def _build_table(rows: List[List[str]], styles) -> Table:
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.4,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
    )
    hdr_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    data = []
    for r_idx, row in enumerate(rows):
        st = hdr_style if r_idx == 0 else cell_style
        data.append([Paragraph(_escape_inline_md(c.strip()), st) for c in row])

    ncols = max(len(r) for r in data)
    avail_w = A4[0] - 3.6 * cm
    col_w = [avail_w / ncols] * ncols
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def build_pdf() -> Path:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#0f2942"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "ReportSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#2b6cb0"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f2942"),
        spaceBefore=12,
        spaceAfter=5,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1e4e79"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=10,
    )

    doc = SimpleDocTemplate(
        str(PDF_OUTPUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.9 * cm,
        title="PathSnap Review 2 Consolidated Technical Report",
        author="Kruthik Reddy — PathSnap Team 13",
    )

    story = []
    # Cover Header Banner
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("PES UNIVERSITY — UE24CS352A: MACHINE LEARNING", caption_style))
    story.append(Paragraph("PathSnap: Distributed Map-Matching &amp; Route Reconstruction", title_style))
    story.append(Paragraph("Review 2 Consolidated Technical Report — Team 13", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a5f"), spaceAfter=10))

    lines = MD_SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    # Skip top two markdown title lines (already rendered in cover banner)
    while i < len(lines) and (lines[i].startswith("# ") or lines[i].startswith("## Review 2")):
        i += 1

    # Section-based chart insertion hooks so every genuine visualization appears right next to its section!
    figure_inserts = {
        "## 13. Actual Experimental Results (MLP `mlp_128_64_32`)": [
            (
                REPO_ROOT / "ml" / "mlp" / "plots" / "loss_curves.png",
                "Figure 2: Actual epoch-wise training and validation BCE loss curves across all 5 GroupKFold CV folds (* marks early-stopping best epoch).",
            ),
            (
                REPO_ROOT / "ml" / "mlp" / "plots" / "performance_curves.png",
                "Figure 3: Actual epoch-wise train vs validation Macro F1-score and validation observation-level Top-1 accuracy across 5 CV folds.",
            ),
            (
                REPO_ROOT / "ml" / "mlp" / "plots" / "student_test_per_trip_comparison.png",
                "Figure 4: Per-trip MLP Observation-Level Top-1 Accuracy and Route Continuity across all 10 held-out student-test trips.",
            ),
        ],
        "## 14. Multi-Model Comparison": [
            (
                REPO_ROOT / "ml" / "comparison" / "plots" / "multi_model_comparison.png",
                "Figure 5: Unified 4-model comparison on the 10 held-out student-test trips (28 features): Classification/Top-1 scores (left) and Log Loss vs inference latency (right).",
            ),
            (
                REPO_ROOT / "ml" / "comparison" / "plots" / "multi_model_roc_pr.png",
                "Figure 6: Actual ROC curves (AUC up to 0.9897) and Precision-Recall curves (MLP AP = 0.9371) on the 172,897 student-test candidates.",
            ),
            (
                REPO_ROOT / "ml" / "comparison" / "plots" / "multi_model_confusion_matrices.png",
                "Figure 7: Candidate-level confusion matrices across all four comparable models on student-test.",
            ),
        ],
    }

    current_heading = ""
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line == "---":
            continue

        # Embedded Markdown image
        img_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", line)
        if img_match:
            cap, rel_p = img_match.group(1), img_match.group(2)
            full_p = (REPORTS_DIR / rel_p).resolve()
            if full_p.exists():
                story.append(
                    KeepTogether(
                        [
                            Spacer(1, 0.15 * cm),
                            _scaled_image(full_p, max_width_cm=16.8, max_height_cm=9.5),
                            Paragraph(f"Figure 1: {cap}", caption_style),
                        ]
                    )
                )
            continue

        # Section Headings
        if line.startswith("## "):
            # Insert queued figures at the end of the previous section if any
            if current_heading in figure_inserts:
                for fig_p, fig_cap in figure_inserts[current_heading]:
                    if fig_p.exists():
                        story.append(
                            KeepTogether(
                                [
                                    Spacer(1, 0.15 * cm),
                                    _scaled_image(fig_p, max_width_cm=16.8, max_height_cm=8.5),
                                    Paragraph(fig_cap, caption_style),
                                ]
                            )
                        )
            current_heading = line
            story.append(Paragraph(_escape_inline_md(line[3:]), h1_style))
            continue

        if line.startswith("### "):
            story.append(Paragraph(_escape_inline_md(line[4:]), h2_style))
            continue

        # Markdown Table
        if line.startswith("|"):
            tbl_lines = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            parsed_rows = []
            for tl in tbl_lines:
                cells = [c.strip() for c in tl.strip("|").split("|")]
                # Skip separator row (---)
                if all(re.match(r"^:?-+:?$", c) for c in cells):
                    continue
                parsed_rows.append(cells)
            if parsed_rows:
                story.append(Spacer(1, 0.1 * cm))
                story.append(_build_table(parsed_rows, styles))
                story.append(Spacer(1, 0.25 * cm))
            continue

        # Bullet / Numbered list block
        if line.startswith("- ") or re.match(r"^\d+\.\s+", line):
            items = [line]
            while i < len(lines):
                nxt = lines[i].strip()
                if nxt.startswith("- ") or re.match(r"^\d+\.\s+", nxt):
                    items.append(nxt)
                    i += 1
                else:
                    break
            flow_items = []
            for it in items:
                cleaned = re.sub(r"^(-\s+|\d+\.\s+)", "", it)
                flow_items.append(
                    ListItem(
                        Paragraph(_escape_inline_md(cleaned), body_style),
                        leftIndent=14,
                    )
                )
            story.append(ListFlowable(flow_items, bulletType="bullet", start="•", spaceAfter=4))
            continue

        # Normal paragraph
        story.append(Paragraph(_escape_inline_md(line), body_style))

    doc.build(story, onFirstPage=_add_page_chrome, onLaterPages=_add_page_chrome)
    return PDF_OUTPUT


if __name__ == "__main__":
    out = build_pdf()
    print(f"Successfully built PDF report: {out} ({out.stat().st_size:,} bytes)")
