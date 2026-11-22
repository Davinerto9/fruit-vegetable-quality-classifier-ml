"""Export an executive/technical PDF from the recorded modeling JSON files.

Run from the repository root:
    python scripts/export_modeling_summary_pdf.py

Requires ReportLab (`python -m pip install reportlab`). The script reads only
existing result JSON files; it does not train models or access the test split.
"""

from __future__ import annotations

import json
import csv
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "output" / "pdf" / "modeling-session-summary.pdf"


def read_json(filename: str) -> dict:
    path = RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el resultado requerido: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_row(label: str, payload: dict, metric_key: str = "validation") -> list[str]:
    values = payload.get(metric_key, payload)
    return [
        label,
        f"{values['f1_macro']:.4f}",
        f"{values['balanced_accuracy']:.4f}",
    ]


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E0EA"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526174"))
    canvas.drawString(18 * mm, 10 * mm, "Clasificador de calidad de frutas y verduras · uso interno del equipo")
    canvas.drawRightString(192 * mm, 10 * mm, f"Página {doc.page}")
    canvas.restoreState()


def make_table(data: list[list[str]], widths: list[float], font_size: int = 8) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFD8E3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def main() -> None:
    logistic = read_json("logistic_baseline.json")
    svm = read_json("svm.json")
    forest = read_json("random_forest.json")
    cnn = read_json("cnn_rgb.json")
    audit = read_json("validation_audit.json")
    pair = audit["rf_cnn_pair_status_counts"]
    rf_cnn_errors = sum(pair[key] for key in ("rf_only_wrong", "cnn_only_wrong", "both_wrong"))
    error_csv = RESULTS / "validation_error_cases.csv"
    with error_csv.open(encoding="utf-8-sig", newline="") as stream:
        all_model_error_rows = max(sum(1 for _ in csv.reader(stream)) - 1, 0)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=28, textColor=colors.HexColor("#16324F"),
        alignment=TA_CENTER, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", parent=styles["Normal"], fontSize=10, leading=15,
        alignment=TA_CENTER, textColor=colors.HexColor("#526174"), spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontSize=14, leading=18,
        textColor=colors.HexColor("#16324F"), spaceBefore=12, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=13,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Callout", parent=styles["BodyText"], fontSize=9.5, leading=14,
        backColor=colors.HexColor("#FFF4D6"), borderColor=colors.HexColor("#E6B94A"),
        borderWidth=0.6, borderPadding=8, spaceBefore=5, spaceAfter=9,
    ))

    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    story = []
    story.append(Spacer(1, 25 * mm))
    story.append(Paragraph("Resumen de modelado y auditoría", styles["CoverTitle"]))
    story.append(Paragraph("Clasificador de calidad de frutas y verduras", styles["Subtitle"]))
    story.append(Paragraph(f"Corte: {date.today().strftime('%d/%m/%Y')} · Proyecto académico 2026-2", styles["Subtitle"]))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "Estado: los cuatro modelos ya tienen checkpoints y métricas iniciales de validation. "
        "La revisión de errores sigue abierta. El conjunto test no se ha consultado.",
        styles["Callout"],
    ))
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("Decisión de cierre de sesión", styles["Section"]))
    story.append(Paragraph(
        "No congelar todavía. Primero alinear scikit-learn a 1.7.2, repetir el audit de validation, "
        "revisar los 871 casos señalados y resolver/documentar el sesgo de fuente y la situación de mango. "
        "Después se congela el experimento y se abre test una sola vez.", styles["BodyText"]
    ))
    story.append(PageBreak())

    story.append(Paragraph("1. Modelos y resultados", styles["Section"]))
    story.append(Paragraph(
        f"Validación: {audit['n_rows']:,} filas, {audit['n_groups']:,} grupos. "
        f"Los cuatro experimentos usan el mismo split; el audit reporta test_used={audit.get('test_used')}.",
        styles["BodyText"],
    ))
    rows = [["Modelo", "F1 macro", "Balanced accuracy"]]
    rows.extend([
        metric_row("Regresión logística", logistic),
        metric_row("SVM lineal", svm),
        metric_row("Random Forest", forest),
        metric_row("CNN RGB", cnn),
    ])
    story.append(make_table(rows, [72 * mm, 35 * mm, 48 * mm], 9))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "RF es el mejor resultado agregado actual (F1 macro 0.8424). CNN RGB registra 0.7988. "
        "Estas cifras son de validation y no equivalen a rendimiento final de generalización.", styles["BodyText"]
    ))

    story.append(Paragraph("Rendimiento por clase: RF y CNN", styles["Section"]))
    rf_report = forest["validation"]["classification_report"]
    cnn_report = cnn["validation"]["classification_report"]
    class_rows = [["Clase", "Soporte", "F1 RF", "F1 CNN"]]
    for name in ("good", "regular", "bad"):
        class_rows.append([
            name,
            str(int(rf_report[name]["support"])),
            f"{rf_report[name]['f1-score']:.4f}",
            f"{cnn_report[name]['f1-score']:.4f}",
        ])
    story.append(make_table(class_rows, [40 * mm, 35 * mm, 35 * mm, 35 * mm], 9))

    story.append(Paragraph("2. Auditoría de desacuerdos", styles["Section"]))
    overlap_rows = [["Estado", "Filas"]]
    labels = [
        ("Ambos correctos", "both_correct"),
        ("RF falla; CNN acierta", "rf_only_wrong"),
        ("CNN falla; RF acierta", "cnn_only_wrong"),
        ("Ambos fallan", "both_wrong"),
    ]
    overlap_rows += [[title, str(pair[key])] for title, key in labels]
    story.append(make_table(overlap_rows, [90 * mm, 40 * mm], 9))
    story.append(Paragraph(
        f"La unión de fallos de RF/CNN suma {rf_cnn_errors:,} filas ({(rf_cnn_errors / audit['n_rows']):.1%} de validation). "
        f"Jaccard de errores: {audit['rf_cnn_error_jaccard']:.4f}. Esto describe el desacuerdo observado; "
        f"no significa que esas etiquetas estén equivocadas. El CSV de errores de los cuatro modelos "
        f"contiene {all_model_error_rows:,} filas al incluir también errores de regresión logística y SVM.", styles["BodyText"]
    ))
    story.append(PageBreak())

    story.append(Paragraph("3. Riesgos y limitaciones", styles["Section"]))
    facts = [
        "<b>Asociación clase-fuente:</b> los 705 casos `regular` de validation (251 grupos) proceden 100% de AgriFreshNET. La evaluación de esa clase no demuestra generalización a otras fuentes.",
        "<b>Versión de scikit-learn:</b> los modelos clásicos se guardaron con 1.7.2 y el audit se ejecutó con 1.9.1, que emitió InconsistentVersionWarning. Repetir el audit con 1.7.2 antes de confiar en predicciones por imagen.",
        "<b>Mango:</b> dos ejemplos aceptados están juntos en train; no hay evaluación independiente y quedan cuatro máscaras pendientes.",
        "<b>Revisión manual:</b> la cola de 871 errores aún no tiene decisiones anotadas. La copia `validation_error_cases_reviewed.csv` no incluye campos de revisión.",
        "<b>Alcance de generalización:</b> se revisó visualmente una muestra estratificada pequeña frente a decenas de miles de candidatos; los filtros automáticos no reemplazan inspección visual.",
        "<b>Compute CNN:</b> las métricas se obtuvieron al cargar el checkpoint; TensorFlow corrió en CPU porque no encontró las bibliotecas GPU.",
    ]
    for fact in facts:
        story.append(Paragraph("• " + fact, styles["BodySmall"]))

    story.append(Paragraph("4. Próximos pasos", styles["Section"]))
    steps = [
        "Alinear el entorno a scikit-learn 1.7.2 y repetir únicamente la auditoría de validation.",
        "Revisar los 871 casos con imagen original, recorte/máscara, etiqueta, fuente, producto y predicciones; registrar decisión, razón y responsable.",
        "Resolver o excluir/documentar las máscaras pendientes de mango y aceptar una limitación explícita para `regular` sin cobertura multifuente.",
        "Si se cambian datos o pipeline, actualizar solo los experimentos afectados y volver a comparar en validation. No reentrenar CNN si no cambió nada que lo requiera.",
        "Congelar versiones, split y checkpoints con hashes; luego ejecutar el comando protegido sobre test una sola vez. No usar test para elegir modelo.",
        "Completar interfaz Streamlit, informe IEEE, anexos/referencias, diagrama CRISP-DM y video de sustentación.",
    ]
    for i, step in enumerate(steps, 1):
        story.append(Paragraph(f"{i}. {step}", styles["BodySmall"]))

    story.append(Paragraph("5. Avance global aproximado: 55%", styles["Section"]))
    story.append(Paragraph(
        "Estimación de gestión, no métrica objetiva: preparación del split, cuatro modelos y primera evaluación están listos. "
        "El avance restante pesa en la revisión de 871 casos, alineación de versiones, sesgo/fuentes y mango, congelamiento, "
        "test final y entrega académica/despliegue.", styles["BodyText"]
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Fuentes locales: experiments/results/{logistic_baseline,svm,random_forest,cnn_rgb,validation_audit}.json. "
        "Este informe no contiene resultados de test.", styles["BodySmall"]
    ))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=22 * mm,
        title="Resumen de modelado y auditoría",
        author="Proyecto clasificador de calidad de frutas y verduras",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"PDF generado: {OUT}")


if __name__ == "__main__":
    main()
