"""
Genera las figuras vectoriales (PDF) del informe IEEE a partir de los resultados guardados.

Fase CRISP-DM: Evaluación
Uso: python -m src.evaluation.report_figures

Lee results/evaluacion_val.json, results/evaluacion_test.json y results/logs/cnn_epocas.csv;
no vuelve a evaluar ningún modelo, así que no gasta test. Escribe en results/figures/:
- informe_f1_modelos.pdf: F1 macro del estado en val y test, y errores mala→buena en test;
- informe_confusion_test.pdf: matriz de confusión del modelo final en test, normalizada por fila;
- informe_cnn_curvas.pdf: pérdida de la CNN en train y en el holdout de parada temprana.
"""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import config

MODEL_NAMES = {
    "logistic_regression": "Logística",
    "svm_linear": "SVM",
    "random_forest": "RF",
    "cnn": "CNN",
    "ensemble": "Ensamble",
}
QUALITY_NAMES = {"good": "buena", "regular": "regular", "bad": "mala"}
COLUMN_WIDTH = 3.5  # pulgadas, ancho de una columna IEEE


def set_style() -> None:
    """Aplica el estilo de figura IEEE: serif, 8 pt y paleta apta para daltonismo."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42,
    })


def load_evaluation(partition: str) -> dict:
    """
    Lee el resumen de evaluación de una partición.

    Parámetros:
        partition: "val" o "test".

    Retorna:
        El diccionario de results/evaluacion_<partition>.json.
    """
    with (config.RESULTS / f"evaluacion_{partition}.json").open(encoding="utf-8") as file:
        return json.load(file)


def save(fig: plt.Figure, name: str) -> None:
    """Guarda la figura en config.FIGURES como PDF vectorial y la cierra."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(val: dict, test: dict) -> None:
    """
    Barras del F1 macro del estado por modelo en val y test (con el IC 95 % por grupos del
    modelo final) y, al lado, cuántas frutas malas pasa cada modelo como buenas en test.

    Parámetros:
        val, test: resúmenes de evaluación de cada partición.
    """
    models = list(MODEL_NAMES)
    colors = sns.color_palette("colorblind", 2)
    x = np.arange(len(models))
    fig, (left, right) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.2),
                                      gridspec_kw={"width_ratios": [1.6, 1]})
    for offset, (label, summary, color, hatch) in zip(
            (-0.2, 0.2), (("Val", val, colors[0], ""), ("Test", test, colors[1], "///"))):
        scores = [summary["models"][m]["f1_macro"] for m in models]
        left.bar(x + offset, scores, 0.4, label=label, color=color, hatch=hatch, edgecolor="black",
                 linewidth=0.4)
        final = summary["models"][summary["final_model"]]
        low, high = final["f1_macro_ci95"]
        i = models.index(summary["final_model"])
        left.errorbar(i + offset, final["f1_macro"],
                      yerr=[[final["f1_macro"] - low], [high - final["f1_macro"]]],
                      color="black", capsize=2, linewidth=0.8)
    left.set_xticks(x, [MODEL_NAMES[m] for m in models], rotation=45, ha="right")
    left.set_ylabel("F1 macro del estado")
    left.set_ylim(0, 1)
    left.legend(loc="upper left", frameon=False, ncol=2)

    errors = [test["models"][m]["bad_to_good"] for m in models]
    right.bar(x, errors, 0.6, color=colors[1], hatch="///", edgecolor="black", linewidth=0.4)
    for i, value in enumerate(errors):
        right.text(i, value + 3, str(value), ha="center", fontsize=6)
    right.set_xticks(x, [MODEL_NAMES[m] for m in models], rotation=45, ha="right")
    right.set_ylabel("Malas predichas como buenas")
    right.set_ylim(0, max(errors) * 1.2)
    save(fig, "informe_f1_modelos")


def plot_confusion(summary: dict) -> None:
    """
    Matriz de confusión del modelo final normalizada por fila, con el porcentaje y el conteo.

    Parámetros:
        summary: resumen de evaluación (se usa el modelo indicado en "final_model").
    """
    result = summary["models"][summary["final_model"]]
    counts = np.array(result["confusion_matrix"])
    rates = counts / counts.sum(axis=1, keepdims=True)
    names = [QUALITY_NAMES[q] for q in result["labels"]]
    annotations = [[f"{rates[i, j]:.0%}\n({counts[i, j]})" for j in range(3)] for i in range(3)]
    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    sns.heatmap(rates, annot=annotations, fmt="", cmap="Greys", vmin=0, vmax=1, cbar=False,
                xticklabels=names, yticklabels=names, ax=ax, linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 7})
    ax.set_xlabel("Estado predicho")
    ax.set_ylabel("Estado real")
    save(fig, "informe_confusion_test")


def plot_cnn_curves() -> None:
    """Pérdida por época de la CNN en train y en el holdout de parada temprana (10 % de train)."""
    with (config.LOGS / "cnn_epocas.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) + 1 for row in rows]
    train = [float(row["loss"]) for row in rows]
    holdout = [float(row["val_loss"]) for row in rows]
    best = epochs[int(np.argmin(holdout))]
    colors = sns.color_palette("colorblind", 2)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.9))
    ax.plot(epochs, train, color=colors[0], label="Train")
    ax.plot(epochs, holdout, color=colors[1], linestyle="--", label="Holdout de parada (10 % de train)")
    ax.axvline(best, color="gray", linewidth=0.6, linestyle=":")
    ax.text(best + 0.4, 0.9 * max(holdout), f"mejor época: {best}", fontsize=6.5)
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropía cruzada ponderada")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "informe_cnn_curvas")


if __name__ == "__main__":
    set_style()
    val_summary, test_summary = load_evaluation("val"), load_evaluation("test")
    plot_model_comparison(val_summary, test_summary)
    plot_confusion(test_summary)
    plot_cnn_curves()
    print(f"Figuras del informe en {config.FIGURES.relative_to(config.ROOT)}")
