"""Genera las figuras 2, 3 y 4 del informe a partir de los resultados del proyecto.

Uso, desde la raíz del repositorio:
    python informe/generar_figuras.py

Lee results/evaluacion_val.json, results/evaluacion_test.json y
results/entrenamiento_cnn.json, y escribe los PDF en informe/figuras/.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results"
OUT = HERE / "figuras"
OUT.mkdir(exist_ok=True)

# Tipografía tipo Times, como el texto de IEEEtran.
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.titlesize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "hatch.linewidth": 0.8})

MODELS = ["logistic_regression", "svm_linear", "random_forest", "cnn", "ensemble"]
NAMES = ["Logística", "SVM", "RF", "CNN", "Ensamble"]
BLUE, ORANGE = "#0072B2", "#E69F00"

val = json.loads((RESULTS / "evaluacion_val.json").read_text(encoding="utf-8"))
test = json.loads((RESULTS / "evaluacion_test.json").read_text(encoding="utf-8"))
cnn = json.loads((RESULTS / "entrenamiento_cnn.json").read_text(encoding="utf-8"))
final = val.get("final_model", "random_forest")

# Fig. 2: F1 macro en val y test, y malas predichas como buenas en test.
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.6), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.45})
x = np.arange(len(MODELS))
f_val = [val["models"][m]["f1_macro"] for m in MODELS]
f_test = [test["models"][m]["f1_macro"] for m in MODELS]
w = 0.4
bar = dict(edgecolor="black", linewidth=0.5)
a.bar(x - w / 2, f_val, w, color=BLUE, label="Val", **bar)
a.bar(x + w / 2, f_test, w, color=ORANGE, hatch="///", label="Test", **bar)
i = MODELS.index(final)
for shift, part, f in ((-w / 2, val, f_val), (w / 2, test, f_test)):
    lo, hi = part["models"][final]["f1_macro_ci95"]
    a.errorbar(i + shift, f[i], yerr=[[f[i] - lo], [hi - f[i]]], color="black", capsize=2.5, lw=0.9)
a.set_ylim(0, 1)
a.set_yticks(np.arange(0, 1.01, 0.2))
a.set_ylabel("F1 macro del estado")
a.set_xticks(x, NAMES, rotation=45, ha="right")
a.legend(ncol=2, loc="upper center", frameon=False, handlelength=1.6)
b2g = [test["models"][m]["bad_to_good"] for m in MODELS]
bars = b.bar(x, b2g, 0.7, color=ORANGE, hatch="///", **bar)
b.bar_label(bars, fontsize=7.5, padding=1)
b.set_ylabel("Malas predichas como buenas")
b.set_xticks(x, NAMES, rotation=45, ha="right")
b.set_ylim(0, max(b2g) * 1.2)
fig.savefig(OUT / "f1_y_errores.pdf", bbox_inches="tight")
plt.close(fig)

# Fig. 3: matriz de confusión del modelo final en test, normalizada por fila.
cm = np.array(test["models"][final]["confusion_matrix"])
rate = cm / cm.sum(1, keepdims=True)
labels = ["buena", "regular", "mala"]
fig, ax = plt.subplots(figsize=(2.6, 2.1))
ax.imshow(rate, cmap="Greys", vmin=0, vmax=1)
for r in range(3):
    for c in range(3):
        ax.text(c, r, f"{rate[r, c]:.0%}\n({cm[r, c]})", ha="center", va="center", fontsize=8,
                color="white" if rate[r, c] > 0.5 else "black")
# Líneas blancas entre celdas.
ax.set_xticks(np.arange(-0.5, 3), minor=True)
ax.set_yticks(np.arange(-0.5, 3), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_xticks(range(3), labels)
ax.set_yticks(range(3), labels, rotation=90, va="center")
ax.tick_params(axis="x", length=0)
ax.set_xlabel("Estado predicho")
ax.set_ylabel("Estado real")
fig.savefig(OUT / "confusion_rf_test.pdf", bbox_inches="tight")
plt.close(fig)

# Fig. 4: pérdida de la CNN por época.
h = cnn["history"]
epochs = np.arange(1, len(h["loss"]) + 1)
fig, ax = plt.subplots(figsize=(3.4, 2.0))
ax.plot(epochs, h["loss"], color=BLUE, lw=1.4, label="Train")
ax.plot(epochs, h["val_loss"], color=ORANGE, lw=1.4, ls="--", label="Holdout de parada (10 % de train)")
best = cnn["best_epoch"]
ax.axvline(best, color="grey", lw=0.6, ls=":")
ax.text(best + 0.5, ax.get_ylim()[1] * 0.92, f"mejor época: {best}", fontsize=6.5, va="top")
ax.set_xlabel("Época")
ax.set_ylabel("Entropía cruzada ponderada")
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
fig.savefig(OUT / "perdida_cnn.pdf", bbox_inches="tight")
plt.close(fig)
print("Figuras en", OUT)
