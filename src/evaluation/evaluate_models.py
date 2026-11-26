"""
Mide los modelos guardados en val o, una sola vez al final, en test.

Fase CRISP-DM: Evaluación
Uso:
    python -m src.main evaluar                     # val (se puede repetir)
    python -m src.main congelar                    # anota la huella de modelos y reparto
    python -m src.main evaluar --test --confirmo   # test, UNA sola vez

Qué mide
    Estado: regresión logística, SVM lineal, Random Forest, CNN y el ensamble
            p = 0,5 * p_RF + 0,5 * p_CNN (peso fijado antes de ver resultados).
            F1 macro (promedio del F1 de good, regular y bad, para que la clase con menos
            fotos pese igual), matriz de confusión, recall de "bad" y cuántas frutas malas
            pasaron como buenas (el error más caro para el negocio).
    Fruta:  accuracy y F1 macro del Random Forest de fruta.
    Intervalo de confianza 95 % del F1 del ensamble remuestreando GRUPOS completos
    (bootstrap): fotos del mismo grupo no son independientes.

Cómo se protege test
    1. "congelar" guarda en results/pretest_freeze.sha256 la huella (SHA-256) de splits.csv y
       de los 5 modelos. Si después cambia un solo byte, test se niega a correr.
    2. Test exige --confirmo y crea results/test_evaluation.lock. Si el candado existe, no se
       repite (ni se puede reentrenar: ver train_models.py).
    3. Los modelos se cargan ANTES de crear el candado: un archivo faltante no gasta la
       única oportunidad.

Salidas: results/evaluacion_<val|test>.json y results/predicciones_<val|test>.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.training.train_models import (fruit_metrics, image_dataset, load_features,
                                       quality_metrics, tensorflow)

FREEZE = "pretest_freeze.sha256"
LOCK = "test_evaluation.lock"
QUALITY_MODELS = ("logistic_regression", "svm_linear", "random_forest", "cnn")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def frozen_files() -> list[Path]:
    """Archivos que definen el resultado de test: el reparto y los 5 modelos."""
    return [config.SPLITS_CSV, *(config.MODELS_DIR / name for name in config.MODEL_FILES.values())]


def freeze() -> Path:
    """Escribe la huella de cada archivo de frozen_files() en results/pretest_freeze.sha256."""
    missing = [str(path) for path in frozen_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan archivos para congelar: " + ", ".join(missing))
    if (config.RESULTS / LOCK).exists():
        raise RuntimeError("Test ya se evaluó: no se vuelve a congelar.")
    lines = [f"{digest(path)}  {path.relative_to(config.ROOT).as_posix()}"
             for path in frozen_files()]
    out = config.RESULTS / FREEZE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Huellas guardadas en {out}. Súbelo a Git antes de evaluar test.")
    return out


def verify_freeze(freeze_path: Path, paths: list[Path]) -> None:
    """Cada archivo debe estar en el freeze y conservar su huella."""
    if not freeze_path.is_file():
        raise FileNotFoundError(f"No existe {freeze_path}: corre primero `python -m src.main congelar`.")
    listed = {}
    for line in freeze_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, relative = line.split(maxsplit=1)
            listed[relative.lstrip("*").strip().replace("\\", "/")] = expected
    for path in paths:
        try:
            relative = Path(path).resolve().relative_to(config.ROOT.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        if relative not in listed:
            raise RuntimeError(f"{relative} no está en el freeze: no se evalúa en test.")
        if not Path(path).is_file() or digest(Path(path)) != listed[relative]:
            raise RuntimeError(f"Cambió {relative} después de congelar.")


def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Ya existe {lock_path}: test ya se evaluó y no se repite.") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat()}, file)


def ensemble_probabilities(p_rf: np.ndarray, rf_classes, p_cnn: np.ndarray,
                           cnn_weight: float) -> np.ndarray:
    """
    Voto suave (1 - w) * p_RF + w * p_CNN, con columnas en el orden de config.QUALITY_CLASSES.

    scikit-learn ordena classes_ alfabéticamente (bad, good, regular); se reordena antes de
    sumar, igual que en src/prediction/predictor.py.
    """
    order = [list(rf_classes).index(label) for label in config.QUALITY_CLASSES]
    return (1 - cnn_weight) * p_rf[:, order] + cnn_weight * p_cnn


def group_bootstrap(y, pred, groups, score, n: int = 2000, seed: int = config.SEED) -> list[float]:
    """IC 95 % de score(y, pred) sacando grupos completos con reemplazo."""
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n):
        take = np.concatenate([members[g] for g in rng.choice(unique, len(unique))])
        values.append(score(y[take], pred[take]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def load_all(include_cnn: bool = True) -> dict:
    """Carga los modelos de models/. Falla antes de tocar test si falta alguno."""
    paths = {key: config.MODELS_DIR / name for key, name in config.MODEL_FILES.items()}
    missing = [str(p) for key, p in paths.items() if not p.is_file() and (include_cnn or key != "cnn")]
    if missing:
        raise FileNotFoundError("Faltan modelos (corre `python -m src.main entrenar`): "
                                + ", ".join(missing))
    models = {key: joblib.load(path)["model"] for key, path in paths.items() if key != "cnn"}
    if include_cnn:
        models["cnn"] = tensorflow().keras.models.load_model(paths["cnn"], compile=False)
    return models


def run(partition: str = "val", confirm: bool = False) -> dict:
    is_test = partition == "test"
    if is_test:
        if not confirm:
            raise ValueError("Test se evalúa una sola vez: agrega --confirmo.")
        verify_freeze(config.RESULTS / FREEZE, frozen_files())
    models = load_all()
    if is_test:
        acquire_lock(config.RESULTS / LOCK)  # todo lo que podía fallar ya cargó

    X, meta, _, _ = load_features()
    selected = (meta["split"] == partition).to_numpy()
    X, meta = X[selected], meta[selected].reset_index()
    rows = meta.to_dict("records")
    y = meta["quality"].to_numpy()

    labels = np.asarray(config.QUALITY_CLASSES)
    rf = models["random_forest"]
    p_rf = rf.predict_proba(X)
    p_cnn = models["cnn"].predict(image_dataset(tensorflow(), rows, 64, cache=False), verbose=0)
    p_ensemble = ensemble_probabilities(p_rf, rf.classes_, p_cnn, config.APP_ENSEMBLE_CNN_WEIGHT)
    predictions = {
        "logistic_regression": models["logistic_regression"].predict(X).astype(str),
        "svm_linear": models["svm_linear"].predict(X).astype(str),
        "random_forest": np.asarray(rf.classes_)[p_rf.argmax(axis=1)].astype(str),
        "cnn": labels[p_cnn.argmax(axis=1)],
        "ensemble": labels[p_ensemble.argmax(axis=1)],
    }
    fruit_prediction = models["fruit_rf"].predict(X).astype(str)

    report = {
        "partition": partition, "test_used": is_test, "n": len(rows),
        "n_groups": int(meta["group"].nunique()),
        "ensemble_cnn_weight": config.APP_ENSEMBLE_CNN_WEIGHT,
        "models": {name: quality_metrics(y, pred, meta["source"]) for name, pred in predictions.items()},
        "fruit": fruit_metrics(meta["product"], fruit_prediction),
    }
    f1 = lambda a, b: quality_metrics(a, b)["f1_macro"]  # noqa: E731
    report["models"]["ensemble"]["f1_macro_ci95"] = group_bootstrap(
        y, predictions["ensemble"], meta["group"], f1, n=1000)

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    name = "test" if is_test else "val"
    with (config.RESULTS / f"predicciones_{name}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "source", "group", "product", "fruit_prediction", "quality",
                         *(f"{m}_prediction" for m in predictions)])
        for i, row in enumerate(rows):
            writer.writerow([row["filename"], row["source"], row["group"], row["product"],
                             fruit_prediction[i], row["quality"],
                             *(predictions[m][i] for m in predictions)])
    out = config.RESULTS / f"evaluacion_{name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{name}: {len(rows)} recortes en {report['n_groups']} grupos")
    for model, item in report["models"].items():
        print(f"  {model:>20}: F1 macro {item['f1_macro']:.3f} | malas como buenas "
              f"{item['bad_to_good']}")
    lo, hi = report["models"]["ensemble"]["f1_macro_ci95"]
    print(f"  ensamble, IC 95 % por grupos: [{lo:.3f}; {hi:.3f}]")
    print(f"  fruta: accuracy {report['fruit']['accuracy']:.3f}")
    print(f"Resumen en {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--confirmo", action="store_true")
    parser.add_argument("--congelar", action="store_true")
    args = parser.parse_args()
    if args.congelar:
        freeze()
    else:
        run("test" if args.test else "val", args.confirmo)
