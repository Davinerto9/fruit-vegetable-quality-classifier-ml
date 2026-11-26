"""
Entrena todos los modelos y los guarda en la carpeta models/.

Fase CRISP-DM: Modelado
Uso:
    python -m src.main entrenar                 # los 4 clásicos y la CNN
    python -m src.main entrenar --solo clasicos
    python -m src.main entrenar --solo cnn      # necesita TensorFlow (mejor con GPU)

Entradas: data/processed/crops/splits.csv y data/processed/features.npz.
Salidas:
    models/calidad_regresion_logistica.joblib, calidad_svm_lineal.joblib,
    calidad_random_forest.joblib, fruta_random_forest.joblib, calidad_cnn.keras
    results/entrenamiento_clasicos.json, results/entrenamiento_cnn.json
    results/logs/cnn_epocas.csv

Reglas para que la nota sea honesta
    1. Solo se entrena con train. Ningún grupo de train aparece en val (se verifica).
    2. Modelos clásicos: GridSearchCV elige los hiperparámetros con validación cruzada de 3
       partes por GRUPOS (StratifiedGroupKFold): en cada vuelta se entrena con 2/3 de los
       grupos de train y se mide en el 1/3 restante. Gana la combinación con mejor F1 macro
       promedio y se reentrena con todo train.
    3. CNN: para decidir cuándo dejar de entrenar se aparta el 10 % de los grupos de train.
    4. Val se mide UNA vez, al final, solo para reportar: no se usa para elegir nada.
    5. Test no se abre aquí. Si test ya se evaluó (results/test_evaluation.lock), no se
       permite reentrenar, porque la nota de test dejaría de ser una prueba a ciegas.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.models.model_definitions import CNN_SETTINGS, build_cnn, classic_models

# GitHub rechaza archivos de más de 100 MB.
GITHUB_LIMIT_MB = 95


# --- Datos ---------------------------------------------------------------------------------


def read_split(split_path: Path = config.SPLITS_CSV) -> pd.DataFrame:
    """splits.csv como tabla, verificando que train y val no compartan grupos."""
    table = pd.read_csv(split_path, dtype=str, encoding="utf-8-sig")
    train_groups = set(table.loc[table["split"] == "train", "group"])
    val_groups = set(table.loc[table["split"] == "val", "group"])
    if train_groups & val_groups:
        raise ValueError("Hay grupos compartidos entre train y val.")
    return table


def load_features(split_path: Path = config.SPLITS_CSV,
                  features_path: Path = config.FEATURES_NPZ):
    """
    Atributos y datos de cada recorte, en el mismo orden.

    Retorna:
        (X, tabla con quality/product/source/group/split, máscara train, máscara val).
    """
    with np.load(features_path, allow_pickle=False) as cache:
        X, names = cache["X"].astype(np.float32), cache["filenames"].astype(str)
    meta = read_split(split_path).set_index("filename").reindex(names)
    if meta["split"].isna().any():
        raise ValueError("features.npz tiene recortes que no están en splits.csv; "
                         "vuelve a correr: python -m src.main atributos")
    return X, meta, (meta["split"] == "train").to_numpy(), (meta["split"] == "val").to_numpy()


def check_not_after_test() -> None:
    if (config.RESULTS / "test_evaluation.lock").exists():
        raise RuntimeError("Test ya se evaluó: reentrenar invalidaría esa nota.")


# --- Métricas ------------------------------------------------------------------------------


def quality_metrics(y_true, y_pred, sources=None) -> dict:
    """F1 macro, matriz de confusión y errores graves (una fruta mala que pasa como buena)."""
    y_true, y_pred = np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)
    labels = config.QUALITY_CLASSES
    bad = y_true == "bad"
    result = {
        "n": int(len(y_true)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "recall_bad": float((y_pred[bad] == "bad").mean()) if bad.any() else None,
        "bad_to_good": int((y_pred[bad] == "good").sum()),
    }
    if sources is not None:
        sources = np.asarray(sources)
        # Kaggle solo tiene good y bad; AgriFreshNET tiene las tres.
        for source, name, labs in (("ryandpark", "f1_kaggle", ["good", "bad"]),
                                   ("agrifreshnet", "f1_agrifreshnet", labels)):
            mask = sources == source
            if mask.any():
                result[name] = float(f1_score(y_true[mask], y_pred[mask], labels=labs,
                                              average="macro", zero_division=0))
    return result


def fruit_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# --- Modelos clásicos ----------------------------------------------------------------------


def save_classic(model, target: str, params: dict, path: Path) -> float:
    """Guarda el modelo comprimido y devuelve su tamaño en MB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "target": target, "class_order": list(model.classes_),
                 "params": params, "sklearn": sklearn.__version__}, path, compress=3)
    size = path.stat().st_size / 2**20
    if size > GITHUB_LIMIT_MB:
        print(f"  AVISO: {path.name} pesa {size:.0f} MB; GitHub no acepta más de 100 MB.")
    return size


def train_classic(split_path: Path = config.SPLITS_CSV,
                  features_path: Path = config.FEATURES_NPZ,
                  only: list[str] | None = None) -> dict:
    """GridSearchCV por grupos para cada modelo clásico; guarda modelos y resumen."""
    check_not_after_test()
    X, meta, train, val = load_features(split_path, features_path)
    groups = meta["group"].to_numpy()[train]
    sources = meta["source"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    report = {"cv": "StratifiedGroupKFold(3) por grupo, sobre train", "scoring": "f1_macro",
              "n_train": int(train.sum()), "n_val": int(val.sum()),
              "sklearn": sklearn.__version__, "test_used": False, "models": {}}

    for name, (target, estimator, grid) in classic_models().items():
        if only and name not in only:
            continue
        y = meta[target].to_numpy()
        started = time.perf_counter()
        search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, refit=True,
                              return_train_score=True)
        search.fit(X[train], y[train], groups=groups)
        best = search.best_index_
        prediction = search.predict(X[val]).astype(str)
        validation = (quality_metrics(y[val], prediction, sources[val]) if target == "quality"
                      else fruit_metrics(y[val], prediction))
        path = config.MODELS_DIR / config.MODEL_FILES[name]
        size = save_classic(search.best_estimator_, target, search.best_params_, path)
        report["models"][name] = {
            "target": target, "file": path.name, "size_mb": round(size, 1),
            "seconds": round(time.perf_counter() - started, 1),
            "best_params": search.best_params_,
            "cv_f1_mean": float(search.cv_results_["mean_test_score"][best]),
            "cv_f1_std": float(search.cv_results_["std_test_score"][best]),
            "train_f1_mean": float(search.cv_results_["mean_train_score"][best]),
            "grid": [{"params": p, "cv_f1_mean": float(m), "cv_f1_std": float(s)}
                     for p, m, s in zip(search.cv_results_["params"],
                                        search.cv_results_["mean_test_score"],
                                        search.cv_results_["std_test_score"])],
            "validation": validation,
        }
        item = report["models"][name]
        print(f"{name:>20}: CV F1 {item['cv_f1_mean']:.3f} ± {item['cv_f1_std']:.3f} "
              f"con {search.best_params_} | val F1 {validation['f1_macro']:.3f} -> {path.name}")

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS / "entrenamiento_clasicos.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                   encoding="utf-8")
    print(f"Resumen en {out}")
    return report


# --- CNN -----------------------------------------------------------------------------------


def tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("La CNN necesita TensorFlow: pip install -r requirements.txt") from error
    return tf


def image_dataset(tf, rows: list[dict], batch_size: int, shuffle: bool = False,
                  cache: bool = True, base: Path = config.CROPS_DIR):
    """
    Lotes de (recorte RGB / 255 con el fondo en negro, número de clase).

    Es el mismo preprocesamiento que usa la app (src/prediction/predictor.py).
    """
    class_to_id = {name: i for i, name in enumerate(config.QUALITY_CLASSES)}
    crops = [str(base / row["crop"]) for row in rows]
    masks = [str(base / row["crop_mask"]) for row in rows]
    labels = np.asarray([class_to_id.get(row.get("quality"), 0) for row in rows], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((crops, masks, labels))

    def decode(crop_path, mask_path, label):
        crop = tf.io.decode_image(tf.io.read_file(crop_path), channels=3, expand_animations=False)
        mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False)
        return crop, mask, label

    def prepare(crop, mask, label):
        crop = tf.image.resize(crop, config.IMG_SIZE, method="bilinear")
        mask = tf.image.resize(mask, config.IMG_SIZE, method="nearest")
        return crop * tf.cast(mask > 127, tf.float32) / 255.0, label

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if cache:
        ds = ds.cache()  # guarda en memoria los PNG ya leídos: las épocas siguientes van rápido
    if shuffle:
        ds = ds.shuffle(min(len(rows), 8192), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def early_stop_holdout(train_rows: list[dict], fraction: float, seed: int = config.SEED):
    """
    Separa ≈fraction de los GRUPOS de train (con las 3 calidades) para decidir cuándo parar.

    Retorna:
        (filas para ajustar los pesos, filas para la parada temprana), sin grupos compartidos.
    """
    n_splits = max(2, round(1 / fraction))
    y = [row["quality"] for row in train_rows]
    groups = [row["group"] for row in train_rows]
    _, held = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
                   .split(np.zeros(len(train_rows)), y, groups))
    held = set(held.tolist())
    fit = [row for i, row in enumerate(train_rows) if i not in held]
    stop = [row for i, row in enumerate(train_rows) if i in held]
    return fit, stop


def train_cnn(split_path: Path = config.SPLITS_CSV, settings: dict = CNN_SETTINGS) -> dict:
    """Entrena la CNN y guarda la mejor época en models/calidad_cnn.keras."""
    check_not_after_test()
    tf = tensorflow()
    tf.keras.utils.set_random_seed(config.SEED)
    rows = read_split(split_path).to_dict("records")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    fit_rows, stop_rows = early_stop_holdout(train_rows, settings["early_stop_fraction"])

    ids = np.asarray([config.QUALITY_CLASSES.index(row["quality"]) for row in fit_rows])
    weights = compute_class_weight("balanced", classes=np.arange(len(config.QUALITY_CLASSES)),
                                   y=ids)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    path = config.MODELS_DIR / config.MODEL_FILES["cnn"]
    path.parent.mkdir(parents=True, exist_ok=True)
    config.LOGS.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=settings["patience"],
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=settings["reduce_lr_patience"],
                                             min_lr=settings["min_lr"], verbose=1),
        tf.keras.callbacks.CSVLogger(str(config.LOGS / "cnn_epocas.csv")),
    ]
    model = build_cnn(tf, settings)
    batch = settings["batch_size"]
    started = time.perf_counter()
    # "validation_data" de Keras es aquí el 10 % apartado de train, NO la partición val.
    history = model.fit(image_dataset(tf, fit_rows, batch, shuffle=True),
                        validation_data=image_dataset(tf, stop_rows, batch),
                        epochs=settings["max_epochs"], class_weight=class_weight,
                        callbacks=callbacks, verbose=2)
    seconds = time.perf_counter() - started

    probabilities = model.predict(image_dataset(tf, val_rows, batch, cache=False), verbose=0)
    prediction = np.asarray(config.QUALITY_CLASSES)[probabilities.argmax(axis=1)]
    validation = quality_metrics([row["quality"] for row in val_rows], prediction,
                                 [row["source"] for row in val_rows])
    losses = history.history["val_loss"]
    report = {
        "file": path.name, "settings": {k: list(v) if isinstance(v, tuple) else v
                                        for k, v in settings.items()},
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "n_fit": len(fit_rows), "n_early_stop": len(stop_rows), "n_val": len(val_rows),
        "class_weight": class_weight, "epochs_run": len(losses),
        "best_epoch": int(np.argmin(losses)) + 1, "seconds": round(seconds, 1),
        "devices": [d.name for d in tf.config.list_logical_devices()],
        "test_used": False, "validation": validation,
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
    }
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS / "entrenamiento_cnn.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"CNN: mejor época {report['best_epoch']} de {report['epochs_run']} | "
          f"val F1 {validation['f1_macro']:.3f} -> {path.name}")
    return report


def run(only: str | None = None) -> None:
    if only in (None, "clasicos"):
        train_classic()
    if only in (None, "cnn"):
        train_cnn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo", choices=("clasicos", "cnn"))
    run(parser.parse_args().solo)
