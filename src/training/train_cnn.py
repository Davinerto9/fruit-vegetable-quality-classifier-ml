"""Entrena una CNN RGB pequeña usando el split de grupos vigente.

Requiere TensorFlow (declarado en requirements.txt). El conjunto test no se abre.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src import config


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "La CNN necesita TensorFlow. Instala el entorno Python 3.11 del proyecto "
            "con `conda env create -f environment.yml` y vuelve a ejecutar este módulo."
        ) from error
    return tf


def _load_rows(split_path: Path):
    with split_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or any(row.get("decision") != "candidate" for row in rows):
        raise ValueError("splits.csv debe contener filas candidate.")
    names = [row["filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("splits.csv contiene archivos duplicados.")
    return rows


def _metrics(y_true, y_pred):
    class_ids = list(range(len(config.QUALITY_CLASSES)))
    return {
        "n": int(len(y_true)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=class_ids,
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                               labels=class_ids).tolist(),
        "confusion_matrix_labels": config.QUALITY_CLASSES,
        "classification_report": classification_report(
            y_true, y_pred, labels=class_ids,
            target_names=config.QUALITY_CLASSES, output_dict=True, zero_division=0,
        ),
    }


def _dataset(tf, partition, class_to_id, batch_size, shuffle=False, cache=False):
    """
    Lote de recortes RGB con el fondo anulado por la máscara.

    Con cache=True se guardan en memoria los PNG ya decodificados (uint8, sin modificar) y el
    redimensionado y la máscara se aplican después, así que los tensores son idénticos a los
    de cache=False. El orden también es igual: shuffle recibe la misma secuencia de elementos
    con la misma semilla. Sirve para no releer los PNG en cada época.
    """
    crop_paths = [str(config.CROPS_DIR / row["crop"]) for row in partition]
    mask_paths = [str(config.CROPS_DIR / row["crop_mask"]) for row in partition]
    labels = np.asarray([class_to_id[row["quality"]] for row in partition], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((crop_paths, mask_paths, labels))

    def decode(crop_path, mask_path, label):
        crop = tf.io.decode_image(tf.io.read_file(crop_path), channels=3,
                                  expand_animations=False)
        mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1,
                                  expand_animations=False)
        return crop, mask, label

    def prepare(crop, mask, label):
        crop = tf.image.resize(crop, config.IMG_SIZE, method="bilinear")
        mask = tf.image.resize(mask, config.IMG_SIZE, method="nearest")
        crop = crop * tf.cast(mask > 127, tf.float32) / 255.0
        return crop, label

    if cache:
        ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).cache()
        if shuffle:
            ds = ds.shuffle(min(len(partition), 8192), seed=config.SEED,
                            reshuffle_each_iteration=True)
        ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    else:
        if shuffle:
            ds = ds.shuffle(min(len(partition), 8192), seed=config.SEED,
                            reshuffle_each_iteration=True)
        ds = ds.map(lambda c, m, y: prepare(*decode(c, m, y)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _validate_partitions(rows):
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Se requieren train y val.")
    train_groups = {row["group"] for row in train_rows}
    val_groups = {row["group"] for row in val_rows}
    if train_groups & val_groups:
        raise ValueError("Hay grupos compartidos entre train y validation.")
    return train_rows, val_rows


def _stem(run_name: str) -> str:
    """Nombre base de los artefactos: cnn_rgb (baseline) o cnn_rgb_<run_name>."""
    return f"cnn_rgb_{run_name}" if run_name else "cnn_rgb"


def train(split_path: Path, epochs: int, batch_size: int, run_name: str = "", cache: bool = False,
          patience: int = 6, reduce_lr_patience: int = 0, reduce_lr_factor: float = 0.5,
          min_lr: float = 1e-5):
    """Entrena la CNN. reduce_lr_patience=0 desactiva ReduceLROnPlateau (comportamiento original)."""
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(config.SEED)
    rows = _load_rows(split_path)
    train_rows, val_rows = _validate_partitions(rows)

    class_to_id = {name: index for index, name in enumerate(config.QUALITY_CLASSES)}
    class_ids = np.asarray([class_to_id[row["quality"]] for row in train_rows])
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_to_id)),
                                   y=class_ids)
    class_weight = {index: float(weight) for index, weight in enumerate(weights)}

    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    x = inputs
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(len(config.QUALITY_CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="fruit_quality_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    stem = _stem(run_name)
    checkpoint = config.CHECKPOINTS / f"{stem}.keras"
    result_path = config.RESULTS / f"{stem}.json"
    history_path = config.LOGS / f"{stem}_history.json"
    epochs_csv = config.LOGS / f"{stem}_epochs.csv"
    if run_name and any(path.exists() for path in (checkpoint, result_path, epochs_csv)):
        raise FileExistsError(f"Ya existen artefactos de {stem}; usa otro --run-name.")
    for path in (checkpoint, result_path, history_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    class EpochTimer(tf.keras.callbacks.Callback):
        """Registra la duración de cada época (costo de cómputo, PI2)."""

        def __init__(self):
            super().__init__()
            self.seconds = []

        def on_epoch_begin(self, epoch, logs=None):
            self._start = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.seconds.append(time.perf_counter() - self._start)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                        restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(epochs_csv)),
        EpochTimer(),
    ]
    if reduce_lr_patience:
        callbacks.insert(2, tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=reduce_lr_factor, patience=reduce_lr_patience,
            min_lr=min_lr, verbose=1))
    started = time.perf_counter()
    val_ds = _dataset(tf, val_rows, class_to_id, batch_size, cache=cache)
    history = model.fit(_dataset(tf, train_rows, class_to_id, batch_size, shuffle=True, cache=cache),
                        validation_data=val_ds,
                        epochs=epochs, class_weight=class_weight, callbacks=callbacks,
                        verbose=1)
    duration = time.perf_counter() - started

    true_labels = np.asarray([class_to_id[row["quality"]] for row in val_rows])
    probabilities = model.predict(val_ds, verbose=0)
    prediction = probabilities.argmax(axis=1)
    result = {
        "model": stem,
        "run_name": run_name or None,
        "epoch_seconds": [round(s, 2) for s in callbacks[-1].seconds],
        "input": "128x128 RGB crops with background removed using the stored mask",
        "architecture": "3 Conv2D/BatchNorm/ReLU/MaxPool blocks, global average pooling, dropout, softmax",
        "split_file": str(split_path),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "seed": config.SEED, "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size, "class_weight": class_weight, "dataset_cache": cache,
        "optimizer": {"name": "adam", "initial_learning_rate": 1e-3},
        "early_stopping": {"monitor": "val_loss", "patience": patience, "restore_best_weights": True},
        "reduce_lr_on_plateau": ({"monitor": "val_loss", "factor": reduce_lr_factor,
                                  "patience": reduce_lr_patience, "min_lr": min_lr}
                                 if reduce_lr_patience else None),
        "devices": [device.name for device in tf.config.list_logical_devices()],
        "test_used": False, "validation": _metrics(true_labels, prediction),
        "fit_seconds": round(duration, 2), "checkpoint": str(checkpoint),
        "history": {key: [float(value) for value in values]
                    for key, values in history.history.items()},
    }
    history_path.write_text(json.dumps(result["history"], indent=2) + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"CNN validation F1 macro={result['validation']['f1_macro']:.4f}; test usado=False")


def evaluate_checkpoint(
    split_path: Path,
    checkpoint_path: Path,
    results_path: Path,
    batch_size: int,
):
    """Calcula métricas de validation desde un checkpoint sin ejecutar fit()."""
    tf = _tensorflow()
    rows = _load_rows(split_path)
    _, val_rows = _validate_partitions(rows)
    class_to_id = {name: index for index, name in enumerate(config.QUALITY_CLASSES)}
    y_true = np.asarray([class_to_id[row["quality"]] for row in val_rows], dtype=np.int32)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    probabilities = model.predict(
        _dataset(tf, val_rows, class_to_id, batch_size), verbose=1,
    )
    y_pred = probabilities.argmax(axis=1)

    history_csv = config.LOGS / f"{checkpoint_path.stem}_epochs.csv"
    history_rows = []
    if history_csv.is_file():
        with history_csv.open(encoding="utf-8-sig", newline="") as file:
            history_rows = list(csv.DictReader(file))
    best_epoch = None
    best_val_loss = None
    if history_rows and "val_loss" in history_rows[-1]:
        best = min(history_rows, key=lambda row: float(row["val_loss"]))
        best_epoch = int(float(best.get("epoch", 0))) + 1
        best_val_loss = float(best["val_loss"])

    results_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "cnn_rgb",
        "input": "128x128 RGB crops with background removed using the stored mask",
        "split_file": str(split_path),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "seed": config.SEED,
        "checkpoint": str(checkpoint_path),
        "checkpoint_evaluation_only": True,
        "checkpoint_best_epoch_from_log": best_epoch,
        "checkpoint_best_val_loss_from_log": best_val_loss,
        "training_log": str(history_csv) if history_rows else None,
        "test_used": False,
        "validation": _metrics(y_true, y_pred),
    }
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(
        f"Checkpoint evaluado sin reentrenar: {checkpoint_path}\n"
        f"Validation F1 macro={result['validation']['f1_macro']:.4f}; "
        f"balanced accuracy={result['validation']['balanced_accuracy']:.4f}; "
        "test usado=False"
    )
    print(f"Métricas guardadas: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=config.CROPS_DIR / "splits.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--evaluate-checkpoint", action="store_true",
        help="Evalúa el checkpoint guardado en validation sin volver a entrenar.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=config.CHECKPOINTS / "cnn_rgb.keras",
    )
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Salida de --evaluate-checkpoint; por defecto <checkpoint>_eval.json (no sobrescribe el JSON de entrenamiento).",
    )
    parser.add_argument("--patience", type=int, default=6, help="Paciencia de EarlyStopping.")
    parser.add_argument("--reduce-lr-patience", type=int, default=0,
                        help="Paciencia de ReduceLROnPlateau; 0 lo desactiva.")
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--cache", action="store_true",
                        help="Guarda en memoria los recortes decodificados (mismos tensores, menos I/O).")
    parser.add_argument("--run-name", default="",
                        help="Versiona checkpoint, resultados y logs como cnn_rgb_<run-name>.")
    args = parser.parse_args()
    if args.evaluate_checkpoint:
        results = args.results or config.RESULTS / f"{args.checkpoint.stem}_eval.json"
        evaluate_checkpoint(args.split, args.checkpoint, results, args.batch_size)
    else:
        train(args.split, args.epochs, args.batch_size, args.run_name, args.cache,
              args.patience, args.reduce_lr_patience, args.reduce_lr_factor, args.min_lr)
