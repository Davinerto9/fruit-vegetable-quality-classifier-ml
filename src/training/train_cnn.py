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


def _dataset(tf, partition, class_to_id, batch_size, shuffle=False):
    crop_paths = [str(config.CROPS_DIR / row["crop"]) for row in partition]
    mask_paths = [str(config.CROPS_DIR / row["crop_mask"]) for row in partition]
    labels = np.asarray([class_to_id[row["quality"]] for row in partition], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((crop_paths, mask_paths, labels))

    def load(crop_path, mask_path, label):
        crop = tf.io.decode_image(tf.io.read_file(crop_path), channels=3,
                                  expand_animations=False)
        mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1,
                                  expand_animations=False)
        crop = tf.image.resize(crop, config.IMG_SIZE, method="bilinear")
        mask = tf.image.resize(mask, config.IMG_SIZE, method="nearest")
        crop = crop * tf.cast(mask > 127, tf.float32) / 255.0
        return crop, label

    if shuffle:
        ds = ds.shuffle(min(len(partition), 8192), seed=config.SEED,
                        reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


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


def train(split_path: Path, epochs: int, batch_size: int):
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

    checkpoint = config.CHECKPOINTS / "cnn_rgb.keras"
    result_path = config.RESULTS / "cnn_rgb.json"
    history_path = config.LOGS / "cnn_rgb_history.json"
    for path in (checkpoint, result_path, history_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                        restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(config.LOGS / "cnn_rgb_epochs.csv")),
    ]
    started = time.perf_counter()
    history = model.fit(_dataset(tf, train_rows, class_to_id, batch_size, shuffle=True),
                        validation_data=_dataset(tf, val_rows, class_to_id, batch_size),
                        epochs=epochs, class_weight=class_weight, callbacks=callbacks,
                        verbose=1)
    duration = time.perf_counter() - started

    true_labels = np.asarray([class_to_id[row["quality"]] for row in val_rows])
    probabilities = model.predict(_dataset(tf, val_rows, class_to_id, batch_size), verbose=0)
    prediction = probabilities.argmax(axis=1)
    result = {
        "model": "cnn_rgb",
        "input": "128x128 RGB crops with background removed using the stored mask",
        "architecture": "3 Conv2D/BatchNorm/ReLU/MaxPool blocks, global average pooling, dropout, softmax",
        "split_file": str(split_path),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "seed": config.SEED, "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size, "class_weight": class_weight,
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

    history_csv = config.LOGS / "cnn_rgb_epochs.csv"
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
    parser.add_argument("--results", type=Path, default=config.RESULTS / "cnn_rgb.json")
    args = parser.parse_args()
    if args.evaluate_checkpoint:
        evaluate_checkpoint(args.split, args.checkpoint, args.results, args.batch_size)
    else:
        train(args.split, args.epochs, args.batch_size)
