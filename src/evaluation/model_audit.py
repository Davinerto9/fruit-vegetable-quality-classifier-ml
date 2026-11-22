"""Audita checkpoints en validation; permite una evaluación final protegida en test."""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src import config
from src.features.classic import FEATURE_NAMES, extract_classic_features
from src.training.train_cnn import _dataset


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _verify_freeze(freeze_path: Path) -> None:
    if not freeze_path.is_file():
        raise FileNotFoundError(
            f"No existe el manifiesto de congelación: {freeze_path}. "
            "Completa la auditoría de validation y congela los artefactos primero."
        )
    for line_number, line in enumerate(freeze_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected_hash, relative_path = line.split(maxsplit=1)
            relative_path = relative_path.lstrip("*").strip()
        except ValueError as error:
            raise ValueError(f"Línea inválida en {freeze_path}:{line_number}") from error
        path = config.ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Artefacto congelado faltante: {path}")
        actual_hash = _digest(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Cambió un artefacto congelado: {relative_path}")


def _acquire_test_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "single final test evaluation; test rows may now be accessed",
    }
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"Ya existe {lock_path}; el acceso final a test ya fue iniciado. "
            "No borres el bloqueo para repetir la evaluación."
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _read_partition(split_path: Path, partition: str):
    selected = []
    train_groups = set()
    with split_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("decision") != "candidate":
                continue
            if partition == "val" and row["split"] == "train":
                train_groups.add(row["group"])
            if row["split"] == partition:
                selected.append(row)
    if not selected:
        raise ValueError(f"No hay filas para la partición {partition!r} en {split_path}.")
    filenames = [row["filename"] for row in selected]
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Hay archivos duplicados en {partition}.")
    if partition == "val":
        val_groups = {row["group"] for row in selected}
        if train_groups & val_groups:
            raise ValueError("Hay grupos compartidos entre train y validation.")
    return selected


def _metrics(y_true, y_pred):
    labels = config.QUALITY_CLASSES
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for index, label in enumerate(labels):
        support = int(np.count_nonzero(y_true == label))
        predicted_count = int(np.count_nonzero(y_pred == label))
        true_positive = int(matrix[index, index])
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall
            else (0.0 if recall is not None else None)
        )
        per_class[label] = {
            "support_rows": support,
            "predicted_rows": predicted_count,
            "precision": float(precision),
            "recall": None if recall is None else float(recall),
            "f1": None if f1 is None else float(f1),
        }
    supported = [label for label in labels if per_class[label]["support_rows"] > 0]
    macro_f1_supported = float(np.mean([per_class[label]["f1"] for label in supported]))
    return {
        "n_rows": int(len(y_true)),
        "f1_macro_all_classes": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0,
        )),
        "f1_macro_supported_classes": macro_f1_supported,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
        "per_class": per_class,
    }


def _subgroup_metrics(rows, y_true, predictions, field):
    output = {}
    keys = sorted({row[field] for row in rows})
    for key in keys:
        indices = np.asarray([row[field] == key for row in rows], dtype=bool)
        subgroup_rows = [row for row in rows if row[field] == key]
        subgroup_metrics = _metrics(y_true[indices], predictions[indices])
        subgroup_metrics["n_groups"] = len({row["group"] for row in subgroup_rows})
        subgroup_metrics["interpretation"] = (
            "descriptive_only_low_support"
            if subgroup_metrics["n_groups"] < 30
            else "review_metrics_and_group_dependence"
        )
        output[str(key)] = subgroup_metrics
    return output


def _class_source_support(rows):
    classes = config.QUALITY_CLASSES
    sources = sorted({row["source"] for row in rows})
    output = {}
    for label in classes:
        total = sum(row["quality"] == label for row in rows)
        output[label] = {}
        for source in sources:
            subset = [row for row in rows
                      if row["quality"] == label and row["source"] == source]
            output[label][source] = {
                "rows": len(subset),
                "groups": len({row["group"] for row in subset}),
                "share_of_class": (len(subset) / total) if total else None,
            }
    return output


def run(partition: str, split_path: Path, batch_size: int, confirm_final_test: bool):
    is_test = partition == "test"
    freeze_path = config.RESULTS / "pretest_freeze.sha256"
    lock_path = config.RESULTS / "test_evaluation.lock"
    if is_test:
        if not confirm_final_test:
            raise ValueError("Test requiere --confirm-final-test después del freeze.")
        _verify_freeze(freeze_path)
        _acquire_test_lock(lock_path)

    rows = _read_partition(split_path, partition)
    missing_files = []
    for row in rows:
        for key in ("crop", "crop_mask"):
            path = config.CROPS_DIR / row[key]
            if not path.is_file():
                missing_files.append(str(path))
    if missing_files:
        raise FileNotFoundError("Faltan recortes o máscaras; ejemplo: " + missing_files[0])

    y_true = np.asarray([row["quality"] for row in rows], dtype=str)
    X = np.stack([
        extract_classic_features(config.CROPS_DIR / row["crop"],
                                 config.CROPS_DIR / row["crop_mask"])
        for row in rows
    ]).astype(np.float32, copy=False)
    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Esperaba {len(FEATURE_NAMES)} atributos y obtuve {X.shape[1]}.")

    checkpoints = config.CHECKPOINTS
    logistic = joblib.load(checkpoints / "logistic_baseline.joblib")["pipeline"]
    svm = joblib.load(checkpoints / "svm.joblib")["model"]
    random_forest = joblib.load(checkpoints / "random_forest.joblib")["model"]
    predictions = {
        "logistic_regression": logistic.predict(X).astype(str),
        "svm_linear": svm.predict(X).astype(str),
        "random_forest": random_forest.predict(X).astype(str),
    }

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("La CNN requiere TensorFlow en el entorno activo.") from error
    class_to_id = {label: index for index, label in enumerate(config.QUALITY_CLASSES)}
    cnn = tf.keras.models.load_model(checkpoints / "cnn_rgb.keras", compile=False)
    probabilities = cnn.predict(
        _dataset(tf, rows, class_to_id, batch_size), verbose=1,
    )
    predictions["cnn_rgb"] = np.asarray(config.QUALITY_CLASSES)[probabilities.argmax(axis=1)]

    metrics = {}
    for name, prediction in predictions.items():
        metrics[name] = {
            "overall": _metrics(y_true, prediction),
            "by_source": _subgroup_metrics(rows, y_true, prediction, "source"),
            "by_product": _subgroup_metrics(rows, y_true, prediction, "product"),
        }

    comparison = []
    error_cases = []
    for index, row in enumerate(rows):
        rf_correct = predictions["random_forest"][index] == y_true[index]
        cnn_correct = predictions["cnn_rgb"][index] == y_true[index]
        if rf_correct and cnn_correct:
            pair_status = "both_correct"
        elif rf_correct:
            pair_status = "cnn_only_wrong"
        elif cnn_correct:
            pair_status = "rf_only_wrong"
        else:
            pair_status = "both_wrong"
        record = {
            "filename": row["filename"], "product": row["product"],
            "source": row["source"], "group": row["group"],
            "quality": row["quality"], "crop": row["crop"],
            "crop_mask": row["crop_mask"], "rf_cnn_status": pair_status,
            "rf_cnn_predictions_agree": bool(
                predictions["random_forest"][index] == predictions["cnn_rgb"][index]
            ),
        }
        for name, prediction in predictions.items():
            record[f"{name}_prediction"] = str(prediction[index])
            record[f"{name}_correct"] = bool(prediction[index] == y_true[index])
        comparison.append(record)
        if any(not record[f"{name}_correct"] for name in predictions) or not record[
            "rf_cnn_predictions_agree"
        ]:
            error_cases.append(record)

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    stem = "validation" if partition == "val" else "final_test"
    prediction_path = config.RESULTS / f"{stem}_predictions.csv"
    errors_path = config.RESULTS / f"{stem}_error_cases.csv"
    result_path = config.RESULTS / f"{stem}_audit.json"
    fields = list(comparison[0])
    for path, records in ((prediction_path, comparison), (errors_path, error_cases)):
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    pair_status_counts = {
        status: sum(row["rf_cnn_status"] == status for row in comparison)
        for status in ("both_correct", "rf_only_wrong", "cnn_only_wrong", "both_wrong")
    }
    error_union = (
        pair_status_counts["rf_only_wrong"]
        + pair_status_counts["cnn_only_wrong"]
        + pair_status_counts["both_wrong"]
    )
    result = {
        "partition": partition,
        "test_used": is_test,
        "n_rows": len(rows),
        "n_groups": len({row["group"] for row in rows}),
        "class_source_support": _class_source_support(rows),
        "models": metrics,
        "rf_cnn_pair_status_counts": pair_status_counts,
        "rf_cnn_error_jaccard": (
            pair_status_counts["both_wrong"] / error_union if error_union else None
        ),
        "predictions_csv": str(prediction_path),
        "error_cases_csv": str(errors_path),
        "test_lock": str(lock_path) if is_test else None,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Partición={partition}; filas={len(rows)}; grupos={result['n_groups']}; "
          f"test usado={is_test}")
    for name, item in metrics.items():
        overall = item["overall"]
        print(f"{name}: F1 macro={overall['f1_macro_all_classes']:.4f}; "
              f"balanced accuracy={overall['balanced_accuracy']:.4f}")
    print(f"RF/CNN error overlap: {result['rf_cnn_pair_status_counts']}; "
          f"Jaccard={result['rf_cnn_error_jaccard']}")
    print(f"Resumen: {result_path}\nPredicciones: {prediction_path}\nErrores: {errors_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("val", "test"), default="val")
    parser.add_argument("--split", type=Path, default=config.CROPS_DIR / "splits.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    run(args.partition, args.split, args.batch_size, args.confirm_final_test)
