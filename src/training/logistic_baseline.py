"""Entrena la línea base logística y mide solo train y validation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, *, include_report: bool = True) -> dict:
    result = {
        "n": int(len(y_true)),
        "f1_macro": float(f1_score(
            y_true, y_pred, labels=config.QUALITY_CLASSES, average="macro", zero_division=0,
        )),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=config.QUALITY_CLASSES,
        ).tolist(),
        "confusion_matrix_labels": config.QUALITY_CLASSES,
    }
    if include_report:
        result["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=config.QUALITY_CLASSES,
            target_names=config.QUALITY_CLASSES,
            output_dict=True,
            zero_division=0,
        )
    return result


def _subgroup_metrics(y_true, y_pred, values) -> dict:
    output = {}
    for subgroup in sorted(set(values)):
        indices = np.asarray([value == subgroup for value in values], dtype=bool)
        if indices.any():
            subgroup_true = y_true[indices]
            subgroup_pred = y_pred[indices]
            supported_labels = [
                label for label in config.QUALITY_CLASSES if np.any(subgroup_true == label)
            ]
            output[str(subgroup)] = {
                "n": int(indices.sum()),
                "scored_classes": supported_labels,
                "class_support": {
                    label: int(np.count_nonzero(subgroup_true == label))
                    for label in config.QUALITY_CLASSES
                },
                "f1_macro_supported_classes": float(f1_score(
                    subgroup_true,
                    subgroup_pred,
                    labels=supported_labels,
                    average="macro",
                    zero_division=0,
                )),
                "confusion_matrix": confusion_matrix(
                    subgroup_true, subgroup_pred, labels=config.QUALITY_CLASSES,
                ).tolist(),
                "confusion_matrix_labels": config.QUALITY_CLASSES,
            }
    return output


def train(
    feature_path: Path,
    split_path: Path,
    checkpoint_path: Path,
    results_path: Path,
) -> tuple[Path, Path]:
    with np.load(feature_path, allow_pickle=False) as saved:
        X = saved["X"].astype(np.float32, copy=False)
        feature_names = saved["feature_names"].astype(str)
        filenames = saved["filenames"].astype(str)
        y = saved["quality"].astype(str)
        groups = saved["groups"].astype(str)
        sources = saved["sources"].astype(str)
        products = saved["products"].astype(str)

    with split_path.open(encoding="utf-8-sig", newline="") as file:
        split_rows = list(csv.DictReader(file))
    split_by_filename = {row["filename"]: row["split"] for row in split_rows}
    if len(split_by_filename) != len(split_rows):
        raise ValueError("El split contiene nombres de archivo duplicados.")
    if set(filenames) != set(split_by_filename):
        raise ValueError("El caché de atributos no corresponde al split vigente; vuelve a generarlo.")

    partitions = np.asarray([split_by_filename[name] for name in filenames])
    train_indices = np.flatnonzero(partitions == "train")
    val_indices = np.flatnonzero(partitions == "val")
    if not len(train_indices) or not len(val_indices):
        raise ValueError("Se requieren registros en train y val para medir la línea base.")
    if set(y[train_indices]) != set(config.QUALITY_CLASSES):
        raise ValueError("Train debe contener las tres clases de calidad.")
    if set(np.unique(groups[train_indices])) & set(np.unique(groups[val_indices])):
        raise ValueError("Hay grupos compartidos entre train y validation.")

    pipeline = Pipeline((
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=config.SEED,
            solver="lbfgs",
        )),
    ))
    pipeline.fit(X[train_indices], y[train_indices])

    train_pred = pipeline.predict(X[train_indices])
    val_pred = pipeline.predict(X[val_indices])
    train_metrics = _metrics(y[train_indices], train_pred)
    val_metrics = _metrics(y[val_indices], val_pred)
    val_source_metrics = _subgroup_metrics(y[val_indices], val_pred, sources[val_indices])
    val_product_metrics = _subgroup_metrics(y[val_indices], val_pred, products[val_indices])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": feature_names.tolist(),
        "class_order": config.QUALITY_CLASSES,
        "seed": config.SEED,
    }, checkpoint_path)

    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    result = {
        "model": "logistic_regression_baseline",
        "input": "masked RGB/HSV histograms, LBP texture, and normalized shape descriptors",
        "feature_count": int(X.shape[1]),
        "feature_cache": str(feature_path),
        "split_file": str(split_path),
        "split_sha256": digest,
        "seed": config.SEED,
        "scaling": "StandardScaler fit on train only",
        "class_weight": "balanced, computed by scikit-learn from train labels",
        "test_used": False,
        "train": train_metrics,
        "validation": val_metrics,
        "validation_by_source": val_source_metrics,
        "validation_by_product": val_product_metrics,
    }
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Modelo guardado: {checkpoint_path}")
    print(f"Resultados train: F1 macro={train_metrics['f1_macro']:.4f}")
    print(
        f"Resultados validation: F1 macro={val_metrics['f1_macro']:.4f}; "
        f"balanced accuracy={val_metrics['balanced_accuracy']:.4f}"
    )
    print(f"Test usado: {result['test_used']}")
    print(f"Métricas guardadas: {results_path}")
    return checkpoint_path, results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=config.DATA_PROCESSED / "classic_features.npz",
    )
    parser.add_argument("--split", type=Path, default=config.CROPS_DIR / "splits.csv")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=config.CHECKPOINTS / "logistic_baseline.joblib",
    )
    parser.add_argument(
        "--results", type=Path,
        default=config.RESULTS / "logistic_baseline.json",
    )
    args = parser.parse_args()
    train(args.features, args.split, args.checkpoint, args.results)
