"""Ajusta SVM lineal y Random Forest respetando los grupos del split vigente."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src import config


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=config.QUALITY_CLASSES,
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                               labels=config.QUALITY_CLASSES).tolist(),
        "confusion_matrix_labels": config.QUALITY_CLASSES,
        "classification_report": classification_report(
            y_true, y_pred, labels=config.QUALITY_CLASSES,
            target_names=config.QUALITY_CLASSES, output_dict=True, zero_division=0,
        ),
    }


def _load(features_path: Path, split_path: Path):
    with np.load(features_path, allow_pickle=False) as saved:
        X = saved["X"].astype(np.float32, copy=False)
        filenames = saved["filenames"].astype(str)
        y = saved["quality"].astype(str)
        groups = saved["groups"].astype(str)
    with split_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    split_by_name = {row["filename"]: row["split"] for row in rows}
    if len(split_by_name) != len(rows) or set(split_by_name) != set(filenames):
        raise ValueError("El caché de atributos no coincide exactamente con splits.csv.")
    partitions = np.asarray([split_by_name[name] for name in filenames])
    train = np.flatnonzero(partitions == "train")
    val = np.flatnonzero(partitions == "val")
    if set(groups[train]) & set(groups[val]):
        raise ValueError("Hay grupos compartidos entre train y validation.")
    return X, y, groups, train, val


def _evaluate_cv(model_name, estimator_factory, params_list, X, y, groups, folds):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=config.SEED)
    cv_splits = list(splitter.split(X, y, groups))
    records = []
    for params in params_list:
        scores = []
        for fold_idx, (fit_idx, score_idx) in enumerate(cv_splits, start=1):
            estimator = estimator_factory(params)
            estimator.fit(X[fit_idx], y[fit_idx])
            prediction = estimator.predict(X[score_idx])
            score = float(f1_score(y[score_idx], prediction, labels=config.QUALITY_CLASSES,
                                   average="macro", zero_division=0))
            scores.append(score)
            print(f"{model_name} {params} fold {fold_idx}/{folds}: F1 macro={score:.4f}", flush=True)
        records.append({"params": params, "fold_f1_macro": scores,
                        "mean_f1_macro": float(np.mean(scores)),
                        "std_f1_macro": float(np.std(scores))})
    best = max(records, key=lambda row: row["mean_f1_macro"])
    return best, records


def train_all(features_path: Path, split_path: Path, folds: int = 3):
    X, y, groups, train_idx, val_idx = _load(features_path, split_path)
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    if set(np.unique(y_train)) != set(config.QUALITY_CLASSES):
        raise ValueError("Train requiere las tres clases.")

    definitions = {
        "svm": (
            lambda params: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(class_weight="balanced", random_state=config.SEED,
                                          max_iter=5000, dual="auto", **params)),
            ]),
            [{"C": value} for value in (0.1, 1.0, 10.0)],
        ),
        "random_forest": (
            lambda params: RandomForestClassifier(
                n_estimators=160, class_weight="balanced_subsample", n_jobs=-1,
                random_state=config.SEED, **params,
            ),
            [
                {"max_features": "sqrt", "min_samples_leaf": leaf, "max_depth": depth}
                for depth, leaf in ((None, 1), (None, 2), (32, 1))
            ],
        ),
    }

    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    for model_name, (factory, params_list) in definitions.items():
        started = time.perf_counter()
        best, cv_records = _evaluate_cv(model_name, factory, params_list, X_train,
                                        y_train, groups_train, folds)
        model = factory(best["params"])
        model.fit(X_train, y_train)
        train_metrics = _metrics(y_train, model.predict(X_train))
        val_metrics = _metrics(y[val_idx], model.predict(X[val_idx]))
        duration = time.perf_counter() - started

        checkpoint = config.CHECKPOINTS / f"{model_name}.joblib"
        result_path = config.RESULTS / f"{model_name}.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "class_order": config.QUALITY_CLASSES,
                     "seed": config.SEED, "feature_cache": str(features_path)}, checkpoint)
        result = {
            "model": model_name,
            "input": "masked RGB/HSV histograms, LBP texture, and normalized shape descriptors",
            "feature_count": int(X.shape[1]), "feature_cache": str(features_path),
            "split_file": str(split_path), "split_sha256": split_hash,
            "seed": config.SEED, "test_used": False,
            "cross_validation": {"method": "StratifiedGroupKFold", "folds": folds,
                                 "selection_metric": "macro_f1", "candidates": cv_records,
                                 "selected": best},
            "training": train_metrics, "validation": val_metrics,
            "fit_and_search_seconds": round(duration, 2),
            "checkpoint": str(checkpoint),
        }
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"{model_name} seleccionado={best['params']}; "
              f"CV F1={best['mean_f1_macro']:.4f}±{best['std_f1_macro']:.4f}; "
              f"validation F1={val_metrics['f1_macro']:.4f}; test usado=False", flush=True)
        print(f"Resultados: {result_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=config.DATA_PROCESSED / "classic_features.npz")
    parser.add_argument("--split", type=Path, default=config.CROPS_DIR / "splits.csv")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    train_all(args.features, args.split, args.folds)
