"""
Entrena los modelos clásicos con GridSearchCV y validación cruzada por grupos.

Fase CRISP-DM: Modelado
Uso:
    python -m src.training.train_classic --split data/processed/crops/splits_v3_week.csv \
        --features data/processed/classic_features_v3_week.npz --tag v3_week

Qué entrena, todo con los mismos 359 atributos de color, textura y forma:
    Calidad (good / regular / bad): regresión logística, SVM lineal y Random Forest.
    Fruta (12 productos):            Random Forest.

Cómo elige los hiperparámetros: GridSearchCV de scikit-learn con StratifiedGroupKFold(3)
sobre train, métrica F1 macro. Los grupos (fotos de la misma ráfaga o sesión) nunca quedan a
la vez en entrenamiento y evaluación de un fold. Con refit=True, el mejor modelo se reentrena
con todo train.

Validation se usa una sola vez, al final, para reportar; no interviene en ninguna elección.
Test no se toca.

Salidas:
    experiments/checkpoints/<modelo>_<tag>.joblib   {"model", "class_order", "params", ...}
    experiments/results/classic_<tag>.json          CV, parámetros y métricas de validation
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src import config


def candidates() -> dict:
    """Modelos y rejillas. Los nombres de parámetro siguen la convención de scikit-learn."""
    rf_grid = {"min_samples_leaf": [1, 5, 20], "max_depth": [None, 20],
               "max_features": ["sqrt", 0.1]}
    return {
        "logistic_regression": ("quality", make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
            {"logisticregression__C": [0.1, 1, 10]}),
        "svm_linear": ("quality", make_pipeline(
            StandardScaler(), LinearSVC(class_weight="balanced", max_iter=5000)),
            {"linearsvc__C": [0.1, 1, 10]}),
        "random_forest": ("quality", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
            random_state=config.SEED), rf_grid),
        "product_rf": ("product", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
            random_state=config.SEED),
            {"min_samples_leaf": [1, 5], "max_features": ["sqrt", 0.1]}),
    }


def load(split_path: Path, features_path: Path):
    with np.load(features_path, allow_pickle=False) as cache:
        X, names = cache["X"].astype(np.float32), cache["filenames"].astype(str)
    meta = pd.read_csv(split_path, dtype=str).set_index("filename").reindex(names)
    if meta["split"].isna().any():
        raise ValueError("El caché de atributos tiene archivos que no están en el split.")
    train = (meta["split"] == "train").to_numpy()
    val = (meta["split"] == "val").to_numpy()
    if set(meta["group"][train]) & set(meta["group"][val]):
        raise ValueError("Hay grupos compartidos entre train y validation.")
    return X, meta, train, val


def metrics(target: str, y_true, y_pred, sources) -> dict:
    labels = config.QUALITY_CLASSES if target == "quality" else sorted(set(y_true) | set(y_pred))
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                           zero_division=0),
    }
    if target == "quality":
        bad = y_true == "bad"
        result["bad_to_good"] = int((y_pred[bad] == "good").sum())
        result["recall_bad"] = float((y_pred[bad] == "bad").mean()) if bad.any() else None
        for source, name, labs in (("ryandpark", "f1_kaggle_2cl", ["good", "bad"]),
                                   ("agrifreshnet", "f1_agrifreshnet_3cl", config.QUALITY_CLASSES)):
            mask = sources == source
            if mask.any():
                result[name] = float(f1_score(y_true[mask], y_pred[mask], labels=labs,
                                              average="macro", zero_division=0))
    return result


def run(split_path: Path, features_path: Path, tag: str, only: list[str] | None = None) -> dict:
    X, meta, train, val = load(split_path, features_path)
    groups = meta["group"].to_numpy()[train]
    sources = meta["source"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    report = {"split": str(split_path), "features": str(features_path), "tag": tag,
              "sklearn": sklearn.__version__, "cv": "StratifiedGroupKFold(3) por group",
              "scoring": "f1_macro", "test_used": False, "models": {}}

    for name, (target, estimator, grid) in candidates().items():
        if only and name not in only:
            continue
        checkpoint = config.CHECKPOINTS / f"{name}_{tag}.joblib"
        if checkpoint.exists():
            raise FileExistsError(f"Ya existe {checkpoint}; usa otro --tag.")
        y = meta[target].to_numpy()
        started = time.perf_counter()
        search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, refit=True,
                              return_train_score=True)
        search.fit(X[train], y[train], groups=groups)
        best = search.best_index_
        cv_result = {
            "best_params": search.best_params_,
            "cv_f1_macro_mean": float(search.cv_results_["mean_test_score"][best]),
            "cv_f1_macro_std": float(search.cv_results_["std_test_score"][best]),
            "train_f1_macro_mean": float(search.cv_results_["mean_train_score"][best]),
            "all": [{"params": p, "mean": float(m), "std": float(s)} for p, m, s in zip(
                search.cv_results_["params"], search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"])],
        }
        validation = metrics(target, y[val], search.predict(X[val]).astype(str), sources[val])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": search.best_estimator_, "target": target,
                     "class_order": list(search.best_estimator_.classes_),
                     "params": search.best_params_, "split": str(split_path), "tag": tag}, checkpoint)
        report["models"][name] = {"target": target, "checkpoint": str(checkpoint),
                                  "seconds": round(time.perf_counter() - started, 1),
                                  "cv": cv_result, "validation": validation}
        print(f"{name:>20} ({target}): CV F1 {cv_result['cv_f1_macro_mean']:.3f}"
              f"±{cv_result['cv_f1_macro_std']:.3f} con {search.best_params_} | "
              f"validation F1 {validation['f1_macro']:.3f}, accuracy {validation['accuracy']:.3f}")

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS / f"classic_{tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Resumen en {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--only", nargs="+", choices=list(candidates()),
                        help="Entrenar solo estos modelos.")
    args = parser.parse_args()
    run(args.split, args.features, args.tag, args.only)
