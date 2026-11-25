"""
Búsqueda en rejilla agrupada de Random Forest (solo train; validation se usa una vez al final).

Fase CRISP-DM: Modelado
Uso: python -m src.training.rf_grid --features RUTA --split RUTA --tag NOMBRE

Protocolo:
1. StratifiedGroupKFold (3 folds, semilla de config) sobre train, con los grupos del split
   (en v2, ráfagas de Kaggle). Ninguna fila de val/test participa en la selección.
2. Rejilla de regularización: min_samples_leaf x max_depth x max_features, con 160 árboles.
3. Regla de selección: mayor F1 macro medio de CV; entre las configuraciones a menos de una
   desviación estándar de la mejor, se elige la de menos errores bad->good en CV y, si empatan,
   la de menor brecha train-CV (más regularizada).
4. Sensibilidad a n_estimators (160 vs 400) para la configuración elegida.
5. Reentrenamiento en todo train y una sola evaluación de confirmación en validation.
Los artefactos se guardan como random_forest_<tag>.joblib/.json sin tocar los previos.
"""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config

L = config.QUALITY_CLASSES
GRID = {
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_depth": [None, 20, 12],
    "max_features": ["sqrt", 0.1],
}


def _model(params, n_estimators=160):
    return RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=config.SEED, **params)


def _scores(y, p, sources):
    kaggle, agri = sources == "ryandpark", sources == "agrifreshnet"
    bad = y == "bad"
    return {
        "f1_macro": float(f1_score(y, p, labels=L, average="macro", zero_division=0)),
        "f1_kaggle_good_bad": float(f1_score(y[kaggle], p[kaggle], labels=["good", "bad"], average="macro", zero_division=0)),
        "f1_agrifreshnet": float(f1_score(y[agri], p[agri], labels=L, average="macro", zero_division=0)),
        "f1_regular": float(f1_score(y, p, labels=["regular"], average="macro", zero_division=0)),
        "recall_bad": float((p[bad] == "bad").mean()),
        "bad_to_good": int((p[bad] == "good").sum()),
    }


def _cv(params, X, y, groups, sources, splits, n_estimators=160):
    folds = []
    for fit_idx, score_idx in splits:
        model = _model(params, n_estimators).fit(X[fit_idx], y[fit_idx])
        fold = _scores(y[score_idx], model.predict(X[score_idx]), sources[score_idx])
        fold["train_f1_macro"] = float(f1_score(y[fit_idx], model.predict(X[fit_idx]), labels=L, average="macro"))
        fold["mean_depth"] = float(np.mean([tree.get_depth() for tree in model.estimators_]))
        fold["mean_leaves"] = float(np.mean([tree.get_n_leaves() for tree in model.estimators_]))
        folds.append(fold)
    frame = pd.DataFrame(folds)
    summary = {f"mean_{k}": float(v) for k, v in frame.mean().items()}
    summary.update({"std_f1_macro": float(frame.f1_macro.std(ddof=0)),
                    "sum_bad_to_good": int(frame.bad_to_good.sum()),
                    "gap_train_cv": float(frame.train_f1_macro.mean() - frame.f1_macro.mean())})
    return {"params": params, "n_estimators": n_estimators, "folds": folds, **summary}


def select(records):
    """Aplica la regla de selección documentada en el módulo."""
    best = max(records, key=lambda r: r["mean_f1_macro"])
    near = [r for r in records if r["mean_f1_macro"] >= best["mean_f1_macro"] - best["std_f1_macro"]]
    return min(near, key=lambda r: (r["sum_bad_to_good"], r["gap_train_cv"])), best, near


def run(features, split, tag, folds):
    for path in (config.CHECKPOINTS / f"random_forest_{tag}.joblib", config.RESULTS / f"random_forest_{tag}.json"):
        if path.exists():
            raise FileExistsError(f"Ya existe {path}; usa otro --tag.")
    with np.load(features, allow_pickle=False) as z:
        X, filenames = z["X"].astype(np.float32), z["filenames"].astype(str)
    meta = pd.read_csv(split).set_index("filename").reindex(filenames)
    if meta.split.isna().any():
        raise ValueError("El caché tiene archivos que no están en el split.")
    part = meta.split.values
    y, groups, sources = (meta[c].values.astype(str) for c in ("quality", "group", "source"))
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    if set(groups[tr]) & set(groups[va]):
        raise ValueError("Hay grupos compartidos entre train y validation.")
    Xtr, ytr, gtr, str_ = X[tr], y[tr], groups[tr], sources[tr]
    splits = list(StratifiedGroupKFold(folds, shuffle=True, random_state=config.SEED).split(Xtr, ytr, gtr))

    started = time.perf_counter()
    records = []
    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    for i, params in enumerate(combos, 1):
        record = _cv(params, Xtr, ytr, gtr, str_, splits)
        records.append(record)
        print(f"[{i}/{len(combos)}] {params}: CV F1={record['mean_f1_macro']:.4f}±{record['std_f1_macro']:.4f} "
              f"bad->good={record['sum_bad_to_good']} gap={record['gap_train_cv']:.3f} "
              f"depth={record['mean_mean_depth']:.0f}", flush=True)
    chosen, best, near = select(records)
    print(f"Mejor CV: {best['params']} ({best['mean_f1_macro']:.4f}); "
          f"{len(near)} dentro de 1 sd; elegida: {chosen['params']}", flush=True)
    trees_400 = _cv(chosen["params"], Xtr, ytr, gtr, str_, splits, n_estimators=400)
    print(f"n_estimators=400: CV F1={trees_400['mean_f1_macro']:.4f}±{trees_400['std_f1_macro']:.4f}", flush=True)
    n_estimators = 400 if trees_400["mean_f1_macro"] > chosen["mean_f1_macro"] + chosen["std_f1_macro"] else 160

    model = _model(chosen["params"], n_estimators).fit(Xtr, ytr)
    train_scores = _scores(ytr, model.predict(Xtr), str_)
    pv = model.predict(X[va])
    val_scores = _scores(y[va], pv, sources[va])
    val_scores["confusion_matrix"] = confusion_matrix(y[va], pv, labels=L).tolist()
    checkpoint = config.CHECKPOINTS / f"random_forest_{tag}.joblib"
    joblib.dump({"model": model, "class_order": L, "seed": config.SEED, "feature_cache": str(features)}, checkpoint)
    result = {
        "model": "random_forest", "tag": tag, "feature_cache": str(features), "split_file": str(split),
        "split_sha256": hashlib.sha256(Path(split).read_bytes()).hexdigest(), "seed": config.SEED,
        "test_used": False, "selection_rule": select.__doc__ + " Ver docstring del módulo.",
        "grid": GRID, "cv_folds": folds, "candidates": records, "best_cv": best["params"],
        "within_one_sd": [r["params"] for r in near], "selected": chosen["params"],
        "n_estimators_check": {"160": chosen["mean_f1_macro"], "400": trees_400["mean_f1_macro"]},
        "n_estimators": n_estimators, "training": train_scores, "validation": val_scores,
        "seconds": round(time.perf_counter() - started, 1), "checkpoint": str(checkpoint),
    }
    (config.RESULTS / f"random_forest_{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Elegida {chosen['params']} n_estimators={n_estimators}; val F1={val_scores['f1_macro']:.4f} "
          f"recall_bad={val_scores['recall_bad']:.3f} bad->good={val_scores['bad_to_good']} "
          f"Kaggle={val_scores['f1_kaggle_good_bad']:.4f} Agri={val_scores['f1_agrifreshnet']:.4f}; test usado=False")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    run(args.features, args.split, args.tag, args.folds)
