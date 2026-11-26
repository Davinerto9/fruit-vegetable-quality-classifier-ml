"""
Prueba de fuga: ¿cuánto se infla la nota si NO se agrupan las fotos de la misma sesión?

Fase CRISP-DM: Evaluación (diagnóstico, opcional)
Uso: python -m src.main fuga        (o python -m src.evaluation.leak_check)

Idea: se hace la misma validación cruzada de un Random Forest fijo sobre train, cambiando
solo cómo se agrupan las fotos de AgriFreshNET:
    photo   cada foto con sus copias aumentadas (lo mínimo)
    minute  fotos del mismo minuto
    day     fotos del mismo día y producto
    week    fotos de la misma semana y producto (lo que usa el proyecto)
Si el F1 baja al agrupar más grueso, significa que con grupos finos el modelo veía en
evaluación fotos casi iguales a las de entrenamiento: la nota estaba inflada (fuga).

También cuenta, para cada nivel, cuántas fotos de val comparten sesión con train en el
reparto actual. Con el reparto del proyecto, a nivel "week" debe dar 0.

No elige nada: la semana se fijó antes (ver src/data/split.py). Sirve para el informe.
Salida: results/prueba_fuga.txt y results/prueba_fuga.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.data.split import LEVELS, merge_groups

OUT = config.RESULTS
LABELS = config.QUALITY_CLASSES


def crossing_stats(rows: list[dict], level: str) -> dict:
    """Sesiones de AgriFreshNET repartidas entre particiones en el reparto actual."""
    mapping = merge_groups(rows, level)
    agri = [row for row in rows if row["source"] == "agrifreshnet"]
    partitions = defaultdict(set)
    for row in agri:
        partitions[mapping[row["group"]]].add(row["split"])
    train_sessions = {g for g, p in partitions.items() if "train" in p}
    val_rows = [row for row in agri if row["split"] == "val"]
    test_rows = [row for row in agri if row["split"] == "test"]
    return {
        "agri_sessions": len(partitions),
        "sessions_in_more_than_one_partition": sum(len(p) > 1 for p in partitions.values()),
        "val_agri_rows": len(val_rows),
        "val_agri_rows_sharing_session_with_train": sum(
            mapping[row["group"]] in train_sessions for row in val_rows),
        "test_agri_rows_sharing_session_with_train": sum(
            mapping[row["group"]] in train_sessions for row in test_rows),
    }


# Random Forest fijo, para que la única diferencia entre filas de la tabla sea el agrupamiento.
RF_PARAMS = {"n_estimators": 160, "min_samples_leaf": 5, "max_depth": 20, "max_features": 0.1}


def rf_from_grid() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS, class_weight="balanced_subsample", n_jobs=-1,
                                  random_state=config.SEED)


def cv_scores(X, y, source, groups, seed: int, folds: int, model) -> dict:
    scores = defaultdict(list)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y, groups):
        fitted = model.set_params(random_state=seed).fit(X[train_idx], y[train_idx])
        pred = fitted.predict(X[test_idx])
        yt, st = y[test_idx], source[test_idx]
        scores["overall"].append(f1_score(yt, pred, labels=LABELS, average="macro"))
        agri, kaggle = st == "agrifreshnet", st == "ryandpark"
        scores["agrifreshnet_3cl"].append(f1_score(yt[agri], pred[agri], labels=LABELS, average="macro"))
        scores["kaggle_2cl"].append(f1_score(yt[kaggle], pred[kaggle], labels=["good", "bad"],
                                             average="macro"))
    return scores


def main(split_path=config.SPLITS_CSV, features_path=config.FEATURES_NPZ, seeds=(42, 43, 44),
         folds=3) -> None:
    seeds = list(seeds)
    split = pd.read_csv(split_path, encoding="utf-8-sig", dtype=str)
    # Se parte de los grupos SIN unir por sesión, para poder comparar los niveles.
    if "group_original" in split:
        split["group"] = split["group_original"]
    rows = split.to_dict("records")
    cache = np.load(features_path, allow_pickle=True)
    by_name = split.set_index("filename")
    names = cache["filenames"]
    keep = by_name.reindex(names)["split"].to_numpy() == "train"
    X = cache["X"][keep]
    train_rows = by_name.reindex(names[keep]).reset_index().to_dict("records")
    y = np.asarray([row["quality"] for row in train_rows])
    source = np.asarray([row["source"] for row in train_rows])

    model = rf_from_grid()
    report = {"split": str(split_path), "seeds": seeds, "folds": folds,
              "rf_params": {k: v for k, v in model.get_params().items()
                            if k in ("n_estimators", "min_samples_leaf", "max_depth", "max_features")},
              "levels": {}}
    lines = [f"Prueba de fuga AgriFreshNET | reparto {split_path.name} | RF {report['rf_params']}",
             f"CV: StratifiedGroupKFold({folds}) x semillas {seeds}; media ± desv. sobre folds y semillas", ""]
    for level in LEVELS:
        mapping = merge_groups(train_rows, level)
        groups = np.asarray([mapping[row["group"]] for row in train_rows])
        pooled = defaultdict(list)
        for seed in seeds:
            for key, values in cv_scores(X, y, source, groups, seed, folds, model).items():
                pooled[key].extend(values)
        stats = crossing_stats(rows, level)
        summary = {key: {"mean": float(np.mean(v)), "std": float(np.std(v))} for key, v in pooled.items()}
        report["levels"][level] = {"train_groups": int(len(set(groups))), "crossing": stats, "cv": summary}
        lines.append(
            f"{level:>6} | grupos train {len(set(groups)):5d} | "
            f"global {summary['overall']['mean']:.4f}±{summary['overall']['std']:.4f} | "
            f"agri {summary['agrifreshnet_3cl']['mean']:.4f}±{summary['agrifreshnet_3cl']['std']:.4f} | "
            f"kaggle {summary['kaggle_2cl']['mean']:.4f}±{summary['kaggle_2cl']['std']:.4f} | "
            f"val agri que comparte sesión con train: "
            f"{stats['val_agri_rows_sharing_session_with_train']}/{stats['val_agri_rows']}"
        )
    base = report["levels"]["photo"]["cv"]["agrifreshnet_3cl"]["mean"]
    lines.append("")
    for level in LEVELS[1:]:
        delta = report["levels"][level]["cv"]["agrifreshnet_3cl"]["mean"] - base
        lines.append(f"Δ agri ({level} − photo) = {delta:+.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prueba_fuga.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "prueba_fuga.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    main(seeds=args.seeds, folds=args.folds)
