"""
Escalera de fuga en AgriFreshNET: ¿cuánto cambia la CV si se agrupa por minuto, día o semana?

Fase CRISP-DM: Evaluación (diagnóstico)
Uso (desde la raíz, con scikit-learn 1.7.2, como los demás diagnósticos):
    PYTHONPATH=. python experiments/diagnostics/agri_leak_ladder.py
    PYTHONPATH=. python experiments/diagnostics/agri_leak_ladder.py --seeds 42 43 44 --folds 3

Entradas: data/processed/crops/splits_v2_burst.csv, data/processed/classic_features_v2_burst.npz
          y experiments/results/random_forest_v2_burst_grid.json (configuración elegida de RF).
Salida:   experiments/diagnostics/out/agri_leak_ladder.txt y .json

Qué mide, por nivel (photo, minute, day, week; ver src/data/agri_sessions.py):
1. Cruce de particiones en v2: cuántas sesiones de AgriFreshNET tienen filas en train y en
   val/test, y cuántas filas de val de AgriFreshNET comparten sesión con train.
2. CV agrupada (StratifiedGroupKFold) de Random Forest con la configuración de la rejilla,
   repetida con varias semillas, sobre train v2: F1 macro global, de AgriFreshNET (3 clases) y
   de Kaggle (2 clases).

Lectura: si el F1 de AgriFreshNET baja al pasar de "photo" a niveles más gruesos, la CV por
foto estaba inflada por fotos de la misma sesión o de la misma pieza. Es la misma prueba que
reveló la fuga de Kaggle (F0, cv_burst.py).

El nivel del split v3 NO se elige con esta tabla (eso sería seleccionar por desempeño): la
regla fijada en docs/plan_simple.md toma el nivel más grueso cuyo split cumpla cobertura y
proporciones. Esta escalera cuantifica la fuga para el informe.
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
from src.data.agri_sessions import LEVELS, merge_groups

OUT = config.ROOT / "experiments" / "diagnostics" / "out"
LABELS = config.QUALITY_CLASSES


def crossing_stats(rows: list[dict], level: str) -> dict:
    """Sesiones de AgriFreshNET repartidas entre particiones en el split v2."""
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


def rf_from_grid() -> RandomForestClassifier:
    saved = json.loads((config.RESULTS / "random_forest_v2_burst_grid.json").read_text(encoding="utf-8"))
    params = dict(saved["selected"])            # min_samples_leaf, max_depth, max_features
    params["n_estimators"] = int(saved.get("n_estimators", 160))
    params.update(class_weight="balanced_subsample", n_jobs=-1, random_state=config.SEED)
    return RandomForestClassifier(**params)


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


def main(split_path, features_path, seeds, folds) -> None:
    split = pd.read_csv(split_path, encoding="utf-8-sig", dtype=str)
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
    lines = [f"Escalera de fuga AgriFreshNET | split {split_path.name} | RF {report['rf_params']}",
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
        report["levels"][level] = {"train_groups": int(len(set(groups))), "crossing_v2": stats, "cv": summary}
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
    (OUT / "agri_leak_ladder.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "agri_leak_ladder.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=lambda p: config.ROOT / p,
                        default=config.CROPS_DIR / "splits_v2_burst.csv")
    parser.add_argument("--features", type=lambda p: config.ROOT / p,
                        default=config.DATA_PROCESSED / "classic_features_v2_burst.npz")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    main(args.split, args.features, args.seeds, args.folds)
