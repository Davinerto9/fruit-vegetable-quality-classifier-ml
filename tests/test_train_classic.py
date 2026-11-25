"""
Pruebas del entrenamiento simple: GridSearchCV por grupos (calidad y fruta) y el holdout de
parada temprana de la CNN.

Fase CRISP-DM: Modelado
Uso: pytest tests/test_train_classic.py
"""
import csv
import json

import joblib
import numpy as np
import pytest

from src import config
from src.training import train_classic
from src.training.train_cnn import train_holdout

PRODUCTS = ("orange", "banana", "tomato")


def _data(tmp_path, n_groups=60, per_group=4):
    rng = np.random.default_rng(0)
    rows, X = [], []
    for g in range(n_groups):
        quality = config.QUALITY_CLASSES[g % 3]
        product = PRODUCTS[(g // 3) % 3]
        source = "agrifreshnet" if quality == "regular" or g % 2 else "ryandpark"
        split = "val" if g % 5 == 0 else "train"
        for k in range(per_group):
            rows.append({"filename": f"f{g}_{k}.png", "group": f"g{g}", "quality": quality,
                         "product": product, "source": source, "split": split})
            X.append(np.r_[rng.normal(config.QUALITY_CLASSES.index(quality), 0.7, 4),
                           rng.normal(PRODUCTS.index(product), 0.7, 4)])
    split = tmp_path / "splits.csv"
    with split.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    features = tmp_path / "features.npz"
    np.savez(features, X=np.asarray(X), filenames=np.asarray([r["filename"] for r in rows]))
    return split, features, rows


@pytest.fixture()
def small_grid(monkeypatch):
    """Rejillas mínimas para que la prueba sea rápida; el código es el mismo."""
    original = train_classic.candidates

    def small():
        models = original()
        for name, (target, estimator, grid) in models.items():
            if "random_forest" in str(type(estimator)).lower() or name.endswith("rf"):
                estimator.set_params(n_estimators=20)
            models[name] = (target, estimator, {k: v[:2] for k, v in grid.items()})
        return models

    monkeypatch.setattr(train_classic, "candidates", small)


def test_train_classic_quality_and_product(tmp_path, monkeypatch, small_grid):
    monkeypatch.setattr(config, "CHECKPOINTS", tmp_path / "checkpoints")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    split, features, _ = _data(tmp_path)
    report = train_classic.run(split, features, "t")
    assert set(report["models"]) == {"logistic_regression", "svm_linear", "random_forest", "product_rf"}
    rf = report["models"]["random_forest"]
    assert rf["target"] == "quality" and rf["validation"]["f1_macro"] > 0.6
    assert "bad_to_good" in rf["validation"] and "f1_agrifreshnet_3cl" in rf["validation"]
    product = report["models"]["product_rf"]
    assert product["target"] == "product" and product["validation"]["accuracy"] > 0.6
    saved = joblib.load(tmp_path / "checkpoints" / "product_rf_t.joblib")
    assert set(saved["class_order"]) == set(PRODUCTS)
    assert json.loads((tmp_path / "results" / "classic_t.json").read_text())["test_used"] is False
    with pytest.raises(FileExistsError):
        train_classic.run(split, features, "t", only=["random_forest"])


def test_train_classic_refuses_shared_groups(tmp_path):
    split, features, rows = _data(tmp_path)
    rows[1]["split"] = "val"  # misma fila de grupo g0 en val y en train
    rows[0]["split"] = "train"
    with split.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="grupos compartidos"):
        train_classic.load(split, features)


def test_cnn_holdout_takes_whole_groups_from_train():
    rows = [{"group": f"g{g}", "quality": config.QUALITY_CLASSES[g % 3]}
            for g in range(100) for _ in range(3)]
    fit, stop = train_holdout(rows, 0.1)
    assert len(fit) + len(stop) == len(rows)
    assert not {r["group"] for r in fit} & {r["group"] for r in stop}
    assert 0.05 <= len(stop) / len(rows) <= 0.15
    assert {r["quality"] for r in stop} == set(config.QUALITY_CLASSES)
