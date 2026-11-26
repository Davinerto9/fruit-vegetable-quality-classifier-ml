"""
Pruebas del entrenamiento: GridSearchCV por grupos (estado y fruta), modelos guardados en
models/ y el apartado de grupos para la parada temprana de la CNN.

Fase CRISP-DM: Modelado
Uso: pytest tests/test_train_models.py
"""
import csv
import json

import joblib
import numpy as np
import pytest

from src import config
from src.models import model_definitions
from src.training import train_models

PRODUCTS = ("orange", "banana", "tomato")


def write_data(tmp_path, n_groups=60, per_group=4):
    """splits.csv y features.npz sintéticos: los atributos separan estado y fruta."""
    rng = np.random.default_rng(0)
    rows, X = [], []
    for g in range(n_groups):
        quality = config.QUALITY_CLASSES[g % 3]
        product = PRODUCTS[(g // 3) % 3]
        source = "agrifreshnet" if quality == "regular" or g % 2 else "ryandpark"
        part = "val" if g % 5 == 0 else ("test" if g % 7 == 0 else "train")
        for k in range(per_group):
            rows.append({"filename": f"f{g}_{k}.png", "group": f"g{g}", "quality": quality,
                         "product": product, "source": source, "split": part})
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
def project(tmp_path, monkeypatch):
    """Rutas temporales y rejillas mínimas para que la prueba sea rápida; el código es el mismo."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    original = model_definitions.classic_models

    def small():
        models = original()
        for name, (target, estimator, grid) in models.items():
            if name.endswith(("forest", "rf")):
                estimator.set_params(n_estimators=20)
            models[name] = (target, estimator, {k: v[:2] for k, v in grid.items()})
        return models

    monkeypatch.setattr(train_models, "classic_models", small)
    return tmp_path


def test_train_classic_saves_quality_and_fruit_models(project):
    split, features, _ = write_data(project)
    report = train_models.train_classic(split, features)
    assert set(report["models"]) == {"logistic_regression", "svm_linear", "random_forest", "fruit_rf"}
    rf = report["models"]["random_forest"]
    assert rf["target"] == "quality" and rf["validation"]["f1_macro"] > 0.6
    assert "bad_to_good" in rf["validation"] and "f1_agrifreshnet" in rf["validation"]
    fruit = report["models"]["fruit_rf"]
    assert fruit["target"] == "product" and fruit["validation"]["accuracy"] > 0.6
    for key in ("logistic_regression", "svm_linear", "random_forest", "fruit_rf"):
        assert (project / "models" / config.MODEL_FILES[key]).is_file()
    saved = joblib.load(project / "models" / config.MODEL_FILES["fruit_rf"])
    assert set(saved["class_order"]) == set(PRODUCTS)
    summary = json.loads((project / "results" / "entrenamiento_clasicos.json").read_text())
    assert summary["test_used"] is False and summary["n_val"] > 0


def test_training_never_sees_val_or_test(project, monkeypatch):
    """GridSearchCV recibe solo filas de train."""
    split, features, rows = write_data(project)
    seen = []
    original_fit = train_models.GridSearchCV.fit

    def spy(self, X, y, groups=None):
        seen.append(set(groups))
        return original_fit(self, X, y, groups=groups)

    monkeypatch.setattr(train_models.GridSearchCV, "fit", spy)
    train_models.train_classic(split, features, only=["logistic_regression"])
    train_groups = {r["group"] for r in rows if r["split"] == "train"}
    assert seen and seen[0] == train_groups


def test_refuses_shared_groups(project):
    split, features, rows = write_data(project)
    rows[0]["split"], rows[1]["split"] = "train", "val"  # mismo grupo g0 en train y val
    with split.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="grupos compartidos"):
        train_models.load_features(split, features)


def test_refuses_to_retrain_after_test(project):
    split, features, _ = write_data(project)
    (project / "results").mkdir()
    (project / "results" / "test_evaluation.lock").write_text("{}")
    with pytest.raises(RuntimeError, match="Test ya se evaluó"):
        train_models.train_classic(split, features)


def test_cnn_early_stop_holdout_takes_whole_groups_from_train():
    rows = [{"group": f"g{g}", "quality": config.QUALITY_CLASSES[g % 3]}
            for g in range(100) for _ in range(3)]
    fit, stop = train_models.early_stop_holdout(rows, 0.1)
    assert len(fit) + len(stop) == len(rows)
    assert not {r["group"] for r in fit} & {r["group"] for r in stop}
    assert 0.05 <= len(stop) / len(rows) <= 0.15
    assert {r["quality"] for r in stop} == set(config.QUALITY_CLASSES)


def test_quality_metrics_count_bad_fruit_passed_as_good():
    metrics = train_models.quality_metrics(["bad", "bad", "good", "regular"],
                                           ["good", "bad", "good", "regular"])
    assert metrics["bad_to_good"] == 1 and metrics["recall_bad"] == 0.5
