"""
Pruebas de la evaluación y de las protecciones de test (no requieren datos ni TensorFlow).

Fase CRISP-DM: Evaluación
Uso: pytest tests/test_evaluate_models.py
"""
import csv
import json

import numpy as np
import pytest

from src import config
from src.evaluation import evaluate_models


def test_ensemble_reorders_alphabetical_rf_classes():
    p_rf = np.array([[0.7, 0.2, 0.1]])      # columnas de scikit-learn: bad, good, regular
    p_cnn = np.array([[0.1, 0.1, 0.8]])     # columnas de config: good, regular, bad
    combined = evaluate_models.ensemble_probabilities(p_rf, ["bad", "good", "regular"], p_cnn, 0.5)
    np.testing.assert_allclose(combined, 0.5 * np.array([[0.2, 0.1, 0.7]]) + 0.5 * p_cnn)
    assert config.QUALITY_CLASSES[combined.argmax()] == "bad"


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """Un proyecto falso con splits.csv y los 5 archivos de modelo."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "SPLITS_CSV", tmp_path / "data" / "splits.csv")
    config.SPLITS_CSV.parent.mkdir(parents=True)
    config.SPLITS_CSV.write_text("filename,split\na,train\n", encoding="utf-8")
    config.MODELS_DIR.mkdir()
    for name in config.MODEL_FILES.values():
        (config.MODELS_DIR / name).write_bytes(name.encode())
    return tmp_path


def test_freeze_then_verify_accepts_unchanged_files(project):
    freeze = evaluate_models.freeze()
    evaluate_models.verify_freeze(freeze, evaluate_models.frozen_files())


def test_verify_rejects_a_model_changed_after_freezing(project):
    freeze = evaluate_models.freeze()
    (config.MODELS_DIR / config.MODEL_FILES["cnn"]).write_bytes(b"otro modelo")
    with pytest.raises(RuntimeError, match="Cambió"):
        evaluate_models.verify_freeze(freeze, evaluate_models.frozen_files())


def test_verify_rejects_files_outside_the_freeze(project):
    freeze = evaluate_models.freeze()
    extra = project / "models" / "otro.joblib"
    extra.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="no está en el freeze"):
        evaluate_models.verify_freeze(freeze, [extra])


def test_test_needs_confirmation(project):
    with pytest.raises(ValueError, match="--confirmo"):
        evaluate_models.run("test", confirm=False)


def test_missing_model_fails_before_the_test_lock(project):
    """Un modelo faltante no debe gastar la única evaluación de test."""
    evaluate_models.freeze()
    (config.MODELS_DIR / config.MODEL_FILES["random_forest"]).unlink()
    with pytest.raises(RuntimeError):
        evaluate_models.run("test", confirm=True)
    assert not (config.RESULTS / evaluate_models.LOCK).exists()


def test_lock_cannot_be_taken_twice(tmp_path):
    lock = tmp_path / "test_evaluation.lock"
    evaluate_models.acquire_lock(lock)
    with pytest.raises(RuntimeError, match="no se repite"):
        evaluate_models.acquire_lock(lock)


def test_group_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.choice(config.QUALITY_CLASSES, 300)
    pred = np.where(rng.random(300) < 0.8, y, "bad")
    groups = np.repeat(np.arange(60), 5)
    accuracy = lambda a, b: float((a == b).mean())  # noqa: E731
    lo, hi = evaluate_models.group_bootstrap(y, pred, groups, accuracy, n=300)
    assert lo <= accuracy(y, pred) <= hi


def test_leak_check_runs_end_to_end(tmp_path, monkeypatch):
    """Prueba de humo de la prueba de fuga con atributos aleatorios y un RF pequeño."""
    from datetime import date, timedelta

    from src.evaluation import leak_check

    rows = []
    for day in range(60):
        product = ("orange", "tomato", "banana")[day % 3]
        quality = config.QUALITY_CLASSES[day % 3]
        stamp = (date(2025, 1, 1) + timedelta(days=day)).strftime("%Y%m%d")
        for k in range(4):
            rows.append({"filename": f"x/IMG_{stamp}_08{k:02d}00.jpg", "product": product,
                         "group_original": f"a:{day}:{k}", "group": f"a:{day}:{k}",
                         "source": "agrifreshnet", "quality": quality,
                         "split": "val" if (day * 4 + k) % 7 == 0 else "train"})
    for i in range(90):
        rows.append({"filename": f"k/{i}.jpg", "product": "apple", "group_original": f"k:{i // 3}",
                     "group": f"k:{i // 3}", "source": "ryandpark",
                     "quality": ("good", "bad")[i % 2], "split": "train"})
    split = tmp_path / "splits.csv"
    with split.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    features = tmp_path / "features.npz"
    np.savez(features, X=np.random.default_rng(0).random((len(rows), 8)),
             filenames=np.asarray([r["filename"] for r in rows]))
    monkeypatch.setattr(leak_check, "RF_PARAMS", {"n_estimators": 8, "max_depth": 5})
    monkeypatch.setattr(leak_check, "OUT", tmp_path / "out")
    leak_check.main(split, features, seeds=[0], folds=2)
    report = json.loads((tmp_path / "out" / "prueba_fuga.json").read_text())
    assert list(report["levels"]) == ["photo", "minute", "day", "week"]
    assert report["levels"]["week"]["train_groups"] <= report["levels"]["photo"]["train_groups"]
