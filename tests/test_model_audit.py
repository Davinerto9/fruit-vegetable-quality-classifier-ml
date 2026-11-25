"""
Pruebas de las protecciones de src/evaluation/model_audit.py (no requieren datos ni TensorFlow).

Fase CRISP-DM: Evaluación
Uso: pytest tests/test_model_audit.py
"""
import numpy as np
import pytest

from src import config
from src.evaluation import model_audit


def test_ensemble_reorders_alphabetical_rf_classes():
    p_rf = np.array([[0.7, 0.2, 0.1]])      # columnas de scikit-learn: bad, good, regular
    p_cnn = np.array([[0.1, 0.1, 0.8]])     # columnas de config: good, regular, bad
    combined = model_audit.ensemble_probabilities(p_rf, ["bad", "good", "regular"], p_cnn, 0.5)
    expected = 0.5 * np.array([[0.2, 0.1, 0.7]]) + 0.5 * p_cnn
    np.testing.assert_allclose(combined, expected)
    assert config.QUALITY_CLASSES[combined.argmax()] == "bad"


def test_require_frozen_accepts_listed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    artifact = tmp_path / "experiments" / "checkpoints" / "rf.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"x")
    freeze = tmp_path / "freeze.sha256"
    freeze.write_text("abc123  experiments/checkpoints/rf.joblib\n", encoding="utf-8")
    model_audit._require_frozen(freeze, [artifact])


def test_require_frozen_rejects_artifacts_outside_the_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    freeze = tmp_path / "freeze.sha256"
    freeze.write_text("abc123  experiments/checkpoints/rf.joblib\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cnn.keras"):
        model_audit._require_frozen(freeze, [tmp_path / "experiments/checkpoints/cnn.keras"])


def test_missing_checkpoint_fails_before_the_test_lock(tmp_path, monkeypatch):
    """Un checkpoint faltante no debe consumir la única evaluación de test."""
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(model_audit, "_verify_freeze", lambda path: None)
    monkeypatch.setattr(model_audit, "_require_frozen", lambda path, paths: None)
    missing = {name: tmp_path / f"{name}.bin"
               for name in ("logistic_regression", "svm_linear", "random_forest", "cnn_rgb")}
    with pytest.raises(FileNotFoundError):
        model_audit.run("test", tmp_path / "split.csv", 64, True, missing, 0.5, "t")
    assert not (tmp_path / "test_evaluation.lock").exists()
