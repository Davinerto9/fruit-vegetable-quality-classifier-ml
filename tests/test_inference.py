"""
Pruebas de src/app/inference.py con imágenes sintéticas y un modelo pequeño entrenado aquí.

Fase CRISP-DM: Despliegue
Uso: pytest tests/test_inference.py

No requieren los datasets, los checkpoints reales, Streamlit ni TensorFlow.
"""
import cv2
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.app import inference
from src.data import segmentation
from src.features.classic import (
    FEATURE_NAMES,
    extract_classic_features,
    extract_classic_features_from_arrays,
)

BACKGROUND = (230, 230, 230)
ORANGE = (0, 140, 255)  # BGR


def scene(circles, size=(400, 400)):
    img = np.full((*size, 3), BACKGROUND, np.uint8)
    for cx, cy, r in circles:
        cv2.circle(img, (cx, cy), r, ORANGE, -1)
    return img


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """Random Forest entrenado con atributos aleatorios, guardado como los checkpoints reales."""
    rng = np.random.default_rng(config.SEED)
    X = rng.random((60, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(config.QUALITY_CLASSES * 20)
    model = RandomForestClassifier(n_estimators=5, random_state=config.SEED).fit(X, y)
    path = tmp_path_factory.mktemp("checkpoints") / "random_forest.joblib"
    joblib.dump({"model": model, "class_order": config.QUALITY_CLASSES}, path)
    return inference.load_classic_model(path)


def test_features_from_arrays_match_features_from_files(tmp_path):
    """La app (memoria) y el entrenamiento (PNG) producen exactamente los mismos atributos."""
    crop, mask = segmentation.model_inputs(segmentation.process(scene([(200, 200, 80)])))
    cv2.imwrite(str(tmp_path / "crop.png"), crop)
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    from_files = extract_classic_features(tmp_path / "crop.png", tmp_path / "mask.png")
    from_arrays = extract_classic_features_from_arrays(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), mask)
    np.testing.assert_array_equal(from_files, from_arrays)


def test_model_inputs_have_model_size():
    crop, mask = segmentation.model_inputs(segmentation.process(scene([(150, 220, 60)])))
    assert crop.shape == (*config.IMG_SIZE, 3)
    assert mask.shape == config.IMG_SIZE
    assert set(np.unique(mask)) <= {0, 255}


def test_probabilities_follow_project_class_order():
    """scikit-learn ordena las clases alfabéticamente; la app debe reordenarlas."""
    ordered = inference._ordered_probabilities(["bad", "good", "regular"], [0.7, 0.2, 0.1])
    assert list(ordered) == config.QUALITY_CLASSES
    assert ordered == {"good": 0.2, "regular": 0.1, "bad": 0.7}


def test_single_fruit_is_classified(tiny_model):
    analysis = inference.analyze_image(scene([(200, 200, 80)]), [tiny_model])
    assert analysis.accepted and analysis.status == "ok"
    assert abs(analysis.d_norm - 160 / 400) < 0.03
    assert analysis.size == segmentation.size_category(analysis.d_norm)
    (prediction,) = analysis.predictions
    assert prediction.label in config.QUALITY_CLASSES
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert analysis.crop_masked_rgb.shape == (*config.IMG_SIZE, 3)


def test_two_fruits_are_rejected_with_a_message(tiny_model):
    analysis = inference.analyze_image(scene([(110, 200, 60), (290, 200, 60)]), [tiny_model])
    assert not analysis.accepted
    assert analysis.predictions == []
    assert "más de una fruta" in analysis.message


def test_unreadable_bytes_are_rejected(tiny_model):
    analysis = inference.analyze_image(inference.decode_image(b"no es una imagen"), [tiny_model])
    assert not analysis.accepted and analysis.status == "discard: unreadable"


def test_decode_image_reads_encoded_photo():
    ok, encoded = cv2.imencode(".jpg", scene([(200, 200, 80)]))
    assert ok
    assert inference.decode_image(encoded.tobytes()).shape == (400, 400, 3)


@pytest.mark.parametrize("d_norm, expected", [
    (0.10, "small"), (0.329, "small"), (0.33, "medium"), (0.54, "medium"),
    (0.55, "large"), (1.2, "large"), (None, None),
])
def test_size_category_uses_configured_thresholds(d_norm, expected):
    assert segmentation.size_category(d_norm) == expected


def _prediction(key, probabilities):
    label = max(probabilities, key=probabilities.get)
    return inference.Prediction(key, key, label, probabilities, True)


def test_ensemble_is_the_fixed_soft_vote():
    """p = 0,5·p_RF + 0,5·p_CNN, igual que src/evaluation/model_audit.py."""
    rf = _prediction("random_forest", {"good": 0.6, "regular": 0.3, "bad": 0.1})
    cnn = _prediction("cnn_rgb", {"good": 0.1, "regular": 0.2, "bad": 0.7})
    final, note = inference.final_decision([rf, cnn], mode="ensemble", cnn_weight=0.5)
    assert note is None and final.model_key == "ensemble"
    assert final.probabilities == pytest.approx({"good": 0.35, "regular": 0.25, "bad": 0.40})
    assert final.label == "bad"  # ningún modelo solo decide: el voto combinado sí


def test_ensemble_falls_back_to_random_forest_with_a_notice():
    rf = _prediction("random_forest", {"good": 0.6, "regular": 0.3, "bad": 0.1})
    final, note = inference.final_decision([rf], mode="ensemble")
    assert final is rf
    assert "ensamble" in note


def test_single_model_mode_ignores_the_other_model():
    rf = _prediction("random_forest", {"good": 0.6, "regular": 0.3, "bad": 0.1})
    cnn = _prediction("cnn_rgb", {"good": 0.1, "regular": 0.2, "bad": 0.7})
    assert inference.final_decision([rf, cnn], mode="random_forest") == (rf, None)
    assert inference.final_decision([rf, cnn], mode="cnn") == (cnn, None)


def test_analysis_exposes_a_final_decision(tiny_model):
    analysis = inference.analyze_image(scene([(200, 200, 80)]), [tiny_model])
    assert analysis.final is analysis.predictions[0]
    if config.APP_MODEL == "ensemble":
        assert analysis.final_note  # sin CNN, avisa que decide Random Forest


@pytest.fixture(scope="module")
def tiny_product_model(tmp_path_factory):
    rng = np.random.default_rng(1)
    X = rng.random((60, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(["orange", "banana", "apple"] * 20)
    model = RandomForestClassifier(n_estimators=5, random_state=config.SEED).fit(X, y)
    path = tmp_path_factory.mktemp("checkpoints") / "product_rf_t.joblib"
    joblib.dump({"model": model}, path)
    handle = inference.load_classic_model(path, key="product_rf", name="Fruta")
    return inference.ModelHandle(**{**handle.__dict__, "kind": "product"})


def test_analysis_identifies_fruit_and_keeps_quality_separate(tiny_model, tiny_product_model):
    analysis = inference.analyze_image(scene([(200, 200, 80)]), [tiny_model, tiny_product_model])
    assert analysis.product in {"orange", "banana", "apple"}
    probs = list(analysis.product_probabilities.values())
    assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1.0)
    assert analysis.product_confident == (probs[0] >= config.APP_MIN_PRODUCT_CONFIDENCE)
    assert [p.model_key for p in analysis.predictions] == ["random_forest"]  # la fruta no vota calidad


def test_every_training_product_has_a_spanish_name():
    assert set(config.PRODUCTS) <= set(inference.PRODUCT_LABELS_ES)
