"""
Inferencia para la app: segmenta una foto, aplica el mismo preprocesamiento del entrenamiento y
predice la calidad con los checkpoints guardados.

Fase CRISP-DM: Despliegue
Uso:
    from src.app.inference import analyze_image, load_models
    models = load_models()                       # Random Forest y, si hay TensorFlow, la CNN
    analysis = analyze_image(imagen_bgr, models)

El módulo no depende de Streamlit, así que se puede probar con pytest y reutilizar en otra
interfaz (por ejemplo, una simulación de línea de empaque con cámara).

Flujo de analyze_image():
1. src.data.segmentation.process(): estado de la foto, máscara, recorte y d_norm.
2. Si el estado no es "ok" (varias frutas, máscara dudosa, escala de grises), la foto se rechaza
   y se explica por qué, en lugar de forzar una clase.
3. src.data.segmentation.model_inputs(): recorte y máscara de 128x128, igual que en prepare.py.
4. Atributos clásicos (src.features.classic) para Random Forest; recorte x máscara para la CNN.
5. Probabilidades por clase en el orden de config.QUALITY_CLASSES y categoría de tamaño.

No se guarda ninguna imagen: todo ocurre en memoria.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.data import segmentation
from src.features.classic import extract_classic_features_from_arrays

# Textos que la app muestra cuando rechaza una foto.
REJECTION_MESSAGES = {
    "discard: grayscale": "La foto parece estar en blanco y negro. Toma una foto a color.",
    "review: dubious mask": (
        "No se pudo separar la fruta del fondo. Usa un fondo liso y de color distinto al de la "
        "fruta, y que la fruta no toque los bordes de la foto."
    ),
    "fruits": "Se detectó más de una fruta. Deja una sola pieza en la foto.",
    "unreadable": "No se pudo leer el archivo como imagen.",
}

# Nombres en español para la interfaz.
QUALITY_LABELS_ES = {"good": "Buena", "regular": "Regular", "bad": "Mala"}
SIZE_LABELS_ES = {"small": "Pequeño", "medium": "Mediano", "large": "Grande"}


@dataclass
class ModelHandle:
    """Un modelo cargado y lo necesario para describirlo en la interfaz."""

    key: str                      # "random_forest" o "cnn_rgb"
    name: str                     # nombre para mostrar
    kind: str                     # "classic" o "cnn"
    estimator: object             # objeto de scikit-learn o modelo de Keras
    checkpoint: Path
    sha256: str
    validation_f1: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """Predicción de un modelo para una foto."""

    model_key: str
    model_name: str
    label: str                    # clase en config.QUALITY_CLASSES
    probabilities: dict[str, float]
    confident: bool               # probabilidad máxima >= config.APP_MIN_CONFIDENCE


@dataclass
class Analysis:
    """Resultado completo para una foto."""

    status: str                   # estado de segmentation.process()
    accepted: bool                # True si la foto se pudo clasificar
    message: str | None           # motivo del rechazo, si aplica
    d_norm: float | None
    size: str | None              # "small", "medium", "large" o None
    overlay_rgb: np.ndarray | None        # foto con el contorno dibujado, para mostrar
    crop_rgb: np.ndarray | None           # recorte 128x128 con fondo
    crop_masked_rgb: np.ndarray | None    # recorte 128x128 sin fondo (lo que "ve" el modelo)
    predictions: list[Prediction] = field(default_factory=list)


# --- Carga de modelos ----------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_f1(model_key: str) -> float | None:
    """F1 macro de validation registrado para el modelo, si existe el JSON de resultados."""
    path = config.RESULTS / f"{model_key}.json"
    if not path.is_file():
        return None
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["validation"]["f1_macro"])
    except (KeyError, ValueError, TypeError):
        return None


def load_classic_model(checkpoint: Path, key: str = "random_forest",
                       name: str = "Random Forest") -> ModelHandle:
    """
    Carga un checkpoint .joblib de src.training (clave "model" o "pipeline").

    Avisa si la versión de scikit-learn instalada no es la usada al serializar, porque
    scikit-learn no garantiza compatibilidad de objetos guardados entre versiones.
    """
    import warnings

    import joblib
    import sklearn

    caught_messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        saved = joblib.load(checkpoint)
        caught_messages = [str(item.message) for item in caught
                           if "InconsistentVersionWarning" in type(item.message).__name__]
    estimator = saved.get("model", saved.get("pipeline")) if isinstance(saved, dict) else saved
    if estimator is None or not hasattr(estimator, "predict"):
        raise ValueError(f"El checkpoint no contiene un modelo de scikit-learn: {checkpoint}")

    notes = []
    if sklearn.__version__ != config.SKLEARN_CHECKPOINT_VERSION or caught_messages:
        notes.append(
            f"El modelo se guardó con scikit-learn {config.SKLEARN_CHECKPOINT_VERSION} y está "
            f"instalada la {sklearn.__version__}. Instala la versión de requirements.txt para "
            "reproducir exactamente las predicciones."
        )
    return ModelHandle(key=key, name=name, kind="classic", estimator=estimator,
                       checkpoint=checkpoint, sha256=_sha256(checkpoint),
                       validation_f1=_validation_f1(key), warnings=notes)


def load_cnn_model(checkpoint: Path) -> ModelHandle:
    """Carga el checkpoint .keras de la CNN. Requiere TensorFlow."""
    import tensorflow as tf

    model = tf.keras.models.load_model(checkpoint, compile=False)
    return ModelHandle(key="cnn_rgb", name="CNN RGB", kind="cnn", estimator=model,
                       checkpoint=checkpoint, sha256=_sha256(checkpoint),
                       validation_f1=_validation_f1("cnn_rgb"))


def load_models(include_cnn: bool = True) -> tuple[list[ModelHandle], list[str]]:
    """
    Carga los modelos disponibles en config.CHECKPOINTS.

    Retorna:
        (modelos cargados, avisos para mostrar). Un modelo que falta o no carga se informa
        en los avisos en lugar de detener la app.
    """
    models, notices = [], []
    rf_path = config.CHECKPOINTS / "random_forest.joblib"
    if rf_path.is_file():
        try:
            models.append(load_classic_model(rf_path))
        except Exception as error:  # noqa: BLE001 - se muestra al usuario
            notices.append(f"No se pudo cargar Random Forest: {error}")
    else:
        notices.append(f"No se encontró {rf_path.relative_to(config.ROOT)}. Entrena con "
                       "`python -m src.training.classical_models`.")

    cnn_path = config.CHECKPOINTS / "cnn_rgb.keras"
    if include_cnn and cnn_path.is_file():
        try:
            models.append(load_cnn_model(cnn_path))
        except ImportError:
            notices.append("TensorFlow no está instalado: la CNN no está disponible.")
        except Exception as error:  # noqa: BLE001
            notices.append(f"No se pudo cargar la CNN: {error}")
    return models, notices


# --- Predicción ------------------------------------------------------------------


def _ordered_probabilities(classes, probabilities) -> dict[str, float]:
    """Reordena las probabilidades al orden de config.QUALITY_CLASSES.

    scikit-learn ordena classes_ alfabéticamente (bad, good, regular); la CNN usa el orden de
    config.QUALITY_CLASSES. Sin este paso, la app mostraría probabilidades en la clase equivocada.
    """
    mapping = {str(label): float(value) for label, value in zip(classes, probabilities)}
    return {label: mapping.get(label, 0.0) for label in config.QUALITY_CLASSES}


def predict_classic(model: ModelHandle, crop_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    features = extract_classic_features_from_arrays(crop_rgb, mask)[None, :]
    estimator = model.estimator
    if hasattr(estimator, "predict_proba"):
        return _ordered_probabilities(estimator.classes_, estimator.predict_proba(features)[0])
    # Modelos sin probabilidades (LinearSVC): softmax de las puntuaciones, solo orientativo.
    scores = np.asarray(estimator.decision_function(features)[0], dtype=float)
    scores = np.exp(scores - scores.max())
    return _ordered_probabilities(estimator.classes_, scores / scores.sum())


def predict_cnn(model: ModelHandle, crop_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    # Igual que train_cnn._dataset: recorte RGB / 255 con el fondo anulado por la máscara.
    tensor = crop_rgb.astype(np.float32) / 255.0 * (mask > 127)[..., None]
    probabilities = np.asarray(model.estimator.predict(tensor[None], verbose=0))[0]
    return {label: float(probabilities[index])
            for index, label in enumerate(config.QUALITY_CLASSES)}


def _overlay(image_bgr: np.ndarray, contour: np.ndarray | None, max_side: int = 900) -> np.ndarray:
    """Foto reducida con el contorno de la fruta en verde, en RGB para mostrar."""
    shown = image_bgr.copy()
    if contour is not None:
        thickness = max(2, round(max(shown.shape[:2]) / 250))
        cv2.drawContours(shown, [contour], -1, (60, 200, 60), thickness)
    scale = min(1.0, max_side / max(shown.shape[:2]))
    if scale < 1.0:
        shown = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)


def rejection_message(status: str) -> str:
    if status.startswith("review:") and "fruits" in status:
        return REJECTION_MESSAGES["fruits"]
    return REJECTION_MESSAGES.get(status, f"La foto no pasó los controles ({status}).")


def decode_image(data: bytes) -> np.ndarray | None:
    """Bytes de un archivo (carga o cámara) a imagen BGR; None si no es una imagen válida."""
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR) if array.size else None


def analyze_image(image_bgr: np.ndarray | None, models: list[ModelHandle]) -> Analysis:
    """
    Analiza una foto completa con los modelos dados.

    Parámetros:
        image_bgr: foto BGR uint8 (H, W, 3), como la entrega cv2.imdecode; None si no se leyó.
        models: modelos de load_models().

    Retorna:
        Analysis con el estado, el tamaño y una predicción por modelo, o el motivo del rechazo.
    """
    if image_bgr is None:
        return Analysis("discard: unreadable", False, REJECTION_MESSAGES["unreadable"],
                        None, None, None, None, None)

    result = segmentation.process(image_bgr)
    status = result["status"]
    overlay = _overlay(image_bgr, result.get("contour"))
    if status != "ok":
        return Analysis(status, False, rejection_message(status), result.get("d_norm"),
                        None, overlay, None, None)

    crop_bgr, mask = segmentation.model_inputs(result)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    masked_rgb = crop_rgb * (mask > 127)[..., None].astype(np.uint8)
    d_norm = float(result["d_norm"])

    predictions = []
    for model in models:
        predictor = predict_cnn if model.kind == "cnn" else predict_classic
        probabilities = predictor(model, crop_rgb, mask)
        label = max(probabilities, key=probabilities.get)
        predictions.append(Prediction(
            model_key=model.key, model_name=model.name, label=label,
            probabilities=probabilities,
            confident=probabilities[label] >= config.APP_MIN_CONFIDENCE,
        ))
    return Analysis(status, True, None, d_norm, segmentation.size_category(d_norm),
                    overlay, crop_rgb, masked_rgb, predictions)
