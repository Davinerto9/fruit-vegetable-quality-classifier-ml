"""
Pruebas de las funciones de imagen de src/data/prepare.py con imágenes sintéticas.

Fase CRISP-DM: Preparación de datos
Uso: pytest tests/test_prepare.py
"""
import cv2
import numpy as np

from src.data import prepare


def test_image_stats_detects_blur():
    """Una imagen desenfocada tiene menor nitidez que la original; dimensiones y brillo correctos."""
    rng = np.random.default_rng(42)
    sharp = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    stats = prepare.image_stats(sharp)
    assert (stats["width"], stats["height"]) == (800, 600)
    assert 100 < stats["brightness"] < 155
    assert prepare.image_stats(blurred)["sharpness"] < stats["sharpness"]


def test_manual_segment_scales_contour_to_the_photo():
    """Un contorno trazado sobre una imagen de referencia se escala a la foto real."""
    image = np.zeros((400, 600, 3), np.uint8)
    square = [[30, 20], [90, 20], [90, 60], [30, 60]]  # sobre una referencia de 150 x 100
    result = prepare.manual_segment(image, square, [150, 100])
    x, y, w, h = cv2.boundingRect(result["contour"])
    assert (x, y, w, h) == (120, 80, 241, 161)
    assert result["status"] == "ok" and result["crop"].shape[0] == result["crop"].shape[1]
