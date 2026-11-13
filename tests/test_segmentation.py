"""
Pruebas de src/data/segmentation.py con imágenes sintéticas (no requieren los datasets).

Fase CRISP-DM: Preparación de datos
Uso: pytest tests/test_segmentation.py
"""
import cv2
import numpy as np

from src import config
from src.data import segmentation

BACKGROUND = (230, 230, 230)  # gris claro, como una tela o mesa
ORANGE = (0, 140, 255)  # BGR


def scene(circles: list, size: tuple = (400, 400)) -> np.ndarray:
    """
    Crea una imagen de fondo uniforme con círculos de color (frutas sintéticas).

    Parámetros:
        circles: lista de (cx, cy, radio).
        size: (alto, ancho) de la imagen.

    Retorna:
        Imagen BGR uint8.
    """
    img = np.full((*size, 3), BACKGROUND, np.uint8)
    for cx, cy, r in circles:
        cv2.circle(img, (cx, cy), r, ORANGE, -1)
    return img


def test_single_fruit_is_ok_and_diameter_matches():
    """Una fruta sobre fondo uniforme: status ok y diámetro normalizado cercano al real (2r / lado)."""
    result = segmentation.process(scene([(200, 200, 80)]))
    assert result["status"] == "ok"
    assert abs(result["d_norm"] - 160 / 400) < 0.03


def test_crops_are_square_and_share_framing():
    """Los recortes con y sin fondo son cuadrados y del mismo tamaño."""
    result = segmentation.process(scene([(150, 220, 60)]))
    assert result["crop"].shape == result["crop_nobg"].shape
    assert result["crop"].shape[0] == result["crop"].shape[1]


def test_crop_without_background_is_black_outside_fruit():
    """En el recorte sin fondo, las esquinas (fuera del círculo) quedan en negro."""
    crop = segmentation.process(scene([(200, 200, 80)]))["crop_nobg"]
    assert crop[0, 0].sum() == 0 and crop[-1, -1].sum() == 0


def test_grayscale_photo_is_discarded():
    """Una foto sin color se descarta."""
    gray = cv2.cvtColor(cv2.cvtColor(scene([(200, 200, 80)]), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert segmentation.process(gray)["status"] == "discard: grayscale"


def test_two_separate_fruits_are_flagged():
    """Dos frutas separadas se marcan para revisión."""
    assert segmentation.process(scene([(110, 200, 60), (290, 200, 60)]))["status"] == "review: 2 fruits"


def test_two_touching_fruits_are_flagged():
    """Dos frutas pegadas (que se tocan) también se detectan."""
    assert segmentation.process(scene([(140, 200, 65), (265, 200, 65)]))["status"] == "review: 2 fruits"


def test_square_crop_uses_configured_margin():
    """El lado del recorte es el lado mayor de la caja más el margen de config en cada lado."""
    crop = segmentation.square_crop(np.zeros((300, 300, 3), np.uint8), (100, 100, 50, 80))
    assert crop.shape[0] == int(80 * (1 + 2 * config.CROP_MARGIN))
