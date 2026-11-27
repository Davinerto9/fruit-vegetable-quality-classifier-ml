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


def _disk(bite: bool = False, blob: bool = False) -> np.ndarray:
    """Máscara de una fruta redonda. bite: le falta un mordisco abierto, como cuando la
    podredumbre quedó fuera (media luna). blob: una sombra pegada por fuera."""
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 70, 255, -1)
    if bite:
        cv2.circle(mask, (100, 60), 50, 0, -1)
    if blob:
        cv2.circle(mask, (100, 178), 14, 255, -1)
    return mask


def _rescue(monkeypatch, before: np.ndarray, after: np.ndarray, fruits=(1, 1)) -> np.ndarray:
    """Corre rescue_dark_rot con máscaras controladas: before con margen 60, after con 40."""
    monkeypatch.setattr(segmentation, "initial_mask", lambda lab, c, margin=60: after)
    monkeypatch.setattr(segmentation, "refine_grabcut", lambda small, mask: mask)
    monkeypatch.setattr(segmentation, "main_object", lambda mask: mask)
    counts = iter(fruits[::-1])  # se llama primero con la nueva y luego con la original
    monkeypatch.setattr(segmentation, "count_fruits", lambda small, initial, mask: next(counts))
    _, mask = segmentation.rescue_dark_rot(None, None, None, before, before)
    return mask


def test_rot_lost_from_a_round_fruit_is_recovered(monkeypatch):
    """Media luna (falta la parte podrida) -> con margen 40 vuelve a ser redonda: se acepta."""
    crescent, full = _disk(bite=True), _disk()
    assert segmentation.solidity(crescent) < 0.9 <= segmentation.solidity(full)
    assert np.array_equal(_rescue(monkeypatch, crescent, full), full)


def test_a_shadow_does_not_count_as_rot(monkeypatch):
    """Si lo nuevo es una sombra pegada, la forma no se llena: se queda la máscara original."""
    crescent = _disk(bite=True)
    with_shadow = cv2.bitwise_or(crescent, _disk(blob=True) - _disk())
    assert np.array_equal(_rescue(monkeypatch, crescent, with_shadow), crescent)


def test_rescue_never_adds_fruits(monkeypatch):
    crescent, full = _disk(bite=True), _disk()
    assert np.array_equal(_rescue(monkeypatch, crescent, full, fruits=(1, 2)), crescent)


def test_full_shapes_are_left_alone(monkeypatch):
    full = _disk()
    monkeypatch.setattr(segmentation, "initial_mask", lambda *a, **k: pytest.fail("no debía repetir"))
    _, mask = segmentation.rescue_dark_rot(None, None, None, full, full)
    assert mask is full
