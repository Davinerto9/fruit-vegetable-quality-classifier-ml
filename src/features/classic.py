"""Atributos visuales clásicos calculados sobre el recorte y la máscara de la fruta."""

from pathlib import Path

import numpy as np
from PIL import Image

COLOR_BINS = 16
LBP_BINS = 256
COLOR_CHANNELS = ("r", "g", "b", "h", "s", "v")
SHAPE_FEATURES = (
    "bbox_aspect_ratio",
    "bbox_fill_ratio",
    "mask_area_ratio",
    "perimeter_over_sqrt_area",
    "circularity",
    "elongation",
    "mask_edge_density",
)

FEATURE_NAMES = tuple(
    f"{channel}_hist_{index:02d}"
    for channel in COLOR_CHANNELS
    for index in range(COLOR_BINS)
) + tuple(f"lbp_{index:03d}" for index in range(LBP_BINS)) + SHAPE_FEATURES


def _normalized_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0, 256))
    total = int(histogram.sum())
    if total:
        histogram = histogram / total
    return histogram.astype(np.float32)


def _lbp_histogram(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Histograma LBP 8 vecinos, conservando solo píxeles interiores de la máscara."""
    height, width = gray.shape
    histogram = np.zeros(LBP_BINS, dtype=np.float32)
    if height < 3 or width < 3:
        return histogram

    center = gray[1:-1, 1:-1]
    valid = mask[1:-1, 1:-1].copy()
    codes = np.zeros(center.shape, dtype=np.uint8)
    bit = 0
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)):
        y0, y1 = 1 + dy, height - 1 + dy
        x0, x1 = 1 + dx, width - 1 + dx
        neighbor = gray[y0:y1, x0:x1]
        neighbor_mask = mask[y0:y1, x0:x1]
        valid &= neighbor_mask
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
        bit += 1

    if not valid.any():
        return histogram
    counts = np.bincount(codes[valid], minlength=LBP_BINS).astype(np.float32)
    histogram = counts / counts.sum()
    return histogram


def _shape_features(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("La máscara no contiene píxeles de fruta.")

    height, width = mask.shape
    box_width = int(xs.max() - xs.min() + 1)
    box_height = int(ys.max() - ys.min() + 1)
    area = float(len(xs))
    box_area = float(box_width * box_height)
    area_ratio = area / float(height * width)
    fill_ratio = area / box_area

    horizontal = np.count_nonzero(mask[:, 1:] != mask[:, :-1])
    horizontal += np.count_nonzero(mask[:, 0]) + np.count_nonzero(mask[:, -1])
    vertical = np.count_nonzero(mask[1:, :] != mask[:-1, :])
    vertical += np.count_nonzero(mask[0, :]) + np.count_nonzero(mask[-1, :])
    perimeter = float(horizontal + vertical)
    circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0

    if len(xs) > 1:
        points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
        largest = max(float(eigenvalues[-1]), 1e-8)
        elongation = float(np.sqrt(max(float(eigenvalues[0]), 0.0) / largest))
    else:
        elongation = 0.0

    edge = np.zeros_like(mask)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge_pixels = float(np.count_nonzero(edge & mask))
    perimeter_scale = perimeter / max(np.sqrt(area), 1e-8)
    return np.asarray((
        box_width / max(box_height, 1),
        fill_ratio,
        area_ratio,
        perimeter_scale,
        circularity,
        elongation,
        edge_pixels / area,
    ), dtype=np.float32)


def extract_classic_features(crop_path: Path, mask_path: Path) -> np.ndarray:
    """Extrae histogramas RGB/HSV, textura LBP y descriptores de forma."""
    with Image.open(crop_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 127

    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"Recorte y máscara tienen tamaños distintos: {crop_path}")
    if not mask.any():
        raise ValueError(f"La máscara está vacía: {mask_path}")

    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)
    pixels_rgb = rgb[mask]
    pixels_hsv = hsv[mask]
    color_features = [
        _normalized_histogram(pixels_rgb[:, channel], COLOR_BINS)
        for channel in range(3)
    ] + [
        _normalized_histogram(pixels_hsv[:, channel], COLOR_BINS)
        for channel in range(3)
    ]

    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    features = np.concatenate((*color_features, _lbp_histogram(gray, mask), _shape_features(mask)))
    if features.shape[0] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError(f"Vector de atributos inválido: {crop_path}")
    return features.astype(np.float32, copy=False)
