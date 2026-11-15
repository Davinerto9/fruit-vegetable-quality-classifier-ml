"""
Genera los recortes candidatos y el manifiesto de preparación a partir del inventario.

Fase CRISP-DM: Preparación de datos (y estadísticas de imagen para la Comprensión de datos)
Uso: python -m src.data.prepare [--pilot]

Por cada foto etiquetada del inventario:
- segmenta con src.data.segmentation.process, o aplica el contorno manual de data/own/contours.json;
- guarda el recorte 128x128 (con fondo, entrada de la CNN) y su máscara (para quitar el fondo
  al extraer características del SVM/RF) en config.CROPS_DIR;
- registra en manifest.csv el estado de segmentación, la decisión (candidate, review o excluded),
  el diámetro normalizado, el número de frutas detectadas y estadísticas de calidad de imagen
  (dimensiones, brillo, contraste y nitidez) que usa el EDA para buscar atajos entre clases.
"""

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

import cv2
import numpy as np

from src import config
from src.data import segmentation

# Estadísticas de imagen que se agregan al manifiesto (ver image_stats).
STAT_COLUMNS = ("width", "height", "brightness", "contrast", "sharpness")


def read_image(path: Path) -> np.ndarray | None:
    """
    Lee una imagen, también en rutas de Windows con caracteres Unicode (cv2.imread falla ahí).

    Parámetros:
        path: ruta de la imagen.

    Retorna:
        Imagen BGR (H, W, 3), o None si el archivo está vacío o no es una imagen válida.
    """
    encoded = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def write_png(path: Path, image: np.ndarray) -> None:
    """
    Guarda una imagen PNG, también en rutas de Windows con caracteres Unicode.

    Parámetros:
        path: ruta de destino; se crean las carpetas que falten.
        image: imagen BGR o máscara en escala de grises.
    """
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"No se pudo codificar el recorte: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(path)


def manual_segment(image: np.ndarray, points: list, reference_size: list) -> dict:
    """
    Aplica un contorno trazado a mano sobre la imagen original (fotos propias mal segmentadas).

    Parámetros:
        image: foto original BGR (H, W, 3).
        points: puntos [x, y] del contorno, medidos sobre una imagen de tamaño reference_size.
        reference_size: [ancho, alto] de la imagen donde se trazó el contorno.

    Retorna:
        dict con las mismas claves que segmentation.process: status, mask, contour, crop y d_norm.
    """
    height, width = image.shape[:2]
    reference_width, reference_height = reference_size
    if width * reference_height != height * reference_width:
        raise ValueError("El contorno manual no corresponde a la orientación de la foto.")
    scale = np.array([width / reference_width, height / reference_height])
    contour = np.rint(np.asarray(points) * scale).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return {
        "status": "ok",
        "mask": mask,
        "contour": contour,
        "crop": segmentation.square_crop(image, cv2.boundingRect(contour)),
        "d_norm": segmentation.normalized_diameter(contour, image.shape),
    }


def image_stats(image: np.ndarray) -> dict:
    """
    Estadísticas de calidad de la foto completa, medidas tras reducirla a config.STATS_SIDE.

    Parámetros:
        image: foto BGR (H, W, 3).

    Retorna:
        dict con width y height (px originales), brightness (media del gris, 0-255), contrast
        (desviación estándar del gris) y sharpness (varianza del laplaciano; baja = borrosa).

    Fórmula: sharpness = Var(∇²I), con ∇² el laplaciano discreto 3x3 de la imagen en gris I.
    Fuente: J. L. Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy: a
        comparative study", ICPR 2000, https://doi.org/10.1109/ICPR.2000.903548
    """
    height, width = image.shape[:2]
    scale = config.STATS_SIDE / max(height, width)
    gray = cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    return {"width": width, "height": height, "brightness": round(float(gray.mean()), 2),
            "contrast": round(float(gray.std()), 2),
            "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)}


def prepare_record(row: dict[str, str], output: Path, manual: dict, *, pilot: bool) -> dict:
    """
    Prepara una fila del inventario: segmenta, decide si es candidata y guarda recorte y máscara.

    Parámetros:
        row: fila de index.csv.
        output: carpeta donde se escriben images/ y masks/.
        manual: contenido de data/own/contours.json (contornos manuales y su tamaño de referencia).
        pilot: si es True también segmenta las fotos no etiquetadas, solo para diagnóstico.

    Retorna:
        La fila con las columnas de STAT_COLUMNS y segmentation_status, segmentation_method,
        decision, detail, d_norm, n_fruits, crop y crop_mask.
    """
    record = {
        **row,
        "segmentation_status": "",
        "segmentation_method": "",
        "decision": "excluded",
        "detail": "",
        "d_norm": "",
        "n_fruits": "",
        **dict.fromkeys(STAT_COLUMNS, ""),
        "crop": "",
        "crop_mask": "",
    }
    eligible = row["status"] == "labeled" and row["quality"] in config.QUALITY_CLASSES
    if not eligible:
        record["detail"] = row["status"] if row["status"] != "labeled" else "invalid_quality"
        if not pilot:
            return record

    try:
        image = read_image(config.ROOT / row["filename"])
        if image is None:
            record["segmentation_status"] = "discard: unreadable"
            return record
        record.update(image_stats(image))
        points = manual["contours"].get(row["filename"])
        if points is None:
            result = segmentation.process(image)
            record["segmentation_method"] = "automatic"
        else:
            result = manual_segment(image, points, manual["reference_size"])
            record["segmentation_method"] = "manual"
    except (OSError, cv2.error) as error:
        record["segmentation_status"] = "review: processing error"
        if eligible:
            record["decision"] = "review"
            record["detail"] = str(error).strip()
        return record

    record["segmentation_status"] = result["status"]
    fruits = re.search(r"(\d+) fruits", result["status"])
    record["n_fruits"] = int(fruits[1]) if fruits else 1
    if not eligible:
        return record
    if result["status"] != "ok":
        record["decision"] = "review" if result["status"].startswith("review:") else "excluded"
        return record

    original = Path(row["filename"])
    crop = Path("images") / original.with_name(original.name + ".png")
    mask = Path("masks") / original.with_name(original.name + ".png")
    box = cv2.boundingRect(result["contour"])
    crop_mask = segmentation.square_crop(result["mask"], box, cv2.BORDER_CONSTANT)
    height, width = config.IMG_SIZE
    interpolation = cv2.INTER_AREA if result["crop"].shape[0] >= height else cv2.INTER_LINEAR
    write_png(output / crop, cv2.resize(result["crop"], (width, height), interpolation=interpolation))
    write_png(output / mask, cv2.resize(crop_mask, (width, height), interpolation=cv2.INTER_NEAREST))
    record.update(
        decision="candidate", d_norm=float(result["d_norm"]),
        crop=crop.as_posix(), crop_mask=mask.as_posix(),
    )
    return record


def publish(staging: Path, output: Path) -> None:
    """
    Mueve la carpeta temporal terminada a su ubicación final (reintenta si Windows la bloquea).

    Parámetros:
        staging: carpeta temporal con images/, masks/ y manifest.csv.
        output: carpeta final (config.CROPS_DIR o pilot-crops); no debe existir.
    """
    # Windows puede bloquear brevemente una carpeta con archivos recién escritos.
    for attempt in range(4):
        try:
            staging.rename(output)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or output.exists() or attempt == 3:
                raise
            sleep(0.2 * (attempt + 1))


def run(pilot: bool) -> None:
    """
    Prepara todo el inventario (o una muestra piloto) en paralelo y publica el resultado.

    Parámetros:
        pilot: True = fotos propias + 1 de cada 300 externas, en data/processed/pilot-crops/.
    """
    with config.INDEX_CSV.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows or not fields:
        raise ValueError("El inventario está vacío. Ejecuta python -m src.data.inventory.")

    output = config.CROPS_DIR
    if pilot:
        output = output.with_name("pilot-crops")
        own = [row for row in rows if row["source"] == "own"]
        external = [row for row in rows if row["source"] != "own" and row["status"] == "labeled"]
        rows = own + external[::300]
    if output.exists():
        raise FileExistsError(f"Archiva o mueve el resultado anterior antes de repetir: {output}")

    with config.OWN_CONTOURS.open(encoding="utf-8") as file:
        manual = json.load(file)

    # Evita multiplicar los hilos internos de OpenCV por los del procesamiento paralelo.
    cv2.setNumThreads(1)
    columns = fields + [
        "segmentation_status", "segmentation_method", "decision", "detail", "d_norm", "n_fruits",
        *STAT_COLUMNS, "crop", "crop_mask",
    ]
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            with ThreadPoolExecutor(max_workers=6) as pool:
                records = pool.map(lambda row: prepare_record(row, staging, manual, pilot=pilot), rows)
                for count, record in enumerate(records, start=1):
                    writer.writerow(record)
                    if not pilot and count % 2000 == 0:
                        print(f"{count}/{len(rows)} fotos procesadas", flush=True)
        publish(staging, output)

    print(f"{len(rows)} fotos registradas en {(output / 'manifest.csv').relative_to(config.ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Usar las fotos propias y una muestra externa.")
    run(parser.parse_args().pilot)
