"""
Fotos de celular (pool compartido de la clase): ingesta y etiquetado ciego del estado.

Fase CRISP-DM: Comprensión y preparación de datos
Uso:
    # 1) descargar la carpeta de Drive ("Photos of Fruits and Vegetables | APO III") y
    #    descomprimirla en data/external/raw/<Producto>/...
    python -m src.data.external_pool ingest --raw data/external/raw
    # 2) hojas de etiquetado ciego (A obligatoria; B opcional, solo para medir el acuerdo)
    python -m src.data.external_pool sheets
    # 3) etiquetas finales y evaluación única: ver src/evaluation/external_eval.py

Salidas en data/external/ (fuera de Git, como data/processed):
    manifest.csv           una fila por foto: blind_id, fruta (de la carpeta), sesión, estado de
                           la segmentación, d_norm, recorte y máscara
    crops/<blind_id>.png   recorte 128x128 (mismo preprocesamiento que el entrenamiento)
    crops/<blind_id>_mask.png
    preview/<blind_id>.jpg vista previa (lado mayor 1024 px, orientación corregida) para etiquetar;
                           las HEIC no se ven en el navegador
    labeling/annotator_A.csv, annotator_B.csv   hojas para etiquetar (sin ninguna predicción)
    labeling/sheet_A.html, sheet_B.html          galería con la foto completa y el blind_id

Decisiones de diseño (ver docs/plan_simple.md):
- La fruta se conoce por la carpeta de Drive (Apple, Banana, …): esa etiqueta es gratis.
- El estado (good / regular / bad) sí hay que etiquetarlo para poder medir. Se hace a ciegas:
  blind_id aleatorio, sin nombre de archivo y sin abrir la app, para que el modelo no influya.
- Ninguna foto del pool se usa para entrenar ni para ajustar nada: todo el pool es una prueba
  externa que se evalúa una sola vez (src.evaluation.external_eval).
- La sesión (misma cámara y mismo día, según EXIF) se guarda para los intervalos de confianza:
  fotos de una misma sesión no son independientes.
- Productos que el modelo no conoce (p. ej. mango) se marcan in_scope=False y se reportan aparte.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from src import config
from src.data import segmentation

EXTERNAL = config.ROOT / "data" / "external"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
FIELDS = ("blind_id", "original", "sha256", "product", "in_scope", "session",
          "status", "d_norm", "crop", "crop_mask", "width", "height")
_TAGS = {name: tag for tag, name in ExifTags.TAGS.items()}


def _register_heif() -> bool:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return False
    register_heif_opener()
    return True


def read_photo(path: Path) -> tuple[np.ndarray, dict]:
    """
    Foto como BGR uint8 con la orientación EXIF aplicada, y sus metadatos de cámara.

    Las fotos de celular suelen guardarse giradas con una etiqueta EXIF; sin aplicarla, la
    segmentación y el modelo verían la fruta de lado. Las HEIC de iPhone requieren pillow-heif.
    """
    if path.suffix.lower() in {".heic", ".heif"} and not _register_heif():
        raise RuntimeError("Para leer HEIC instala pillow-heif (está en requirements.txt).")
    with Image.open(path) as image:
        exif = image.getexif()
        meta = {
            "make": str(exif.get(_TAGS["Make"], "")).strip(),
            "model": str(exif.get(_TAGS["Model"], "")).strip(),
            "datetime": str(exif.get_ifd(0x8769).get(_TAGS["DateTimeOriginal"], "")
                            or exif.get(_TAGS["DateTime"], "")).strip(),
        }
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), meta


def session_of(path: Path, meta: dict) -> str:
    """
    Sesión de captura: misma cámara y mismo día según EXIF. Sin EXIF (p. ej. reenviadas por
    WhatsApp), se usa la carpeta y la serie del nombre (IMG_31xx → bloque de 50).
    """
    product_dir = path.parent.name.lower()
    if meta.get("datetime"):
        day = meta["datetime"][:10].replace(":", "")
        camera = (meta.get("make", "") + " " + meta.get("model", "")).strip() or "unknown"
        return f"{product_dir}:{camera}:{day}"
    stem = path.stem.lower()
    digits = "".join(ch for ch in stem if ch.isdigit())
    if stem.startswith("img_") and digits:
        return f"{product_dir}:series:{int(digits) // 50}"
    return f"{product_dir}:noexif"


def _write(path: Path, image: np.ndarray, extension: str = ".png") -> None:
    """Escribe con imencode + tofile: cv2.imwrite falla con rutas Unicode en Windows."""
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if extension == ".jpg" else []
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise RuntimeError(f"No se pudo codificar {path}")
    encoded.tofile(str(path))


def blind_id(sha: str) -> str:
    """Identificador ciego estable: no revela producto, sesión ni nombre original."""
    return "EXT-" + hashlib.sha256(("blind:" + sha).encode()).hexdigest()[:8].upper()


def ingest(raw: Path, out: Path = EXTERNAL) -> Path:
    manifest_path = out / "manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"Ya existe {manifest_path}; la ingesta no se repite sobre el mismo pool.")
    photos = sorted(p for p in raw.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not photos:
        raise FileNotFoundError(f"No hay fotos en {raw}")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)
    rows, seen = [], {}
    for path in photos:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha in seen:
            print(f"Duplicado omitido: {path} = {seen[sha]}")
            continue
        seen[sha] = path
        product_name = path.parent.name.lower().strip()
        product = config.PRODUCT_NAMES.get(product_name, product_name)
        image, meta = read_photo(path)
        result = segmentation.process(image)
        scale = min(1.0, 1024 / max(image.shape[:2]))
        preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _write(out / "preview" / f"{blind_id(sha)}.jpg", preview, ".jpg")
        row = {"blind_id": blind_id(sha), "original": path.relative_to(raw).as_posix(), "sha256": sha,
               "product": product, "in_scope": str(product in config.PRODUCTS),
               "session": session_of(path, meta), "status": result["status"],
               "d_norm": "" if result.get("d_norm") is None else f"{result['d_norm']:.4f}",
               "crop": "", "crop_mask": "", "width": image.shape[1], "height": image.shape[0]}
        if result["status"] == "ok":
            crop, mask = segmentation.model_inputs(result)
            row["crop"], row["crop_mask"] = f"crops/{row['blind_id']}.png", f"crops/{row['blind_id']}_mask.png"
            _write(out / row["crop"], crop)
            _write(out / row["crop_mask"], mask)
        rows.append(row)

    with manifest_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by = Counter((r["product"], r["status"] == "ok") for r in rows)
    print(f"{len(rows)} fotos ingeridas en {manifest_path}")
    for (product, ok), n in sorted(by.items()):
        print(f"  {product:12s} {'ok' if ok else 'rechazada':9s} {n}")
    return manifest_path


def read_manifest(out: Path = EXTERNAL) -> list[dict[str, str]]:
    with (out / "manifest.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def sheets(out: Path = EXTERNAL) -> None:
    """
    Hojas de etiquetado: A (obligatoria) y B (opcional, para medir el acuerdo), en orden
    aleatorio distinto, sin nombre original, sin sesión y sin ninguna salida del modelo. Solo
    fotos con segmentación ok de frutas que el modelo conoce.
    """
    rows = [r for r in read_manifest(out) if r["status"] == "ok" and r["in_scope"] == "True"]
    labeling = out / "labeling"
    labeling.mkdir(parents=True, exist_ok=True)
    for annotator, seed in (("A", 1), ("B", 2)):
        order = rows[:]
        random.Random(seed).shuffle(order)
        sheet = labeling / f"annotator_{annotator}.csv"
        if sheet.exists():
            raise FileExistsError(f"Ya existe {sheet}: no se regeneran hojas en uso.")
        with sheet.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("blind_id", "product", "quality", "size",
                                                      "unusable", "notes"))
            writer.writeheader()
            writer.writerows({"blind_id": r["blind_id"], "product": r["product"], "quality": "",
                              "size": "", "unusable": "", "notes": ""} for r in order)
        cards = "\n".join(
            f'<figure><img loading="lazy" src="../preview/{r["blind_id"]}.jpg">'
            f'<figcaption>{r["blind_id"]} · {html.escape(r["product"])}</figcaption></figure>'
            for r in order)
        (labeling / f"sheet_{annotator}.html").write_text(
            "<!DOCTYPE html><meta charset='utf-8'><title>Etiquetado ciego "
            f"{annotator}</title><style>body{{font-family:sans-serif}}figure{{display:inline-block;"
            "width:300px;margin:8px}}img{width:300px;height:300px;object-fit:contain;background:#eee}"
            "</style><p>Etiqueta calidad (good / regular / bad) y tamaño (small / medium / large) "
            "según docs/criterios-calidad.md, en annotator_" + annotator + ".csv. Marca unusable=1 si "
            "la foto no permite juzgar. No consultes al otro anotador ni la app.</p>" + cards,
            encoding="utf-8")
    print(f"Hojas en {labeling}: {len(rows)} fotos. Llena annotator_A.csv; annotator_B.csv es "
          "opcional (otra persona, sin mirar A) para reportar el acuerdo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--raw", type=Path, default=EXTERNAL / "raw")
    sub.add_parser("sheets")
    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.raw)
    else:
        sheets()
