"""
Construye data/processed/index.csv a partir de las tres fuentes locales (ryandpark, AgriFreshNET
y fotos propias), con etiqueta, grupo para las particiones y estado de cada imagen.

Fase CRISP-DM: Comprensión de datos (inventario) y Preparación de datos (etiquetas y grupos)
Uso: python -m src.main inventario      (o python -m src.data.inventory)

El grupo (`group`) es la unidad que no se separa al dividir en train/val/test:
- ryandpark: fotos de la misma ráfaga (mismo minuto, o números consecutivos si no hay fecha);
- AgriFreshNET: la foto original con sus copias aumentadas y variantes del teléfono (~2);
- propias: la sesión o la pieza anotada en labels.csv;
- en todas, las copias con contenido idéntico se unen en un solo grupo.
"""
import csv
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile

from src import config


@dataclass
class ImageRecord:
    """Una fila del inventario (las columnas de index.csv, en este orden)."""
    filename: str
    product: str
    quality: str
    source: str
    size: str
    id_fruta: str
    group: str
    status: str


def images(directory: Path) -> list[Path]:
    """
    Lista las imágenes de una carpeta (sin subcarpetas), en orden alfabético.

    Parámetros:
        directory: carpeta de un producto y una calidad.

    Retorna:
        Rutas cuya extensión, en minúscula, está en config.IMAGE_EXTENSIONS.
    """
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def own_records() -> list[ImageRecord]:
    """
    Registros de las fotos propias según data/own/labels.csv.

    Retorna:
        Un registro por fila. El grupo es el de la anotación, o la pieza (id_fruta), o la foto.

    Lanza:
        FileNotFoundError si falta una foto; ValueError si la calidad no está en QUALITY_CLASSES.
    """
    records = []
    with config.OWN_LABELS.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            path = config.DATA_OWN / row["filename"]
            if not path.is_file():
                raise FileNotFoundError(f"Foto propia indicada en labels.csv: {path}")

            quality = row["quality"]
            if quality and quality not in config.QUALITY_CLASSES:
                raise ValueError(f"Calidad desconocida en {path.name}: {quality!r}")

            fruit_id = row.get("id_fruta") or ""
            group = row.get("group") or fruit_id or path.stem
            status = row.get("status") or ("labeled" if quality else "pending_quality")
            if status == "needs_instance_labels":
                quality = ""
            records.append(ImageRecord(
                path.relative_to(config.ROOT).as_posix(), row["product"], quality,
                "own", row["size"], fruit_id, f"own:{group}", status,
            ))
    return records


def capture_group(stem: str) -> str | None:
    """
    Ráfaga de captura de una foto de ryandpark, deducida de su nombre.

    Las fotos de una misma escena se tomaron con segundos de diferencia (la misma fruta desde
    varios ángulos); si quedan en particiones distintas, el modelo se evalúa con casi la misma
    imagen que vio al entrenar.

    Parámetros:
        stem: nombre del archivo sin extensión y en minúscula, p. ej. "img_20190902_111542_1".

    Retorna:
        "AAAAMMDD_HHMM" si el nombre tiene fecha y hora; "img_NNNx" (bloques de
        config.CAPTURE_NUMBER_BLOCK números) si solo tiene un consecutivo; None si no aplica.
    """
    stamp = re.search(config.CAPTURE_STAMP, stem)
    if stamp:
        return f"{stamp[1]}_{stamp[2]}"
    number = re.fullmatch(r"img_(\d+)", stem)
    if number:
        return f"img_{int(number[1]) // config.CAPTURE_NUMBER_BLOCK}x"
    return None


def kaggle_records() -> list[ImageRecord]:
    """
    Registros de ryandpark. Las carpetas Good y Bad quedan etiquetadas; Mixed se inventaría
    sin calidad (`needs_instance_labels`) porque cada foto tiene varias piezas.

    Retorna:
        Un registro por imagen, agrupado por ráfaga de captura (ver capture_group) o, si el
        nombre no la indica, por nombre de origen.
    """
    records = []
    folders = dict(config.RYANDPARK_QUALITY)
    if (config.RYANDPARK_DIR / config.RYANDPARK_MIXED_DIR).is_dir():
        folders[config.RYANDPARK_MIXED_DIR] = ""

    for category, quality in folders.items():
        status = "labeled" if quality else "needs_instance_labels"
        for directory in sorted((config.RYANDPARK_DIR / category).iterdir()):
            if not directory.is_dir():
                continue
            product = config.PRODUCT_NAMES[directory.name.split("_")[0].lower()]
            photos = images(directory)
            stems = {path.stem.lower() for path in photos}
            for path in photos:
                stem = path.stem.lower()
                # Las copias de granada usan cinco dígitos. También hay pares _1 y _01.
                for suffix in (r"_\d{5}$", r"_0?1$"):
                    base = re.sub(suffix, "", stem)
                    if base in stems:
                        stem = base
                capture = capture_group(stem)
                group = f"capture:{capture}" if capture else f"{category}:{product}:{stem}"
                records.append(ImageRecord(
                    path.relative_to(config.ROOT).as_posix(), product, quality,
                    "ryandpark", "", "", f"ryandpark:{group}", status,
                ))
    return records


def agrifreshnet_records() -> list[ImageRecord]:
    """
    Registros de AgriFreshNET. La carpeta indica etapa y producto, p. ej. "Semi fresh Orange(9-20)".

    Retorna:
        Un registro por imagen. Las copias "aug_N_" y las variantes "~2" comparten grupo con
        su foto original.
    """
    records = []
    for directory in sorted(config.AGRIFRESHNET_DIR.iterdir()):
        if not directory.is_dir():
            continue
        name = directory.name.lower().replace("_", " ").split("(", 1)[0].strip()
        stage, product_name = name.rsplit(maxsplit=1)
        quality = config.AGRIFRESHNET_QUALITY[stage]
        product = config.PRODUCT_NAMES[product_name]
        for path in images(directory):
            stem = re.sub(config.AUG_PREFIX, "", path.stem.lower())
            # El teléfono añade ~2, ~3, etc. a variantes del mismo archivo con fecha y hora.
            stem = re.sub(r"^(img_\d{8}_\d{6,}.*?)~\d+$", r"\1", stem)
            records.append(ImageRecord(
                path.relative_to(config.ROOT).as_posix(), product, quality,
                "agrifreshnet", "", "", f"agrifreshnet:{quality}:{product}:{stem}", "labeled",
            ))
    return records


def content_digest(path: Path) -> bytes:
    """
    Huella SHA-256 del contenido de un archivo.

    Parámetros:
        path: ruta del archivo.

    Retorna:
        Los 32 bytes del resumen.
    """
    with path.open("rb") as file:
        return file_digest(file, "sha256").digest()


def merge_external_duplicates(records: list[ImageRecord]) -> None:
    """
    Une en un solo grupo las copias idénticas, aunque estén en carpetas o categorías distintas,
    y marca `conflicting_quality` en los grupos donde un mismo contenido tiene dos etiquetas.

    Solo se calcula el SHA-256 de los archivos que comparten tamaño en bytes (candidatos a copia).
    Una ráfaga que contiene fotos distintas de calidades distintas NO es un conflicto.

    Parámetros:
        records: registros externos; se modifican en el lugar (group y status).
    """
    by_size = defaultdict(list)
    aliases = {}
    for record in records:
        path = config.ROOT / record.filename
        by_size[path.stat().st_size].append((path, record))

    def canonical(group: str) -> str:
        while group in aliases:
            group = aliases[group]
        return group

    candidates = [entry for bucket in by_size.values() if len(bucket) > 1 for entry in bucket]
    seen = {}
    labels_by_digest = defaultdict(set)
    group_of_digest = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = pool.map(content_digest, (path for path, _ in candidates))
        for (_, record), digest in zip(candidates, digests):
            group = canonical(record.group)
            previous = canonical(seen.setdefault(digest, group))
            if group != previous:
                first, second = sorted((group, previous))
                aliases[second] = first
            if record.quality:
                labels_by_digest[digest].add(record.quality)
            group_of_digest[digest] = record.group

    for record in records:
        record.group = canonical(record.group)
    conflicts = {canonical(group_of_digest[d]) for d, labels in labels_by_digest.items() if len(labels) > 1}
    for record in records:
        if record.group in conflicts:
            record.status = "conflicting_quality"


def build_inventory() -> list[ImageRecord]:
    """
    Reúne las tres fuentes, une los duplicados externos y comprueba que no haya rutas repetidas.

    Retorna:
        Registros ordenados por fuente y ruta.
    """
    external = kaggle_records() + agrifreshnet_records()
    merge_external_duplicates(external)
    records = own_records() + external
    if len({record.filename for record in records}) != len(records):
        raise ValueError("El inventario contiene rutas de imagen duplicadas.")
    return sorted(records, key=lambda record: (record.source, record.filename))


def write_inventory(records: list[ImageRecord]) -> None:
    """
    Escribe config.INDEX_CSV. Primero escribe un archivo temporal y luego lo reemplaza, para
    no dejar un índice a medias si el proceso se interrumpe.

    Parámetros:
        records: registros del inventario.
    """
    config.INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=config.INDEX_CSV.parent, delete=False,
    )
    temporary = Path(file.name)
    try:
        with file:
            writer = csv.DictWriter(file, fieldnames=list(ImageRecord.__annotations__))
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        temporary.replace(config.INDEX_CSV)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    inventory = build_inventory()
    write_inventory(inventory)
    print(f"{len(inventory)} imágenes registradas en {config.INDEX_CSV.relative_to(config.ROOT)}")
