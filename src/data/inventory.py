"""Construye data/processed/index.csv a partir de las tres fuentes locales."""

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
    filename: str
    product: str
    quality: str
    source: str
    size: str
    id_fruta: str
    group: str
    status: str


def images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def own_records() -> list[ImageRecord]:
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


def kaggle_records() -> list[ImageRecord]:
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
                records.append(ImageRecord(
                    path.relative_to(config.ROOT).as_posix(), product, quality,
                    "ryandpark", "", "", f"ryandpark:{category}:{product}:{stem}", status,
                ))
    return records


def agrifreshnet_records() -> list[ImageRecord]:
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
    with path.open("rb") as file:
        return file_digest(file, "sha256").digest()


def merge_external_duplicates(records: list[ImageRecord]) -> None:
    """Agrupa copias idénticas incluso cuando se encuentran en categorías distintas."""
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
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = pool.map(content_digest, (path for path, _ in candidates))
        for (_, record), digest in zip(candidates, digests):
            group = canonical(record.group)
            previous = canonical(seen.setdefault(digest, group))
            if group != previous:
                first, second = sorted((group, previous))
                aliases[second] = first

    labels_by_group = defaultdict(set)
    for record in records:
        record.group = canonical(record.group)
        if record.quality:
            labels_by_group[record.group].add(record.quality)

    for record in records:
        if len(labels_by_group[record.group]) > 1:
            record.status = "conflicting_quality"


def build_inventory() -> list[ImageRecord]:
    external = kaggle_records() + agrifreshnet_records()
    merge_external_duplicates(external)
    records = own_records() + external
    if len({record.filename for record in records}) != len(records):
        raise ValueError("El inventario contiene rutas de imagen duplicadas.")
    return sorted(records, key=lambda record: (record.source, record.filename))


def write_inventory(records: list[ImageRecord]) -> None:
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
