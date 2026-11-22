"""Construye el caché de atributos clásicos desde los recortes del split vigente."""

import argparse
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src import config
from src.features.classic import FEATURE_NAMES, extract_classic_features


def build(split_path: Path, output_path: Path) -> Path:
    with split_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"El split está vacío: {split_path}")
    if any(row.get("decision") != "candidate" for row in rows):
        raise ValueError("El split debe contener únicamente filas candidate.")
    filenames = [row["filename"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("El split contiene archivos repetidos.")

    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    for index, row in enumerate(rows, start=1):
        matrix[index - 1] = extract_classic_features(
            config.CROPS_DIR / row["crop"],
            config.CROPS_DIR / row["crop_mask"],
        )
        if index % 2000 == 0:
            print(f"{index}/{len(rows)} recortes procesados", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(
        dir=output_path.parent, prefix=f".{output_path.stem}-", suffix=".npz", delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        np.savez_compressed(
            temporary_path,
            X=matrix,
            feature_names=np.asarray(FEATURE_NAMES, dtype=str),
            filenames=np.asarray(filenames, dtype=str),
            quality=np.asarray([row["quality"] for row in rows], dtype=str),
            groups=np.asarray([row["group"] for row in rows], dtype=str),
            products=np.asarray([row["product"] for row in rows], dtype=str),
            sources=np.asarray([row["source"] for row in rows], dtype=str),
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Atributos guardados en {output_path} ({len(rows)} filas x {len(FEATURE_NAMES)} atributos)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=config.CROPS_DIR / "splits.csv")
    parser.add_argument(
        "--output", type=Path, default=config.DATA_PROCESSED / "classic_features.npz",
    )
    args = parser.parse_args()
    build(args.split, args.output)
