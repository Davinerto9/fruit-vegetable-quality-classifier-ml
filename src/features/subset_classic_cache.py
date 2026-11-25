"""
Deriva un caché de atributos clásicos para otro split sin volver a extraerlos.

Fase CRISP-DM: Preparación de datos
Uso: python -m src.features.subset_classic_cache --split RUTA --output RUTA

Los atributos dependen solo del recorte y su máscara (no tienen parámetros ajustados),
así que para un split con los mismos recortes basta con seleccionar sus filas. Se
reemplazan quality, group, product y source por los del split nuevo, y se falla si
algún archivo del split no está en el caché o si cambió su etiqueta.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from src import config


def subset(cache_path: Path, split_path: Path, output_path: Path) -> Path:
    if output_path.resolve() == cache_path.resolve():
        raise ValueError("La salida no puede sobrescribir el caché de origen.")
    with split_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    with np.load(cache_path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    position = {name: i for i, name in enumerate(data["filenames"].astype(str))}
    missing = [row["filename"] for row in rows if row["filename"] not in position]
    if missing:
        raise ValueError(f"{len(missing)} archivos del split no están en el caché; ej. {missing[0]}")
    order = np.asarray([position[row["filename"]] for row in rows])
    if any(data["quality"][i] != row["quality"] for i, row in zip(order, rows)):
        raise ValueError("Hay etiquetas distintas entre el caché y el split.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=data["X"][order],
        feature_names=data["feature_names"],
        filenames=np.asarray([row["filename"] for row in rows], dtype=str),
        quality=np.asarray([row["quality"] for row in rows], dtype=str),
        groups=np.asarray([row["group"] for row in rows], dtype=str),
        products=np.asarray([row["product"] for row in rows], dtype=str),
        sources=np.asarray([row["source"] for row in rows], dtype=str),
    )
    print(f"Caché derivado: {output_path} ({len(rows)} filas de {len(position)})")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=config.DATA_PROCESSED / "classic_features.npz")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subset(args.cache, args.split, args.output)
