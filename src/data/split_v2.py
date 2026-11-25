"""
Split v2: reparte los recortes candidatos usando los grupos vigentes de index.csv.

Fase CRISP-DM: Preparación de datos
Uso: python -m src.data.split_v2 [--output RUTA]

Motivo: crops/manifest.csv se generó desde un index.csv anterior al agrupamiento por
ráfaga de ryandpark (commit 90f498a). Por eso splits.csv (v1) usa un grupo por archivo
en Kaggle, y fotos de una misma ráfaga quedaron en particiones distintas. Este módulo
toma el manifiesto tal cual (recortes, decisiones y máscaras no cambian), reemplaza la
columna group por la de index.csv, aplica las mismas exclusiones visuales que
src.data.split y llama al mismo assign_groups. No modifica splits.csv ni manifest.csv:
escribe un archivo nuevo, por defecto crops/splits_v2_burst.csv.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from src import config
from src.data.split import PARTITIONS, assign_groups

DEFAULT_OUTPUT = config.CROPS_DIR / "splits_v2_burst.csv"


def regroup(manifest_rows: list[dict[str, str]], index_groups: dict[str, str]) -> list[dict[str, str]]:
    """
    Devuelve copias de las filas del manifiesto con el grupo vigente de index.csv.

    La columna original se conserva como group_v1 para trazabilidad.
    """
    missing = [row["filename"] for row in manifest_rows if row["filename"] not in index_groups]
    if missing:
        raise ValueError(f"{len(missing)} filas del manifiesto no están en index.csv; ej. {missing[0]}")
    return [dict(row, group_v1=row["group"], group=index_groups[row["filename"]]) for row in manifest_rows]


def select_candidates(rows: list[dict[str, str]], excluded_files: set[str]) -> list[dict[str, str]]:
    """Candidatos sin los grupos que contienen un archivo excluido en la revisión visual."""
    excluded_groups = {row["group"] for row in rows if row["filename"] in excluded_files}
    return [row for row in rows if row["decision"] == "candidate" and row["group"] not in excluded_groups]


def run(manifest: Path, index: Path, output: Path) -> Path:
    if output.resolve() in {(config.CROPS_DIR / "splits.csv").resolve(), manifest.resolve()}:
        raise ValueError("El split v2 no puede sobrescribir splits.csv ni el manifiesto.")
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; muévelo antes de regenerar.")

    with index.open(encoding="utf-8", newline="") as file:
        index_groups = {row["filename"]: row["group"] for row in csv.DictReader(file)}
    with manifest.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames)
        manifest_rows = list(reader)
    with config.SEGMENTATION_EXCLUSIONS.open(encoding="utf-8", newline="") as file:
        excluded_files = {row["reviewed_file"] for row in csv.DictReader(file)}

    rows = select_candidates(regroup(manifest_rows, index_groups), excluded_files)
    if any(not row["group"] or row["quality"] not in config.QUALITY_CLASSES for row in rows):
        raise ValueError("Todos los candidatos necesitan grupo y calidad válida.")
    assignment = assign_groups(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*columns, "group_v1", "split"])
        writer.writeheader()
        writer.writerows(dict(row, split=assignment[row["group"]]) for row in rows)

    candidates = sum(row["decision"] == "candidate" for row in manifest_rows)
    print(f"Split v2 guardado en {output}")
    print(f"Candidatos: {candidates}; excluidos por revisión visual (grupo completo): {candidates - len(rows)}")
    for partition in PARTITIONS:
        subset = [row for row in rows if assignment[row["group"]] == partition]
        print(f"{partition}: {len(subset)} archivos, {len({row['group'] for row in subset})} grupos")
        for name, key in (("calidad", "quality"), ("fuente", "source"), ("producto", "product")):
            counts = Counter(row[key] for row in subset)
            print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=config.CROPS_DIR / "manifest.csv")
    parser.add_argument("--index", type=Path, default=config.INDEX_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.manifest, args.index, args.output)
