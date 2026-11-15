"""
Divide los recortes candidatos en train, val y test sin separar las fotos de un mismo grupo.

Fase CRISP-DM: Preparación de datos
Uso: python -m src.data.split [--manifest RUTA]

Lee el manifiesto de src.data.prepare, quita los grupos apartados en la revisión visual
(config.SEGMENTATION_EXCLUSIONS) y escribe splits.csv junto al manifiesto. Cada grupo completo
va a una sola partición, con proporciones cercanas a 1 - VAL_SIZE - TEST_SIZE / VAL_SIZE / TEST_SIZE
por archivo, calidad, fuente y producto.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src import config

PARTITIONS = ("train", "val", "test")


def assign_groups(rows: list[dict[str, str]]) -> dict[str, str]:
    """
    Asigna cada grupo a una partición comparando 256 permutaciones aleatorias de los grupos.

    Para cada permutación cuenta, por partición, los archivos de cada estrato (total, calidad,
    fuente x calidad y producto) y elige la de menor puntaje, en este orden de prioridad:
    1) que train tenga las tres calidades; 2) que val y test también las tengan;
    3) el menor error cuadrático ponderado frente a las proporciones esperadas.

    Parámetros:
        rows: filas candidatas con group, quality, source y product.

    Retorna:
        dict grupo -> "train" | "val" | "test".

    Fórmula: E = sum_s w_s * ((n_ps - p_p * N_s) / max(N_s, 20))^2, con n_ps los archivos del
        estrato s en la partición p, N_s el total del estrato, p_p la proporción objetivo y
        w_s = 3 para total y calidad, 1 para el resto.
    """
    groups = sorted({row["group"] for row in rows})
    if len(groups) < len(PARTITIONS):
        raise ValueError("Se requieren al menos tres grupos candidatos.")

    quality = sorted({row["quality"] for row in rows})
    strata = (
        [("total", "*")]
        + [("quality", label) for label in quality]
        + sorted({("source_quality", (row["source"], row["quality"])) for row in rows})
        + sorted({("product", row["product"]) for row in rows})
    )
    group_index = {group: i for i, group in enumerate(groups)}
    stratum_index = {stratum: i for i, stratum in enumerate(strata)}
    counts = np.zeros((len(groups), len(strata)), dtype=np.int64)
    for row in rows:
        keys = (
            ("total", "*"), ("quality", row["quality"]),
            ("source_quality", (row["source"], row["quality"])), ("product", row["product"]),
        )
        counts[group_index[row["group"]], [stratum_index[key] for key in keys]] += 1

    proportions = np.array([1 - config.VAL_SIZE - config.TEST_SIZE, config.VAL_SIZE, config.TEST_SIZE])
    test_groups = max(1, round(len(groups) * config.TEST_SIZE))
    val_groups = max(1, round(len(groups) * config.VAL_SIZE))
    totals = counts.sum(axis=0)
    expected = proportions[:, None] * totals
    quality_cols = slice(1, 1 + len(quality))
    possible_coverage = np.minimum(3, (counts[:, quality_cols] > 0).sum(axis=0))
    weights = np.array([3 if kind in ("total", "quality") else 1 for kind, _ in strata])

    rng = np.random.default_rng(config.SEED)
    best_score = None
    best_parts = None
    # Compara asignaciones de grupos completos. Prioriza cobertura de calidad y
    # después el balance por archivo, fuente y producto.
    for _ in range(256):
        order = rng.permutation(len(groups))
        test, val, train = np.split(order, [test_groups, test_groups + val_groups])
        parts = (train, val, test)
        actual = np.array([counts[part].sum(axis=0) for part in parts])
        has_quality = actual[:, quality_cols] > 0
        coverage = has_quality.sum(axis=0)
        share_error = (actual - expected) / np.maximum(totals, 20)
        score = (
            int((~has_quality[0]).sum()),
            int((possible_coverage - coverage).sum()),
            float((weights * share_error**2).sum()),
        )
        if best_score is None or score < best_score:
            best_parts, best_score = parts, score

    return {
        groups[i]: partition
        for partition, part in zip(PARTITIONS, best_parts)
        for i in part
    }


def run(manifest: Path) -> Path:
    """
    Genera splits.csv con los candidatos no excluidos y su partición, e imprime la cobertura.

    Las exclusiones se buscan por archivo revisado (reviewed_file) y se aparta el grupo
    completo al que pertenece hoy ese archivo, para que sigan valiendo si cambia el agrupamiento.

    Parámetros:
        manifest: manifest.csv de src.data.prepare.

    Retorna:
        Ruta del splits.csv escrito.
    """
    with config.SEGMENTATION_EXCLUSIONS.open(encoding="utf-8", newline="") as file:
        excluded_files = {row["reviewed_file"] for row in csv.DictReader(file)}
    with manifest.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        all_rows = list(reader)
    excluded_groups = {row["group"] for row in all_rows if row["filename"] in excluded_files}
    candidates = [row for row in all_rows if row["decision"] == "candidate"]
    rows = [row for row in candidates if row["group"] not in excluded_groups]
    if not rows:
        raise ValueError("El manifiesto no contiene recortes candidatos.")
    if any(not row["group"] or row["quality"] not in config.QUALITY_CLASSES for row in rows):
        raise ValueError("Todos los recortes candidatos necesitan un grupo y una calidad válida.")
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("El manifiesto contiene rutas de imagen repetidas.")

    assignment = assign_groups(rows)
    output = manifest.with_name("splits.csv")
    file = NamedTemporaryFile(
        dir=output.parent, delete=False, mode="w", encoding="utf-8", newline="",
    )
    temporary = Path(file.name)
    try:
        with file:
            writer = csv.DictWriter(file, fieldnames=[*columns, "split"])
            writer.writeheader()
            writer.writerows(dict(row, split=assignment[row["group"]]) for row in rows)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Particiones guardadas en {output}")
    print(f"Recortes candidatos excluidos por revisión visual: {len(candidates) - len(rows)}")
    for partition in PARTITIONS:
        subset = [row for row in rows if assignment[row["group"]] == partition]
        qualities = Counter(row["quality"] for row in subset)
        sources = Counter(row["source"] for row in subset)
        products = Counter(row["product"] for row in subset)
        print(f"{partition}: {len(subset)} archivos, {len({row['group'] for row in subset})} grupos")
        for name, counts in (("calidad", qualities), ("fuente", sources), ("producto", products)):
            print(f"  {name}: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        missing = set(config.QUALITY_CLASSES) - qualities.keys()
        if missing:
            print(f"  clases ausentes: {', '.join(sorted(missing))}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=config.CROPS_DIR / "manifest.csv")
    run(parser.parse_args().manifest)
