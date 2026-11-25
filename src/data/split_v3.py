"""
Split v3: el split v2 por ráfagas, con los grupos de AgriFreshNET unidos por sesión de captura.

Fase CRISP-DM: Preparación de datos
Uso:
    python -m src.data.split_v3 --level week               # escribe crops/splits_v3_week.csv
    python -m src.data.split_v3 --level week --dry-run     # solo informa cobertura y proporciones
    python -m src.data.split_v3 --choose                   # aplica la regla de docs/plan_simple.md

Motivo: AgriFreshNET fotografía las mismas piezas durante días (carpetas "Fresh X(1-9)",
"Semi fresh X(9-20)", "Rotten X(24-35)"). En v2 cada foto original es su propio grupo, así que
fotos de la misma sesión, o de la misma pieza en días seguidos, pueden quedar en train y en
val. Este módulo une esos grupos (src/data/agri_sessions.merge_groups) y vuelve a repartir con
el mismo assign_groups y la misma semilla que v2 (con más permutaciones, TRIALS), sobre los
mismos recortes. Corta por archivos (cut="files") en lugar de por número de grupos. Kaggle y las fotos
propias conservan sus grupos.

Regla de elección del nivel (fijada antes de ver resultados de modelos; --choose):
se toma el nivel MÁS GRUESO, en el orden week > day > minute, cuyo split cumpla:
    1. train contiene las tres calidades y todos los productos;
    2. val y test contienen las tres calidades;
    3. por calidad, la proporción de cada partición se desvía como máximo MAX_DEVIATION
       (5 puntos porcentuales) del objetivo 70/15/15.
No se mira el F1 para elegir: la escalera de agri_leak_ladder.py solo cuantifica la fuga.
No modifica splits_v2_burst.csv: escribe un archivo nuevo.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src import config
from src.data.agri_sessions import merge_groups
from src.data.split import PARTITIONS, assign_groups

SOURCE_SPLIT = config.CROPS_DIR / "splits_v2_burst.csv"
CHOICE_ORDER = ("week", "day", "minute")
MAX_DEVIATION = 0.05
# Permutaciones de assign_groups: los grupos de v3 son más grandes y desiguales que los de v2.
TRIALS = 4096


def regroup(rows: list[dict[str, str]], level: str) -> list[dict[str, str]]:
    """Copias de las filas con el grupo v3; el grupo v2 queda en group_v2."""
    mapping = merge_groups(rows, level)
    return [dict(row, group_v2=row["group"], group=mapping[row["group"]]) for row in rows]


def check(rows: list[dict[str, str]], assignment: dict[str, str]) -> dict:
    """
    Cobertura y desvío de proporciones de una asignación.

    Retorna:
        dict con ok (bool), problems (lista de textos) y deviation (máximo desvío por calidad).
    """
    targets = dict(zip(PARTITIONS, (1 - config.VAL_SIZE - config.TEST_SIZE,
                                     config.VAL_SIZE, config.TEST_SIZE)))
    problems = []
    by_partition = {p: [row for row in rows if assignment[row["group"]] == p] for p in PARTITIONS}
    all_products = {row["product"] for row in rows}
    for partition, subset in by_partition.items():
        qualities = {row["quality"] for row in subset}
        missing = set(config.QUALITY_CLASSES) - qualities
        if missing:
            problems.append(f"{partition} sin calidades {sorted(missing)}")
    missing_products = all_products - {row["product"] for row in by_partition["train"]}
    if missing_products:
        problems.append(f"train sin productos {sorted(missing_products)}")

    totals = Counter(row["quality"] for row in rows)
    deviation = 0.0
    for partition, subset in by_partition.items():
        counts = Counter(row["quality"] for row in subset)
        for quality, total in totals.items():
            deviation = max(deviation, abs(counts[quality] / total - targets[partition]))
    if deviation > MAX_DEVIATION:
        problems.append(f"desvío de proporciones {deviation:.3f} > {MAX_DEVIATION}")
    return {"ok": not problems, "problems": problems, "deviation": deviation}


def build(rows: list[dict[str, str]], level: str) -> tuple[list[dict[str, str]], dict[str, str], dict]:
    regrouped = regroup(rows, level)
    assignment = assign_groups(regrouped, trials=TRIALS, cut="files")
    return regrouped, assignment, check(regrouped, assignment)


def report(rows, assignment, level, verdict) -> None:
    print(f"Nivel {level}: {len({r['group'] for r in rows})} grupos "
          f"({len({r['group_v2'] for r in rows})} en v2). "
          f"{'CUMPLE' if verdict['ok'] else 'NO CUMPLE: ' + '; '.join(verdict['problems'])} "
          f"(desvío máx. {verdict['deviation']:.3f})")
    for partition in PARTITIONS:
        subset = [row for row in rows if assignment[row["group"]] == partition]
        print(f"  {partition}: {len(subset)} archivos, {len({r['group'] for r in subset})} grupos; "
              + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r['quality'] for r in subset).items()))
              + " | " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r['source'] for r in subset).items())))


def run(level: str | None, choose: bool, source: Path, output: Path | None, dry_run: bool) -> Path | None:
    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = [c for c in reader.fieldnames if c != "split"]
        rows = [{k: v for k, v in row.items() if k != "split"} for row in reader]

    levels = CHOICE_ORDER if choose else (level,)
    chosen = None
    for candidate in levels:
        regrouped, assignment, verdict = build(rows, candidate)
        report(regrouped, assignment, candidate, verdict)
        if verdict["ok"]:
            chosen = (candidate, regrouped, assignment)
            break
        if not choose:
            chosen = (candidate, regrouped, assignment)
    if chosen is None:
        print("Ningún nivel cumple la regla: se mantiene v2 y se documenta la limitación.")
        return None
    level, regrouped, assignment = chosen
    if choose:
        print(f"Regla de elección: nivel {level}.")
    if dry_run:
        return None

    output = output or config.CROPS_DIR / f"splits_v3_{level}.csv"
    if output.resolve() in {source.resolve(), (config.CROPS_DIR / "splits.csv").resolve()}:
        raise ValueError("El split v3 no puede sobrescribir v1 ni v2.")
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; muévelo antes de regenerar.")
    with output.open("x", encoding="utf-8", newline="") as file:
        fields = [*columns, *(["group_v2"] if "group_v2" not in columns else []), "split"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(row, split=assignment[row["group"]]) for row in regrouped)
    print(f"Split v3 guardado en {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", choices=("minute", "day", "week"))
    parser.add_argument("--choose", action="store_true", help="Aplica la regla de elección.")
    parser.add_argument("--source", type=Path, default=SOURCE_SPLIT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.choose and not args.level:
        parser.error("Indica --level o --choose.")
    run(args.level, args.choose, args.source, args.output, args.dry_run)
