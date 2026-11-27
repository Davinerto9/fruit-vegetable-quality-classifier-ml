"""
Reparte los recortes en train (70 %), val (15 %) y test (15 %) sin que fotos "hermanas"
queden en partes distintas.

Fase CRISP-DM: Preparación de datos
Uso: python -m src.main dividir      (o python -m src.data.split)

Entrada: data/processed/crops/manifest.csv (src.data.prepare).
Salida:  data/processed/crops/splits.csv, con una columna split = train | val | test.

Por qué no se reparte foto por foto
    Si dos fotos casi iguales quedan una en train y otra en test, el modelo "se acuerda" de
    la primera y acierta la segunda sin haber aprendido nada: la nota sale inflada. Eso es
    una fuga. Por eso se reparte por GRUPOS: todas las fotos de un grupo van juntas.

Qué es un grupo
    1. Kaggle (ryandpark): una ráfaga, fotos tomadas en el mismo minuto o con números
       seguidos (lo calcula src.data.inventory).
    2. AgriFreshNET: todas las fotos del mismo producto tomadas la misma SEMANA. Este dataset
       fotografía las mismas piezas durante días mientras se dañan (carpetas "Fresh X(1-9)",
       "Rotten X(24-35)" = días desde la compra), así que fotos de días seguidos pueden ser
       la misma fruta. La semana se lee de la fecha del nombre (IMG_20251102_...).
    3. Fotos propias: cada fruta física.
    4. Fotos de celular de la mitad "entrenar" (src/evaluation/phone_test.py): cada sesión.
       Van todas a train; la otra mitad es la prueba final y nunca entra aquí.
    Además, si un archivo se apartó en la revisión visual (data/annotations/
    segmentation_exclusions.csv), se aparta su grupo completo.

Cómo se elige el reparto
    Se prueban TRIALS órdenes aleatorios de los grupos (semilla fija) y se queda el que mejor
    cumple, en este orden: train con las 3 calidades y todos los productos; val y test con las
    3 calidades; proporciones lo más cerca posible de 70/15/15 por calidad, fuente y producto.
    Al final se verifica que cada calidad quede a menos de MAX_DEVIATION (5 puntos) de su
    proporción objetivo; si no, se detiene con un error en vez de entrenar con un mal reparto.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src import config

PARTITIONS = ("train", "val", "test")
# Agrupación de AgriFreshNET. "week" es la más gruesa que todavía deja las 3 calidades en
# cada partición; se fijó antes de mirar resultados de modelos.
AGRI_LEVEL = "week"
TRIALS = 4096
MAX_DEVIATION = 0.05


# --- Sesiones de AgriFreshNET (fecha en el nombre del archivo) -----------------------------

LEVELS = ("photo", "minute", "day", "week")

_AUG = re.compile(r"^aug_\d+_", re.IGNORECASE)
_PATTERNS = (
    # IMG_AAAAMMDD_HHMMSS[mmm][_...]
    re.compile(r"^img_(?P<d>20\d{6})_(?P<h>\d{2})(?P<m>\d{2})\d{2}", re.IGNORECASE),
    # IMG-AAAAMMDD-WA####  (WhatsApp: no trae hora)
    re.compile(r"^img-(?P<d>20\d{6})-wa\d+", re.IGNORECASE),
    # WhatsApp Image AAAA-MM-DD at HH.MM.SS
    re.compile(r"^whatsapp image (?P<y>20\d{2})-(?P<mo>\d{2})-(?P<dd>\d{2}) at "
               r"(?P<h>\d{1,2})\.(?P<m>\d{2})", re.IGNORECASE),
)


def parse_stamp(filename: str) -> tuple[str, str | None] | None:
    """
    Fecha y, si existe, hora:minuto de una foto de AgriFreshNET.

    Parámetros:
        filename: ruta o nombre del archivo.

    Retorna:
        ("AAAAMMDD", "HHMM") o ("AAAAMMDD", None) si solo hay fecha; None si el nombre no
        tiene fecha reconocible.
    """
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = _AUG.sub("", name)
    for pattern in _PATTERNS:
        match = pattern.match(name)
        if not match:
            continue
        parts = match.groupdict()
        day = parts.get("d") or f"{parts['y']}{parts['mo']}{parts['dd']}"
        hour = parts.get("h")
        minute = f"{int(hour):02d}{parts['m']}" if hour is not None else None
        return day, minute
    return None


def session_key(filename: str, product: str, level: str) -> str | None:
    """
    Clave de sesión de una foto de AgriFreshNET para el nivel pedido.

    Parámetros:
        filename: ruta o nombre del archivo.
        product: producto canónico (columna product del split).
        level: uno de LEVELS.

    Retorna:
        La clave, o None si el nivel es "photo", si el nombre no tiene fecha, o si el nivel
        es "minute" y el nombre no tiene hora (en ese caso se usa el día, más conservador).
    """
    if level not in LEVELS:
        raise ValueError(f"Nivel desconocido: {level}; usa uno de {LEVELS}")
    if level == "photo":
        return None
    stamp = parse_stamp(filename)
    if stamp is None:
        return None
    day, minute = stamp
    if level == "minute":
        # Sin hora no se puede separar por minuto: se agrupa el día completo.
        return (f"agrifreshnet:{product}:minute:{day}_{minute}" if minute
                else f"agrifreshnet:{product}:day:{day}")
    if level == "day":
        return f"agrifreshnet:{product}:day:{day}"
    year, week, _ = date(int(day[:4]), int(day[4:6]), int(day[6:])).isocalendar()
    return f"agrifreshnet:{product}:week:{year}W{week:02d}"


def merge_groups(rows: list[dict[str, str]], level: str) -> dict[str, str]:
    """
    Nuevo grupo por fila: une el grupo vigente con la sesión de AgriFreshNET (union-find).

    Si dos grupos vigentes comparten sesión, quedan en el mismo grupo nuevo; si un grupo
    vigente toca dos sesiones (por ejemplo, copias idénticas en carpetas de días distintos),
    también las une. Las filas de otras fuentes conservan su grupo.

    Parámetros:
        rows: filas con filename, source, product y group.
        level: nivel de sesión (LEVELS).

    Retorna:
        dict grupo vigente -> grupo nuevo. El nombre nuevo es determinista: el menor
        identificador del componente, con prefijo "sesion:" si el componente une dos o más grupos.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for row in rows:
        find(row["group"])
        if row["source"] == "agrifreshnet":
            key = session_key(row["filename"], row["product"], level)
            if key:
                union(row["group"], "session:" + key)

    members: dict[str, set[str]] = {}
    for node in list(parent):
        members.setdefault(find(node), set()).add(node)
    mapping = {}
    for root, nodes in members.items():
        groups = sorted(node for node in nodes if not node.startswith("session:"))
        name = groups[0] if len(groups) == 1 else "sesion:" + groups[0]
        for group in groups:
            mapping[group] = name
    return mapping


# --- Reparto -------------------------------------------------------------------------------


def assign_groups(rows: list[dict[str, str]], trials: int = 256, cut: str = "groups") -> dict[str, str]:
    """
    Asigna cada grupo completo a una partición comparando varios órdenes aleatorios.

    Para cada permutación cuenta, por partición, los archivos de cada estrato (total, calidad,
    fuente x calidad y producto) y elige la de menor puntaje, en este orden de prioridad:
    1) que train tenga las tres calidades; 2) que train represente todos los productos;
    3) que val y test también tengan las tres calidades; 4) el menor error cuadrático
    ponderado frente a las proporciones esperadas.

    Parámetros:
        rows: filas candidatas con group, quality, source y product.
        trials: órdenes aleatorios a comparar.
        cut: "files" (el que usa el proyecto) corta cada orden donde el acumulado de archivos
            alcanza 15 % y 30 %; "groups" toma un número fijo de grupos (solo sirve si los
            grupos son de tamaño parecido).

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
    product_cols = [
        i for i, (kind, _) in enumerate(strata) if kind == "product"
    ]
    weights = np.array([3 if kind in ("total", "quality") else 1 for kind, _ in strata])

    rng = np.random.default_rng(config.SEED)
    best_score = None
    best_parts = None
    # Compara asignaciones de grupos completos. Prioriza cobertura de calidad y
    # después el balance por archivo, fuente y producto.
    for _ in range(trials):
        order = rng.permutation(len(groups))
        if cut == "files":
            cumulative = np.cumsum(counts[order, 0])
            n_test = int(np.searchsorted(cumulative, config.TEST_SIZE * totals[0])) + 1
            n_val = int(np.searchsorted(cumulative, (config.TEST_SIZE + config.VAL_SIZE) * totals[0])) + 1
            test, val, train = np.split(order, [n_test, max(n_val, n_test + 1)])
        else:
            test, val, train = np.split(order, [test_groups, test_groups + val_groups])
        parts = (train, val, test)
        actual = np.array([counts[part].sum(axis=0) for part in parts])
        has_quality = actual[:, quality_cols] > 0
        has_product = actual[:, product_cols] > 0
        coverage = has_quality.sum(axis=0)
        share_error = (actual - expected) / np.maximum(totals, 20)
        score = (
            int((~has_quality[0]).sum()),
            int((~has_product[0]).sum()),
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




# --- Programa ---------------------------------------------------------------------------------


def load_candidates(manifest: Path) -> tuple[list[dict[str, str]], list[str], int]:
    """
    Filas candidatas del manifiesto, sin los grupos apartados en la revisión visual.

    Retorna:
        (filas, columnas del manifiesto, cuántas filas se apartaron).
    """
    with config.SEGMENTATION_EXCLUSIONS.open(encoding="utf-8", newline="") as file:
        excluded_files = {row["reviewed_file"] for row in csv.DictReader(file)}
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames)
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
    return rows, columns, len(candidates) - len(rows)


def make_split(rows: list[dict[str, str]], level: str = AGRI_LEVEL,
               trials: int = TRIALS) -> tuple[list[dict[str, str]], dict]:
    """
    Agrupa por sesión de AgriFreshNET, reparte y verifica.

    Retorna:
        (filas con group_original, group y split; veredicto de check()).
    """
    mapping = merge_groups(rows, level)
    grouped = [dict(row, group_original=row["group"], group=mapping[row["group"]]) for row in rows]
    assignment = assign_groups(grouped, trials=trials, cut="files")
    verdict = check(grouped, assignment)
    return [dict(row, split=assignment[row["group"]]) for row in grouped], verdict


def load_phone_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    """Fotos de celular de la mitad "entrenar" (si ya se etiquetaron), todas para train."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        phone = [row for row in csv.DictReader(file) if row.get("filename")]
    return [dict({column: row.get(column, "") for column in columns},
                 group_original=row["group"], split="train") for row in phone]


def print_summary(rows: list[dict[str, str]]) -> None:
    for partition in PARTITIONS:
        subset = [row for row in rows if row["split"] == partition]
        print(f"{partition}: {len(subset)} recortes, {len({r['group'] for r in subset})} grupos")
        for name, field in (("calidad", "quality"), ("fuente", "source"), ("fruta", "product")):
            counts = Counter(row[field] for row in subset)
            print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def run(manifest: Path = config.CROPS_DIR / "manifest.csv",
        output: Path = config.SPLITS_CSV) -> Path:
    """Escribe splits.csv. Falla si el reparto no cumple las reglas de check()."""
    rows, columns, excluded = load_candidates(manifest)
    rows, verdict = make_split(rows)
    if not verdict["ok"]:
        raise RuntimeError("El reparto no cumple las reglas: " + "; ".join(verdict["problems"]))
    phone_dir = manifest.parent.parent / config.PHONE_TRAIN_DIR.name
    phone = load_phone_rows(phone_dir / config.PHONE_TRAIN_CSV.name, columns)
    rows += phone
    output.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=output.parent, delete=False, mode="w", encoding="utf-8",
                              newline="")
    temporary = Path(file.name)
    try:
        with file:
            fields = [*columns, *(["group_original"] if "group_original" not in columns else []),
                      "split"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Reparto guardado en {output}")
    print(f"Recortes apartados por la revisión visual: {excluded}")
    print(f"Fotos de celular agregadas a train: {len(phone)}")
    print(f"Desvío máximo de proporciones por calidad: {verdict['deviation']:.3f}")
    print_summary(rows)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=config.CROPS_DIR / "manifest.csv")
    run(parser.parse_args().manifest)
