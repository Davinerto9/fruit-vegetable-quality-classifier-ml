"""
Sesiones de captura de AgriFreshNET deducidas del nombre de archivo.

Fase CRISP-DM: Preparación de datos
Uso:
    from src.data.agri_sessions import session_key, LEVELS
    session_key("data/raw/agrifreshnet/.../aug_48_IMG_20251102_080441757_HDR_AE.jpg",
                "orange", "day")      # -> "agrifreshnet:orange:day:20251102"

Por qué existe: AgriFreshNET es una serie temporal. Las carpetas "Fresh Orange(1-9)",
"Semi fresh Orange(9-20)" y "Rotten Orange(24-35)" indican días desde la compra: las mismas
piezas se fotografían varias veces mientras se deterioran. Los grupos de index.csv solo unen
una foto con sus copias aumentadas, así que fotos del mismo minuto, del mismo día o de la
misma pieza en días seguidos pueden caer en particiones distintas. Este módulo da la clave
de sesión con la que se comprueba esa fuga (experiments/diagnostics/agri_leak_ladder.py) y
se agrupa el split v3 (src/data/split_v3.py).

Niveles, de más fino a más grueso:
    photo   el grupo vigente (foto original + aumentos); no agrega nada
    minute  misma fecha y misma hora:minuto
    day     mismo producto y mismo día
    week    mismo producto y misma semana ISO (aproxima "la misma pieza en días seguidos")

Formatos reconocidos (sin el prefijo aug_N_ ni el sufijo ~N):
    IMG_20251102_080441.jpg, IMG_20251102_080441757_HDR_AE.jpg   fecha y hora
    IMG-20260207-WA0007.jpg                                       solo fecha (WhatsApp)
    WhatsApp Image 2026-01-16 at 10.21.05.jpeg                    fecha y hora
Los nombres sin fecha ("Banana (9).jpeg") no tienen clave: conservan su grupo.
"""
from __future__ import annotations

import re
from datetime import date

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
        identificador del componente, con prefijo "v3:" si el componente une dos o más grupos.
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
        name = groups[0] if len(groups) == 1 else "v3:" + groups[0]
        for group in groups:
            mapping[group] = name
    return mapping
