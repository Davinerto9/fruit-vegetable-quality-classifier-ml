"""
Fotos de celular del pool de la clase: la mitad entrena el modelo y la otra mitad es la
prueba final, etiquetada a ciegas.

Fase CRISP-DM: Preparación de datos y Evaluación
Uso (en orden):
    # 0) descargar la carpeta de Drive "Photos of Fruits and Vegetables | APO III" en
    #    data/external/raw/<Fruta>/... (una carpeta por fruta)
    python -m src.main celular ingerir     # recorta cada foto, le da un código ciego y la
                                           # asigna a la mitad "entrenar" o "prueba"
    python -m src.main celular hojas       # hojas para etiquetar el estado, sin el modelo
    #    llenar data/external/labeling/annotator_A.csv (B es opcional: otra persona)
    python -m src.main celular acuerdo     # solo si hay B: kappa y desacuerdos a resolver
    python -m src.main celular finalizar   # etiqueta final y pasa la mitad "entrenar" al
                                           # entrenamiento (data/processed/celular/)
    python -m src.main dividir; python -m src.main atributos; python -m src.main entrenar
    python -m src.main celular evaluar --confirmo   # la mitad "prueba", UNA sola vez

Por qué así
- La fruta se sabe por la carpeta de Drive (Apple, Banana, ...): esa etiqueta es gratis.
- Las dos mitades se separan por SESIÓN, no foto por foto: misma fruta, misma cámara y fotos con
  menos de 10 minutos entre sí. Son casi iguales, y si quedaran una en cada mitad la prueba
  saldría inflada. Una fruta con una sola sesión queda entera en "prueba".
- Las fotos idénticas a alguna de data/own/ (que ya entrena) se apartan al ingerir.
  En cada fruta se reparten las sesiones al azar (semilla fija) hasta tener ≈50 % de las fotos
  en cada mitad. Se hace al ingerir, ANTES de etiquetar y sin mirar ningún resultado.
- La mitad "entrenar" se suma a train: el modelo aprende cómo se ven las fotos de celular.
- La mitad "prueba" no se usa para entrenar ni para ajustar nada y se evalúa una sola vez. Es
  la medida más cercana al uso real, porque el fondo ya no delata la fuente.
- El estado (good / regular / bad) se pone a mano en las dos mitades, a CIEGAS: cada foto tiene
  un código EXT-XXXXXXXX y la hoja no dice a qué mitad pertenece ni qué opina el modelo.
- Como se toman varias fotos seguidas de la misma pieza, la hoja va en orden de toma y basta
  escribir el estado en la primera foto de cada pieza; las vacías copian la anterior.
- El intervalo de confianza remuestrea sesiones completas.
- Frutas que el modelo no conoce (p. ej. kiwi) quedan fuera (in_scope=False).

Archivos (en data/external/, fuera de Git): manifest.csv (con la columna half), crops/,
preview/, labeling/, labels_final.csv. Resultado: results/evaluacion_celular.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import random
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src import config
from src.data import segmentation
from src.evaluation.evaluate_models import group_bootstrap

EXTERNAL = config.ROOT / "data" / "external"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
FIELDS = ("blind_id", "original", "sha256", "product", "in_scope", "session", "half",
          "status", "d_norm", "crop", "crop_mask", "width", "height")
HALVES = ("entrenar", "prueba")
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


# Dos fotos de la misma fruta y la misma cámara separadas por más de esto son sesiones distintas.
SESSION_GAP_MINUTES = 10


def session_of(path: Path, meta: dict) -> str:
    """
    Sesión base: carpeta de la fruta, cámara y día según EXIF. Sin EXIF (p. ej. reenviadas por
    WhatsApp), se usa la carpeta y la serie del nombre (IMG_31xx → bloque de 50).
    split_by_time() la parte después en bloques de tiempo.
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


def split_by_time(rows: list[dict], times: dict[str, str],
                  gap_minutes: int = SESSION_GAP_MINUTES) -> None:
    """
    Parte cada sesión base en bloques: una pausa de más de gap_minutes abre un bloque nuevo
    (":b1", ":b2", ...). Así, una misma persona que fotografía la misma fruta a las 10:37 y
    otra vez a las 11:48 deja dos sesiones, y las fotos de un mismo bloque (segundos o minutos
    de diferencia, casi iguales) quedan siempre juntas.

    Parámetros:
        rows: filas del manifiesto; se modifica su campo "session".
        times: blind_id -> "AAAA:MM:DD HH:MM:SS" (EXIF). Las filas sin hora no se tocan.
    """
    def minutes(stamp: str) -> float:
        moment = datetime.strptime(stamp[:19], "%Y:%m:%d %H:%M:%S")
        return moment.timestamp() / 60

    by_session: dict[str, list[dict]] = {}
    for row in rows:
        if times.get(row["blind_id"]):
            by_session.setdefault(row["session"], []).append(row)
    for base, members in by_session.items():
        members.sort(key=lambda r: times[r["blind_id"]])
        block, previous = 1, None
        for row in members:
            now = minutes(times[row["blind_id"]])
            if previous is not None and now - previous > gap_minutes:
                block += 1
            row["session"] = f"{base}:b{block}"
            previous = now


def own_hashes() -> dict[str, str]:
    """SHA-256 de las fotos de data/own/, que ya entrenan: no pueden estar en el pool."""
    folder = config.ROOT / "data" / "own"
    return {hashlib.sha256(path.read_bytes()).hexdigest(): path.name
            for path in sorted(folder.glob("*")) if path.suffix.lower() in EXTENSIONS}


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


def assign_halves(rows: list[dict], seed: int = config.SEED,
                  start: Counter | None = None) -> dict[str, str]:
    """
    Asigna cada sesión completa a "entrenar" o "prueba", fruta por fruta.

    Las sesiones de cada fruta se barajan con semilla fija y cada una va a la mitad que lleva
    menos fotos de esa fruta (empezando por "prueba", para que una fruta con una sola sesión
    quede para medir). Solo usa fruta y sesión: ninguna etiqueta de estado.

    Parámetros:
        start: fotos que ya tiene cada (fruta, mitad) de una ingesta anterior, para que las
            sesiones nuevas equilibren contando con ellas.

    Retorna:
        dict sesión -> mitad.
    """
    rng = random.Random(seed)
    start = start or Counter()
    halves = {}
    for product in sorted({r["product"] for r in rows}):
        sizes = Counter(r["session"] for r in rows if r["product"] == product)
        sessions = sorted(sizes)
        rng.shuffle(sessions)
        count = {half: start[(product, half)] for half in ("prueba", "entrenar")}
        for session in sessions:
            half = min(("prueba", "entrenar"), key=lambda h: count[h])
            halves[session] = half
            count[half] += sizes[session]
    return halves


def ingest(raw: Path, out: Path = EXTERNAL) -> Path:
    """
    Recorta cada foto nueva, le da un código ciego y asigna su sesión a una mitad.

    Si ya hubo una ingesta, solo procesa las fotos que no estén en manifest.csv (se reconocen
    por su contenido): las fotos anteriores conservan su código, su sesión y su mitad, y las
    sesiones nuevas se reparten contando con lo que ya hay. No se permite después de evaluar.
    """
    manifest_path = out / "manifest.csv"
    if LOCK.exists():
        raise RuntimeError("Las fotos de celular ya se evaluaron: no se agregan más.")
    old_rows = read_manifest(out) if manifest_path.exists() else []
    known = {r["sha256"] for r in old_rows}
    photos = sorted(p for p in raw.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not photos:
        raise FileNotFoundError(f"No hay fotos en {raw}")
    photos = [p for p in photos if hashlib.sha256(p.read_bytes()).hexdigest() not in known]
    if old_rows:
        print(f"Ya había {len(old_rows)} fotos ingeridas; fotos nuevas: {len(photos)}.")
    if not photos:
        return manifest_path
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)
    rows, seen, times, own = [], {}, {}, own_hashes()
    for path in photos:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha in seen:
            print(f"Duplicado omitido: {path} = {seen[sha]}")
            continue
        seen[sha] = path
        product_name = path.parent.name.lower().strip()
        product = config.PRODUCT_NAMES.get(product_name, product_name)
        if sha in own:
            # La misma foto ya está en data/own/ y entrena: si cayera en la prueba sería trampa.
            rows.append({"blind_id": blind_id(sha), "original": path.relative_to(raw).as_posix(),
                         "sha256": sha, "product": product,
                         "in_scope": str(product in config.PRODUCTS), "session": "", "half": "",
                         "status": f"duplicate: data/own/{own[sha]}", "d_norm": "", "crop": "",
                         "crop_mask": "", "width": "", "height": ""})
            continue
        image, meta = read_photo(path)
        times[blind_id(sha)] = meta.get("datetime", "")
        result = segmentation.process(image)
        scale = min(1.0, 1024 / max(image.shape[:2]))
        preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _write(out / "preview" / f"{blind_id(sha)}.jpg", preview, ".jpg")
        row = {"blind_id": blind_id(sha), "original": path.relative_to(raw).as_posix(), "sha256": sha,
               "product": product, "in_scope": str(product in config.PRODUCTS),
               "session": session_of(path, meta), "half": "", "status": result["status"],
               "d_norm": "" if result.get("d_norm") is None else f"{result['d_norm']:.4f}",
               "crop": "", "crop_mask": "", "width": image.shape[1], "height": image.shape[0]}
        if result["status"] == "ok":
            crop, mask = segmentation.model_inputs(result)
            row["crop"], row["crop_mask"] = f"crops/{row['blind_id']}.png", f"crops/{row['blind_id']}_mask.png"
            _write(out / row["crop"], crop)
            _write(out / row["crop_mask"], mask)
        rows.append(row)

    split_by_time(rows, times)
    old_sessions = {r["session"] for r in old_rows if r["session"]}
    for row in rows:
        if row["session"] in old_sessions:  # mismo nombre que una sesión anterior: es otra
            row["session"] += ":nueva"
    usable = [r for r in rows if r["status"] == "ok" and r["in_scope"] == "True"]
    before = Counter((r["product"], r["half"]) for r in old_rows if r["half"])
    halves = assign_halves(usable, start=before)
    for row in usable:
        row["half"] = halves[row["session"]]
    temporary = manifest_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in FIELDS} for r in old_rows] + rows)
    temporary.replace(manifest_path)
    def state(row):
        return "ok" if row["status"] == "ok" else (
            "ya en data/own" if row["status"].startswith("duplicate") else "rechazada")
    by = Counter((r["product"], state(r)) for r in rows)
    print(f"{len(rows)} fotos ingeridas en {manifest_path}")
    for (product, name), n in sorted(by.items()):
        print(f"  {product:12s} {name:15s} {n}")
    everything = [r for r in old_rows + rows if r["status"] == "ok" and r["in_scope"] == "True"]
    split = Counter((r["product"], r["half"]) for r in everything)
    sessions = Counter(r["product"] for r in {r["session"]: r for r in everything}.values())
    print("Mitades con todas las fotos (fotos):")
    for product in sorted({r["product"] for r in everything}):
        print(f"  {product:12s} {sessions[product]:2d} sesiones | entrenar "
              f"{split[(product, 'entrenar')]:4d} | prueba {split[(product, 'prueba')]:4d}")
    return manifest_path


def read_manifest(out: Path = EXTERNAL) -> list[dict[str, str]]:
    with (out / "manifest.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


SHEET_FIELDS = ("n", "blind_id", "product", "quality", "unusable", "notes")


def _sheet_has_labels(path: Path) -> bool:
    return path.is_file() and any(_quality(row) or row.get("unusable", "").strip()
                                  for row in _read_rows(path))


def sheets(out: Path = EXTERNAL) -> None:
    """
    Hojas de etiquetado: A (obligatoria) y B (opcional, para medir el acuerdo).

    Las fotos van en el orden en que se tomaron, agrupadas por fruta y numeradas (#1, #2, ...).
    Como se toman varias fotos seguidas de la misma pieza, basta escribir el estado en la
    PRIMERA foto de cada pieza: una casilla vacía significa "igual que la foto anterior de la
    misma fruta" (finalize() la rellena). No se muestra nombre original, sesión, mitad ni nada
    del modelo. Solo fotos con segmentación ok de frutas que el modelo conoce.

    Si una hoja ya tiene etiquetas (de este formato o del anterior), se rehace conservando cada
    etiqueta en su foto y se guarda una copia; así se pueden agregar fotos nuevas después.
    """
    rows = [r for r in read_manifest(out) if r["status"] == "ok" and r["in_scope"] == "True"]
    rows.sort(key=lambda r: (r["product"], r["original"]))  # IMG_3175, IMG_3176...: orden de toma
    labeling = out / "labeling"
    labeling.mkdir(parents=True, exist_ok=True)
    for annotator in ("A", "B"):
        sheet = labeling / f"annotator_{annotator}.csv"
        previous: dict[str, dict[str, str]] = {}
        new_ids: set[str] = set()
        if _sheet_has_labels(sheet):
            old_rows = _read_rows(sheet)
            previous = {r["blind_id"]: r for r in old_rows}
            name = "formato_anterior" if "n" not in old_rows[0] else "respaldo"
            backup = labeling / f"annotator_{annotator}_{name}.csv"
            backup.write_bytes(sheet.read_bytes())
            kept = sum(1 for r in old_rows if _quality(r) or r.get("unusable"))
            new_ids = {r["blind_id"] for r in rows} - set(previous)
            print(f"Hoja {annotator}: se conservan {kept} etiquetas (copia en {backup.name}); "
                  f"fotos nuevas: {len(new_ids)}.")
        with sheet.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SHEET_FIELDS)
            writer.writeheader()
            first_new, last = [], None
            for i, r in enumerate(rows, start=1):
                old = previous.get(r["blind_id"], {})
                writer.writerow({"n": i, "blind_id": r["blind_id"], "product": r["product"],
                                 "quality": _quality(old), "unusable": old.get("unusable", ""),
                                 "notes": old.get("notes", "")})
                is_new = r["blind_id"] in new_ids
                if is_new and last != (r["product"], True):
                    first_new.append(f"#{i} ({r['product']})")
                last = (r["product"], is_new)
            if first_new:
                print(f"  Las fotos nuevas empiezan en {', '.join(first_new)}: escribe el estado "
                      "en esa foto aunque sea igual al de la anterior.")
        sections, current = [], None
        for i, r in enumerate(rows, start=1):
            if r["product"] != current:
                current = r["product"]
                sections.append(f"<h2>{html.escape(current)}</h2>")
            sections.append(f'<figure><img loading="lazy" src="../preview/{r["blind_id"]}.jpg">'
                            f'<figcaption><b>#{i}</b> · {r["blind_id"]}</figcaption></figure>')
        (labeling / f"sheet_{annotator}.html").write_text(
            "<!DOCTYPE html><meta charset='utf-8'><title>Etiquetado ciego "
            f"{annotator}</title><style>body{{font-family:sans-serif;max-width:1400px}}"
            "figure{display:inline-block;width:210px;margin:4px}img{width:210px;height:210px;"
            "object-fit:contain;background:#eee}figcaption{font-size:14px}h2{margin-top:28px}"
            "</style><h1>Etiquetado ciego " + annotator + "</h1><p>Las fotos están en el orden "
            "en que se tomaron. En <b>annotator_" + annotator + ".csv</b> escribe en la columna "
            "<b>quality</b> <code>good</code>, <code>regular</code> o <code>bad</code> "
            "(docs/criterios-calidad.md) <b>solo en la primera foto de cada pieza nueva</b> (cada "
            "manzana, limón... distinto): una casilla vacía significa «igual que la foto anterior». "
            "Si la pieza cambia y tiene el mismo estado, igual escríbelo: así queda claro. Escribe <code>1</code> en "
            "<b>unusable</b> si una foto no deja juzgar. No abras la app ni mires la hoja de la "
            "otra persona.</p>" + "".join(sections), encoding="utf-8")
    print(f"Hojas en {labeling}: {len(rows)} fotos en orden de toma. En annotator_A.csv escribe "
          "el estado solo en la primera foto de cada pieza (las vacías copian la anterior). "
          "annotator_B.csv es opcional: otra persona, sin mirar A, para medir el acuerdo.")


LABELING = EXTERNAL / "labeling"
LOCK = config.RESULTS / "celular_evaluation.lock"
CLASSES = np.asarray(config.QUALITY_CLASSES)


def _read_rows(path: Path) -> list[dict[str, str]]:
    """
    Lee un CSV aunque Excel lo haya guardado con ";" o en codificación de Windows.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    first = text.splitlines()[0] if text else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in rows]


def fill_down(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    En una hoja de etiquetado, una casilla de quality vacía toma el valor de la foto anterior
    de la misma fruta (orden de la columna n). La primera foto de cada fruta debe tenerlo.
    """
    rows = sorted(rows, key=lambda r: int(r["n"]))
    last: dict[str, str] = {}
    for row in rows:
        if row.get("unusable"):
            continue
        quality = row.get("quality", "").lower()
        if quality:
            last[row["product"]] = quality
        elif row["product"] in last:
            row["quality"] = last[row["product"]]
        else:
            raise ValueError(f"La foto #{row['n']} es la primera de {row['product']} y no tiene "
                             "estado: escríbelo ahí.")
    return rows


def _read(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows = _read_rows(path)
    if rows and "n" in rows[0] and any(r.get("quality") for r in rows):
        rows = fill_down(rows)
    return {row["blind_id"]: row for row in rows}


def _quality(row: dict[str, str]) -> str:
    return row.get("quality", "").strip().lower()


def agreement(labeling: Path = LABELING) -> dict:
    """Kappa de Cohen entre A y B y lista de desacuerdos (to_adjudicate.csv)."""
    a, b = _read(labeling / "annotator_A.csv"), _read(labeling / "annotator_B.csv")
    ids = [i for i in sorted(set(a) & set(b)) if _quality(a[i]) and _quality(b[i])
           and not (a[i].get("unusable") or b[i].get("unusable"))]
    if not ids:
        raise ValueError("No hay fotos etiquetadas por A y por B.")
    qa, qb = [_quality(a[i]) for i in ids], [_quality(b[i]) for i in ids]
    invalid = sorted({q for q in qa + qb if q not in config.QUALITY_CLASSES})
    if invalid:
        raise ValueError(f"Etiquetas inválidas: {invalid} (usa good, regular o bad).")
    kappa = cohen_kappa_score(qa, qb, labels=config.QUALITY_CLASSES)
    disagreements = [i for i, x, y in zip(ids, qa, qb) if x != y]
    with (labeling / "to_adjudicate.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("blind_id", "product", "annotator_A", "annotator_B", "quality"))
        writer.writeheader()
        for i in disagreements:
            writer.writerow({"blind_id": i, "product": a[i]["product"], "annotator_A": _quality(a[i]),
                             "annotator_B": _quality(b[i]), "quality": ""})
    result = {"n_both": len(ids), "kappa": float(kappa),
              "agreement": 1 - len(disagreements) / len(ids), "disagreements": len(disagreements)}
    (labeling / "agreement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"κ = {kappa:.2f}; acuerdo {result['agreement']:.0%}; "
          f"{len(disagreements)} desacuerdos en to_adjudicate.csv")
    return result


def finalize(labeling: Path = LABELING, out: Path = EXTERNAL) -> Path:
    """
    Etiqueta final por foto: la de A; si B también etiquetó y no coincide, la resuelta en
    to_adjudicate.csv. Las marcadas unusable se omiten.
    """
    a, b = _read(labeling / "annotator_A.csv"), _read(labeling / "annotator_B.csv")
    adjudicated = _read(labeling / "to_adjudicate.csv")
    rows = []
    for i, row in sorted(a.items()):
        if row.get("unusable") or (i in b and b[i].get("unusable")):
            continue
        qa = _quality(row)
        if not qa:
            raise ValueError(f"{i} no tiene etiqueta de A.")
        qb = _quality(b[i]) if i in b else ""
        if qb and qb != qa:
            quality, origin = _quality(adjudicated.get(i, {})), "adjudicated"
            if not quality:
                raise ValueError(f"{i}: A y B no coinciden y sigue sin adjudicar (corre agreement).")
        else:
            quality, origin = qa, "A+B" if qb else "A"
        if quality not in config.QUALITY_CLASSES:
            raise ValueError(f"{i}: etiqueta inválida {quality!r}")
        rows.append({"blind_id": i, "quality": quality, "source_of_label": origin})
    path = out / "labels_final.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("blind_id", "quality", "source_of_label"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} etiquetas finales en {path}: {dict(Counter(r['quality'] for r in rows))}")
    export_training(out)
    return path


def labeled_rows(out: Path = EXTERNAL, half: str | None = None) -> list[dict[str, str]]:
    """Fotos con segmentación ok, fruta conocida y etiqueta final; opcionalmente de una mitad."""
    labels = _read(out / "labels_final.csv")
    return [dict(r, quality=labels[r["blind_id"]]["quality"]) for r in read_manifest(out)
            if r["status"] == "ok" and r["in_scope"] == "True" and r["blind_id"] in labels
            and (half is None or r["half"] == half)]


def export_training(out: Path = EXTERNAL, crops_dir: Path | None = None) -> Path:
    """
    Copia los recortes de la mitad "entrenar" a data/processed/celular/ (al lado de crops/)
    y escribe celular_train.csv, que src/data/split.py agrega a train. Cada sesión es un grupo.
    Las rutas quedan relativas a crops/ ("../celular/..."), como las del resto de recortes.
    """
    crops_dir = crops_dir or config.CROPS_DIR
    target = crops_dir.parent / config.PHONE_TRAIN_DIR.name
    (target / "images").mkdir(parents=True, exist_ok=True)
    (target / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in labeled_rows(out, "entrenar"):
        name = f"{r['blind_id']}.png"
        (target / "images" / name).write_bytes((out / r["crop"]).read_bytes())
        (target / "masks" / name).write_bytes((out / r["crop_mask"]).read_bytes())
        crop = f"../{target.name}/images/{name}"
        mask = f"../{target.name}/masks/{name}"
        rows.append({"filename": f"data/external/raw/{r['original']}", "product": r["product"],
                     "quality": r["quality"], "source": "celular", "size": "",
                     "group": f"celular:{r['session']}", "status": "labeled",
                     "decision": "candidate", "d_norm": r["d_norm"], "crop": crop,
                     "crop_mask": mask, "blind_id": r["blind_id"]})
    path = target / config.PHONE_TRAIN_CSV.name
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["filename"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} fotos de la mitad 'entrenar' pasan al entrenamiento ({path}). "
          "Ahora corre: dividir, atributos y entrenar.")
    return path


def check_halves_apart(out: Path = EXTERNAL, split_path: Path | None = None) -> None:
    """Ninguna foto ni sesión de la mitad "prueba" puede estar en el reparto de entrenamiento."""
    split_path = split_path or config.SPLITS_CSV
    test = labeled_rows(out, "prueba")
    if not split_path.is_file():
        return
    with split_path.open(encoding="utf-8-sig", newline="") as file:
        used = [r for r in csv.DictReader(file) if r.get("source") == "celular"]
    files = {r["filename"] for r in used}
    groups = {r["group"] for r in used}
    leaked = [r["blind_id"] for r in test if f"data/external/raw/{r['original']}" in files
              or f"celular:{r['session']}" in groups]
    if leaked:
        raise RuntimeError(f"{len(leaked)} fotos de la mitad 'prueba' están en el entrenamiento.")


def evaluation_rows(out: Path = EXTERNAL) -> list[dict[str, str]]:
    """Las fotos que se miden: solo la mitad "prueba"."""
    return labeled_rows(out, "prueba")


def session_bootstrap(y, pred, sessions, score, n=2000, seed=config.SEED) -> list[float]:
    """IC 95 % de score(y, pred) remuestreando sesiones completas."""
    return group_bootstrap(y, pred, sessions, score, n=n, seed=seed)


def quality_f1(y, pred):
    return f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)


def summarize(y, pred, sessions) -> dict:
    bad = y == "bad"
    return {"f1_macro": float(quality_f1(y, pred)),
            "f1_macro_ci95": session_bootstrap(y, pred, sessions, quality_f1),
            "recall_bad": float((pred[bad] == "bad").mean()) if bad.any() else None,
            "bad_to_good": int((pred[bad] == "good").sum())}


def _acquire_lock() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("Las fotos de celular ya se evaluaron; no se repite.") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat()}, file)


def evaluate(confirm: bool) -> dict:
    """Evalúa una sola vez fruta y estado sobre las fotos de celular etiquetadas."""
    from src.data.features import extract_classic_features
    from src.evaluation.evaluate_models import load_all
    from src.training.train_models import image_dataset, tensorflow

    if not confirm:
        raise ValueError("La evaluación es única: agrega --confirmo.")
    rows = evaluation_rows()
    if not rows:
        raise ValueError("No hay fotos evaluables (¿falta labels_final.csv?).")
    check_halves_apart()
    models = load_all()
    _acquire_lock()  # todo lo que podía fallar ya cargó

    X = np.stack([extract_classic_features(EXTERNAL / r["crop"], EXTERNAL / r["crop_mask"])
                  for r in rows])
    rf = models["random_forest"]
    order = [list(rf.classes_).index(c) for c in CLASSES]
    p_rf = rf.predict_proba(X)[:, order]
    p_cnn = models["cnn"].predict(image_dataset(tensorflow(), rows, 64, cache=False,
                                                base=EXTERNAL), verbose=0)
    w = config.APP_ENSEMBLE_CNN_WEIGHT
    y = np.asarray([r["quality"] for r in rows])
    sessions = np.asarray([r["session"] for r in rows])
    y_product = np.asarray([r["product"] for r in rows])
    product_pred = models["fruit_rf"].predict(X).astype(str)

    report = {
        "n": len(rows), "sessions": int(len(np.unique(sessions))),
        "class_counts": dict(Counter(y.tolist())), "products": dict(Counter(y_product.tolist())),
        "fruit": {"accuracy": float(accuracy_score(y_product, product_pred)),
                  "accuracy_ci95": session_bootstrap(y_product, product_pred, sessions,
                                                     accuracy_score),
                  "f1_macro": float(f1_score(y_product, product_pred, average="macro",
                                             zero_division=0))},
        "quality": {"random_forest": summarize(y, CLASSES[p_rf.argmax(1)], sessions),
                    "cnn": summarize(y, CLASSES[p_cnn.argmax(1)], sessions),
                    "ensemble": summarize(y, CLASSES[((1 - w) * p_rf + w * p_cnn).argmax(1)],
                                          sessions)},
    }
    # Por fruta, separando las que tuvieron fotos de celular en el entrenamiento: en esas, la
    # prueba puede incluir las mismas piezas fotografiadas en otro momento (más optimista).
    trained = {r["product"] for r in labeled_rows(half="entrenar")}
    final = {"random_forest": p_rf, "cnn": p_cnn, "ensemble": (1 - w) * p_rf + w * p_cnn}
    quality_pred = CLASSES[final[config.APP_MODEL].argmax(1)]
    report["final_model"] = config.APP_MODEL
    report["by_product"] = {
        product: {"n": int(mask.sum()), "phone_photos_in_training": product in trained,
                  "fruit_accuracy": float(accuracy_score(y_product[mask], product_pred[mask])),
                  "quality_f1_final": float(quality_f1(y[mask], quality_pred[mask]))}
        for product in sorted(set(y_product)) for mask in [y_product == product]}
    path = config.RESULTS / "evaluacion_celular.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lo, hi = report["fruit"]["accuracy_ci95"]
    print(f"Fruta: accuracy {report['fruit']['accuracy']:.3f} [{lo:.3f}; {hi:.3f}]")
    for product, item in report["by_product"].items():
        seen = "con fotos de celular en train" if item["phone_photos_in_training"] else "nunca vista en celular"
        print(f"  {product:10s} n={item['n']:3d} | fruta {item['fruit_accuracy']:.2f} | "
              f"estado F1 {item['quality_f1_final']:.2f} | {seen}")
    for name, item in report["quality"].items():
        lo, hi = item["f1_macro_ci95"]
        print(f"Estado, {name:>13}: F1 {item['f1_macro']:.3f} [{lo:.3f}; {hi:.3f}] | "
              f"malas como buenas {item['bad_to_good']}")
    return report


def run(command: str, raw: Path | None = None, confirm: bool = False):
    actions = {"ingerir": lambda: ingest(raw or EXTERNAL / "raw"), "hojas": sheets,
               "acuerdo": agreement, "finalizar": finalize, "evaluar": lambda: evaluate(confirm)}
    return actions[command]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paso", choices=("ingerir", "hojas", "acuerdo", "finalizar", "evaluar"))
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--confirmo", action="store_true")
    args = parser.parse_args()
    run(args.paso, args.raw, args.confirmo)
