"""
Prueba final con fotos de celular del pool de la clase, etiquetadas a ciegas.

Fase CRISP-DM: Evaluación
Uso (en orden):
    # 0) descargar la carpeta de Drive "Photos of Fruits and Vegetables | APO III" en
    #    data/external/raw/<Fruta>/... (una carpeta por fruta)
    python -m src.main celular ingerir     # recorta cada foto y le da un código ciego
    python -m src.main celular hojas       # hojas para etiquetar el estado, sin el modelo
    #    llenar data/external/labeling/annotator_A.csv (B es opcional: otra persona)
    python -m src.main celular acuerdo     # solo si hay B: kappa y desacuerdos a resolver
    python -m src.main celular finalizar   # etiqueta final de cada foto
    python -m src.main celular evaluar --confirmo   # UNA sola vez

Por qué así
- La fruta se sabe por la carpeta de Drive (Apple, Banana, ...): esa etiqueta es gratis.
- El estado (good / regular / bad) hay que ponerlo a mano para poder medir. Se hace a CIEGAS:
  cada foto tiene un código EXT-XXXXXXXX y la hoja no muestra nada del modelo, para que lo
  que diga la app no influya en quien etiqueta.
- Ninguna foto de celular se usa para entrenar ni para ajustar nada: son una prueba externa
  de cómo funciona el modelo con fotos "de la vida real", y se evalúan una sola vez.
- Fotos de la misma sesión (misma cámara, mismo día) se parecen entre sí: el intervalo de
  confianza remuestrea sesiones completas.
- Frutas que el modelo no conoce (p. ej. mango) quedan fuera (in_scope=False).

Archivos (en data/external/, fuera de Git): manifest.csv, crops/, preview/, labeling/,
labels_final.csv. Resultado: results/evaluacion_celular.json.
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
FIELDS = ("blind_id", "original", "sha256", "product", "in_scope", "session",
          "status", "d_norm", "crop", "crop_mask", "width", "height")
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


def session_of(path: Path, meta: dict) -> str:
    """
    Sesión de captura: misma cámara y mismo día según EXIF. Sin EXIF (p. ej. reenviadas por
    WhatsApp), se usa la carpeta y la serie del nombre (IMG_31xx → bloque de 50).
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


def ingest(raw: Path, out: Path = EXTERNAL) -> Path:
    manifest_path = out / "manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"Ya existe {manifest_path}; la ingesta no se repite sobre el mismo pool.")
    photos = sorted(p for p in raw.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not photos:
        raise FileNotFoundError(f"No hay fotos en {raw}")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)
    rows, seen = [], {}
    for path in photos:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha in seen:
            print(f"Duplicado omitido: {path} = {seen[sha]}")
            continue
        seen[sha] = path
        product_name = path.parent.name.lower().strip()
        product = config.PRODUCT_NAMES.get(product_name, product_name)
        image, meta = read_photo(path)
        result = segmentation.process(image)
        scale = min(1.0, 1024 / max(image.shape[:2]))
        preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _write(out / "preview" / f"{blind_id(sha)}.jpg", preview, ".jpg")
        row = {"blind_id": blind_id(sha), "original": path.relative_to(raw).as_posix(), "sha256": sha,
               "product": product, "in_scope": str(product in config.PRODUCTS),
               "session": session_of(path, meta), "status": result["status"],
               "d_norm": "" if result.get("d_norm") is None else f"{result['d_norm']:.4f}",
               "crop": "", "crop_mask": "", "width": image.shape[1], "height": image.shape[0]}
        if result["status"] == "ok":
            crop, mask = segmentation.model_inputs(result)
            row["crop"], row["crop_mask"] = f"crops/{row['blind_id']}.png", f"crops/{row['blind_id']}_mask.png"
            _write(out / row["crop"], crop)
            _write(out / row["crop_mask"], mask)
        rows.append(row)

    with manifest_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by = Counter((r["product"], r["status"] == "ok") for r in rows)
    print(f"{len(rows)} fotos ingeridas en {manifest_path}")
    for (product, ok), n in sorted(by.items()):
        print(f"  {product:12s} {'ok' if ok else 'rechazada':9s} {n}")
    return manifest_path


def read_manifest(out: Path = EXTERNAL) -> list[dict[str, str]]:
    with (out / "manifest.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def sheets(out: Path = EXTERNAL) -> None:
    """
    Hojas de etiquetado: A (obligatoria) y B (opcional, para medir el acuerdo), en orden
    aleatorio distinto, sin nombre original, sin sesión y sin ninguna salida del modelo. Solo
    fotos con segmentación ok de frutas que el modelo conoce.
    """
    rows = [r for r in read_manifest(out) if r["status"] == "ok" and r["in_scope"] == "True"]
    labeling = out / "labeling"
    labeling.mkdir(parents=True, exist_ok=True)
    for annotator, seed in (("A", 1), ("B", 2)):
        order = rows[:]
        random.Random(seed).shuffle(order)
        sheet = labeling / f"annotator_{annotator}.csv"
        if sheet.exists():
            raise FileExistsError(f"Ya existe {sheet}: no se regeneran hojas en uso.")
        with sheet.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("blind_id", "product", "quality", "size",
                                                      "unusable", "notes"))
            writer.writeheader()
            writer.writerows({"blind_id": r["blind_id"], "product": r["product"], "quality": "",
                              "size": "", "unusable": "", "notes": ""} for r in order)
        cards = "\n".join(
            f'<figure><img loading="lazy" src="../preview/{r["blind_id"]}.jpg">'
            f'<figcaption>{r["blind_id"]} · {html.escape(r["product"])}</figcaption></figure>'
            for r in order)
        (labeling / f"sheet_{annotator}.html").write_text(
            "<!DOCTYPE html><meta charset='utf-8'><title>Etiquetado ciego "
            f"{annotator}</title><style>body{{font-family:sans-serif}}figure{{display:inline-block;"
            "width:300px;margin:8px}}img{width:300px;height:300px;object-fit:contain;background:#eee}"
            "</style><p>Etiqueta calidad (good / regular / bad) y tamaño (small / medium / large) "
            "según docs/criterios-calidad.md, en annotator_" + annotator + ".csv. Marca unusable=1 si "
            "la foto no permite juzgar. No consultes al otro anotador ni la app.</p>" + cards,
            encoding="utf-8")
    print(f"Hojas en {labeling}: {len(rows)} fotos. Llena annotator_A.csv; annotator_B.csv es "
          "opcional (otra persona, sin mirar A) para reportar el acuerdo.")



LABELING = EXTERNAL / "labeling"
LOCK = config.RESULTS / "celular_evaluation.lock"
CLASSES = np.asarray(config.QUALITY_CLASSES)


def _read(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {row["blind_id"]: row for row in csv.DictReader(file)}


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
    return path


def evaluation_rows(out: Path = EXTERNAL) -> list[dict[str, str]]:
    labels = _read(out / "labels_final.csv")
    return [dict(r, quality=labels[r["blind_id"]]["quality"]) for r in read_manifest(out)
            if r["status"] == "ok" and r["in_scope"] == "True" and r["blind_id"] in labels]


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
    path = config.RESULTS / "evaluacion_celular.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lo, hi = report["fruit"]["accuracy_ci95"]
    print(f"Fruta: accuracy {report['fruit']['accuracy']:.3f} [{lo:.3f}; {hi:.3f}]")
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
