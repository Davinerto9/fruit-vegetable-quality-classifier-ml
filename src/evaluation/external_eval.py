"""
Fotos de celular: etiquetas finales y evaluación única del modelo congelado.

Fase CRISP-DM: Evaluación
Uso:
    # (opcional) si dos personas etiquetaron: acuerdo (kappa) y desacuerdos para resolver
    python -m src.evaluation.external_eval agreement
    # etiquetas finales: A sola, o A y B con los desacuerdos resueltos en to_adjudicate.csv
    python -m src.evaluation.external_eval finalize
    # evaluación, UNA sola vez (crea experiments/results/external_evaluation.lock)
    python -m src.evaluation.external_eval evaluate --confirm-external

Qué mide, con los modelos de config (APP_RF_CHECKPOINT, APP_CNN_CHECKPOINT,
APP_PRODUCT_CHECKPOINT):
    Fruta:  accuracy y F1 macro del Random Forest de fruta (etiqueta = carpeta de Drive).
    Estado: F1 macro, recall de "bad" y errores bad→good de RF, CNN y su promedio 50/50.
Con IC 95% por bootstrap de sesiones: fotos de la misma sesión no son independientes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src import config
from src.data.external_pool import EXTERNAL, read_manifest

LABELING = EXTERNAL / "labeling"
LOCK = config.RESULTS / "external_evaluation.lock"
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
    """IC 95% de score(y, pred) remuestreando sesiones completas."""
    rng = np.random.default_rng(seed)
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    values = [score(y[take], pred[take]) for take in
              (np.concatenate([index[s] for s in rng.choice(unique, len(unique))]) for _ in range(n))]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


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
    import joblib

    from src.features.classic import extract_classic_features
    from src.training.train_cnn import _dataset, _tensorflow

    if not confirm:
        raise ValueError("La evaluación es única: agrega --confirm-external.")
    rows = evaluation_rows()
    if not rows:
        raise ValueError("No hay fotos evaluables (¿falta labels_final.csv?).")
    paths = {name: config.CHECKPOINTS / getattr(config, name) for name in
             ("APP_RF_CHECKPOINT", "APP_CNN_CHECKPOINT", "APP_PRODUCT_CHECKPOINT")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Faltan modelos: " + ", ".join(missing))
    tf = _tensorflow()
    rf = joblib.load(paths["APP_RF_CHECKPOINT"])["model"]
    product_rf = joblib.load(paths["APP_PRODUCT_CHECKPOINT"])["model"]
    cnn = tf.keras.models.load_model(paths["APP_CNN_CHECKPOINT"], compile=False)
    _acquire_lock()  # todo lo que podía fallar ya cargó

    X = np.stack([extract_classic_features(EXTERNAL / r["crop"], EXTERNAL / r["crop_mask"]) for r in rows])
    order = [list(rf.classes_).index(c) for c in CLASSES]
    p_rf = rf.predict_proba(X)[:, order]
    # _dataset une CROPS_DIR / ruta; con una ruta absoluta, pathlib usa la absoluta.
    ds_rows = [dict(r, crop=str((EXTERNAL / r["crop"]).resolve()),
                    crop_mask=str((EXTERNAL / r["crop_mask"]).resolve())) for r in rows]
    p_cnn = cnn.predict(_dataset(tf, ds_rows, {c: i for i, c in enumerate(CLASSES)}, 64), verbose=0)
    w = config.APP_ENSEMBLE_CNN_WEIGHT
    y = np.asarray([r["quality"] for r in rows])
    sessions = np.asarray([r["session"] for r in rows])
    y_product = np.asarray([r["product"] for r in rows])
    product_pred = product_rf.predict(X).astype(str)

    report = {
        "n": len(rows), "sessions": int(len(np.unique(sessions))),
        "class_counts": dict(Counter(y.tolist())), "products": dict(Counter(y_product.tolist())),
        "fruit": {"accuracy": float(accuracy_score(y_product, product_pred)),
                  "accuracy_ci95": session_bootstrap(y_product, product_pred, sessions, accuracy_score),
                  "f1_macro": float(f1_score(y_product, product_pred, average="macro", zero_division=0))},
        "quality": {"random_forest": summarize(y, CLASSES[p_rf.argmax(1)], sessions),
                    "cnn": summarize(y, CLASSES[p_cnn.argmax(1)], sessions),
                    "ensemble_50_50": summarize(y, CLASSES[((1 - w) * p_rf + w * p_cnn).argmax(1)], sessions)},
        "checkpoints": {k: str(v) for k, v in paths.items()},
    }
    path = config.RESULTS / "external_evaluation.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lo, hi = report["fruit"]["accuracy_ci95"]
    print(f"Fruta: accuracy {report['fruit']['accuracy']:.3f} [{lo:.3f}; {hi:.3f}]")
    for name, item in report["quality"].items():
        lo, hi = item["f1_macro_ci95"]
        print(f"Estado, {name:>15}: F1 {item['f1_macro']:.3f} [{lo:.3f}; {hi:.3f}] | bad→good {item['bad_to_good']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("agreement")
    sub.add_parser("finalize")
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--confirm-external", action="store_true")
    args = parser.parse_args()
    if args.command == "agreement":
        agreement()
    elif args.command == "finalize":
        finalize()
    else:
        evaluate(args.confirm_external)
