"""
Pruebas de las fotos de celular: ingesta, etiquetado ciego, etiquetas finales y evaluación.

Fase CRISP-DM: Preparación de datos / Evaluación
Uso: pytest tests/test_phone_test.py
"""
import csv

import cv2
import numpy as np
import pytest
from PIL import Image

from src import config
from src.evaluation import phone_test


def _photo(path, color=(255, 140, 0), size=(400, 300), center=(200, 150), radius=70,
           date="2026:11:20 10:00:00", orientation=None):
    """Foto RGB de una fruta sobre fondo claro, con EXIF de fecha y, opcional, orientación."""
    image = np.full((size[1], size[0], 3), 230, np.uint8)
    cv2.circle(image, center, radius, color, -1)
    pil = Image.fromarray(image)
    exif = pil.getexif()
    exif[0x010F], exif[0x0110] = "Apple", "iPhone 13"
    exif.get_ifd(0x8769)[0x9003] = date
    if orientation:
        exif[0x0112] = orientation
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, exif=exif)


@pytest.fixture()
def pool(tmp_path):
    raw = tmp_path / "raw"
    for product, color in (("Orange", (255, 140, 0)), ("Banana", (240, 220, 40))):
        for day in (18, 19, 20, 21):
            for k in range(3):
                _photo(raw / product / f"{product}_{day}_{k}.jpg", color=color,
                       date=f"2026:11:{day} 10:0{k}:00", center=(180 + 10 * k, 150))
    _photo(raw / "Mango" / "m1.jpg", color=(250, 180, 30), radius=60)
    _photo(raw / "Orange" / "two.jpg", center=(120, 150), radius=50)
    image = cv2.imread(str(raw / "Orange" / "two.jpg"))
    cv2.circle(image, (300, 150), 50, (0, 140, 255), -1)
    cv2.imwrite(str(raw / "Orange" / "two.jpg"), image)
    out = tmp_path / "external"
    phone_test.ingest(raw, out)
    return out, phone_test.read_manifest(out)


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rot.jpg"
    _photo(path, size=(400, 300), orientation=6)  # rotar 90°: la imagen mostrada es vertical
    image, meta = phone_test.read_photo(path)
    assert image.shape[:2] == (400, 300)
    assert meta["model"] == "iPhone 13" and meta["datetime"].startswith("2026:11:20")


def test_ingest_builds_blind_manifest(pool):
    out, rows = pool
    assert len(rows) == 26
    assert all(r["blind_id"].startswith("EXT-") and len(r["blind_id"]) == 12 for r in rows)
    assert "partition" not in rows[0]
    assert {r["product"]: r["in_scope"] for r in rows}["mango"] == "False"
    assert len({r["session"] for r in rows if r["product"] == "orange"}) >= 4  # una por día
    two = next(r for r in rows if r["original"] == "Orange/two.jpg")
    assert two["status"] != "ok" and two["crop"] == ""
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok and all((out / r["crop"]).is_file() and (out / "preview" / f"{r['blind_id']}.jpg").is_file()
                      for r in ok)


def test_ingest_refuses_to_run_twice(pool, tmp_path):
    out, _ = pool
    with pytest.raises(FileExistsError):
        phone_test.ingest(tmp_path / "raw", out)


def test_sheets_are_blind_and_only_known_fruits(pool):
    out, rows = pool
    phone_test.sheets(out)
    test_ids = {r["blind_id"] for r in rows if r["status"] == "ok" and r["in_scope"] == "True"}
    for annotator in ("A", "B"):
        with (out / "labeling" / f"annotator_{annotator}.csv").open(encoding="utf-8") as file:
            sheet = list(csv.DictReader(file))
        assert {r["blind_id"] for r in sheet} == test_ids
        assert set(sheet[0]) == {"blind_id", "product", "quality", "size", "unusable", "notes"}
        page = (out / "labeling" / f"sheet_{annotator}.html").read_text(encoding="utf-8")
        assert ".jpg" in page and not any(r["original"] in page for r in rows)
    order_a = [r["blind_id"] for r in csv.DictReader((out / "labeling/annotator_A.csv").open(encoding="utf-8"))]
    order_b = [r["blind_id"] for r in csv.DictReader((out / "labeling/annotator_B.csv").open(encoding="utf-8"))]
    assert order_a != order_b


def _fill(path, labels):
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["quality"] = labels(row["blind_id"])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_single_annotator_is_enough(pool):
    out, _ = pool
    phone_test.sheets(out)
    lab = out / "labeling"
    ids = [r["blind_id"] for r in _fill(lab / "annotator_A.csv", lambda i: "good")]
    path = phone_test.finalize(lab, out)
    final = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["blind_id"] for r in final] == sorted(ids)
    assert {r["source_of_label"] for r in final} == {"A"}
    assert len(phone_test.evaluation_rows(out)) == len(ids)


def test_two_annotators_kappa_and_adjudication(pool):
    out, _ = pool
    phone_test.sheets(out)
    lab = out / "labeling"
    ids = [r["blind_id"] for r in _fill(lab / "annotator_A.csv", lambda i: "good")]
    disagree = ids[0]
    _fill(lab / "annotator_B.csv", lambda i: "bad" if i == disagree else "good")
    result = phone_test.agreement(lab)
    assert result["disagreements"] == 1 and result["n_both"] == len(ids)
    with pytest.raises(ValueError, match="adjudicar"):
        phone_test.finalize(lab, out)
    _fill(lab / "to_adjudicate.csv", lambda i: "regular")
    final = list(csv.DictReader(phone_test.finalize(lab, out).open(encoding="utf-8")))
    resolved = next(r for r in final if r["blind_id"] == disagree)
    assert resolved["quality"] == "regular" and resolved["source_of_label"] == "adjudicated"


def test_evaluation_requires_confirmation():
    with pytest.raises(ValueError, match="confirmo"):
        phone_test.evaluate(confirm=False)


def test_session_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = np.asarray(config.QUALITY_CLASSES * 40)
    pred = np.where(rng.random(len(y)) < 0.8, y, "regular")
    sessions = np.asarray([f"s{i % 12}" for i in range(len(y))])
    lo, hi = phone_test.session_bootstrap(y, pred, sessions, phone_test.quality_f1, n=300)
    from sklearn.metrics import f1_score
    point = f1_score(y, pred, labels=config.QUALITY_CLASSES, average="macro")
    assert lo <= point <= hi


def test_heic_without_pillow_heif_explains_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_test, "_register_heif", lambda: False)
    with pytest.raises(RuntimeError, match="pillow-heif"):
        phone_test.read_photo(tmp_path / "x.HEIC")
