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


@pytest.fixture(autouse=True)
def lock_aside(tmp_path, monkeypatch):
    """El candado real del repositorio (results/) no debe afectar a las pruebas."""
    monkeypatch.setattr(phone_test, "LOCK", tmp_path / "celular_evaluation.lock")


@pytest.fixture()
def crops(tmp_path, monkeypatch):
    """Carpeta de recortes del entrenamiento, temporal (finalizar copia ahí la mitad entrenar)."""
    path = tmp_path / "crops"
    path.mkdir()
    monkeypatch.setattr(config, "CROPS_DIR", path)
    monkeypatch.setattr(config, "SPLITS_CSV", path / "splits.csv")
    return path


@pytest.fixture()
def pool(tmp_path):
    raw = tmp_path / "raw"
    for product, color in (("Orange", (255, 140, 0)), ("Banana", (240, 220, 40))):
        for day in (18, 19, 20, 21):
            for k in range(3):
                _photo(raw / product / f"{product}_{day}_{k}.jpg", color=color,
                       date=f"2026:11:{day} 10:0{k}:00", center=(180 + 10 * k, 150))
    _photo(raw / "Kiwi" / "m1.jpg", color=(250, 180, 30), radius=60)
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
    assert {r["product"]: r["in_scope"] for r in rows}["kiwi"] == "False"
    assert len({r["session"] for r in rows if r["product"] == "orange"}) >= 4  # una por día
    two = next(r for r in rows if r["original"] == "Orange/two.jpg")
    assert two["status"] != "ok" and two["crop"] == ""
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok and all((out / r["crop"]).is_file() and (out / "preview" / f"{r['blind_id']}.jpg").is_file()
                      for r in ok)


def test_ingesting_again_changes_nothing(pool, tmp_path):
    out, before = pool
    phone_test.ingest(tmp_path / "raw", out)
    assert phone_test.read_manifest(out) == before


def test_sheets_are_blind_ordered_and_only_known_fruits(pool):
    out, rows = pool
    phone_test.sheets(out)
    test_ids = {r["blind_id"] for r in rows if r["status"] == "ok" and r["in_scope"] == "True"}
    for annotator in ("A", "B"):
        with (out / "labeling" / f"annotator_{annotator}.csv").open(encoding="utf-8") as file:
            sheet = list(csv.DictReader(file))
        assert {r["blind_id"] for r in sheet} == test_ids
        assert tuple(sheet[0]) == phone_test.SHEET_FIELDS
        assert [int(r["n"]) for r in sheet] == list(range(1, len(sheet) + 1))
        products = [r["product"] for r in sheet]
        assert products == sorted(products)  # agrupadas por fruta
        page = (out / "labeling" / f"sheet_{annotator}.html").read_text(encoding="utf-8")
        assert ".jpg" in page and "#1" in page
        assert not any(r["original"] in page or (r["half"] and r["half"] in page) for r in rows)


def test_rebuilding_sheets_keeps_every_label(pool):
    out, _ = pool
    phone_test.sheets(out)
    phone_test.sheets(out)  # vacías: se rehacen sin problema
    _fill(out / "labeling" / "annotator_A.csv", lambda i: "good")
    phone_test.sheets(out)  # con etiquetas: se rehace conservándolas
    with (out / "labeling" / "annotator_A.csv").open(encoding="utf-8") as file:
        assert {r["quality"] for r in csv.DictReader(file)} == {"good"}
    assert (out / "labeling" / "annotator_A_respaldo.csv").is_file()


def test_new_mango_photos_are_added_without_touching_previous_ones(pool, tmp_path, capsys):
    out, before = pool
    phone_test.sheets(out)
    _fill(out / "labeling" / "annotator_A.csv", lambda i: "regular")
    raw = tmp_path / "raw"
    for k in range(4):  # fotos nuevas de mango, todas seguidas: una sola sesión
        _photo(raw / "Mango" / f"IMG_{k}.jpg", color=(40, 160, 230), radius=80,
               date=f"2026:11:26 14:5{k}:00", center=(190 + 5 * k, 150))
    phone_test.ingest(raw, out)
    after = phone_test.read_manifest(out)
    old = {r["blind_id"]: r for r in before}
    assert all({k: r[k] for k in ("half", "session", "status")} ==
               {k: old[r["blind_id"]][k] for k in ("half", "session", "status")}
               for r in after if r["blind_id"] in old)
    mango = [r for r in after if r["product"] == "mango"]
    assert len(mango) == 4 and {r["in_scope"] for r in mango} == {"True"}
    assert {r["half"] for r in mango if r["status"] == "ok"} == {"prueba"}  # sesión única
    phone_test.ingest(raw, out)  # repetir no duplica
    assert len(phone_test.read_manifest(out)) == len(after)
    capsys.readouterr()
    phone_test.sheets(out)
    message = capsys.readouterr().out
    assert "fotos nuevas empiezan en" in message and "(mango)" in message
    with (out / "labeling" / "annotator_A.csv").open(encoding="utf-8") as file:
        sheet = list(csv.DictReader(file))
    assert {r["quality"] for r in sheet if r["product"] != "mango"} == {"regular"}
    assert {r["quality"] for r in sheet if r["product"] == "mango"} == {""}


def test_no_new_photos_after_the_phone_evaluation(pool, tmp_path, monkeypatch):
    out, _ = pool
    lock = tmp_path / "celular.lock"
    lock.write_text("{}")
    monkeypatch.setattr(phone_test, "LOCK", lock)
    with pytest.raises(RuntimeError, match="ya se evaluaron"):
        phone_test.ingest(tmp_path / "raw", out)


def test_empty_cells_copy_the_previous_photo_of_the_same_fruit():
    rows = [{"n": "3", "product": "apple", "quality": "", "unusable": ""},
            {"n": "1", "product": "apple", "quality": "good", "unusable": ""},
            {"n": "2", "product": "apple", "quality": "", "unusable": "1"},
            {"n": "4", "product": "apple", "quality": "BAD", "unusable": ""},
            {"n": "5", "product": "apple", "quality": "", "unusable": ""},
            {"n": "6", "product": "lime", "quality": "regular", "unusable": ""},
            {"n": "7", "product": "lime", "quality": "", "unusable": ""}]
    filled = phone_test.fill_down(rows)
    assert [r["quality"].lower() for r in filled] == ["good", "", "good", "bad", "bad",
                                                      "regular", "regular"]


def test_first_photo_of_a_fruit_needs_a_label():
    rows = [{"n": "1", "product": "apple", "quality": "good", "unusable": ""},
            {"n": "2", "product": "lime", "quality": "", "unusable": ""}]
    with pytest.raises(ValueError, match="#2 es la primera de lime"):
        phone_test.fill_down(rows)


def test_reads_excel_semicolons_and_windows_encoding(tmp_path):
    path = tmp_path / "annotator_A.csv"
    path.write_bytes("n;blind_id;product;quality;unusable;notes\r\n"
                     "1;EXT-1;apple;good;;mancha pequeña\r\n2;EXT-2;apple;;;\r\n".encode("cp1252"))
    rows = phone_test._read(path)
    assert rows["EXT-2"]["quality"] == "good" and rows["EXT-1"]["notes"] == "mancha pequeña"


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


def test_single_annotator_is_enough(pool, crops):
    out, _ = pool
    phone_test.sheets(out)
    lab = out / "labeling"
    ids = [r["blind_id"] for r in _fill(lab / "annotator_A.csv", lambda i: "good")]
    path = phone_test.finalize(lab, out)
    final = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["blind_id"] for r in final] == sorted(ids)
    assert {r["source_of_label"] for r in final} == {"A"}
    halves = {r["blind_id"]: r["half"] for r in phone_test.read_manifest(out)}
    assert len(phone_test.evaluation_rows(out)) == sum(halves[i] == "prueba" for i in ids)


def test_two_annotators_kappa_and_adjudication(pool, crops):
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


def test_halves_keep_sessions_whole_and_cover_every_fruit(pool):
    _, rows = pool
    usable = [r for r in rows if r["status"] == "ok" and r["in_scope"] == "True"]
    assert {r["half"] for r in usable} == {"entrenar", "prueba"}
    assert all(r["half"] == "" for r in rows if r not in usable)
    by_session = {}
    for r in usable:
        by_session.setdefault(r["session"], set()).add(r["half"])
    assert all(len(h) == 1 for h in by_session.values())
    for product in ("orange", "banana"):
        halves = [r["half"] for r in usable if r["product"] == product]
        assert halves.count("entrenar") == halves.count("prueba")  # 4 sesiones iguales


def test_assign_halves_sends_a_lone_session_to_test():
    rows = [{"product": "kiwi", "session": "s1"}] * 5
    assert phone_test.assign_halves(rows) == {"s1": "prueba"}


def test_finalize_exports_only_the_training_half(pool, crops):
    out, rows = pool
    phone_test.sheets(out)
    _fill(out / "labeling" / "annotator_A.csv", lambda i: "good")
    phone_test.finalize(out / "labeling", out)
    with (crops.parent / "celular" / "celular_train.csv").open(encoding="utf-8") as file:
        exported = list(csv.DictReader(file))
    train_ids = {r["blind_id"] for r in rows if r["half"] == "entrenar"}
    assert {r["blind_id"] for r in exported} == train_ids
    assert all((crops / r["crop"]).is_file() and (crops / r["crop_mask"]).is_file() for r in exported)
    assert all(r["source"] == "celular" and r["group"].startswith("celular:") for r in exported)
    test_ids = {r["blind_id"] for r in phone_test.evaluation_rows(out)}
    assert test_ids and not test_ids & train_ids


def test_a_test_photo_in_training_is_detected(pool, crops):
    out, rows = pool
    phone_test.sheets(out)
    _fill(out / "labeling" / "annotator_A.csv", lambda i: "good")
    phone_test.finalize(out / "labeling", out)
    with (crops.parent / "celular" / "celular_train.csv").open(encoding="utf-8") as file:
        exported = list(csv.DictReader(file))
    split = crops / "splits.csv"
    leaked = next(r for r in rows if r["half"] == "prueba")
    exported.append(dict(exported[0], filename=f"data/external/raw/{leaked['original']}",
                         group="celular:otra"))
    with split.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    with pytest.raises(RuntimeError, match="mitad 'prueba'"):
        phone_test.check_halves_apart(out, split)
    split.unlink()
    phone_test.check_halves_apart(out, split)  # sin reparto no hay nada que revisar


def test_a_long_pause_opens_a_new_session():
    rows = [{"blind_id": i, "session": "apple:iphone:20261114"} for i in "abcd"]
    times = {"a": "2026:11:14 10:37:41", "b": "2026:11:14 10:38:32",
             "c": "2026:11:14 11:48:27", "d": "2026:11:14 11:49:37"}
    phone_test.split_by_time(rows, times)
    assert [r["session"] for r in rows] == ["apple:iphone:20261114:b1"] * 2 + \
        ["apple:iphone:20261114:b2"] * 2


def test_photos_already_in_data_own_are_set_aside(tmp_path):
    raw = tmp_path / "raw"
    for k in range(3):
        _photo(raw / "Banana" / f"b{k}.jpg", color=(240, 220, 40), date=f"2026:11:18 10:0{k}:00")
    own = sorted((config.ROOT / "data" / "own").glob("banana_good_*.jpg"))[0]
    (raw / "Banana" / "copia.jpg").write_bytes(own.read_bytes())
    out = tmp_path / "external"
    phone_test.ingest(raw, out)
    rows = phone_test.read_manifest(out)
    copy = next(r for r in rows if r["original"] == "Banana/copia.jpg")
    assert copy["status"] == f"duplicate: data/own/{own.name}" and copy["half"] == ""
    phone_test.sheets(out)
    sheet = (out / "labeling" / "annotator_A.csv").read_text(encoding="utf-8")
    assert copy["blind_id"] not in sheet


def test_labeling_only_the_first_photo_of_each_fruit_is_enough(pool, crops):
    out, _ = pool
    phone_test.sheets(out)
    sheet = out / "labeling" / "annotator_A.csv"
    with sheet.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    seen = set()
    for row in rows:
        row["quality"] = "bad" if row["product"] not in seen else ""
        seen.add(row["product"])
    with sheet.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    final = list(csv.DictReader(phone_test.finalize(out / "labeling", out).open(encoding="utf-8")))
    assert len(final) == len(rows) and {r["quality"] for r in final} == {"bad"}


def test_labels_from_the_old_sheet_format_are_kept(pool):
    out, rows = pool
    lab = out / "labeling"
    lab.mkdir()
    ok = [r for r in rows if r["status"] == "ok" and r["in_scope"] == "True"]
    with (lab / "annotator_A.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("blind_id", "product", "quality", "size",
                                                  "unusable", "notes"))
        writer.writeheader()
        for i, r in enumerate(reversed(ok)):
            writer.writerow({"blind_id": r["blind_id"], "product": r["product"],
                             "quality": "bad" if i == 0 else "", "size": "small",
                             "unusable": "", "notes": "nota" if i == 0 else ""})
    phone_test.sheets(out)
    new = {r["blind_id"]: r for r in csv.DictReader((lab / "annotator_A.csv").open(encoding="utf-8"))}
    labeled = ok[-1]["blind_id"]
    assert new[labeled]["quality"] == "bad" and new[labeled]["notes"] == "nota"
    assert sum(1 for r in new.values() if r["quality"]) == 1
    assert (lab / "annotator_A_formato_anterior.csv").is_file()
