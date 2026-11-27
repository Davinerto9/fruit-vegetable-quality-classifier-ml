"""
Pruebas del agrupamiento y la detección de conflictos de src/data/inventory.py (sin datasets reales).

Fase CRISP-DM: Preparación de datos
Uso: pytest tests/test_inventory.py
"""
from src import config
from src.data import inventory
from src.data.inventory import ImageRecord


def test_capture_group_joins_shots_of_the_same_minute():
    """Los tres formatos de fecha de ryandpark caen en la misma ráfaga si comparten minuto."""
    a = inventory.capture_group("img20200728182100_27396")
    b = inventory.capture_group("img20200728182159")
    c = inventory.capture_group("img_20200728_182130_1")
    assert a == b == c == "20200728_1821"
    assert inventory.capture_group("img_20200728_182201_1") == "20200728_1822"


def test_capture_group_blocks_of_numbered_photos():
    """Las fotos sin fecha se agrupan por bloques de números consecutivos."""
    assert inventory.capture_group("img_2427") == inventory.capture_group("img_2420")
    assert inventory.capture_group("img_2427") != inventory.capture_group("img_2430")
    assert inventory.capture_group("foto_sin_patron") is None


def record(filename: str, quality: str, group: str) -> ImageRecord:
    """Registro externo mínimo para las pruebas."""
    return ImageRecord(filename, "orange", quality, "agrifreshnet", "", "", group, "labeled")


def test_identical_copies_with_two_labels_are_conflicts(tmp_path, monkeypatch):
    """Dos copias idénticas con etiquetas distintas se unen y se marcan en conflicto."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / "a.jpg").write_bytes(b"misma foto")
    (tmp_path / "b.jpg").write_bytes(b"misma foto")
    records = [record("a.jpg", "good", "g1"), record("b.jpg", "regular", "g2")]
    inventory.merge_external_duplicates(records)
    assert records[0].group == records[1].group
    assert {r.status for r in records} == {"conflicting_quality"}


def test_different_photos_in_one_burst_are_not_conflicts(tmp_path, monkeypatch):
    """Una ráfaga con fotos distintas de dos calidades no es un conflicto de etiqueta."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / "a.jpg").write_bytes(b"fruta buena")
    (tmp_path / "b.jpg").write_bytes(b"fruta mala!")  # mismo tamaño, contenido distinto
    records = [record("a.jpg", "good", "burst"), record("b.jpg", "bad", "burst")]
    inventory.merge_external_duplicates(records)
    assert {r.status for r in records} == {"labeled"}


def test_afruitdb_sessions_group_bursts_and_keep_grades_together(tmp_path, monkeypatch):
    """Fotos a segundos de distancia van juntas, aunque sean de calidades distintas; una
    pausa larga abre otra sesión. La uva birmana no es una fruta del modelo y queda fuera."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "AFRUITDB_DIR", tmp_path / "afruitdb")
    files = {"Apple/1st grade/IMG_20230930_172415.jpg": b"a",
             "Apple/2nd grade/IMG_20230930_172421.jpg": b"b",
             "Apple/2nd grade/IMG_20230930_181000.jpg": b"c",
             "Apple/3rd grade/foto.jpg": b"d",
             "Burmese_grape/1st grade/IMG_20230930_172415.jpg": b"e"}
    for name, content in files.items():
        path = tmp_path / "afruitdb" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    records = {r.filename.split("afruitdb/")[1]: r for r in inventory.afruitdb_records()}
    assert set(records) == {name for name in files if name.startswith("Apple")}
    first, second, later, loose = (records[n] for n in sorted(files)[:4])
    assert first.quality == "good" and second.quality == "regular" and loose.quality == "bad"
    assert first.group == second.group == "afruitdb:apple:20230930_1724"
    assert later.group == "afruitdb:apple:20230930_1810"
    assert loose.group == "afruitdb:apple:foto"
    assert {r.source for r in records.values()} == {"afruitdb"}


def test_quality_metrics_report_regular_by_source():
    from src.training.train_models import quality_metrics
    metrics = quality_metrics(["regular", "regular", "regular", "good"],
                              ["regular", "good", "regular", "good"],
                              ["agrifreshnet", "afruitdb", "afruitdb", "afruitdb"])
    assert metrics["recall_regular_by_source"] == {"afruitdb": 0.5, "agrifreshnet": 1.0}
    assert "f1_afruitdb" in metrics
