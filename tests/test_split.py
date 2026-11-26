"""
Pruebas del reparto por grupos (src/data/split.py) con datos sintéticos.

Fase CRISP-DM: Preparación de datos
Uso: pytest tests/test_split.py
"""
import csv
from datetime import date, timedelta

import pytest

from src import config
from src.data import split
from src.data.split import assign_groups, merge_groups, parse_stamp, session_key

AGRI = "data/raw/agrifreshnet/Processed Data"


def synthetic_rows() -> list[dict[str, str]]:
    """60 grupos de 1 a 3 fotos, repartidos entre tres calidades y dos fuentes."""
    rows = []
    for g in range(60):
        quality = ("good", "regular", "bad")[g % 3]
        for k in range(1 + g % 3):
            rows.append({"group": f"g{g}", "quality": quality, "source": ("a", "b")[g % 2],
                         "product": "banana", "filename": f"g{g}_{k}.jpg"})
    return rows


def test_each_group_goes_to_a_single_partition():
    """Ningún grupo aparece en dos particiones (sin fuga entre train, val y test)."""
    rows = synthetic_rows()
    assignment = assign_groups(rows, cut="files")
    assert set(assignment) == {row["group"] for row in rows}
    assert set(assignment.values()) == set(split.PARTITIONS)


def test_every_partition_has_all_qualities_and_is_reproducible():
    rows = synthetic_rows()
    assignment = assign_groups(rows, cut="files")
    for partition in split.PARTITIONS:
        assert {r["quality"] for r in rows if assignment[r["group"]] == partition} == \
            {"good", "regular", "bad"}
    assert assign_groups(rows, cut="files") == assignment


def test_file_cut_balances_a_giant_group():
    rows = [{"group": f"g{i}", "quality": config.QUALITY_CLASSES[i % 3], "source": "s",
             "product": "p"} for i in range(60)]
    rows += [{"group": "big", "quality": q, "source": "s", "product": "p"}
             for q in config.QUALITY_CLASSES for _ in range(20)]
    by_files = assign_groups(rows, trials=512, cut="files")
    share = sum(by_files[row["group"]] == "train" for row in rows) / len(rows)
    assert 0.6 <= share <= 0.8


# --- Sesiones de AgriFreshNET -----------------------------------------------------------------


@pytest.mark.parametrize("name, expected", [
    (f"{AGRI}/Fresh Orange(1-9)/aug_48_IMG_20251102_080441757_HDR_AE.jpg", ("20251102", "0804")),
    (f"{AGRI}/Fresh Orange(1-9)/IMG_20251102_080459.jpg", ("20251102", "0804")),
    (f"{AGRI}/Rotten Tomato(24-35)/aug_290_IMG-20260207-WA0007.jpg", ("20260207", None)),
    ("WhatsApp Image 2026-01-16 at 9.05.11.jpeg", ("20260116", "0905")),
    (f"{AGRI}/Fresh Banana/Banana (9).jpeg", None),
])
def test_parse_stamp_formats(name, expected):
    assert parse_stamp(name) == expected


def test_session_levels_get_coarser():
    name = "aug_3_IMG_20251102_080441757_HDR.jpg"
    assert session_key(name, "orange", "photo") is None
    assert session_key(name, "orange", "minute") == "agrifreshnet:orange:minute:20251102_0804"
    assert session_key(name, "orange", "day") == "agrifreshnet:orange:day:20251102"
    assert session_key(name, "orange", "week") == "agrifreshnet:orange:week:2025W44"
    assert session_key("IMG-20260207-WA0007.jpg", "tomato", "minute") == \
        "agrifreshnet:tomato:day:20260207"


def _row(filename, group, source="agrifreshnet", product="orange", quality="good"):
    return {"filename": filename, "group": group, "source": source, "product": product,
            "quality": quality}


def test_merge_groups_unites_same_session_and_keeps_other_sources():
    rows = [
        _row("IMG_20251102_080441.jpg", "agri:a"),
        _row("aug_1_IMG_20251102_080441.jpg", "agri:a"),
        _row("IMG_20251102_093000.jpg", "agri:b"),                    # mismo día, otra hora
        _row("IMG_20251102_093000.jpg", "agri:c", product="tomato"),  # otro producto
        _row("Banana (9).jpeg", "agri:d"),                            # sin fecha
        _row("img_20190902_111542.jpg", "kaggle:x", source="ryandpark"),
    ]
    assert merge_groups(rows, "minute")["agri:b"] == "agri:b"
    day = merge_groups(rows, "day")
    assert day["agri:a"] == day["agri:b"] == "sesion:agri:a"
    assert day["agri:c"] == "agri:c" and day["agri:d"] == "agri:d"
    assert day["kaggle:x"] == "kaggle:x"


def test_merge_groups_is_transitive_through_shared_groups():
    """Un grupo con fotos de dos días une ambos días."""
    rows = [_row("IMG_20251102_080441.jpg", "agri:a"), _row("IMG_20251103_080441.jpg", "agri:a"),
            _row("IMG_20251103_120000.jpg", "agri:b")]
    assert len(set(merge_groups(rows, "day").values())) == 1


# --- Programa completo ------------------------------------------------------------------------


def synthetic_manifest(path, n_days=60):
    """Manifiesto con varias fotos por día de AgriFreshNET y ráfagas de Kaggle."""
    rows = []
    for day in range(n_days):
        product = ("orange", "tomato", "banana")[day % 3]
        quality = config.QUALITY_CLASSES[day % 3]
        stamp = (date(2024, 1, 1) + timedelta(weeks=day)).strftime("%Y%m%d")  # una semana cada uno
        for k in range(5):
            rows.append(dict(_row(f"{AGRI}/x/IMG_{stamp}_{8 + day % 10:02d}{k:02d}00.jpg",
                                  f"agrifreshnet:{quality}:{product}:{day}_{k}",
                                  product=product, quality=quality), decision="candidate"))
    for i in range(120):
        rows.append(dict(_row(f"k/{i}.jpg", f"ryandpark:capture:{i // 3}", source="ryandpark",
                              product="apple", quality=("good", "bad")[i % 2]),
                         decision="candidate"))
    rows.append(dict(_row("k/roto.jpg", "ryandpark:roto", source="ryandpark", product="apple"),
                     decision="excluded"))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


@pytest.fixture()
def exclusions(tmp_path, monkeypatch):
    path = tmp_path / "exclusions.csv"
    path.write_text("group,reviewed_file,reason\nx,k/0.jpg,multiple_pieces\n", encoding="utf-8")
    monkeypatch.setattr(config, "SEGMENTATION_EXCLUSIONS", path)


def test_run_keeps_every_week_in_one_partition(tmp_path, exclusions):
    manifest = tmp_path / "manifest.csv"
    synthetic_manifest(manifest)
    output = split.run(manifest, tmp_path / "splits.csv")
    with output.open(encoding="utf-8") as file:
        written = list(csv.DictReader(file))
    assert {"group_original", "split"} <= set(written[0])
    assert all(row["decision"] == "candidate" for row in written)
    # La foto k/0.jpg se apartó, y con ella toda su ráfaga (k/0, k/1, k/2).
    assert not {"k/0.jpg", "k/1.jpg", "k/2.jpg"} & {row["filename"] for row in written}
    weeks = {}
    for row in written:
        if row["source"] == "agrifreshnet":
            weeks.setdefault(session_key(row["filename"], row["product"], "week"), set()).add(row["split"])
    assert all(len(parts) == 1 for parts in weeks.values())


def test_run_refuses_a_bad_split(tmp_path, exclusions, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    synthetic_manifest(manifest)
    monkeypatch.setattr(split, "MAX_DEVIATION", 0.0)
    with pytest.raises(RuntimeError, match="no cumple"):
        split.run(manifest, tmp_path / "splits.csv")
