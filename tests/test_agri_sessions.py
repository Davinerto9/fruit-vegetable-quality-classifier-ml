"""
Pruebas de las sesiones de AgriFreshNET y del split v3 con datos sintéticos.

Fase CRISP-DM: Preparación de datos
Uso: pytest tests/test_agri_sessions.py
"""
import csv

import pytest

from src import config
from src.data import split_v3
from src.data.agri_sessions import merge_groups, parse_stamp, session_key
from src.data.split import assign_groups

AGRI = "data/raw/agrifreshnet/Processed Data"


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


def test_minute_level_without_hour_falls_back_to_day():
    assert session_key("IMG-20260207-WA0007.jpg", "tomato", "minute") == "agrifreshnet:tomato:day:20260207"


def _row(filename, group, source="agrifreshnet", product="orange", quality="good"):
    return {"filename": filename, "group": group, "source": source, "product": product,
            "quality": quality}


def test_merge_groups_unites_same_session_and_keeps_other_sources():
    rows = [
        _row("IMG_20251102_080441.jpg", "agri:a"),
        _row("aug_1_IMG_20251102_080441.jpg", "agri:a"),
        _row("IMG_20251102_093000.jpg", "agri:b"),          # mismo día, otra hora
        _row("IMG_20251102_093000.jpg", "agri:c", product="tomato"),  # otro producto
        _row("Banana (9).jpeg", "agri:d"),                   # sin fecha
        _row("img_20190902_111542.jpg", "kaggle:x", source="ryandpark"),
    ]
    minute = merge_groups(rows, "minute")
    assert minute["agri:a"] == "agri:a" and minute["agri:b"] == "agri:b"
    day = merge_groups(rows, "day")
    assert day["agri:a"] == day["agri:b"] == "v3:agri:a"
    assert day["agri:c"] == "agri:c" and day["agri:d"] == "agri:d"
    assert day["kaggle:x"] == "kaggle:x"


def test_merge_groups_is_transitive_through_shared_groups():
    """Un grupo con fotos de dos días une ambos días (copias idénticas en carpetas distintas)."""
    rows = [
        _row("IMG_20251102_080441.jpg", "agri:a"),
        _row("IMG_20251103_080441.jpg", "agri:a"),
        _row("IMG_20251103_120000.jpg", "agri:b"),
    ]
    assert len(set(merge_groups(rows, "day").values())) == 1


def _synthetic_split(path, n_days=60):
    """Split v2 sintético: varias fotos por día y producto en AgriFreshNET, más Kaggle."""
    rows = []
    for day in range(n_days):
        product = ("orange", "tomato", "banana")[day % 3]
        quality = config.QUALITY_CLASSES[day % 3]
        stamp = f"202511{(day % 28) + 1:02d}"
        for k in range(5):
            rows.append(dict(_row(f"{AGRI}/x/IMG_{stamp}_{8 + day % 10:02d}{k:02d}00.jpg",
                                  f"agrifreshnet:{quality}:{product}:{day}_{k}",
                                  product=product, quality=quality),
                             decision="candidate", split="train"))
    for i in range(120):
        quality = ("good", "bad")[i % 2]
        rows.append(dict(_row(f"k/{i}.jpg", f"ryandpark:capture:{i // 3}", source="ryandpark",
                              product="apple", quality=quality), decision="candidate", split="train"))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_split_v3_keeps_every_session_in_one_partition(tmp_path):
    source = tmp_path / "splits_v2_burst.csv"
    rows = _synthetic_split(source)
    output = split_v3.run("day", False, source, tmp_path / "v3.csv", dry_run=False)
    with output.open(encoding="utf-8") as file:
        written = list(csv.DictReader(file))
    assert len(written) == len(rows)
    assert {"group_v2", "split"} <= set(written[0])
    partitions_by_day = {}
    for row in written:
        if row["source"] == "agrifreshnet":
            key = session_key(row["filename"], row["product"], "day")
            partitions_by_day.setdefault(key, set()).add(row["split"])
    assert all(len(parts) == 1 for parts in partitions_by_day.values())


def test_split_v3_never_overwrites_v2(tmp_path):
    source = tmp_path / "splits_v2_burst.csv"
    _synthetic_split(source)
    with pytest.raises(ValueError):
        split_v3.run("day", False, source, source, dry_run=False)


def test_assign_groups_default_is_unchanged_and_file_cut_balances():
    rows = [{"group": f"g{i}", "quality": config.QUALITY_CLASSES[i % 3], "source": "s",
             "product": "p"} for i in range(60)]
    # Un grupo gigante: con cut="groups" su partición queda muy sobredimensionada.
    rows += [{"group": "big", "quality": q, "source": "s", "product": "p"}
             for q in config.QUALITY_CLASSES for _ in range(20)]
    assert assign_groups(rows) == assign_groups(rows, trials=256, cut="groups")
    by_files = assign_groups(rows, trials=512, cut="files")
    share = sum(by_files[row["group"]] == "train" for row in rows) / len(rows)
    assert 0.6 <= share <= 0.8


def test_leak_ladder_runs_end_to_end(tmp_path, monkeypatch):
    """Prueba de humo del diagnóstico con atributos aleatorios y un RF pequeño."""
    import importlib.util
    import json

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "agri_leak_ladder", config.ROOT / "experiments/diagnostics/agri_leak_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)

    split_path = tmp_path / "splits_v2_burst.csv"
    rows = _synthetic_split(split_path)
    for i, row in enumerate(rows):  # unos pocos en val para las estadísticas de cruce
        row["split"] = "val" if i % 7 == 0 else "train"
    with split_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rng = np.random.default_rng(0)
    features = tmp_path / "features.npz"
    np.savez(features, X=rng.random((len(rows), 8)),
             filenames=np.asarray([r["filename"] for r in rows]))
    results = tmp_path / "results"
    results.mkdir()
    (results / "random_forest_v2_burst_grid.json").write_text(json.dumps(
        {"selected": {"min_samples_leaf": 2, "max_depth": 5, "max_features": "sqrt"},
         "n_estimators": 8}))
    monkeypatch.setattr(config, "RESULTS", results)
    monkeypatch.setattr(ladder, "OUT", tmp_path / "out")

    ladder.main(split_path, features, seeds=[0], folds=2)
    report = json.loads((tmp_path / "out" / "agri_leak_ladder.json").read_text())
    assert list(report["levels"]) == ["photo", "minute", "day", "week"]
    assert report["levels"]["week"]["train_groups"] <= report["levels"]["photo"]["train_groups"]
