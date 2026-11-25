"""
Pruebas del reagrupamiento del split v2 (src/data/split_v2.py) con datos sintéticos.

Uso: pytest tests/test_split_v2.py
"""
import pytest

from src.data import split_v2


def manifest_rows():
    """Una ráfaga de 3 fotos con grupos v1 por archivo, más una foto suelta."""
    return [
        {"filename": f"img_{n}.jpg", "group": f"file:{n}", "decision": "candidate",
         "quality": "good", "source": "ryandpark", "product": "banana"}
        for n in (1, 2, 3, 4)
    ]


def test_regroup_uses_index_groups_and_keeps_v1():
    index = {"img_1.jpg": "burst:a", "img_2.jpg": "burst:a", "img_3.jpg": "burst:a", "img_4.jpg": "burst:b"}
    rows = split_v2.regroup(manifest_rows(), index)
    assert [row["group"] for row in rows] == ["burst:a"] * 3 + ["burst:b"]
    assert rows[0]["group_v1"] == "file:1"


def test_regroup_fails_if_index_is_missing_rows():
    with pytest.raises(ValueError):
        split_v2.regroup(manifest_rows(), {"img_1.jpg": "burst:a"})


def test_exclusion_removes_the_whole_burst():
    index = {"img_1.jpg": "burst:a", "img_2.jpg": "burst:a", "img_3.jpg": "burst:a", "img_4.jpg": "burst:b"}
    rows = split_v2.select_candidates(split_v2.regroup(manifest_rows(), index), {"img_2.jpg"})
    assert [row["filename"] for row in rows] == ["img_4.jpg"]
