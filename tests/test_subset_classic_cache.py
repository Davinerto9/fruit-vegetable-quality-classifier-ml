"""Pruebas de src/features/subset_classic_cache.py con un caché sintético."""
import csv

import numpy as np
import pytest

from src.features.subset_classic_cache import subset


def _write(tmp_path, rows, quality_in_cache=("good", "bad", "regular")):
    cache = tmp_path / "cache.npz"
    np.savez(cache, X=np.arange(6, dtype=np.float32).reshape(3, 2), feature_names=np.array(["a", "b"]),
             filenames=np.array(["f0", "f1", "f2"]), quality=np.array(quality_in_cache),
             groups=np.array(["g0", "g1", "g2"]), products=np.array(["p"] * 3), sources=np.array(["s"] * 3))
    split = tmp_path / "split.csv"
    with split.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "quality", "group", "product", "source"])
        writer.writeheader()
        writer.writerows(rows)
    return cache, split


def test_selects_rows_in_split_order_with_new_groups(tmp_path):
    rows = [{"filename": "f2", "quality": "regular", "group": "burst", "product": "p", "source": "s"},
            {"filename": "f0", "quality": "good", "group": "burst", "product": "p", "source": "s"}]
    cache, split = _write(tmp_path, rows)
    out = subset(cache, split, tmp_path / "out.npz")
    with np.load(out) as saved:
        assert saved["X"].tolist() == [[4, 5], [0, 1]]
        assert saved["groups"].tolist() == ["burst", "burst"]


def test_fails_on_label_mismatch(tmp_path):
    rows = [{"filename": "f0", "quality": "bad", "group": "g", "product": "p", "source": "s"}]
    cache, split = _write(tmp_path, rows)
    with pytest.raises(ValueError):
        subset(cache, split, tmp_path / "out.npz")
