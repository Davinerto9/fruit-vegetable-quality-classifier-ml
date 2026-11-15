"""
Pruebas de la partición por grupos de src/data/split.py con datos sintéticos.

Fase CRISP-DM: Preparación de datos
Uso: pytest tests/test_split.py
"""
from src.data import split


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
    assignment = split.assign_groups(rows)
    assert set(assignment) == {row["group"] for row in rows}
    assert set(assignment.values()) == set(split.PARTITIONS)


def test_every_partition_has_all_qualities():
    """Train, val y test contienen ejemplos de las tres calidades."""
    rows = synthetic_rows()
    assignment = split.assign_groups(rows)
    for partition in split.PARTITIONS:
        qualities = {row["quality"] for row in rows if assignment[row["group"]] == partition}
        assert qualities == {"good", "regular", "bad"}


def test_assignment_is_reproducible():
    """Con la semilla de config, dos ejecuciones dan la misma partición."""
    rows = synthetic_rows()
    assert split.assign_groups(rows) == split.assign_groups(rows)
