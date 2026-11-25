"""Prueba de la regla de selección de src/training/rf_grid.py."""
from src.training.rf_grid import select


def rec(name, f1, sd, bad_to_good, gap):
    return {"params": name, "mean_f1_macro": f1, "std_f1_macro": sd,
            "sum_bad_to_good": bad_to_good, "gap_train_cv": gap}


def test_prefers_fewer_bad_to_good_within_one_sd():
    records = [rec("best", 0.81, 0.01, 300, 0.19), rec("close", 0.805, 0.01, 250, 0.10),
               rec("far", 0.78, 0.01, 100, 0.05)]
    chosen, best, near = select(records)
    assert best["params"] == "best"
    assert chosen["params"] == "close"
    assert {r["params"] for r in near} == {"best", "close"}


def test_tie_breaks_on_smaller_gap():
    records = [rec("a", 0.81, 0.01, 200, 0.19), rec("b", 0.81, 0.01, 200, 0.12)]
    assert select(records)[0]["params"] == "b"
