"""
CNN v2 vs RF v2 en validation v2 (mismas filas). Nunca test.

Uso: PYTHONPATH=. python experiments/diagnostics/eval_cnn_vs_rf_v2.py [--cnn-run v2_burst_baseline] [--rf-tag v2_burst_grid]
Escribe experiments/results/cnn_vs_rf_<cnn-run>.json y las predicciones por recorte en
experiments/diagnostics/out/val_v2_predictions_<cnn-run>.csv.
"""
import argparse
import csv
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score

from src import config
from src.training.train_cnn import _dataset

L = config.QUALITY_CLASSES


def summary(y, p, groups):
    sup = [c for c in L if (y == c).any()]
    bad = y == "bad"
    return {"n": int(len(y)), "groups": int(len(set(groups))), "classes_present": sup,
            "f1_macro_supported": float(f1_score(y, p, labels=sup, average="macro", zero_division=0)),
            "f1_per_class": {c: float(f1_score(y, p, labels=[c], average="macro", zero_division=0)) for c in sup},
            "recall_bad": float((p[bad] == "bad").mean()) if bad.any() else None,
            "bad_to_good": int((p[bad] == "good").sum()),
            "regular_to_good": int((p[y == "regular"] == "good").sum()),
            "predicted_regular": int((p == "regular").sum()),
            "confusion_matrix": confusion_matrix(y, p, labels=L).tolist()}


def paired(y, a, b, groups, labels, rng, reps=2000):
    idx = pd.Series(np.arange(len(y))).groupby(groups).indices
    keys = list(idx)
    diffs = []
    for _ in range(reps):
        take = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        diffs.append(f1_score(y[take], b[take], labels=labels, average="macro")
                     - f1_score(y[take], a[take], labels=labels, average="macro"))
    obs = f1_score(y, b, labels=labels, average="macro") - f1_score(y, a, labels=labels, average="macro")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": float(obs), "ci95": [float(lo), float(hi)], "p_delta_le_0": float(np.mean(np.array(diffs) <= 0))}


def main(cnn_run, rf_tag):
    split = config.CROPS_DIR / "splits_v2_burst.csv"
    rows = [r for r in csv.DictReader(open(split, encoding="utf-8-sig")) if r["split"] == "val"]
    y = np.asarray([r["quality"] for r in rows]); g = np.asarray([r["group"] for r in rows])
    src = np.asarray([r["source"] for r in rows]); prod = np.asarray([r["product"] for r in rows])
    c2i = {c: i for i, c in enumerate(L)}
    cnn = tf.keras.models.load_model(config.CHECKPOINTS / f"cnn_rgb_{cnn_run}.keras", compile=False)
    p_cnn = np.asarray(L)[cnn.predict(_dataset(tf, rows, c2i, 64), verbose=0).argmax(1)]
    with np.load(config.DATA_PROCESSED / "classic_features_v2_burst.npz") as z:
        pos = {n: i for i, n in enumerate(z["filenames"].astype(str))}
        X = z["X"][[pos[r["filename"]] for r in rows]]
    p_rf = joblib.load(config.CHECKPOINTS / f"random_forest_{rf_tag}.joblib")["model"].predict(X).astype(str)
    rng = np.random.default_rng(config.SEED)
    out = {"split": str(split), "test_used": False, "cnn": f"cnn_rgb_{cnn_run}", "rf": f"random_forest_{rf_tag}", "models": {}}
    for name, p in (("cnn", p_cnn), ("rf", p_rf)):
        out["models"][name] = {"overall": summary(y, p, g),
                               "by_source": {k: summary(y[src == k], p[src == k], g[src == k]) for k in sorted(set(src))},
                               "by_product": {k: summary(y[prod == k], p[prod == k], g[prod == k]) for k in sorted(set(prod))}}
    out["paired_delta_cnn_minus_rf"] = {
        "global": paired(y, p_rf, p_cnn, g, L, rng),
        "ryandpark": paired(y[src == "ryandpark"], p_rf[src == "ryandpark"], p_cnn[src == "ryandpark"], g[src == "ryandpark"], ["good", "bad"], rng),
        "agrifreshnet": paired(y[src == "agrifreshnet"], p_rf[src == "agrifreshnet"], p_cnn[src == "agrifreshnet"], g[src == "agrifreshnet"], L, rng),
    }
    rc, cc = p_rf == y, p_cnn == y
    out["pair_status"] = {"both_correct": int((rc & cc).sum()), "rf_only_wrong": int((~rc & cc).sum()),
                          "cnn_only_wrong": int((rc & ~cc).sum()), "both_wrong": int((~rc & ~cc).sum())}
    (config.RESULTS / f"cnn_vs_rf_{cnn_run}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame({"filename": [r["filename"] for r in rows], "source": src, "product": prod, "group": g, "quality": y,
                  "cnn_prediction": p_cnn, "rf_prediction": p_rf}).to_csv(
        config.ROOT / "experiments" / "diagnostics" / "out" / f"val_v2_predictions_{cnn_run}.csv", index=False)
    for name in ("cnn", "rf"):
        o = out["models"][name]["overall"]; bs = out["models"][name]["by_source"]
        print(f"{name}: F1={o['f1_macro_supported']:.4f} recall_bad={o['recall_bad']:.3f} bad->good={o['bad_to_good']} "
              f"F1_regular={o['f1_per_class']['regular']:.4f} regular->good={o['regular_to_good']} pred_regular_kaggle={bs['ryandpark']['predicted_regular']} | "
              + " ".join(f"{k}={v['f1_macro_supported']:.4f}" for k, v in bs.items()))
    for k, v in out["paired_delta_cnn_minus_rf"].items():
        print(f"Δ(CNN−RF) {k}: {v['delta']:+.4f} IC95=[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print("pair status", out["pair_status"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-run", default="v2_burst_baseline")
    ap.add_argument("--rf-tag", default="v2_burst_grid")
    a = ap.parse_args()
    main(a.cnn_run, a.rf_tag)
