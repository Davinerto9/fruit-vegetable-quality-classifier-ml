"""
Ensamble RF + CNN en validation v2 (nunca test).

Regla fijada antes de mirar resultados: voto suave con pesos iguales,
p = 0.5 * p_RF + 0.5 * p_CNN, y la clase es el argmax. No se ajusta nada en validation.
Además se reporta la curva por peso w (p = w * p_CNN + (1 - w) * p_RF), solo como
análisis descriptivo de sensibilidad: elegir w con esa curva sería seleccionar sobre validation.

Uso: PYTHONPATH=. python experiments/diagnostics/eval_ensemble_v2.py [--cnn-run v2_burst_e3b] [--rf-tag v2_burst_grid]
"""
import argparse
import csv
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from src import config
from src.training.train_cnn import _dataset

L = config.QUALITY_CLASSES


def summary(y, p):
    bad, reg = y == "bad", y == "regular"
    sup = [c for c in L if (y == c).any()]
    return {"f1_macro": float(f1_score(y, p, labels=sup, average="macro", zero_division=0)),
            "f1_regular": float(f1_score(y, p, labels=["regular"], average="macro", zero_division=0)) if reg.any() else None,
            "recall_bad": float((p[bad] == "bad").mean()), "bad_to_good": int((p[bad] == "good").sum()),
            "regular_to_good": int((p[reg] == "good").sum()), "n": int(len(y))}


def paired(y, a, b, g, labels, rng, reps=2000):
    idx = pd.Series(np.arange(len(y))).groupby(g).indices
    keys = list(idx)
    d = []
    for _ in range(reps):
        t = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        d.append(f1_score(y[t], b[t], labels=labels, average="macro") - f1_score(y[t], a[t], labels=labels, average="macro"))
    obs = f1_score(y, b, labels=labels, average="macro") - f1_score(y, a, labels=labels, average="macro")
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(obs), "ci95": [float(lo), float(hi)]}


def main(cnn_run, rf_tag):
    rows = [r for r in csv.DictReader(open(config.CROPS_DIR / "splits_v2_burst.csv", encoding="utf-8-sig")) if r["split"] == "val"]
    y = np.asarray([r["quality"] for r in rows]); g = np.asarray([r["group"] for r in rows])
    src = np.asarray([r["source"] for r in rows])
    cnn = tf.keras.models.load_model(config.CHECKPOINTS / f"cnn_rgb_{cnn_run}.keras", compile=False)
    p_cnn = cnn.predict(_dataset(tf, rows, {c: i for i, c in enumerate(L)}, 64), verbose=0)  # columnas en orden L
    rf = joblib.load(config.CHECKPOINTS / f"random_forest_{rf_tag}.joblib")["model"]
    with np.load(config.DATA_PROCESSED / "classic_features_v2_burst.npz") as z:
        pos = {n: i for i, n in enumerate(z["filenames"].astype(str))}
        X = z["X"][[pos[r["filename"]] for r in rows]]
    p_rf = rf.predict_proba(X)[:, [list(rf.classes_).index(c) for c in L]]  # reordenado a L
    assert np.allclose(p_rf.sum(1), 1) and np.allclose(p_cnn.sum(1), 1, atol=1e-4)
    labels = np.asarray(L)
    pred = {"rf": labels[p_rf.argmax(1)], "cnn": labels[p_cnn.argmax(1)],
            "ensemble_0.5": labels[(0.5 * p_rf + 0.5 * p_cnn).argmax(1)]}
    rng = np.random.default_rng(config.SEED)
    out = {"rule": "soft vote, equal weights (fixed a priori)", "cnn": cnn_run, "rf": rf_tag, "test_used": False,
           "models": {}, "paired": {}, "weight_curve_descriptive_only": {}}
    for name, p in pred.items():
        out["models"][name] = {"overall": summary(y, p),
                               "by_source": {s: summary(y[src == s], p[src == s]) for s in ("ryandpark", "agrifreshnet")}}
        print(f"{name:13s} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in out["models"][name]["overall"].items())
              + f" | Kaggle={out['models'][name]['by_source']['ryandpark']['f1_macro']:.4f}"
              + f" Agri={out['models'][name]['by_source']['agrifreshnet']['f1_macro']:.4f}")
    for base in ("rf", "cnn"):
        for scope, m, lab in (("global", np.ones(len(y), bool), L), ("ryandpark", src == "ryandpark", ["good", "bad"]),
                              ("agrifreshnet", src == "agrifreshnet", L)):
            r = paired(y[m], pred[base][m], pred["ensemble_0.5"][m], g[m], lab, rng)
            out["paired"][f"ensemble_minus_{base}_{scope}"] = r
            print(f"Δ(ensamble − {base}) {scope:12s} {r['delta']:+.4f} IC95=[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    for w in np.round(np.linspace(0, 1, 11), 1):
        p = labels[(w * p_cnn + (1 - w) * p_rf).argmax(1)]
        out["weight_curve_descriptive_only"][str(w)] = summary(y, p)
    print("curva descriptiva w_CNN -> F1 / bad->good:",
          " ".join(f"{w}:{v['f1_macro']:.3f}/{v['bad_to_good']}" for w, v in out["weight_curve_descriptive_only"].items()))
    agree = pred["rf"] == pred["cnn"]
    print(f"acuerdo RF-CNN: {agree.mean():.3f}; accuracy cuando acuerdan {np.mean(pred['rf'][agree] == y[agree]):.3f}, "
          f"cuando no: ensamble {np.mean(pred['ensemble_0.5'][~agree] == y[~agree]):.3f} "
          f"(RF {np.mean(pred['rf'][~agree] == y[~agree]):.3f}, CNN {np.mean(pred['cnn'][~agree] == y[~agree]):.3f}, n={int((~agree).sum())})")
    out["agreement"] = float(agree.mean())
    (config.RESULTS / f"ensemble_rf_{rf_tag}_cnn_{cnn_run}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-run", default="v2_burst_e3b")
    ap.add_argument("--rf-tag", default="v2_burst_grid")
    a = ap.parse_args()
    main(a.cnn_run, a.rf_tag)
