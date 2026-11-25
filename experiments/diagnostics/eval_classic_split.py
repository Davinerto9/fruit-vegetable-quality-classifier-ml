"""
Evalúa checkpoints clásicos en validation de un split dado (nunca test).

Uso (desde la raíz, con scikit-learn 1.7.2):
  PYTHONPATH=. python experiments/diagnostics/eval_classic_split.py --tag v2_burst \
      --features data/processed/classic_features_v2_burst.npz --split data/processed/crops/splits_v2_burst.csv
  PYTHONPATH=. python experiments/diagnostics/eval_classic_split.py --tag ""   # v1

Escribe experiments/results/classic_eval_<tag|v1>.json con métricas globales, por fuente
y por producto, recall de bad, errores bad->good, F1 de regular y brecha train/val.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from src import config

L = config.QUALITY_CLASSES


def summary(t, p):
    sup = [c for c in L if (t == c).any()]
    bad = t == "bad"
    reg = t == "regular"
    return {
        "n": int(len(t)),
        "classes_present": sup,
        "f1_macro_supported": float(f1_score(t, p, labels=sup, average="macro", zero_division=0)),
        "f1_per_class": {c: float(f1_score(t, p, labels=[c], average="macro", zero_division=0)) for c in sup},
        "recall_bad": float((p[bad] == "bad").mean()) if bad.any() else None,
        "bad_to_good": int((p[bad] == "good").sum()),
        "regular_to_good": int((p[reg] == "good").sum()),
        "predicted_regular": int((p == "regular").sum()),
        "confusion_matrix": confusion_matrix(t, p, labels=L).tolist(),
    }


def main(tag, features, split):
    s = pd.read_csv(split)
    with np.load(features, allow_pickle=False) as z:
        X, fn = z["X"], z["filenames"].astype(str)
    part = s.set_index("filename").split.reindex(fn).values
    meta = s.set_index("filename").reindex(fn)
    y = meta.quality.values.astype(str); src = meta.source.values.astype(str); prod = meta["product"].values.astype(str)
    grp = meta.group.values.astype(str)
    tr, va = part == "train", part == "val"
    assert not (set(grp[tr]) & set(grp[va])), "grupos compartidos train/val"
    suffix = f"_{tag}" if tag else ""
    models = {
        "logistic_regression": joblib.load(config.CHECKPOINTS / f"logistic_baseline{suffix}.joblib")["pipeline"],
        "svm_linear": joblib.load(config.CHECKPOINTS / f"svm{suffix}.joblib")["model"],
        "random_forest": joblib.load(config.CHECKPOINTS / f"random_forest{suffix}.joblib")["model"],
    }
    out = {"tag": tag or "v1", "split": str(split), "features": str(features), "test_used": False,
           "val_rows": int(va.sum()), "val_groups": int(len(set(grp[va]))),
           "val_groups_by_source": {k: int(len(set(grp[va & (src == k)]))) for k in sorted(set(src))},
           "models": {}}
    for name, m in models.items():
        pv, pt = m.predict(X[va]).astype(str), m.predict(X[tr]).astype(str)
        r = {"train": summary(y[tr], pt), "val": summary(y[va], pv), "by_source": {}, "by_product": {}}
        r["train_val_gap_f1"] = r["train"]["f1_macro_supported"] - r["val"]["f1_macro_supported"]
        for key, arr, field in (("by_source", src, None), ("by_product", prod, None)):
            for k in sorted(set(arr[va])):
                mk = arr[va] == k
                r[key][k] = summary(y[va][mk], pv[mk])
                r[key][k]["groups"] = int(len(set(grp[va][mk])))
        out["models"][name] = r
        v = r["val"]
        print(f"{name}: val F1={v['f1_macro_supported']:.4f} recall_bad={v['recall_bad']:.3f} "
              f"bad->good={v['bad_to_good']} F1_regular={v['f1_per_class']['regular']:.4f} "
              f"train F1={r['train']['f1_macro_supported']:.4f} gap={r['train_val_gap_f1']:.4f} | "
              + " ".join(f"{k}:{d['f1_macro_supported']:.4f}(n={d['n']},g={d['groups']},{'/'.join(d['classes_present'])})"
                         for k, d in r["by_source"].items()))
    path = config.RESULTS / f"classic_eval_{tag or 'v1'}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("guardado", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--features", type=Path, default=config.DATA_PROCESSED / "classic_features.npz")
    ap.add_argument("--split", type=Path, default=config.CROPS_DIR / "splits.csv")
    a = ap.parse_args()
    main(a.tag, a.features, a.split)
