"""Bootstrap pareado por grupos (val v2): RF v2_burst_grid vs RF v2_burst, mismas filas. Nunca test."""
import joblib, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src import config
rng = np.random.default_rng(config.SEED)
z = np.load(config.DATA_PROCESSED / "classic_features_v2_burst.npz")
s = pd.read_csv(config.CROPS_DIR / "splits_v2_burst.csv").set_index("filename").reindex(z["filenames"])
va = (s.split == "val").values; tr = (s.split == "train").values
y, g, src = s.quality.values[va], s.group.values[va], s.source.values[va]
P = {t: joblib.load(config.CHECKPOINTS / f"random_forest_{t}.joblib")["model"] for t in ("v2_burst", "v2_burst_grid")}
pred = {t: m.predict(z["X"][va]) for t, m in P.items()}
for t, m in P.items():
    tp = m.predict(z["X"][tr]); yt = s.quality.values[tr]
    d = [e.get_depth() for e in m.estimators_]; l = [e.get_n_leaves() for e in m.estimators_]
    print(f"{t}: train F1={f1_score(yt, tp, average='macro'):.4f} val F1={f1_score(y, pred[t], average='macro'):.4f} "
          f"F1 regular={f1_score(y, pred[t], labels=['regular'], average='macro'):.4f} depth med={np.median(d):.0f} leaves med={np.median(l):.0f}")
L3, L2 = ["good", "regular", "bad"], ["good", "bad"]
for name, mask, labels in (("global", np.ones(len(y), bool), L3), ("ryandpark", src == "ryandpark", L2), ("agrifreshnet", src == "agrifreshnet", L3)):
    yy, gg = y[mask], g[mask]; a, b = pred["v2_burst"][mask], pred["v2_burst_grid"][mask]
    idx = pd.Series(np.arange(len(yy))).groupby(gg).indices; groups = list(idx)
    diffs = []
    for _ in range(2000):
        take = np.concatenate([idx[k] for k in rng.choice(groups, len(groups))])
        diffs.append(f1_score(yy[take], b[take], labels=labels, average="macro") - f1_score(yy[take], a[take], labels=labels, average="macro"))
    obs = f1_score(yy, b, labels=labels, average="macro") - f1_score(yy, a, labels=labels, average="macro")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"Δ F1 (grid − base) {name:12s} groups={len(groups):4d} Δ={obs:+.4f} IC95=[{lo:+.4f}, {hi:+.4f}] P(Δ≤0)={np.mean(np.array(diffs) <= 0):.3f}")
for t in pred:
    bad = y == "bad"; print(t, "recall bad", round((pred[t][bad] == "bad").mean(), 3), "bad->good", int((pred[t][bad] == "good").sum()),
          "regular->good", int((pred[t][y == "regular"] == "good").sum()), "pred regular Kaggle", int((pred[t][src == "ryandpark"] == "regular").sum()))
