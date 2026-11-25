"""Δ F1 pareado (bootstrap de grupos) entre dos corridas CNN sobre val v2. Uso: paired_cnn_runs.py BASE NUEVA"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src import config
base, new = sys.argv[1], sys.argv[2]
out = config.ROOT / "experiments" / "diagnostics" / "out"
a = pd.read_csv(out / f"val_v2_predictions_{base}.csv"); b = pd.read_csv(out / f"val_v2_predictions_{new}.csv")
assert (a.filename == b.filename).all()
rng = np.random.default_rng(config.SEED)
for name, m, labels in (("global", np.ones(len(a), bool), ["good", "regular", "bad"]), ("ryandpark", (a.source == "ryandpark").values, ["good", "bad"]), ("agrifreshnet", (a.source == "agrifreshnet").values, ["good", "regular", "bad"])):
    y, pa, pb, g = a.quality.values[m], a.cnn_prediction.values[m], b.cnn_prediction.values[m], a.group.values[m]
    idx = pd.Series(np.arange(len(y))).groupby(g).indices; keys = list(idx); d = []
    for _ in range(2000):
        t = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        d.append(f1_score(y[t], pb[t], labels=labels, average="macro") - f1_score(y[t], pa[t], labels=labels, average="macro"))
    obs = f1_score(y, pb, labels=labels, average="macro") - f1_score(y, pa, labels=labels, average="macro")
    lo, hi = np.percentile(d, [2.5, 97.5]); print(f"Δ F1 ({new} − {base}) {name:12s} {obs:+.4f} IC95=[{lo:+.4f}, {hi:+.4f}]")
