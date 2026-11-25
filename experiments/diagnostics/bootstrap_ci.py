"""IC 95% por bootstrap de grupos (no de filas) del F1 macro de RF en validation, v1 y v2. Nunca test."""
import joblib, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src import config
rng = np.random.default_rng(config.SEED)
cfg = {"v1": ("classic_features.npz", "splits.csv", ""), "v2_burst": ("classic_features_v2_burst.npz", "splits_v2_burst.csv", "_v2_burst")}
for tag, (feat, split, suf) in cfg.items():
    z = np.load(config.DATA_PROCESSED / feat); s = pd.read_csv(config.CROPS_DIR / split).set_index("filename").reindex(z["filenames"])
    va = (s.split == "val").values
    rf = joblib.load(config.CHECKPOINTS / f"random_forest{suf}.joblib")["model"]
    d = pd.DataFrame({"y": s.quality.values[va], "p": rf.predict(z["X"][va]), "g": s.group.values[va], "src": s.source.values[va]})
    for name, sub, labels in (("global", d, ["good", "regular", "bad"]), ("ryandpark", d[d.src == "ryandpark"], ["good", "bad"]), ("agrifreshnet", d[d.src == "agrifreshnet"], ["good", "regular", "bad"])):
        groups = sub.g.unique(); idx = sub.groupby("g").indices; arr_y, arr_p = sub.y.values, sub.p.values
        stats = []
        for _ in range(1000):
            take = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
            stats.append(f1_score(arr_y[take], arr_p[take], labels=labels, average="macro", zero_division=0))
        pt = f1_score(arr_y, arr_p, labels=labels, average="macro")
        lo, hi = np.percentile(stats, [2.5, 97.5])
        print(f"{tag:9s} RF {name:12s} groups={len(groups):4d} F1={pt:.4f}  IC95=[{lo:.4f}, {hi:.4f}]")
