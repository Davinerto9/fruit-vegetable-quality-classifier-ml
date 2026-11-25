"""Comprueba que --augment solo aplica simetrías exactas del cuadrado, conserva etiquetas y es reproducible."""
import csv
from collections import Counter
import numpy as np
import tensorflow as tf
from src import config
from src.training.train_cnn import _dataset
rows = [r for r in csv.DictReader(open(config.CROPS_DIR / "splits_v2_burst.csv", encoding="utf-8-sig")) if r["split"] == "train"][:512]
c2i = {c: i for i, c in enumerate(config.QUALITY_CLASSES)}
def dihedral(x):
    out = []
    for t in (x, x.transpose(1, 0, 2)):
        out += [t, t[:, ::-1], t[::-1], t[::-1, ::-1]]
    return out
plain = np.concatenate([x.numpy() for x, _ in _dataset(tf, rows, c2i, 64)])
def run():
    tf.keras.utils.set_random_seed(config.SEED)
    ds = _dataset(tf, rows, c2i, 64, augment=True)
    return [np.concatenate([x.numpy() for x, _ in ds]) for _ in range(2)], np.concatenate([y.numpy() for _, y in ds])
(ep1, ep2), labels = run()
(ep1b, _), _ = run()
kinds = Counter()
ok = 0
for a, x in zip(ep1, plain):
    hits = [k for k, d in enumerate(dihedral(x)) if np.array_equal(a, d)]
    ok += bool(hits); kinds[hits[0] if hits else -1] += 1
print(f"imágenes={len(ep1)} exactamente una simetría del original={ok}/{len(ep1)}")
print("frecuencia por simetría (0=identidad):", dict(sorted(kinds.items())))
print("etiquetas iguales al original:", np.array_equal(labels, np.asarray([c2i[r['quality']] for r in rows])))
print("suma de píxeles conservada:", np.allclose(ep1.sum((1, 2, 3)), plain.sum((1, 2, 3))))
print("pasada 1 != pasada 2 (varía entre épocas):", not np.array_equal(ep1, ep2))
print("reproducible con la misma semilla:", np.array_equal(ep1, ep1b))

# Con shuffle (como en train): la simetría aplicada a cada imagen debe cambiar entre épocas.
key = {round(float(s), 3): i for i, s in enumerate(plain.sum((1, 2, 3)))}
tf.keras.utils.set_random_seed(config.SEED)
ds = _dataset(tf, rows, c2i, 64, shuffle=True, augment=True)
per_epoch = []
for _ in range(3):
    xs = np.concatenate([x.numpy() for x, _ in ds]); kind = {}
    for a in xs:
        i = key.get(round(float(a.sum()), 3))
        if i is not None:
            kind[i] = next(k for k, d in enumerate(dihedral(plain[i])) if np.array_equal(a, d))
    per_epoch.append(kind)
common = set(per_epoch[0]) & set(per_epoch[1]) & set(per_epoch[2])
changed = sum(len({per_epoch[e][i] for e in range(3)}) > 1 for i in common)
print(f"shuffle=True: imágenes identificadas={len(common)}; con simetría distinta en alguna de 3 épocas={changed}")
