"""Comprueba que _dataset(cache=True) da los mismos tensores y orden que cache=False (2 pasadas, con shuffle)."""
import csv
import numpy as np
import tensorflow as tf
from src import config
from src.training.train_cnn import _dataset
rows = [r for r in csv.DictReader(open(config.CROPS_DIR / "splits_v2_burst.csv", encoding="utf-8-sig")) if r["split"] == "train"][:700]
c2i = {c: i for i, c in enumerate(config.QUALITY_CLASSES)}
for shuffle in (False, True):
    tf.keras.utils.set_random_seed(config.SEED); a = _dataset(tf, rows, c2i, 64, shuffle=shuffle, cache=False)
    tf.keras.utils.set_random_seed(config.SEED); b = _dataset(tf, rows, c2i, 64, shuffle=shuffle, cache=True)
    for epoch in range(2):
        xa = np.concatenate([x.numpy() for x, _ in a]); ya = np.concatenate([y.numpy() for _, y in a])
        xb = np.concatenate([x.numpy() for x, _ in b]); yb = np.concatenate([y.numpy() for _, y in b])
        print(f"shuffle={shuffle} pasada={epoch + 1}: n={len(xa)} labels iguales={np.array_equal(ya, yb)} "
              f"pixeles iguales={np.array_equal(xa, xb)} max|dif|={np.abs(xa - xb).max():.2e}")
