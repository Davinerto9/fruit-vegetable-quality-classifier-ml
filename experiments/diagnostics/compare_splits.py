"""Compara split v1 (splits.csv) y v2 (splits_v2_burst.csv): integridad y cobertura. Solo metadatos."""
import pandas as pd
from src import config
pd.set_option('display.width', 250)
v1 = pd.read_csv(config.CROPS_DIR / 'splits.csv'); v2 = pd.read_csv(config.CROPS_DIR / 'splits_v2_burst.csv')
print('rows v1', len(v1), 'v2', len(v2), '| groups v1', v1.group.nunique(), 'v2', v2.group.nunique())
print('v2 groups in >1 split:', (v2.groupby('group').split.nunique() > 1).sum())
print('v2: v1-groups split across v2 groups:', (v2.groupby('group_v1').group.nunique() > 1).sum())
idx = pd.read_csv(config.INDEX_CSV)[['filename', 'group']].rename(columns={'group': 'burst'})
k = v2[v2.source == 'ryandpark'].merge(idx, on='filename')
print('v2 kaggle bursts in >1 split:', (k.groupby('burst').split.nunique() > 1).sum())
print('rows only in v1:', len(set(v1.filename) - set(v2.filename)), '| only in v2:', len(set(v2.filename) - set(v1.filename)))
for name, d in (('v1', v1), ('v2', v2)):
    print(f'\n== {name}: share of rows per split'); print((d.split.value_counts(normalize=True)).round(3).to_dict())
    t = d.groupby(['split', 'source', 'quality']).agg(rows=('filename', 'size'), groups=('group', 'nunique')).unstack('quality', fill_value=0)
    print(t)
o = v2[v2.source == 'own']; print('\nv2 own photos'); print(o.groupby(['split', 'group', 'quality']).size())
