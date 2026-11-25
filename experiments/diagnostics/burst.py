import pandas as pd
from src import config
R=str(config.ROOT)+'/'
i=pd.read_csv(R+'data/processed/index.csv'); s=pd.read_csv(R+'data/processed/crops/splits.csv')
k=s[s.source=='ryandpark'].merge(i[['filename','group']].rename(columns={'group':'idx_group'}),on='filename',how='left')
print('kaggle rows',len(k),'missing in index',k.idx_group.isna().sum())
print('split groups',k.group.nunique(),'burst groups (index)',k.idx_group.nunique())
g=k.groupby('idx_group').split.nunique(); print('burst groups spanning >1 split:',(g>1).sum(),'of',len(g))
bad=k[k.idx_group.isin(g[g>1].index)]
print('rows in split-crossing bursts:',len(bad)); print(pd.crosstab(bad.split,bad.quality))
v=k[k.split=='val']; vb=v[v.idx_group.isin(g[g>1].index)]
tg=set(k[k.split=='train'].idx_group)
print('val kaggle rows whose burst also has train rows:',v.idx_group.isin(tg).sum(),'of',len(v))
t=k[k.split=='test']; print('test kaggle rows whose burst has train rows:',t.idx_group.isin(tg).sum(),'of',len(t))
v.assign(burst_in_train=v.idx_group.isin(tg))[['filename','burst_in_train']].to_csv(R+'experiments/diagnostics/out/'+'val_burst.csv',index=False)
