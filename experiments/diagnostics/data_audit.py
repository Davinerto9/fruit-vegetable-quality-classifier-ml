import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
from src import config
R=str(config.ROOT)+'/'
s=pd.read_csv(R+'data/processed/crops/splits.csv'); m=pd.read_csv(R+'data/processed/crops/manifest.csv')
print('splits rows',len(s),'groups',s.group.nunique())
print(s.groupby('split').agg(rows=('filename','size'),groups=('group','nunique')))
print('\n== source x quality (rows / groups) per split')
for sp in ['train','val','test']:
    d=s[s.split==sp]; print('--',sp)
    print(pd.crosstab(d.source,d.quality,margins=True)[['good','regular','bad','All']])
    print(d.groupby(['source','quality']).group.nunique().unstack(fill_value=0))
print('\n== proportions by source (all splits)')
print(pd.crosstab(s.source,s.quality,normalize='index').round(3))
print('\n== product x source x quality (val)')
v=s[s.split=='val']; print(pd.crosstab([v['product'],v.source],v.quality))
print('\n== product x source (all) ')
print(pd.crosstab(s['product'],s.source))
print('\n== own photos')
own_m=m[m.source=='own']; print('manifest own rows',len(own_m)); print(own_m.groupby(['decision']).size())
own=s[s.source=='own']; print(own.groupby(['split','product','quality']).agg(rows=('filename','size'),groups=('group','nunique')))
print(own[['filename','product','quality','group','split']].sort_values('split').to_string())
print('own groups x split'); print(own.groupby('group').split.unique())
print('own excluded/not candidate:'); print(own_m[own_m.decision!='candidate'][['filename','quality','decision','detail']].to_string())
cand_own=own_m[own_m.decision=='candidate']; print('own candidates not in splits:', sorted(set(cand_own.filename)-set(own.filename)))
print('\n== group leakage checks')
g=s.groupby('group').split.nunique(); print('groups in >1 split:', (g>1).sum())
# AgriFreshNET aug originals
a=s[s.source=='agrifreshnet'].copy()
a['base']=a.filename.str.replace(r'aug_\d+_','',regex=True).str.lower()
bb=a.groupby('base').split.nunique(); print('agri base names in >1 split:',(bb>1).sum())
a['stem']=a.filename.str.split('/').str[-1].str.replace(r'aug_\d+_','',regex=True).str.lower()
# same original file name across different quality folders?
print('agri aug rows', a.filename.str.contains('aug_').sum(), 'of', len(a))
print('\n== image stats by source x quality (manifest stats cols)')
for c in ['width','height','brightness','contrast','sharpness','d_norm']:
    print(c); print(s.groupby(['source','quality'])[c].median().unstack().round(3))

