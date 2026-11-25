import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src import config
R=str(config.ROOT)+'/'
z=np.load(R+'data/processed/classic_features.npz'); s=pd.read_csv(R+'data/processed/crops/splits.csv'); i=pd.read_csv(R+'data/processed/index.csv')
fn=z['filenames']; sp=s.set_index('filename').split.reindex(fn).values; burst=i.set_index('filename').group.reindex(fn).values
tr=sp=='train'; X=z['X'][tr]; y=z['quality'][tr]; src=z['sources'][tr]; g_old=z['groups'][tr]; g_new=burst[tr]
L=['good','regular','bad']
for name,g in (('file groups (current split)',g_old),('burst groups (index.csv)',g_new)):
    res=[]
    for k,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(X,y,g)):
        rf=RandomForestClassifier(n_estimators=160,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=42).fit(X[a],y[a])
        p=rf.predict(X[b]); kb=src[b]=='ryandpark'; ab=src[b]=='agrifreshnet'
        res.append((f1_score(y[b],p,labels=L,average='macro'),f1_score(y[b][kb],p[kb],labels=['good','bad'],average='macro'),f1_score(y[b][ab],p[ab],labels=L,average='macro')))
    r=np.array(res); print(name,'| overall %.4f±%.4f | kaggle(2cl) %.4f±%.4f | agri(3cl) %.4f±%.4f'%(r[:,0].mean(),r[:,0].std(),r[:,1].mean(),r[:,1].std(),r[:,2].mean(),r[:,2].std()))
