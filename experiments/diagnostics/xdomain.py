import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from src import config
R=str(config.ROOT)+'/'
z=np.load(R+'data/processed/classic_features.npz'); s=pd.read_csv(R+'data/processed/crops/splits.csv')
sp=s.set_index('filename').split.reindex(z['filenames']).values
X=z['X']; y=z['quality']; src=z['sources']; prod=z['products']
tr=sp=='train'; va=sp=='val'
def fit(m): return RandomForestClassifier(n_estimators=160,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=42).fit(X[m],y[m])
L=['good','regular','bad']
# A: train Agri only (3 classes) -> eval Kaggle val (good/bad)
mA=tr&(src=='agrifreshnet'); rfA=fit(mA)
for name,mm in (('Kaggle val (all products)',va&(src=='ryandpark')),('Kaggle val banana+orange',va&(src=='ryandpark')&np.isin(prod,['banana','orange'])),('Agri val (in-domain)',va&(src=='agrifreshnet'))):
    p=rfA.predict(X[mm]); t=y[mm]
    pb=np.where(p=='regular','regular',p)
    sup=[c for c in L if (t==c).any()]
    print(f'A train=Agri -> {name}: n={mm.sum()} F1(supported)={f1_score(t,p,labels=sup,average="macro"):.4f} cm={confusion_matrix(t,p,labels=L).tolist()}')
    # collapse regular->bad as "not good" to probe good-vs-notgood ranking
    pg=rfA.predict_proba(X[mm])[:,list(rfA.classes_).index('good')]
    if set(np.unique(t))>={'good','bad'}:
        from sklearn.metrics import roc_auc_score
        mk=np.isin(t,['good','bad']); print('    AUC P(good) good-vs-bad:',round(roc_auc_score(t[mk]=='good',pg[mk]),4))
# B: train Kaggle only (2 classes) -> eval Agri val good/bad
mB=tr&(src=='ryandpark'); rfB=fit(mB)
for name,mm in (('Agri val good+bad',va&(src=='agrifreshnet')&np.isin(y,['good','bad'])),('Agri val banana+orange good+bad',va&(src=='agrifreshnet')&np.isin(y,['good','bad'])&np.isin(prod,['banana','orange'])),('Kaggle val (in-domain)',va&(src=='ryandpark'))):
    p=rfB.predict(X[mm]); t=y[mm]
    from sklearn.metrics import roc_auc_score
    pg=rfB.predict_proba(X[mm])[:,list(rfB.classes_).index('good')]
    print(f'B train=Kaggle -> {name}: n={mm.sum()} F1(good,bad)={f1_score(t,p,labels=["good","bad"],average="macro"):.4f} AUC={roc_auc_score(t=="good",pg):.4f} cm={confusion_matrix(t,p,labels=["good","bad"]).tolist()}')
    if 'Agri val good+bad'==name:
        # Agri regular: where does a Kaggle-only model place them? P(good)
        mr=va&(src=='agrifreshnet')&(y=='regular'); pr=rfB.predict_proba(X[mr])[:,list(rfB.classes_).index('good')]
        mg=va&(src=='agrifreshnet')&(y=='good'); mb=va&(src=='agrifreshnet')&(y=='bad')
        print('    mean P(good) under Kaggle-only model: agri good',round(rfB.predict_proba(X[mg])[:,1].mean() if list(rfB.classes_)[1]=='good' else rfB.predict_proba(X[mg])[:,list(rfB.classes_).index('good')].mean(),3),
              'agri regular',round(pr.mean(),3),'agri bad',round(rfB.predict_proba(X[mb])[:,list(rfB.classes_).index('good')].mean(),3))
