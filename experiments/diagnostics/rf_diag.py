import numpy as np, pandas as pd, joblib, json
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from src import config
R=str(config.ROOT)+'/'
z=np.load(R+'data/processed/classic_features.npz'); s=pd.read_csv(R+'data/processed/crops/splits.csv')
sp=s.set_index('filename').split.reindex(z['filenames']).values
X=z['X']; y=z['quality']; src=z['sources']; prod=z['products']; names=z['feature_names']
tr=sp=='train'; va=sp=='val'   # test rows never used
print('X',X.shape,'train',tr.sum(),'val',va.sum())
rf=joblib.load(R+'experiments/checkpoints/random_forest.joblib')['model']
print('params',{k:v for k,v in rf.get_params().items() if k in('n_estimators','max_depth','min_samples_leaf','min_samples_split','max_features','class_weight','bootstrap','oob_score')})
d=[e.get_depth() for e in rf.estimators_]; l=[e.get_n_leaves() for e in rf.estimators_]
print('tree depth min/med/max',min(d),np.median(d),max(d),'leaves med',np.median(l),'train rows',tr.sum())
sel=json.load(open(R+'experiments/results/random_forest.json'))['cross_validation']['selected']; print('selected',sel['params'],sel['mean_f1_macro'])
L=['good','regular','bad']
def rep(mask,tag):
    p=rf.predict(X[mask]); t=y[mask]
    sup=[c for c in L if (t==c).any()]
    print(f"{tag}: n={mask.sum()} F1(all3)={f1_score(t,p,labels=L,average='macro',zero_division=0):.4f} F1(supported {sup})={f1_score(t,p,labels=sup,average='macro',zero_division=0):.4f} cm={confusion_matrix(t,p,labels=L).tolist()}")
for part,m in (('train',tr),('val',va)):
    rep(m,part)
    for so in ['ryandpark','agrifreshnet','own']: 
        mm=m&(src==so)
        if mm.any(): rep(mm,f'  {part}/{so}')
# Kaggle binary F1 when restricting to good/bad but counting regular predictions as errors
vk=va&(src=='ryandpark'); p=rf.predict(X[vk]); print('val kaggle predicted regular:',(p=='regular').sum())
va_ag=va&(src=='agrifreshnet'); p=rf.predict(X[va_ag]); print('val agri pred counts',pd.Series(p).value_counts().to_dict())
# shared products banana/orange
for pr in ['banana','orange']:
    for so in ['ryandpark','agrifreshnet']:
        mm=va&(prod==pr)&(src==so); p=rf.predict(X[mm]); print(pr,so,'n',mm.sum(),'pred',pd.Series(p).value_counts().to_dict(),'true',pd.Series(y[mm]).value_counts().to_dict())
imp=pd.Series(rf.feature_importances_,index=names).sort_values(ascending=False)
print('top20 importances'); print(imp.head(20).round(4).to_string())
fam=imp.groupby(lambda n: n.split('_')[0] if n.startswith(('lbp','r_','g_','b_','h_','s_','v_')) else 'shape').sum(); print('importance by family'); print(fam.round(3))
# ---- Source probe: can the same 359 features identify the dataset? (train fit, val eval, exclude own)
m_tr=tr&(src!='own'); m_va=va&(src!='own')
probe=RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=42,min_samples_leaf=2).fit(X[m_tr],src[m_tr])
ps=probe.predict(X[m_va]); print('SOURCE probe val bal.acc',round(balanced_accuracy_score(src[m_va],ps),4),confusion_matrix(src[m_va],ps,labels=['ryandpark','agrifreshnet']).tolist())
# Within products shared by both sources and classes good/bad only (removes product and regular confound)
m2tr=m_tr&np.isin(prod,['banana','orange'])&np.isin(y,['good','bad']); m2va=m_va&np.isin(prod,['banana','orange'])&np.isin(y,['good','bad'])
probe2=RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=42,min_samples_leaf=2).fit(X[m2tr],src[m2tr])
ps2=probe2.predict(X[m2va]); print('SOURCE probe (banana/orange, good+bad only) val bal.acc',round(balanced_accuracy_score(src[m2va],ps2),4),'n',m2va.sum())
pi=pd.Series(probe.feature_importances_,index=names).sort_values(ascending=False); print('probe top10'); print(pi.head(10).round(4).to_string())
# shape-only and color-only source probes
for fam_name,cols in (('shape',[i for i,n in enumerate(names) if not n.startswith(('lbp','r_','g_','b_','h_','s_','v_'))]),('lbp',[i for i,n in enumerate(names) if n.startswith('lbp')]),('color',[i for i,n in enumerate(names) if n.startswith(('r_','g_','b_','h_','s_','v_'))])):
    pr_=RandomForestClassifier(n_estimators=150,n_jobs=-1,random_state=42,min_samples_leaf=2).fit(X[m2tr][:,cols],src[m2tr])
    print(' probe family',fam_name,'bal.acc',round(balanced_accuracy_score(src[m2va],pr_.predict(X[m2va][:,cols])),4))
