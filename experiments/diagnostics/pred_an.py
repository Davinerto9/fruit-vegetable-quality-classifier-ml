import pandas as pd, numpy as np
from src import config
pd.set_option('display.width',250)
R=str(config.ROOT)+'/'
S=R+'experiments/diagnostics/out/'
p=pd.read_csv(R+'experiments/results/validation_predictions.csv',encoding='utf-8-sig')
M=['logistic_regression','svm_linear','random_forest','cnn_rgb']
print('== predicted regular by source'); print(pd.DataFrame({m:p[p[m+'_prediction']=='regular'].source.value_counts() for m in M}).fillna(0).astype(int))
for m in M:
    b=p[p.quality=='bad']; r=p[p.quality=='regular']
    print(f"{m}: recall bad={(b[m+'_prediction']=='bad').mean():.3f} bad->good={(b[m+'_prediction']=='good').sum()} ({(b[m+'_prediction']=='good').mean():.3%}) regular->good={(r[m+'_prediction']=='good').sum()}  | bad->good by source", b[b[m+'_prediction']=='good'].source.value_counts().to_dict())
print(p.rf_cnn_status.value_counts().to_dict())
print('both_wrong by source x quality'); bw=p[p.rf_cnn_status=='both_wrong']; print(pd.crosstab(bw.source,bw.quality))
print('error union by product (RF or CNN wrong) / support')
u=p[p.rf_cnn_status!='both_correct']; print((u['product'].value_counts()/p['product'].value_counts()).round(3).sort_values(ascending=False).to_string())
c=pd.read_csv(S+'crop_stats.csv',dtype={'dhash':str}); c=c[c.split!='test']  # test rows excluded from descriptive analysis too
print('\n== crop stats medians by source x quality (train+val)')
for col in ['raw_w','raw_h','crop_w','mask_frac','fruit_bright','fruit_contrast','fruit_sat','lap_var_128','bg_bright']:
    print(col); print(c.groupby(['source','quality'])[col].median().unstack()[['good','regular','bad']].round(2))
print('jpeg share by source',c.groupby('source').jpeg.mean().round(3).to_dict())
c2=pd.read_csv(S+'crop_stats.csv',dtype={'dhash':str})
dup=c2.groupby('md5').agg(n=('split','size'),splits=('split','nunique'),q=('quality','nunique'),src=('source','nunique'))
print('exact md5 dup sets',(dup.n>1).sum(),'across splits',(dup.splits>1).sum(),'with conflicting labels',(dup.q>1).sum(),'across sources',(dup.src>1).sum())
# near dup: identical dhash across splits (exclude degenerate)
hd=c2[c2.dhash.str.count('1').between(8,56)].groupby('dhash').agg(n=('split','size'),splits=('split','nunique'),groups=('group','nunique'),q=('quality','nunique'))
print('identical dhash sets',(hd.n>1).sum(),'spanning >1 split',(hd.splits>1).sum(),'spanning >1 group',(hd.groups>1).sum(),'conflicting quality',(hd.q>1).sum())
x=c2[c2.dhash.isin(hd[hd.splits>1].index)].sort_values('dhash')[['dhash','filename','split','group','quality']]
x.to_csv(S+'neardup_cross_split.csv',index=False); print(x.head(12).to_string())
