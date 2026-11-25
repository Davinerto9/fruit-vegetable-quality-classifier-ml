import pandas as pd, numpy as np, hashlib, cv2
from multiprocessing import Pool
from src import config
R=str(config.ROOT)+'/'
s=pd.read_csv(R+'data/processed/crops/splits.csv')
def f(row):
    fn,crop,mask=row
    raw=open(R+fn,'rb').read(); md5=hashlib.md5(raw).hexdigest()
    rimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    rh,rw=rimg.shape[:2]
    c=cv2.imread(R+'data/processed/crops/'+crop); mk=cv2.imread(R+'data/processed/crops/'+mask,0)>127
    h,w=c.shape[:2]; g=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
    inner=g[mk]; bg=g[~mk]
    lap=cv2.Laplacian(cv2.resize(g,(128,128)),cv2.CV_64F)
    m128=cv2.resize(mk.astype(np.uint8),(128,128),interpolation=cv2.INTER_NEAREST)>0
    hsv=cv2.cvtColor(c,cv2.COLOR_BGR2HSV)
    # background of crop (outside mask) -> describes acquisition background
    bgc=c[~mk].mean(0) if (~mk).any() else np.array([np.nan]*3)
    small=cv2.resize(np.where(mk,g,0).astype(np.uint8),(9,8),interpolation=cv2.INTER_AREA)
    dh=(small[:,1:]>small[:,:-1]).flatten(); dhash=''.join('1' if b else '0' for b in dh)
    return dict(filename=fn,md5=md5,raw_w=rw,raw_h=rh,crop_w=w,crop_h=h,mask_frac=mk.mean(),
        fruit_bright=inner.mean(),fruit_contrast=inner.std(),fruit_sat=hsv[...,1][mk].mean(),
        lap_var_128=lap[m128].var() if m128.any() else np.nan, bg_bright=bg.mean() if bg.size else np.nan,
        bg_b=bgc[0],bg_g=bgc[1],bg_r=bgc[2],jpeg=fn.lower().endswith(('jpg','jpeg')),dhash=dhash)
if __name__=='__main__':
    with Pool(10) as p: out=p.map(f, list(zip(s.filename,s.crop,s.crop_mask)), chunksize=64)
    d=pd.DataFrame(out).merge(s[['filename','source','quality','product','group','split']],on='filename')
    d.to_csv(R+'experiments/diagnostics/out/'+'crop_stats.csv',index=False)
    print('done',len(d))
