import sys,os,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score
import agents as A
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')+'/'
d=A.load(); f=A.perceive(d); Y=d[A.FT].copy(); yb=d['Machine failure'].values; FT=A.FT
strat=d['Machine failure'].astype(str)+d['Type']
def metr(pb,pt,ytb,Yt):
    o={'P':precision_score(ytb,pb,zero_division=0),'R':recall_score(ytb,pb),'F1':f1_score(ytb,pb)}
    for t in FT:
        m=Yt[t].values==1; o['rec_'+t]=float(np.asarray(pt[t])[m].mean())
    # 类型识别宏F1（仅三类机理故障+TWF）
    o['typeF1']=float(np.mean([f1_score(Yt[t],pt[t],zero_division=0) for t in ['HDF','PWF','OSF']]))
    return o
def run_data(ftr,Ytr,fte,cols,seed,Xtr=None,Xte=None):
    Xtr=ftr[cols] if Xtr is None else Xtr; Xte=fte[cols] if Xte is None else Xte
    from sklearn.ensemble import RandomForestClassifier as RF
    ybt=(Ytr.sum(1)>0).astype(int)
    pb=RF(300,class_weight='balanced_subsample',random_state=seed,n_jobs=2).fit(Xtr,ybt).predict_proba(Xte)[:,1]
    pt=pd.DataFrame(0,index=fte.index,columns=FT)
    for t in FT:
        if Ytr[t].sum()>=2:
            pt[t]=(RF(200,class_weight='balanced_subsample',random_state=seed,n_jobs=2).fit(Xtr,Ytr[t]).predict_proba(Xte)[:,1]>=0.5).astype(int)
    fb=(pb>=0.5).astype(int); pt=pt.mul(fb,axis=0)
    return fb,pt
def rule_only(fte,K):
    r=A.mech(fte,K); fb=(r[['HDF','PWF','OSF']].sum(1)>0).astype(int).values; r['TWF']=0; r['RNF']=0; return fb,r
def collab(ftr,Ytr,fte,seed,K):
    Xtr=pd.concat([ftr[A.RAW],A.evidence(ftr,K)],axis=1); Xte=pd.concat([fte[A.RAW],A.evidence(fte,K)],axis=1)
    return run_data(ftr,Ytr,fte,None,seed,Xtr,Xte)
