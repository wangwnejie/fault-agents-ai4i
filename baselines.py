"""Exp: comparison of classifiers (RF / SVM / GBDT / MLP) x inputs (raw / raw+physical / raw+margin evidence)."""
import sys,os,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import agents as A
d=A.load(); f=A.perceive(d); Y=d[A.FT].copy(); yb=d['Machine failure'].values; FT=A.FT
strat=d['Machine failure'].astype(str)+d['Type']
sss=list(StratifiedShuffleSplit(10,test_size=0.3,random_state=2026).split(f,strat))
def mk(name,seed,n_pos,n):
    if name=='RF': return RandomForestClassifier(300,class_weight='balanced_subsample',random_state=seed,n_jobs=2)
    if name=='SVM': return make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',kernel='rbf',class_weight='balanced',probability=False,random_state=seed))
    if name=='GBDT':
        w=(n-n_pos)/max(n_pos,1)
        return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed,class_weight={0:1,1:min(w,10)})
    if name=='MLP': return make_pipeline(StandardScaler(),MLPClassifier((128,64,32),activation='relu',alpha=1e-3,learning_rate_init=1e-3,max_iter=400,early_stopping=True,validation_fraction=0.15,n_iter_no_change=20,random_state=seed))
def fitpred(name,Xtr,ytr,Xte,seed):
    m=mk(name,seed,int(ytr.sum()),len(ytr))
    if name=='MLP':   # oversample positives for the MLP (no class_weight support)
        pos=np.flatnonzero(ytr==1); rep=np.random.default_rng(seed).choice(pos,max(len(ytr)//10-len(pos),0))
        idx=np.r_[np.arange(len(ytr)),rep]; m.fit(Xtr.iloc[idx],ytr[idx])
    else: m.fit(Xtr,ytr)
    return m.predict(Xte)
def run(name,Xtr,Ytr,Xte,seed):
    if name=='RF':
        import run_lib as L
        return L.run_data(None,Ytr,Xte,None,seed,Xtr,Xte)
    ybt=(Ytr.sum(1)>0).astype(int).values
    fb=fitpred(name,Xtr,ybt,Xte,seed).astype(int)
    pt=pd.DataFrame(0,index=Xte.index,columns=FT)
    for t in ['HDF','PWF','OSF']:
        pt[t]=fitpred(name,Xtr,Ytr[t].values,Xte,seed).astype(int)
    pt=pt.mul(fb,axis=0); return fb,pt
def metr(fb,pt,yt,Yt):
    o={'P':precision_score(yt,fb,zero_division=0),'R':recall_score(yt,fb),'F1':f1_score(yt,fb)}
    for t in ['HDF','PWF','OSF']: o['rec_'+t]=float(pt[t].values[Yt[t].values==1].mean())
    return o
if __name__=='__main__':
    R={}; t0=time.time()
    for name in ['RF','SVM','GBDT','MLP']:
        for inp in ['raw','phys','evid']:
            rows=[]
            for i,(tr,te) in enumerate(sss):
                K=A.perturb(np.random.default_rng(100+i),0.10)
                ftr,fte=f.iloc[tr],f.iloc[te]
                if inp=='raw': Xtr,Xte=ftr[A.RAW],fte[A.RAW]
                elif inp=='phys': Xtr,Xte=ftr[A.ENG],fte[A.ENG]
                else: Xtr=pd.concat([ftr[A.RAW],A.evidence(ftr,K)],axis=1); Xte=pd.concat([fte[A.RAW],A.evidence(fte,K)],axis=1)
                rows.append(metr(*run(name,Xtr,Y.iloc[tr],Xte,i),yb[te],Y.iloc[te]))
            df=pd.DataFrame(rows); R[name+'|'+inp]={'mean':df.mean().to_dict(),'std':df.std().to_dict(),'F1':df.F1.tolist()}
            print(name,inp,{k:round(v,3) for k,v in df.mean().items()},round(time.time()-t0),flush=True)
    json.dump(R,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','res_baselines.json'),'w'),ensure_ascii=False,indent=1)
