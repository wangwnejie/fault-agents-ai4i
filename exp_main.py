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
sss=list(StratifiedShuffleSplit(10,test_size=0.3,random_state=2026).split(f,strat))
OUT={}
# ---- Exp1 main (knowledge error ±10%)
M={k:[] for k in ['数据智能体','数据智能体+物理特征','机理智能体(精确知识)','机理智能体(近似知识)','多智能体协同']}
for i,(tr,te) in enumerate(sss):
    rng=np.random.default_rng(100+i); K=A.perturb(rng,0.10)
    ftr,fte,Ytr,Yte,yt=f.iloc[tr],f.iloc[te],Y.iloc[tr],Y.iloc[te],yb[te]
    M['数据智能体'].append(metr(*run_data(ftr,Ytr,fte,A.RAW,i),yt,Yte))
    M['数据智能体+物理特征'].append(metr(*run_data(ftr,Ytr,fte,A.ENG,i),yt,Yte))
    M['机理智能体(精确知识)'].append(metr(*rule_only(fte,A.BASE),yt,Yte))
    M['机理智能体(近似知识)'].append(metr(*rule_only(fte,K),yt,Yte))
    M['多智能体协同'].append(metr(*collab(ftr,Ytr,fte,i,K),yt,Yte))
OUT['main']={k:{c:[float(pd.DataFrame(v)[c].mean()),float(pd.DataFrame(v)[c].std())] for c in pd.DataFrame(v).columns} for k,v in M.items()}
OUT['main_raw']={k:pd.DataFrame(v)['F1'].tolist() for k,v in M.items()}
for k in M: print(k,{c:round(x[0],3) for c,x in OUT['main'][k].items()},flush=True)
# ---- Exp2 knowledge error sweep
OUT['kerr']={}
for s in [0,0.05,0.1,0.2,0.3]:
    a,b=[],[]
    for i,(tr,te) in enumerate(sss[:5]):
        K=A.perturb(np.random.default_rng(200+i),s) if s>0 else A.BASE
        ftr,fte,Ytr,Yte,yt=f.iloc[tr],f.iloc[te],Y.iloc[tr],Y.iloc[te],yb[te]
        a.append(metr(*rule_only(fte,K),yt,Yte)['F1']); b.append(metr(*collab(ftr,Ytr,fte,i,K),yt,Yte)['F1'])
    OUT['kerr'][s]={'rule':a,'collab':b}; print('kerr',s,np.mean(a),np.mean(b),flush=True)
# ---- Exp3 label scarcity
OUT['scar']={}
for frac in [0.05,0.1,0.25,0.5,1.0]:
    r={'数据智能体':[],'数据智能体+物理特征':[],'多智能体协同':[]}
    for i,(tr,te) in enumerate(sss[:5]):
        rng=np.random.default_rng(300+i); sub=tr if frac==1 else rng.choice(tr,int(len(tr)*frac),replace=False)
        while yb[sub].sum()<3: sub=rng.choice(tr,int(len(tr)*frac),replace=False)
        K=A.perturb(np.random.default_rng(100+i),0.10)
        ftr,fte,Ytr,Yte,yt=f.iloc[sub],f.iloc[te],Y.iloc[sub],Y.iloc[te],yb[te]
        r['数据智能体'].append(metr(*run_data(ftr,Ytr,fte,A.RAW,i),yt,Yte)['F1'])
        r['数据智能体+物理特征'].append(metr(*run_data(ftr,Ytr,fte,A.ENG,i),yt,Yte)['F1'])
        r['多智能体协同'].append(metr(*collab(ftr,Ytr,fte,i,K),yt,Yte)['F1'])
    OUT['scar'][frac]=r; print('scar',frac,{k:round(np.mean(v),3) for k,v in r.items()},int(yb[sub].sum()),flush=True)
json.dump(OUT,open(O+'res2.json','w'),ensure_ascii=False)
