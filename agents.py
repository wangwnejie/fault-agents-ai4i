import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
D=os.environ.get('AI4I_CSV',os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','ai4i2020.csv'))
FT=['TWF','HDF','PWF','OSF','RNF']
def load():
    d=pd.read_csv(D); return d
# ---- 数据感知智能体: 物理量构造
def perceive(d):
    f=pd.DataFrame(index=d.index)
    f['Ta']=d['Air temperature [K]']; f['Tp']=d['Process temperature [K]']; f['n']=d['Rotational speed [rpm]']
    f['M']=d['Torque [Nm]']; f['wear']=d['Tool wear [min]']; f['type']=d['Type'].map({'L':0,'M':1,'H':2})
    f['dT']=f.Tp-f.Ta; f['P']=f.M*f.n*2*np.pi/60; f['strain']=f.wear*f.M
    return f
# ---- 机理诊断智能体: 设备知识规则（阈值可扰动以模拟知识不准确）
BASE=dict(dT=8.6,n=1380,Pl=3500,Ph=9000,osf=(11000,12000,13000),tw=200)
def mech(f,K=BASE):
    r=pd.DataFrame(0,index=f.index,columns=FT)
    r['HDF']=((f.dT<K['dT'])&(f.n<K['n'])).astype(int)
    r['PWF']=((f.P<K['Pl'])|(f.P>K['Ph'])).astype(int)
    lim=f['type'].map({0:K['osf'][0],1:K['osf'][1],2:K['osf'][2]})
    r['OSF']=(f.strain>lim).astype(int)
    r['TWF']=(f.wear>=K['tw']).astype(int)   # 刀具寿命预警区
    return r
def perturb(rng,s):
    j=lambda v: v*(1+rng.uniform(-s,s))
    return dict(dT=j(8.6),n=j(1380),Pl=j(3500),Ph=j(9000),osf=tuple(j(v) for v in (11000,12000,13000)),tw=j(200))
# ---- 数据驱动诊断智能体
RAW=['Ta','Tp','n','M','wear','type']; ENG=RAW+['dT','P','strain']
def data_agent(ftr,ytr,cols,seed):
    m={}
    yb=(ytr[FT].sum(1)>0).astype(int)
    m['bin']=RandomForestClassifier(300,min_samples_leaf=1,class_weight='balanced_subsample',random_state=seed,n_jobs=2).fit(ftr[cols],yb)
    for t in FT:
        if ytr[t].sum()>=2: m[t]=RandomForestClassifier(200,class_weight='balanced_subsample',random_state=seed,n_jobs=2).fit(ftr[cols],ytr[t])
    return m
def data_pred(m,f,cols):
    p=pd.DataFrame(0.0,index=f.index,columns=FT)
    for t in FT:
        if t in m: p[t]=m[t].predict_proba(f[cols])[:,1]
    return m['bin'].predict_proba(f[cols])[:,1],p
def evidence(f,K):
    """机理智能体向协调者提交的证据：各故障机理阈值裕度（负值表示越限）"""
    e=pd.DataFrame(index=f.index)
    e['m_dT']=f.dT-K['dT']; e['m_n']=f.n-K['n']; e['m_Pl']=f.P-K['Pl']; e['m_Ph']=K['Ph']-f.P
    lim=f['type'].map({0:K['osf'][0],1:K['osf'][1],2:K['osf'][2]}); e['m_osf']=(lim-f.strain)/1000
    e['m_tw']=K['tw']-f.wear
    r=mech(f,K); 
    for t in ['HDF','PWF','OSF','TWF']: e['flag_'+t]=r[t]
    return e
