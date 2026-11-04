import sys,os,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedShuffleSplit
import agents as A
from run_lib import run_data, collab, rule_only
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')+'/'
R2=json.load(open(O+'res2.json')); raw=R2['main_raw']
print('collab vs phys',ttest_rel(raw['多智能体协同'],raw['数据智能体+物理特征']), np.mean(np.array(raw['多智能体协同'])-np.array(raw['数据智能体+物理特征'])), sum(a>b for a,b in zip(raw['多智能体协同'],raw['数据智能体+物理特征'])))
print('collab vs raw',ttest_rel(raw['多智能体协同'],raw['数据智能体']).pvalue)
d=A.load(); f=A.perceive(d); Y=d[A.FT].copy(); yb=d['Machine failure'].values
strat=d['Machine failure'].astype(str)+d['Type']
sss=list(StratifiedShuffleSplit(10,test_size=0.3,random_state=2026).split(f,strat))
print('wear>=200 frac',(f.wear>=200).mean(),'TWF wear range',f.wear[Y.TWF==1].min(),f.wear[Y.TWF==1].max())
CF,CM,CT=10.0,1.0,0.3   # 非计划停机、计划检修、换刀（相对单位）
def cost(fail,typ_twf,act_maint,act_tool):
    """逐条记录成本：故障且未被对应措施覆盖->CF；检修->CM；换刀->CT"""
    c=np.where(act_maint,CM,0.0)+np.where(act_tool,CT,0.0)
    covered=act_maint | (act_tool & typ_twf)
    c=c+np.where(fail & ~covered,CF,0.0)
    return c.sum()/len(fail)*1000
res={k:[] for k in ['事后维修','定期换刀','数据智能体决策','多智能体协同决策']}; ratio={}
for i,(tr,te) in enumerate(sss):
    K=A.perturb(np.random.default_rng(100+i),0.10)
    ftr,fte,Ytr,Yte=f.iloc[tr],f.iloc[te],Y.iloc[tr],Y.iloc[te]
    fail=yb[te]==1; twf=Yte.TWF.values==1; n=len(te); z=np.zeros(n,bool)
    res['事后维修'].append(cost(fail,twf,z,z))
    tool=(fte.wear>=200).values
    res['定期换刀'].append(cost(fail,twf,z,tool))
    fb,_=run_data(ftr,Ytr,fte,A.RAW,i); res['数据智能体决策'].append(cost(fail,twf,fb.astype(bool),z))
    fb2,_=collab(ftr,Ytr,fte,i,K); tool2=(fte.wear>=K['tw']).values & ~fb2.astype(bool)
    res['多智能体协同决策'].append(cost(fail,twf,fb2.astype(bool),tool2))
out={k:[float(np.mean(v)),float(np.std(v))] for k,v in res.items()}; print(out)
# sensitivity to CF/CM ratio for best two
sens={}
for cf in [5,10,20,50]:
    CF=cf; r={}
    for nm in ['事后维修','定期换刀','数据智能体决策','多智能体协同决策']: r[nm]=[]
    for i,(tr,te) in enumerate(sss[:5]):
        K=A.perturb(np.random.default_rng(100+i),0.10); ftr,fte,Ytr,Yte=f.iloc[tr],f.iloc[te],Y.iloc[tr],Y.iloc[te]
        fail=yb[te]==1; twf=Yte.TWF.values==1; z=np.zeros(len(te),bool)
        r['事后维修'].append(cost(fail,twf,z,z)); r['定期换刀'].append(cost(fail,twf,z,(fte.wear>=200).values))
        fb,_=run_data(ftr,Ytr,fte,A.RAW,i); r['数据智能体决策'].append(cost(fail,twf,fb.astype(bool),z))
        fb2,_=collab(ftr,Ytr,fte,i,K); r['多智能体协同决策'].append(cost(fail,twf,fb2.astype(bool),(fte.wear>=K['tw']).values & ~fb2.astype(bool)))
    sens[cf]={k:float(np.mean(v)) for k,v in r.items()}; print(cf,sens[cf])
# case records from split 0
tr,te=sss[0]; K=A.perturb(np.random.default_rng(100),0.10); fte=f.iloc[te]
ev=A.evidence(fte,K); fb2,pt2=collab(f.iloc[tr],Y.iloc[tr],fte,0,K)
cases={}
for t in ['HDF','OSF']:
    idx=[j for j in range(len(te)) if Y.iloc[te[j]][t]==1 and fb2[j]==1][0]
    rid=te[idx]; cases[t]={'UDI':int(d.UDI[rid]),'Type':d.Type[rid],**{c:float(f.loc[rid,c]) for c in ['Ta','Tp','n','M','wear','dT','P','strain']},
        **{c:float(ev.iloc[idx][c]) for c in ev.columns},'pred':{k:int(pt2.iloc[idx][k]) for k in A.FT},'K':{k:(list(v) if isinstance(v,tuple) else float(v)) for k,v in K.items()}}
json.dump({'cost':out,'sens':sens,'cases':cases,'C':[10,1,0.3]},open(O+'res3.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(cases,ensure_ascii=False,indent=0)[:1500])
