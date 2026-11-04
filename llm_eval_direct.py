import json, numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
T=json.load(open('llm_runs/direct_truth.json')); truth={t['id']:t for t in T['truth']}; coll={t['id']:t for t in T['collab']}
def load(v):
    o={}
    for b in range(3): o.update({int(r['id']):r for r in json.load(open(f'llm_runs/out_direct_{v}_{b}.json'))})
    return o
res={}
for nm,P in [('LLM直接诊断(原始测量值)',load('raw')),('LLM直接诊断(含物理量)',load('phys')),('多智能体协同',coll)]:
    ids=sorted(truth); assert all(i in P for i in ids),nm
    y=[truth[i]['fault'] for i in ids]; p=[int(P[i]['fault']) for i in ids]
    # type accuracy on true mechanism faults
    tp=[i for i in ids if set(truth[i]['types'])&{'HDF','PWF','OSF'}]
    typ=np.mean([ set(P[i]['types'])&{'HDF','PWF','OSF'} == set(truth[i]['types'])&{'HDF','PWF','OSF'} for i in tp])
    res[nm]=dict(P=precision_score(y,p),R=recall_score(y,p),F1=f1_score(y,p),acc=float(np.mean(np.array(y)==np.array(p))),type_exact=float(typ),n=len(ids),n_mech=len(tp))
    print(nm,{k:round(v,3) for k,v in res[nm].items()})
json.dump(res,open('results/res_llm_direct.json','w'),ensure_ascii=False,indent=1)
