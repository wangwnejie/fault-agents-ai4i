import json, re
C={c['case']:c for c in json.load(open('llm_runs/report_cases.json'))}
Rp={}; [Rp.update({int(k):v for k,v in json.load(open(f'llm_runs/out_report_{h}.json')).items()}) for h in (0,1)]
def nums_in(o):
    s=set()
    def walk(x):
        if isinstance(x,dict): [walk(v) for v in x.values()]
        elif isinstance(x,list): [walk(v) for v in x]
        elif isinstance(x,(int,float)) and not isinstance(x,bool): s.add(float(x))
        elif isinstance(x,str):
            for m in re.findall(r'-?\d+(?:\.\d+)?',x): s.add(float(m))
    walk(o); return s
tot=0; bad=[]; act_ok=0; type_ok=0; conflict_ok=[]; lens=[]
for k,txt in sorted(Rp.items()):
    msg=C[k]['messages']; allowed=nums_in(msg)
    allowed|={abs(v) for v in allowed}
    found=[float(x) for x in re.findall(r'(?<![A-Za-z])-?\d+(?:\.\d+)?',txt.replace(',','').replace('−','-'))]
    for x in found:
        tot+=1
        ok=any(abs(x-a)<=max(0.051,1e-9) or (abs(a)>=10 and abs(x-round(a))<=0.5) or abs(abs(x)-abs(a))<=0.051 for a in allowed)
        if not ok: bad.append((k,x))
    act=msg['decide']['action']; act_ok+= act in txt
    tys={'HDF':'散热','PWF':'功率','OSF':'过载'}
    t=msg['diagnose']['types']; type_ok+= all((tys[x] in txt or x in txt) for x in t)
    if msg['mechanism']['violations'] and not msg['diagnose']['fault']:
        conflict_ok.append((k, any(w in txt for w in ['不一致','矛盾','分歧','但数据诊断','未判定','提示','复核','冲突'])))
    lens.append(len(re.sub(r'\s','',txt)))
print('reports',len(Rp),'numbers',tot,'unsupported',len(bad),bad)
print('action match',act_ok,'type match',type_ok,'conflict flagged',conflict_ok,'len max',max(lens),'mean',sum(lens)/len(lens))
json.dump(dict(n=len(Rp),numbers=tot,unsupported=bad,action_ok=act_ok,type_ok=type_ok,conflict=conflict_ok,len_max=max(lens),len_mean=sum(lens)/len(lens)),open('results/res_llm_report.json','w'),ensure_ascii=False,indent=1)
