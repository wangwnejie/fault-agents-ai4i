import json, collections
R={r['id']:r for r in json.load(open('llm_runs/plan_requests.json'))}
NAMES={'perceive','mechanism','diagnose','decide'}
def norm(a): return str(a).strip().lower()
def check(plan,req):
    """return dict(valid, correct, seq)"""
    if isinstance(plan,dict) and 'ask' in plan:
        return dict(ask=True,valid=req['kind']=='ask',correct=req['kind']=='ask',seq=('ASK',),err='' if req['kind']=='ask' else 'unneeded_ask')
    if req['kind']=='ask': return dict(ask=False,valid=False,correct=False,seq=tuple(s.get('agent') for s in plan),err='guessed')
    seq=[s.get('agent') for s in plan]; err=[]
    if any(a not in NAMES for a in seq): err.append('unknown_agent')
    pos={a:i for i,a in enumerate(seq)}
    need={'full':['perceive','mechanism','diagnose','decide'],'diag':['perceive','mechanism','diagnose'],'tool':['perceive','mechanism','diagnose','decide']}[req['kind']]
    for a in need:
        if a not in pos: err.append('missing_'+a)
    order=[('perceive','mechanism'),('mechanism','diagnose'),('diagnose','decide'),('perceive','diagnose')]
    for a,b in order:
        if a in pos and b in pos and pos[a]>pos[b]: err.append(f'order_{a}_{b}')
    # dependencies declared
    steps={s.get('step'):s.get('agent') for s in plan}
    for s in plan:
        dep=[steps.get(x) for x in s.get('depends_on',[])]
        if s.get('agent')=='mechanism' and 'perceive' not in dep: err.append('dep_mech')
        if s.get('agent')=='diagnose' and not ('mechanism' in dep): err.append('dep_diag')
        if s.get('agent')=='decide' and 'diagnose' not in dep: err.append('dep_decide')
    valid=not err
    p=[s for s in plan if s.get('agent')=='perceive']
    args_ok=bool(p) and norm(p[0].get('args',{}).get('device_id',''))==norm(req['device']) and all(str(p[0].get('args',{}).get(k,'')).strip() for k in ('start','end'))
    extra=req['kind']=='diag' and 'decide' in pos
    return dict(ask=False,valid=valid,correct=valid and args_ok,seq=tuple(seq),err=','.join(err)+('' if args_ok else ',args'),extra=extra)
out=collections.defaultdict(dict)
for r in (1,2,3):
    for h in (0,1):
        P=json.load(open(f'llm_runs/out_plan_r{r}_{h}.json'))
        for k,v in P.items(): out[int(k)][r]=check(v,R[int(k)])
rows=[]
for i in sorted(out):
    for r in (1,2,3): rows.append((i,R[i]['kind'],r,out[i][r]))
n=len(rows)
summ=dict(n=n,valid=sum(x[3]['valid'] for x in rows),correct=sum(x[3]['correct'] for x in rows))
bykind=collections.defaultdict(lambda:[0,0,0])
for i,k,r,c in rows: bykind[k][0]+=1; bykind[k][1]+=c['valid']; bykind[k][2]+=c['correct']
cons=sum(len({out[i][r]['seq'] for r in (1,2,3)})==1 for i in out)
summ['consistent_requests']=cons; summ['bykind']=dict(bykind)
summ['extra_decide_in_diag']=sum(1 for i,k,r,c in rows if c.get('extra'))
summ['errors']=[(i,r,c['err']) for i,k,r,c in rows if not c['correct']]
print(json.dumps(summ,ensure_ascii=False,indent=1))
json.dump(summ,open('results/res_llm_plan.json','w'),ensure_ascii=False,indent=1)
