"""prepare inputs for LLM experiments (direct diagnosis / planning / report fidelity)"""
import sys,os,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedShuffleSplit
import agents as A, run_lib as L
d=A.load(); f=A.perceive(d); Y=d[A.FT].copy(); yb=d['Machine failure'].values
strat=d['Machine failure'].astype(str)+d['Type']
tr,te=list(StratifiedShuffleSplit(10,test_size=0.3,random_state=2026).split(f,strat))[0]
K=A.perturb(np.random.default_rng(100),0.10)
fb,pt=L.collab(f.iloc[tr],Y.iloc[tr],f.iloc[te],0,K)
ev=A.evidence(f.iloc[te],K)
rng=np.random.default_rng(7)
fail_idx=[j for j in range(len(te)) if yb[te[j]]==1]; norm_idx=[j for j in range(len(te)) if yb[te[j]]==0]
sel=list(rng.choice(fail_idx,60,replace=False))+list(rng.choice(norm_idx,60,replace=False)); rng.shuffle(sel)
osf=K['osf']
kb=("设备知识库（专家经验阈值，可能存在误差）：\n"
    f"① 散热故障HDF：工艺温度与环境温度之差ΔT低于{K['dT']:.2f} K，且转速低于{K['n']:.0f} r/min；\n"
    f"② 功率故障PWF：输出功率P=2πnM/60低于{K['Pl']:.0f} W或高于{K['Ph']:.0f} W；\n"
    f"③ 过载故障OSF：刀具载荷累积量W=刀具磨损时间×扭矩超过限值，L、M、H级产品限值分别为{osf[0]:.0f}、{osf[1]:.0f}、{osf[2]:.0f} min·N·m；\n"
    f"④ 刀具磨损故障TWF：刀具磨损时间达到约{K['tw']:.0f} min后可能发生。")
open('prompts/kb_split0.txt','w').write(kb)
rec=[]; truth=[]; coll=[]
for k,j in enumerate(sel):
    r=te[j]; row=dict(id=k+1,type=d.Type[r],Ta=float(d['Air temperature [K]'][r]),Tp=float(d['Process temperature [K]'][r]),n=int(d['Rotational speed [rpm]'][r]),M=float(d['Torque [Nm]'][r]),wear=int(d['Tool wear [min]'][r]))
    row_p=dict(row,dT=round(f.dT.iloc[r],1),P=round(f.P.iloc[r]),W=round(f.strain.iloc[r]))
    rec.append((row,row_p)); truth.append(dict(id=k+1,fault=int(yb[r]),types=[t for t in A.FT if Y[t].iloc[r]==1]))
    coll.append(dict(id=k+1,fault=int(fb[j]),types=[t for t in ['HDF','PWF','OSF'] if pt[t].iloc[j]==1]))
for v in ('raw','phys'):
    for b in range(3):
        lines=[json.dumps(rec[i][0 if v=='raw' else 1],ensure_ascii=False) for i in range(b*40,(b+1)*40)]
        open(f'llm_runs/direct_{v}_{b}.txt','w').write('\n'.join(lines))
json.dump(dict(truth=truth,collab=coll),open('llm_runs/direct_truth.json','w'),ensure_ascii=False)
# ---- report-fidelity cases: agent messages for 20 test records
pick=[]; 
det=[j for j in range(len(te)) if fb[j]==1 and yb[te[j]]==1]; pick+=list(rng.choice(det,8,replace=False))
tool=[j for j in range(len(te)) if fb[j]==0 and f.wear.iloc[te[j]]>=K['tw']]; pick+=list(rng.choice(tool,4,replace=False))
conf=[j for j in range(len(te)) if fb[j]==0 and ev[['flag_HDF','flag_PWF','flag_OSF']].iloc[j].sum()>0]; pick+=list(rng.choice(conf,2,replace=False))
nor=[j for j in range(len(te)) if fb[j]==0 and yb[te[j]]==0 and f.wear.iloc[te[j]]<K['tw'] and ev[['flag_HDF','flag_PWF','flag_OSF']].iloc[j].sum()==0]; pick+=list(rng.choice(nor,6,replace=False))
cases=[]
for k,j in enumerate(pick):
    r=te[j]; e=ev.iloc[j]
    msg={'perceive':{'device':'M-%02d'%(k+1),'cycle':int(d.UDI[r]),'type':d.Type[r],'Ta_K':float(d['Air temperature [K]'][r]),'Tp_K':float(d['Process temperature [K]'][r]),'n_rpm':int(d['Rotational speed [rpm]'][r]),'M_Nm':float(d['Torque [Nm]'][r]),'wear_min':int(d['Tool wear [min]'][r]),'dT_K':round(float(f.dT.iloc[r]),1),'P_kW':round(float(f.P.iloc[r])/1000,2),'W_minNm':int(round(f.strain.iloc[r]))},
         'mechanism':{'margin_dT_K':round(float(e.m_dT),2),'margin_n_rpm':round(float(e.m_n),1),'margin_P_low_kW':round(float(e.m_Pl)/1000,2),'margin_P_high_kW':round(float(e.m_Ph)/1000,2),'margin_W_kminNm':round(float(e.m_osf),2),'margin_wear_min':round(float(e.m_tw),1),'violations':[t for t in ['HDF','PWF','OSF'] if e['flag_'+t]==1]},
         'diagnose':{'fault':bool(fb[j]),'types':[t for t in ['HDF','PWF','OSF'] if pt[t].iloc[j]==1]}}
    act='计划检修' if fb[j] else ('更换刀具' if f.wear.iloc[r]>=K['tw'] else '继续运行')
    msg['decide']={'action':act,'basis':('诊断为故障' if fb[j] else ('刀具磨损已达寿命阈值' if act=='更换刀具' else '未诊断出故障且刀具寿命裕度充足'))}
    cases.append(dict(case=k+1,truth_fault=int(yb[r]),messages=msg))
json.dump(cases,open('llm_runs/report_cases.json','w'),ensure_ascii=False,indent=1)
print('direct sample faults',sum(t['fault'] for t in truth),'collab correct',sum(int(a['fault']==b['fault']) for a,b in zip(coll,truth)))
print('cases',len(cases))
