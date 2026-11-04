"""Re-run the LLM experiments through the Anthropic Messages API (or any compatible endpoint).
Usage:  export ANTHROPIC_API_KEY=...;  python run_llm.py direct|plan|report  [--model claude-sonnet-5-5]
Outputs are written to llm_runs/ with the same file names as the recorded outputs, then evaluate with llm_eval_*.py.
"""
import os, sys, json, re, urllib.request, argparse
H=os.path.dirname(os.path.abspath(__file__))
def call(system, user, model, max_tokens=4000):
    req=urllib.request.Request('https://api.anthropic.com/v1/messages',
        data=json.dumps({'model':model,'max_tokens':max_tokens,'system':system,'messages':[{'role':'user','content':user}]}).encode(),
        headers={'x-api-key':os.environ['ANTHROPIC_API_KEY'],'anthropic-version':'2023-06-01','content-type':'application/json'})
    out=json.load(urllib.request.urlopen(req,timeout=600))
    return ''.join(b.get('text','') for b in out['content']), out.get('usage',{})
def jsonpart(t):
    m=re.search(r'(\[.*\]|\{.*\})',t,re.S); return json.loads(m.group(1))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('task'); ap.add_argument('--model',default='claude-sonnet-5-5'); a=ap.parse_args()
    coord=open(f'{H}/prompts/coordinator.txt',encoding='utf-8').read(); kb=open(f'{H}/prompts/kb_split0.txt',encoding='utf-8').read(); usage=[]
    if a.task=='direct':
        for v in ('raw','phys'):
            for b in range(3):
                rec=open(f'{H}/llm_runs/direct_{v}_{b}.txt',encoding='utf-8').read()
                u=(kb+'\n\n请根据上述知识库判断下列生产周期是否发生故障及故障类型（HDF、PWF、OSF、TWF，可多选或为空），只输出JSON数组，每项形如{"id":1,"fault":1,"types":["HDF"]}。不得调用任何工具。\n'+rec)
                t,us=call('你是工业设备故障诊断专家。',u,a.model); usage.append(us)
                json.dump(jsonpart(t),open(f'{H}/llm_runs/out_direct_{v}_{b}.json','w',encoding='utf-8'),ensure_ascii=False)
    elif a.task=='plan':
        R=json.load(open(f'{H}/llm_runs/plan_requests.json',encoding='utf-8'))
        for r in (1,2,3):
            for h,(lo,hi) in enumerate([(0,10),(10,20)]):
                o={}
                for q in R[lo:hi]:
                    t,us=call(coord,'运维请求：'+q['req']+'\n本次只输出调用计划。',a.model,1500); usage.append(us); o[str(q['id'])]=jsonpart(t)
                json.dump(o,open(f'{H}/llm_runs/out_plan_r{r}_{h}.json','w',encoding='utf-8'),ensure_ascii=False)
    elif a.task=='report':
        for h in (0,1):
            C=json.load(open(f'{H}/llm_runs/report_in_{h}.json',encoding='utf-8')); o={}
            for c in C:
                t,us=call(coord,'各智能体返回的消息如下，请撰写运维报告：\n'+json.dumps(c['messages'],ensure_ascii=False),a.model,1000); usage.append(us); o[str(c['case'])]=t.strip()
            json.dump(o,open(f'{H}/llm_runs/out_report_{h}.json','w',encoding='utf-8'),ensure_ascii=False)
    json.dump(usage,open(f'{H}/results/usage_{a.task}.json','w'),indent=1); print('done; token usage saved to results/')
