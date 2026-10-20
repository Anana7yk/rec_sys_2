"""One-command supplement: fits -> validation-only fixed curves -> seal -> repeat-final."""
from runtime import *
from assessment import curve,curve_latency,select_coefficients
from horizon2 import greedy_slates
from metrics2 import latency
import argparse

def assess(inputs,out,device,split):
 cfg,rm,items,d=assets(inputs,device,final=split=='final');dc=d['val' if split=='validation' else 'final'];grid=json.loads((B.parents[1]/'configs/study04/supplement.json').read_text())['lambda_grid'];dest=out/split;dest.mkdir(exist_ok=False)
 evaluator=Evaluator(inputs,rm.asset,items['ids']) if split=='final' else None
 rows=[];crows=[];lats=[];slates=[]
 selected=json.loads((out/'SELECTION.json').read_text())['rows'] if split=='final' else []
 for seed in cfg['seeds']:
  models=core_models(inputs,out,seed,device)
  for name,m in models.items():
   policy='pure' if name.startswith('pure') else 'hybrid';ref=models[policy]
   summary,f,rank=evaluate('rl_pure' if name=='pure_rl' else 'base',m,ref,dc,rm,items,cfg,device,evaluator=evaluator,quick=split!='final')
   f.to_csv(dest/f'{seed}_{name}_events.csv',index=False);np.save(dest/f'{seed}_{name}_top20.npy',rank);rows.append(dict(seed=seed,method=name,**summary))
   if split=='final':
    lats.append(dict(seed=seed,method=name,**latency('base',m,ref,dc,rm,items,cfg,device)))
    slates.extend(dict(seed=seed,method=name,**r) for r in greedy_slates('base',m,ref,dc,rm,evaluator,items,cfg,device))
   print(split,'CORE',seed,name,summary['ndcg20'],summary['reward'],flush=True)
  for policy in ['pure','hybrid']:
   m=models[policy];cr=curve(policy,m,dc,rm,items,cfg,device,grid,dest/f'grid_{seed}_{policy}',evaluator)
   crows.extend(dict(seed=seed,**r) for r in cr)
   if split=='final':
    for family in ['full','top100']:
     for lam in grid:
      name=f'{policy}_{family}_{str(float(lam)).replace(".","p")}'
      lats.append(dict(seed=seed,method=name,policy=policy,family=family,lam=lam,**curve_latency(m,dc,rm,cfg,device,family,lam)))
    # Unique chosen coefficients, then named selectors reuse the same generated lists.
    choices=[r for r in selected if r['seed']==seed and r['policy']==policy and r['rule'] in ['tol05','reward_match']];cache={}
    for r in choices:
     k=(r['family'],r['lam'])
     if k not in cache:
      method='mixed_full' if r['family']=='full' else 'mixed100';cache[k]=greedy_slates(method,m,m,dc,rm,evaluator,items,cfg,device,r['lam'])
     slates.extend(dict(seed=seed,method=f'{policy}_{r["family"]}_{r["rule"]}',**z) for z in cache[k])
   print(split,'GRID',seed,policy,'done',flush=True)
 pd.DataFrame(rows).to_csv(dest/'core_metrics.csv',index=False);pd.DataFrame(crows).to_csv(dest/'curve_metrics.csv',index=False)
 if split=='final':pd.DataFrame(lats).to_csv(dest/'latency.csv',index=False);pd.DataFrame(slates).to_csv(dest/'slates.csv',index=False)
 else:
  selection=select_coefficients(pd.DataFrame(rows),pd.DataFrame(crows),cfg);dump(out/'SELECTION.json',dict(rows=selection,criterion='validation only; fixed grid; no final access in this stage',assessment='exploratory_reassessment'))

def seal(inputs,out):
 from provenance import create_seal
 root=B.parents[1]
 files=[root/'docs/PROTOCOL_STUDY04.md',root/'configs/study04/supplement.json',*sorted((root/'recsys').glob('*.py')),B/'runtime.py',B/'assessment.py',B/'run.py',B.parent/'study03/train.py',out/'SELECTION.json',*sorted(out.glob('seed_*/*.pt'))]
 files+=sorted(f for f in inputs.rglob('*') if f.is_file())
 files+=sorted(f for f in (out/'validation').rglob('*') if f.is_file())
 dump(out/'SEAL.json',create_seal(files,out,inputs))

def checkseal(out):
 from provenance import verify_seal
 sealpath=out/'SEAL_REPAIR_01.json' if (out/'SEAL_REPAIR_01.json').exists() else out/'SEAL.json'
 return verify_seal(sealpath,out)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','fit','validation','final']);p.add_argument('--inputs',type=Path,default=B/'inputs');p.add_argument('--out',type=Path,default=B/'runs/study_04');p.add_argument('--device',default='mps');a=p.parse_args();torch.set_num_threads(6);start=time.monotonic();a.out.mkdir(parents=True,exist_ok=a.stage!='all')
 if a.stage in ['all','fit']:fit_pure(a.inputs,a.out,a.device)
 if a.stage in ['all','validation']:assess(a.inputs,a.out,a.device,'validation');seal(a.inputs,a.out)
 if a.stage in ['all','final']:
  checkseal(a.out);dump(a.out/'REPEAT_FINAL_OPENED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),independent=False,seal=digest(a.out/'SEAL.json')));assess(a.inputs,a.out,a.device,'final');checkseal(a.out)
 dump(a.out/f'TIME_{a.stage}.json',dict(seconds=time.monotonic()-start))
if __name__=='__main__':main()
