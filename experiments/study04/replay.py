"""Actual checkpoint inference replay, separately from exact saved-ranking verification."""
from runtime import *
from assessment import rerank_scores
from models2 import logp
from metrics2 import tensor
import argparse

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=B/'inputs');p.add_argument('--out',type=Path,default=B/'runs/study_04');p.add_argument('--report',type=Path,default=B.parents[1]/'validation/study04_replay.json');p.add_argument('--device',default='cpu');p.add_argument('--limit',type=int,default=64);a=p.parse_args();torch.set_num_threads(6)
 cfg,rm,items,d=assets(a.inputs,a.device,final=True);dd={k:v[:a.limit] for k,v in d['final'].items()};x=tensor(dd['x'],a.device);g=tensor(dd['g'],a.device);fb=tensor(dd['fb'],a.device);selection=json.loads((a.out/'SELECTION.json').read_text())['rows'];rows=[]
 for seed in cfg['seeds']:
  models=core_models(a.inputs,a.out,seed,a.device);configs=[(name,m,'base',0,None) for name,m in models.items()]
  for r in selection:
   if r['seed']==seed and r['rule'] in ['tol05','reward_match']:configs.append((f'{r["policy"]}_{r["family"]}_{r["rule"]}',models[r['policy']],r['family'],r['lam'],r))
  for name,m,family,lam,r in configs:
   lp=logp(m,x,g) if family=='base' else rerank_scores(m,x,g,fb,rm,family,lam);rank=lp.topk(20,-1).indices.cpu().numpy()
   prefix=a.out/'final'/f'{seed}_{name}' if r is None else a.out/'final'/f'grid_{seed}_{r["policy"]}'/r['grid_method']
   saved=np.load(str(prefix)+'_top20.npy')[:len(x)];events=pd.read_csv(str(prefix)+'_events.csv').iloc[:len(x)];reward=rm.all(x,fb).gather(1,tensor(rank,a.device)).mean(1).cpu().numpy();fraction=float((saved==rank).mean());assert fraction>=.99,(seed,name,fraction)
   rows.append(dict(seed=seed,method=name,contexts=len(x),exact_rows=int((saved==rank).all(1).sum()),rank_elements_agreement=fraction,max_Top20_reward_difference=float(np.abs(events.reward.to_numpy()-reward).max())))
 dump(a.report,dict(device=a.device,checkpoint_inference=True,full_retraining=False,rows=rows));print('replayed',len(rows),'configs',flush=True)
if __name__=='__main__':main()
