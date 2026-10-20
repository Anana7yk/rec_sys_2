"""Fixed coefficient curves with one shared reference forward per context batch."""
from runtime import *
from models2 import logp,categorical_kl,sync
from metrics2 import tensor,latency

def tag(lam):return str(float(lam)).replace('.','p')

def summarize(frame,ranks,d,n):
 s={k:float(v) for k,v in frame.groupby('uid').mean(numeric_only=True).mean().items() if k!='t'}
 freq=np.bincount(ranks.ravel(),weights=np.repeat(d['w'],20),minlength=n);p=freq/freq.sum()
 s.update(coverage=float((freq>0).mean()),hhi=float(p@p),n_users=int(frame.uid.nunique()),n_events=len(frame))
 for k in [10,20]:
  for metric in ['ndcg','recall']:s[f'conditional_{metric}{k}']=float(frame[frame.in_catalog==1].groupby('uid')[f'{metric}{k}'].mean().mean())
 _,ix=np.unique(d['uid'],return_index=True);rng=np.random.default_rng(4321);over=[]
 for _ in range(min(2000,len(ix)*4)):
  i,j=rng.choice(ix,2,replace=False);a=set(ranks[i]);b=set(ranks[j]);over.append(len(a&b)/len(a|b))
 s['user_jaccard']=float(np.mean(over));return s

def batch_frame(d,rank,rr,expected,kl,items,ep=None,captured=None):
 a=rank.cpu().numpy();r=rr.cpu().numpy();hit=a==(d['target']-3)[:,None];n=len(items['ids'])
 f=pd.DataFrame(dict(uid=d['uid'],t=d['t'],in_catalog=(d['target']>=3).astype(float),reward=r.mean(1),expected_reward=expected.cpu().numpy(),kl=kl.cpu().numpy(),objective=(expected-.1*kl).cpu().numpy(),duration=items['duration'][a].mean(1),logpop=np.log1p(items['pop'][a]).mean(1),top_decile=(a<max(1,int(np.ceil(n*.1)))).mean(1),history_repeat=(a[:,:,None]+3==d['x'][:,None,:]).any(2).mean(1)))
 for k in [10,20]:f[f'recall{k}']=hit[:,:k].any(1).astype(float);f[f'ndcg{k}']=(hit[:,:k]/np.log2(np.arange(k)+2)).sum(1)
 if ep is not None:f['evaluator_reward']=ep.mean(1);f['disagreement']=np.abs(ep-r).mean(1)
 if captured is not None:f['top100_reference_mass']=captured.cpu().numpy()
 return f,a

@torch.no_grad()
def curve(policy,model,d,rm,items,cfg,device,grid,out,evaluator=None):
 model.eval();frames={(family,lam):[] for family in ['full','top100'] for lam in grid};ranks={k:[] for k in frames};start=time.monotonic()
 for begin in range(0,len(d['x']),cfg['batch']):
  dd={k:v[begin:begin+cfg['batch']] for k,v in d.items()};x=tensor(dd['x'],device);g=tensor(dd['g'],device);fb=tensor(dd['fb'],device)
  lp=logp(model,x,g);r=rm.all(x,fb);pool=lp.topk(min(100,lp.shape[-1]),-1).indices;small=lp.gather(1,pool);sr=r.gather(1,pool);captured=small.exp().sum(1)
  batch={}
  for family,lam in frames:
   if family=='full':
    q=lp if lam==0 else (lp+lam*r).log_softmax(-1);rank=q.topk(20,-1).indices;expected=(q.exp()*r).sum(1);kl=categorical_kl(q,lp)
   else:
    q=(small+lam*sr).log_softmax(-1);rank=pool.gather(1,q.topk(20,-1).indices);expected=(q.exp()*sr).sum(1);kl=(q.exp()*(q-small)).sum(1)
   batch[(family,lam)]=(rank,r.gather(1,rank),expected,kl)
  # One evaluator batch for all fixed points; it never selects coefficients.
  pred=None
  if evaluator is not None:
   aa=np.concatenate([z[0].cpu().numpy() for z in batch.values()],axis=1);pred=evaluator.candidates(dd,aa)
  for j,(k,(rank,rr,expected,kl)) in enumerate(batch.items()):
   ep=None if pred is None else pred[:,j*20:(j+1)*20]
   f,a=batch_frame(dd,rank,rr,expected,kl,items,ep,captured);frames[k].append(f);ranks[k].append(a)
 out.mkdir(parents=True,exist_ok=False);rows=[]
 for (family,lam),ff in frames.items():
  f=pd.concat(ff,ignore_index=True);a=np.concatenate(ranks[(family,lam)]);name=f'{policy}_{family}_{tag(lam)}';f.to_csv(out/f'{name}_events.csv',index=False);np.save(out/f'{name}_top20.npy',a)
  rows.append(dict(policy=policy,family=family,lam=lam,method=name,**summarize(f,a,d,len(items['ids']))))
 dump(out/'TIMING.json',dict(seconds=time.monotonic()-start,shared_reference_forward=True,grid=grid));return rows

@torch.no_grad()
def rerank_scores(model,x,g,fb,rm,family,lam):
 lp=logp(model,x,g)
 if family=='full':return lp if lam==0 else (lp+lam*rm.all(x,fb)).log_softmax(-1)
 pool=lp.topk(min(100,lp.shape[-1]),-1).indices;b,k=pool.shape
 z=lp.gather(1,pool)
 if lam:
  reward=rm.actions(x[:,None,:].expand(b,k,-1).reshape(b*k,-1),fb[:,None,:].expand(b,k,-1).reshape(b*k,-1),pool.reshape(-1)).reshape(b,k);z=z+lam*reward
 return torch.full_like(lp,-torch.inf).scatter(1,pool,z).log_softmax(-1)

@torch.no_grad()
def curve_latency(model,d,rm,cfg,device,family,lam):
 x=tensor(d['x'][:cfg['batch']],device);g=tensor(d['g'][:cfg['batch']],device);fb=tensor(d['fb'][:cfg['batch']],device);times=[]
 for j in range(7):
  sync(device);start=time.perf_counter();rerank_scores(model,x,g,fb,rm,family,lam).topk(20,-1);sync(device)
  if j>=2:times.append(time.perf_counter()-start)
 return dict(batch=len(x),median_seconds=float(np.median(times)),repeats=times,rm_candidates_per_context=(0 if not lam else (model.spec['n'] if family=='full' else min(100,model.spec['n']))))

def select_coefficients(core,curves,cfg):
 rows=[]
 for seed in cfg['seeds']:
  for policy in ['pure','hybrid']:
   base=core[(core.seed==seed)&(core.method==policy)].iloc[0];rl=core[(core.seed==seed)&(core.method==policy+'_rl')].iloc[0];target=rl.reward-base.reward
   for family in ['full','top100']:
    f=curves[(curves.seed==seed)&(curves.policy==policy)&(curves.family==family)]
    for tol in [.01,.03,.05]:
     ok=f[f.ndcg20>=base.ndcg20*(1-tol)];assert len(ok)
     c=ok.sort_values(['reward','lam'],ascending=[False,True]).iloc[0]
     rows.append(dict(seed=seed,policy=policy,family=family,rule=f'tol{round(tol*100):02}',tolerance=tol,lam=float(c.lam),grid_method=c.method,validation_ndcg=float(c.ndcg20),validation_reward=float(c.reward),base_ndcg=float(base.ndcg20),base_reward=float(base.reward),target_RL_gain=float(target),adequate_reward_match=None))
    ok=f[f.ndcg20>=base.ndcg20*.95].copy();ok['gap']=np.abs((ok.reward-base.reward)-target)
    c=ok.sort_values(['gap','ndcg20','lam'],ascending=[True,False,True]).iloc[0]
    rows.append(dict(seed=seed,policy=policy,family=family,rule='reward_match',tolerance=.05,lam=float(c.lam),grid_method=c.method,validation_ndcg=float(c.ndcg20),validation_reward=float(c.reward),base_ndcg=float(base.ndcg20),base_reward=float(base.reward),target_RL_gain=float(target),reward_gap=float(c.gap),adequate_reward_match=bool(c.gap<=max(.0005,.25*abs(target)))))
 return rows
