import sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from assessment import rerank_scores,select_coefficients
from models2 import HybridGPT,logp

class Reward:
 def all(self,x,fb):return torch.arange(120,device=x.device).float().expand(len(x),-1)/120
 def actions(self,x,fb,a):return a.float()/120

def test_grid_distribution_matches_sampling_and_original_point():
 torch.manual_seed(2);m=HybridGPT(120,8,16,1,4).eval();x=torch.tensor([[3,4,1,0,0,0,0,0]]);g=torch.tensor([[1,2,3,0,0,0,0,0]]);fb=torch.ones(1,4);r=Reward();lp=logp(m,x,g)
 q=rerank_scores(m,x,g,fb,r,'full',0);assert torch.equal(q,lp)
 q=rerank_scores(m,x,g,fb,r,'full',2);expected=(lp+2*r.all(x,fb)).softmax(-1);assert torch.allclose(q.exp(),expected,atol=1e-7)
 t=rerank_scores(m,x,g,fb,r,'top100',0);pool=lp.topk(100,-1).indices;assert (t.exp()>0).sum()==100
 assert torch.equal(t.topk(20,-1).indices,lp.topk(20,-1).indices)
 torch.testing.assert_close(t.exp().sum(1),torch.ones(1));dist=torch.distributions.Categorical(logits=t);a=dist.sample((10,));assert torch.isin(a,pool).all();assert torch.allclose(dist.log_prob(a),t[0,a[:,0]].reshape(-1,1),atol=1e-6)

def test_selection_uses_guard_and_reward_match_without_final():
 core=pd.DataFrame([dict(seed=11,method='pure',ndcg20=.1,reward=.6),dict(seed=11,method='pure_rl',ndcg20=.098,reward=.603),dict(seed=11,method='hybrid',ndcg20=.2,reward=.7),dict(seed=11,method='hybrid_rl',ndcg20=.198,reward=.703)])
 rows=[]
 for policy,nd,r in [('pure',.1,.6),('hybrid',.2,.7)]:
  for family in ['full','top100']:
   for lam,rel,gain in [(0,1,0),(1,.98,.003),(2,.94,.02)]:rows.append(dict(seed=11,policy=policy,family=family,lam=lam,ndcg20=nd*rel,reward=r+gain,method=f'{policy}_{family}_{lam}'))
 s=select_coefficients(core,pd.DataFrame(rows),dict(seeds=[11]));assert all(r['lam']==0 for r in s if r['rule']=='tol01');assert all(r['lam']==1 for r in s if r['rule']!='tol01');assert all(r['adequate_reward_match'] for r in s if r['rule']=='reward_match')


def test_slate_summary_does_not_overweight_users_with_more_targets():
 from summarize import slate_aggregate
 f=pd.DataFrame(dict(seed=[11]*4,method=['m']*4,uid=[1,1,1,2],ndcg5=[0.,0.,0.,1.],reward=[.2,.2,.2,.8]))
 per_seed,mean=slate_aggregate(f)
 assert mean.loc['m','ndcg5']==.5
 assert np.isclose(mean.loc['m','reward'],.5)
