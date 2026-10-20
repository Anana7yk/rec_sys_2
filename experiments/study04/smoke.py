"""Tiny synthetic frozen-input supplement: real RL/SFT/grid/seal/evaluation on CPU."""
from runtime import *
from run import assess,seal,checkseal
from data2 import weights
from reward3 import design
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=B/'runs/smoke');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);i=a.out/'inputs';i.mkdir();r=a.out/'results';r.mkdir();torch.set_num_threads(2);rng=np.random.default_rng(710);n=128;ids=np.arange(1000,1000+n)
 cfg=dict(seeds=[11],context=8,batch=64,rm_cal_end=20000000,rl_lr=5e-5,beta=.1,gamma=.9,bootstrap=100);dump(i/'config.json',cfg);np.save(i/'catalog_ids.npy',ids);pd.DataFrame(dict(item_id=ids,count=np.arange(n,0,-1),duration=np.full(n,200))).to_csv(i/'catalog.csv',index=False)
 def data(rows,offset):
  raw=np.full((rows,8),-1,dtype=np.int64);g=np.zeros_like(raw)
  for j in range(rows):raw[j,:3]=ids[(j+np.arange(3))%n];raw[j,3]=-2;g[j,:4]=np.arange(1,5)
  return dict(uid=np.arange(rows)%4,t=np.arange(rows)+offset,item=ids[(np.arange(rows)+3)%n],raw=raw,g=g,fb=np.tile(np.array([.6,.2,.3,.7],dtype=np.float32),(rows,1)),w=weights(np.arange(rows)%4),repeat=np.zeros(rows),y=rng.random(rows))
 for name,rows,offset in [('data/train_raw.npz',64,21000000),('data/val_raw.npz',8,23000000),('final/policy_data/holdout_raw.npz',8,25000000)]:
  path=i/name;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data(rows,offset))
 d=map_data(data(64,21000000),ids);pool=i/'pool';pool.mkdir()
 arrays=dict(x=d['x'],g=d['g'],context=np.arange(64),target=d['target']-3,offset=np.arange(0,65,16),users=np.arange(4),uid=np.repeat(np.arange(4),16),t=d['t'],positive_counts=np.bincount(d['target']-3,minlength=n))
 for k,v in arrays.items():np.save(pool/f'{k}.npy',v)
 asset=dict(ids=ids,order=np.arange(n),candidate=rng.normal(scale=.1,size=(n+1,4)),correction=np.zeros(n+1));spec=dict(kind='joint',coef=rng.normal(scale=.01,size=37),cal_a=1.,cal_b=.6);(i/'reward').mkdir();np.savez_compressed(i/'reward/asset.npz',**asset)
 with (i/'reward/specs.pkl').open('wb') as f:pickle.dump({'joint':spec},f)
 dump(i/'reward/selection.json',dict(primary='joint'));dump(i/'reward/evaluator_cal.json',dict(a=1.,b=0.))
 from catboost import CatBoostRegressor
 dd=data(64,21000000);cb=CatBoostRegressor(iterations=4,depth=2,verbose=False,thread_count=2);cb.fit(design(dd,asset,'joint')[:,1:],dd['y']);cb.save_model(str(i/'reward/evaluator.cbm'))
 p=i/'seed_11';p.mkdir();torch.manual_seed(11)
 from models2 import HybridGPT
 m=HybridGPT(n,8,16,1,4);save(p/'pure.pt',m);(p/'pure').mkdir();(p/'pure.pt').rename(p/'pure/best.pt');m.alpha=.25;m.spec['alpha']=.25
 for name in ['base.pt','sft.pt','rl_h1_raw.pt']:save(p/name,m)
 start=time.monotonic();fit_pure(i,r,'cpu',seeds=[11],steps=4);assess(i,r,'cpu','validation');seal(i,r);checkseal(r);assess(i,r,'cpu','final');checkseal(r)
 dump(a.out/'SMOKE_COMPLETE.json',dict(seconds=time.monotonic()-start,scope='synthetic test only; 4 actual RL updates, time-matched SFT, full fixed grids, selections, seal, evaluation and greedy slates; no research labels'))
if __name__=='__main__':main()
