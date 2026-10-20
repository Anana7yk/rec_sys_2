"""Supplement-only fits and frozen-input loading. Shared trainer imported from study03; no writes to its archived run."""
import os,sys,json,pickle,time,hashlib,copy
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parents[1]/'recsys'));sys.path.insert(0,str(B.parent/'study03'));sys.path.insert(0,str(B))
import numpy as np,pandas as pd,torch
from data2 import dump,map_data
from models2 import load,save,freeze
from reward3 import Reward,Evaluator
from metrics2 import evaluate,key
from fit2 import rl
from train import fit

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def assets(inputs,device,final=False):
 cfg=json.loads((inputs/'config.json').read_text());ids=np.load(inputs/'catalog_ids.npy');a=dict(np.load(inputs/'reward/asset.npz'))
 with (inputs/'reward/specs.pkl').open('rb') as f:specs=pickle.load(f)
 sel=json.loads((inputs/'reward/selection.json').read_text());rm=Reward(a,specs[sel['primary']],ids,device);rm.beta=cfg['beta']
 cat=pd.read_csv(inputs/'catalog.csv');items=dict(ids=ids,pop=cat['count'].to_numpy(),duration=cat['duration'].to_numpy(),positive_train=np.load(inputs/'pool/positive_counts.npy'))
 if (inputs/'artists.json').exists():items['artists']=[set(x) for x in json.loads((inputs/'artists.json').read_text())]
 d={s:map_data(dict(np.load(inputs/f'data/{s}_raw.npz')),ids) for s in ['train','val']}
 if final:d['final']=map_data(dict(np.load(inputs/'final/policy_data/holdout_raw.npz')),ids)
 return cfg,rm,items,d

def fit_pure(inputs,out,device,seeds=None,steps=None):
 cfg,rm,items,d=assets(inputs,device);cfg['batch']=64
 for seed in seeds or cfg['seeds']:
  p=out/f'seed_{seed}';p.mkdir(parents=True,exist_ok=False)
  ref=freeze(load(inputs/f'seed_{seed}/pure/best.pt',device));assert ref.alpha==0
  base,_,_=evaluate('base',ref,ref,d['val'],rm,items,cfg,device,quick=True)
  model,seconds=rl(cfg,d,items,rm,ref,p/'pure_rl.pt',device,seed,1,False,steps or 640)
  # fit() expects a root with immutable training inputs and a writable output subdir.
  work=p/'fit_workspace';work.mkdir()
  for name in ['config.json','catalog.csv','catalog_ids.npy','pool','data']:
   (work/name).symlink_to((inputs/name).resolve(),target_is_directory=(inputs/name).is_dir())
  m=fit(work,'sft',seed+3000,ref.spec['d'],.0001,.01,24000,False,device,initial=ref,seconds=seconds)
  cand=[]
  for name,mm in [('initial',ref),('best_ndcg',m),('last',load(work/'sft/last.pt',device))]:
   s,_,_=evaluate('sft',mm,ref,d['val'],rm,items,cfg,device,quick=True);cand.append((name,mm,s))
  name,m,summary=max(cand,key=lambda z:key(z[2],base));save(p/'pure_sft.pt',m)
  dump(p/'pure_sft_selection.json',dict(choice=name,candidates=[dict(checkpoint=n,**s) for n,_,s in cand],criterion='raw Top20 reward under95% NDCG guard, identical to RL'))
  # Keep artifacts in one flat seed directory; links were read-only conveniences.
  import shutil
  shutil.move(str(work/'sft'),str(p/'sft_audit'))
  for f in work.iterdir():f.unlink()
  work.rmdir()
  dump(p/'TRAINING_COMPLETE.json',dict(seed=seed,RL_actions=(steps or 640)*64,paired_RL_compute_seconds=seconds,pure_alpha=m.alpha))
  print('PURE_FITS_DONE',seed,flush=True)

def core_models(inputs,out,seed,device):
 p=inputs/f'seed_{seed}';n=out/f'seed_{seed}'
 return {'pure':freeze(load(p/'pure/best.pt',device)),'pure_sft':freeze(load(n/'pure_sft.pt',device)),'pure_rl':freeze(load(n/'pure_rl.pt',device)),'hybrid':freeze(load(p/'base.pt',device)),'hybrid_sft':freeze(load(p/'sft.pt',device)),'hybrid_rl':freeze(load(p/'rl_h1_raw.pt',device))}
