"""Build supplement inputs entirely from a run inside this project, no old checkout."""
from runtime import *
import argparse,shutil

def prepare(source,out):
 source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 files=['config.json','catalog.csv','catalog_ids.npy','POLICY_SEAL.json','supervised_selection.json','data/train_raw.npz','data/val_raw.npz','final/policy_data/holdout_raw.npz','final/real_diagnostic_contexts.npz','reward_data/val_raw.npz','final/reward_data/holdout_raw.npz']
 files+=[str(p.relative_to(source)) for p in (source/'pool').glob('*.npy')]
 files+=[str(p.relative_to(source)) for p in (source/'reward').iterdir() if p.suffix in ['.npz','.pkl','.cbm','.json']]
 for seed in [11,29,47]:
  files +=[f'seed_{seed}/{p}' for p in ['pure/best.pt','base.pt','rl_h1_raw.pt','sft.pt','rl_h1_raw_timing.json','sft/timing.json','hybrid_selection.json']]
 for relative in files:
  target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,target)
 shutil.copy2(source/'final/metrics.csv',out/'controls.csv')
 import polars as pl
 ids=np.load(out/'catalog_ids.npy');index={int(x):i for i,x in enumerate(ids)};artists=[set() for _ in ids]
 for item,artist in pl.read_parquet(B.parents[1]/'data/raw/artist_item_mapping.parquet').select('item_id','artist_id').iter_rows():
  ix=index.get(int(item))
  if ix is not None:artists[ix].add(int(artist))
 dump(out/'artists.json',[sorted(s) for s in artists]);files+=['controls.csv','artists.json']
 dump(out/'INPUT_MANIFEST.json',dict(source='selected frozen run of study03',files=[dict(path=p,bytes=(out/p).stat().st_size,sha256=digest(out/p)) for p in sorted(files)]))
 print('Supplement inputs prepared',len(files),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study03',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.study03,a.out)
