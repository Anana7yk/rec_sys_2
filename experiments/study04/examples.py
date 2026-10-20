"""Descriptive examples only, never model selection."""
from runtime import *
from data2 import lookup
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=B/'inputs');p.add_argument('--out',type=Path,default=B/'runs/study_04');p.add_argument('--reports',type=Path,default=B.parents[1]/'validation/recomputed/study04');args=p.parse_args();r=args.out;ids=np.load(args.inputs/'catalog_ids.npy');raw=dict(np.load(args.inputs/'final/policy_data/holdout_raw.npz'));seed=11
 before=np.load(r/'final/11_pure_top20.npy');after=np.load(r/'final/11_pure_rl_top20.npy');changed=(before!=after).any(1);users=np.unique(raw['uid'][changed])[:3];lines=['# Examples: actual saved recommendations','', 'Rule fixed for this descriptive view: seed11, first three increasing user IDs with a changed pure Top20. These are illustrative, not a representative sample or evidence of user utility. Numbers are anonymized raw item IDs. Curly-bracketed history groups have no within-timestamp order. The logged next item is a ranking label only; no response is assigned to generated actions.',''];records=[]
 for u in users:
  j=np.flatnonzero((raw['uid']==u)&changed)[0];gg=raw['g'][j];rr=raw['raw'][j];groups=[rr[(gg==g)&(rr>=0)].tolist() for g in np.unique(gg) if g>0];groups=[g for g in groups if g][-2:]
  lines.extend([f'## User {u}, t={raw["t"][j]}','',f'Last available groups: {" → ".join("{"+", ".join(map(str,z))+"}" for z in groups)}. Logged next item: {raw["item"][j]}.','', '| Policy | Top10 item IDs | Top20 proxy | Mean duration, sec |','|---|---|---:|---:|'])
  for name in ['pure','pure_rl','hybrid','hybrid_rl']:
   rank=np.load(r/'final'/f'{seed}_{name}_top20.npy')[j];f=pd.read_csv(r/'final'/f'{seed}_{name}_events.csv').iloc[j];a=ids[rank[:10]].tolist();lines.append(f'| {name} | {a} | {f.reward:.5f} | {f.duration:.1f} |');records.append(dict(uid=int(u),t=int(raw['t'][j]),method=name,top10=a,Top20_proxy=float(f.reward),duration=float(f.duration)))
  lines.append('')
 lines+=['Across all500 users, pure RL shifts mean recommended duration by about−10sec and raises popular-decile concentration; hybrid RL shifts it by about−3sec and reduces history repeats. Individual examples can depart from these averages. Aggregate estimates and uncertainty, not these three lists, determine the conclusion.']
 (args.reports/'EXAMPLES.md').write_text('\n'.join(lines)+'\n');dump(args.reports/'examples.json',records)
if __name__=='__main__':main()
