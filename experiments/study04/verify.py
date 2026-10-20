"""Read-only result, seal and optional byte-for-byte previous-study checks."""
from runtime import *
from run import checkseal
from assessment import tag
from data2 import lookup
import argparse,ast

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=B/'inputs');p.add_argument('--out',type=Path,default=B/'runs/study_04');p.add_argument('--reports',type=Path,default=B.parents[1]/'results/study04');a=p.parse_args();checkseal(a.out)
 checks={}
 checks['seal_verification']=checkseal(a.out)
 manifest=json.loads((a.inputs/'INPUT_MANIFEST.json').read_text())
 for r in manifest['files']:assert digest(a.inputs/r['path'])==r['sha256'],r['path']
 checks['input_manifest_files']=len(manifest['files']);checks['seal_files']=len(json.loads((a.out/'SEAL.json').read_text())['files'])
 ids=np.load(a.inputs/'catalog_ids.npy');cfg=json.loads((a.inputs/'config.json').read_text());raw=dict(np.load(a.inputs/'final/policy_data/holdout_raw.npz'));target=lookup(ids,raw['item'])-3;count=0
 for path in [*a.inputs.glob('data/*_raw.npz'),*a.inputs.glob('reward_data/*_raw.npz'),a.inputs/'final/policy_data/holdout_raw.npz',a.inputs/'final/reward_data/holdout_raw.npz']:
  d=dict(np.load(path));assert np.all(d['ht']<d['t'][:,None]);assert np.all(d['fbmax']<d['t']);assert np.all((d['raw']==-2).sum(1)==1)
 checks['strict_past_and_query_markers']=True
 pool=a.inputs/'pool';tt=np.load(pool/'t.npy',mmap_mode='r');uu=np.load(pool/'uid.npy',mmap_mode='r');yy=np.load(pool/'target.npy',mmap_mode='r');off=np.load(pool/'offset.npy');users=np.load(pool/'users.npy')
 assert tt.max()<cfg['train_end'];assert np.isin(users,cfg['dev_user_ids']).all() and not np.isin(users,cfg['holdout_user_ids']).any();assert yy.max()<len(ids);assert off[0]==0 and off[-1]==len(yy) and (np.diff(off)>0).all();assert np.array_equal(uu[off[:-1]],users) and np.array_equal(uu[off[1:]-1],users)
 checks['full_SFT_pool_train_only_and_disjoint_users']=True
 core=pd.read_csv(a.out/'final/core_metrics.csv');curves=pd.read_csv(a.out/'final/curve_metrics.csv')
 for kind,table in [('core',core),('grid',curves)]:
  for row in table.to_dict('records'):
   seed=int(row['seed']);name=row['method'];prefix=a.out/'final'/f'{seed}_{name}' if kind=='core' else a.out/'final'/f'grid_{seed}_{row["policy"]}'/name
   rank=np.load(str(prefix)+'_top20.npy');f=pd.read_csv(str(prefix)+'_events.csv');assert rank.shape==(len(target),20);assert 0<=rank.min() and rank.max()<len(ids);assert np.all(np.diff(np.sort(rank,1),axis=1)>0);assert np.array_equal(f.uid,raw['uid']) and np.array_equal(f.t,raw['t'])
   hit=rank==target[:,None]
   for k in [10,20]:
    assert np.allclose(f[f'recall{k}'],hit[:,:k].any(1));assert np.allclose(f[f'ndcg{k}'],(hit[:,:k]/np.log2(np.arange(k)+2)).sum(1))
   means=f.groupby('uid').mean(numeric_only=True).mean()
   for metric in ['ndcg10','ndcg20','recall10','recall20','reward','expected_reward','kl','duration','history_repeat','evaluator_reward']:
    if metric in means:assert np.isclose(means[metric],row[metric],atol=1e-12), (seed,name,metric)
   count+=1
 checks['ranking_outputs_independently_verified']=count
 for seed in cfg['seeds']:
  p=a.inputs/f'seed_{seed}';pure=torch.load(p/'pure/best.pt',map_location='cpu',weights_only=False);hybrid=torch.load(p/'base.pt',map_location='cpu',weights_only=False)
  assert pure['spec']['alpha']==0 and hybrid['spec']['alpha']==.25
  assert all(torch.equal(pure['state'][k],hybrid['state'][k]) for k in pure['state'])
  t=json.loads((a.out/f'seed_{seed}/pure_rl_timing.json').read_text());s=json.loads((a.out/f'seed_{seed}/sft_audit/timing.json').read_text());assert t['steps']==640 and t['generated_actions']==40960;assert t['compute_seconds']<=s['compute_seconds']<t['compute_seconds']+max(1,5*s['compute_seconds']/s['steps'])
  for name in ['pure_rl','pure_sft']:assert torch.load(a.out/f'seed_{seed}/{name}.pt',map_location='cpu',weights_only=False)['spec']['alpha']==0
  for policy in ['pure','hybrid']:
   z=np.load(a.out/'final'/f'{seed}_{policy}_top20.npy')
   for family in ['full','top100']:assert np.array_equal(z,np.load(a.out/'final'/f'grid_{seed}_{policy}'/f'{policy}_{family}_0p0_top20.npy'))
 checks['same_initial_parameters_fixed_copy_gates_and_compute_match']=True
 # All frozen controls must reproduce the old full-sample summaries on the same backend.
 old=pd.read_csv(a.inputs/'controls.csv' if (a.inputs/'controls.csv').exists() else B.parents[1]/'results/study03/tables/main_seeds.csv')
 for row in core.to_dict('records'):
  mapping={'pure':'pure','hybrid':'base','hybrid_rl':'rl_h1_raw','hybrid_sft':'sft'}
  if row['method'] in mapping:
   r=old[(old.seed==row['seed'])&(old.method==mapping[row['method']])].iloc[0]
   for col in ['ndcg20','recall20','reward','duration','history_repeat']:assert np.isclose(row[col],r[col],atol=1e-10),(row['method'],col,row[col],r[col])
 checks['study03_frozen_controls_reproduced']=True
 sl=pd.read_csv(a.out/'final/slates.csv')
 for x in sl.actions:
  acts=ast.literal_eval(x);assert len(acts)==5 and len(set(acts))==5 and min(acts)>=0 and max(acts)<len(ids)
 checks['no_service_tokens_or_duplicate_slate_actions']=True
 means=pd.read_csv(a.reports/'tables/selected_mean.csv').set_index('method');seeds=pd.read_csv(a.reports/'tables/selected_seeds.csv');derived=seeds.groupby('method').mean(numeric_only=True)
 for col in ['ndcg20','reward','expected_reward','duration']:assert np.allclose(means.loc[derived.index,col],derived[col],atol=1e-12)
 checks['tables_equal_saved_seed_results']=True
 slmean=pd.read_csv(a.reports/'tables/slate_mean.csv').set_index('method')
 independently=sl.groupby(['seed','method','uid']).mean(numeric_only=True).groupby(['seed','method']).mean(numeric_only=True).groupby('method').mean(numeric_only=True)
 for metric in ['ndcg5','recall5','reward','evaluator_reward']:assert np.allclose(slmean.loc[independently.index,metric],independently[metric],atol=1e-12)
 contrasts=pd.read_csv(a.reports/'tables/slate_bootstrap.csv')
 for row in contrasts.to_dict('records'):
  left,right=row['comparison'].split(' - ');assert np.isclose(row['delta'],slmean.loc[left,row['metric']]-slmean.loc[right,row['metric']],atol=1e-12),(left,right,row['metric'])
 checks['slate_macro_user_means_equal_bootstrap_point_estimates']=True
 dump(B.parents[1]/'validation/study04_verification.json',checks);print(checks,flush=True)
if __name__=='__main__':main()
