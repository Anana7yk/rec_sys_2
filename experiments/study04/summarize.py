"""Derived results only: user bootstrap, validation-selected comparators, RM evidence."""
from runtime import *
from assessment import tag
from reward3 import ci,predict,matched,error_frame,metrics
from data2 import weights
import argparse

CORE=['pure','pure_sft','pure_rl','hybrid','hybrid_sft','hybrid_rl']
METRICS=['ndcg20','recall20','reward','expected_reward','kl','objective','evaluator_reward','duration','history_repeat','top_decile','logpop','disagreement','ndcg5','recall5']

def selector_name(r):return f'{r["policy"]}_{r["family"]}_{r["rule"]}'
def event_path(out,seed,name,selected):
 if name in CORE:return out/'final'/f'{seed}_{name}_events.csv'
 r=next(z for z in selected if z['seed']==seed and selector_name(z)==name)
 return out/'final'/f'grid_{seed}_{r["policy"]}'/f'{r["grid_method"]}_events.csv'

def user_metrics(frames):
 f=pd.concat(frames,ignore_index=True);return f.groupby(['seed','uid']).mean(numeric_only=True).groupby('uid').mean(numeric_only=True)

def bootstrap_contrast(terms,cache,label):
 names=list(terms);ix=cache[names[0]].index
 for n in names:assert cache[n].index.equals(ix)
 cols=[k for k in METRICS if all(k in cache[n] for n in names)];v=sum(terms[n]*cache[n][cols].to_numpy() for n in names)
 rng=np.random.default_rng(42);w=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=2000)/len(ix);boot=w@v;mean=v.mean(0);lo,hi=np.quantile(boot,[.025,.975],axis=0)
 return [dict(comparison=label,metric=c,delta=float(mean[j]),low=float(lo[j]),high=float(hi[j]),users=len(ix)) for j,c in enumerate(cols)]

def rm_evidence(inputs,reports):
 d=dict(np.load(inputs/'final/reward_data/holdout_raw.npz'));ids=np.load(inputs/'catalog_ids.npy');mask=np.isin(d['item'],ids);d={k:v[mask] for k,v in d.items()};d['w']=weights(d['uid']);a=dict(np.load(inputs/'reward/asset.npz'))
 with (inputs/'reward/specs.pkl').open('rb') as f:specs=pickle.load(f)
 evaluator=Evaluator(inputs,a,ids);rows=[];err={};pa={}
 for name in ['mean','history','candidate','joint','residual','evaluator']:
  p=evaluator.raw(d) if name=='evaluator' else predict(d,a,specs[name]);e=error_frame(d,p);m=matched(d,p);err[name]=e.set_index('uid');pa[name]=m.set_index('uid');chance=ci(m.concordance-.5)
  rows.append(dict(kind=name,**metrics(d['y'],p,d['w']),within_mse=e.within.mean(),between_mse=e.between.mean(),matched_concordance=m.concordance.mean(),concordance_low=chance['low']+.5,concordance_high=chance['high']+.5,matched_users=len(m),matched_pairs=int(m.pairs.sum()),users=len(e),rows=len(d['uid'])))
 effects=[]
 for name in ['joint','residual','evaluator']:
  for base in ['mean','history','candidate']:
   for metric in ['mse','within']:
    effects.append(dict(model=name,control=base,metric=metric,**ci(err[name][metric]-err[base][metric])))
   effects.append(dict(model=name,control=base,metric='concordance',**ci(pa[name].concordance-pa[base].concordance)))
 pd.DataFrame(rows).to_csv(reports/'tables/RM_evidence.csv',index=False);pd.DataFrame(effects).to_csv(reports/'tables/RM_contrasts.csv',index=False)
 if not (inputs/'controls.csv').exists():
  old=pd.read_csv(B.parents[1]/'results/study03/tables/reward_action_domain.csv').query("split=='final'").set_index('kind');now=pd.DataFrame(rows).set_index('kind')
  for col in ['rmse','mae','within_mse','between_mse']:assert np.allclose(old.loc[now.index,col],now[col],atol=1e-12)
  dump(reports/'RM_TABLE_VERIFIED.json',dict(published_RM_estimates_unchanged=True,no_refit=True))
 else:dump(reports/'RM_TABLE_VERIFIED.json',dict(fresh_run=True,historical_values_not_forced=True))

def slate_aggregate(sl):
 per_seed=sl.groupby(['seed','method','uid']).mean(numeric_only=True).groupby(['seed','method']).mean(numeric_only=True)
 return per_seed,per_seed.groupby('method').mean(numeric_only=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=B/'inputs');p.add_argument('--out',type=Path,default=B/'runs/study_04');p.add_argument('--reports',type=Path,default=B.parents[1]/'validation/recomputed/study04');a=p.parse_args();(a.reports/'tables').mkdir(parents=True,exist_ok=True)
 sel=json.loads((a.out/'SELECTION.json').read_text())['rows'];pd.DataFrame(sel).to_csv(a.reports/'tables/validation_selection.csv',index=False)
 core=pd.read_csv(a.out/'final/core_metrics.csv');grid=pd.read_csv(a.out/'final/curve_metrics.csv');names=CORE+list(dict.fromkeys(selector_name(r) for r in sel));seeds=sorted(core.seed.unique());cache={};ss=[];admiss=[]
 for name in names:
  ff=[]
  for seed in seeds:
   f=pd.read_csv(event_path(a.out,seed,name,sel));f['seed']=seed;ff.append(f)
   if name in CORE:r=core[(core.seed==seed)&(core.method==name)].iloc[0].to_dict()
   else:
    pick=next(z for z in sel if z['seed']==seed and selector_name(z)==name);r=grid[(grid.seed==seed)&(grid.method==pick['grid_method'])].iloc[0].to_dict();r['method']=name
    base=core[(core.seed==seed)&(core.method==pick['policy'])].iloc[0];rl=core[(core.seed==seed)&(core.method==pick['policy']+'_rl')].iloc[0]
    admiss.append(dict(seed=seed,method=name,lam=pick['lam'],validation_guard_passed=pick['validation_ndcg']>=pick['base_ndcg']*(1-pick['tolerance']),repeat_final_loss=1-r['ndcg20']/base.ndcg20,repeat_final_guard_passed=r['ndcg20']>=base.ndcg20*(1-pick['tolerance']),validation_reward_match_adequate=pick.get('adequate_reward_match'),repeat_final_reward_gain=r['reward']-base.reward,repeat_final_RL_gain=rl.reward-base.reward,reward_gain_gap=(r['reward']-rl.reward)))
   ss.append(r)
  cache[name]=user_metrics(ff)
 pd.DataFrame(ss).to_csv(a.reports/'tables/selected_seeds.csv',index=False);ss=pd.DataFrame(ss);ss.groupby('method').mean(numeric_only=True).to_csv(a.reports/'tables/selected_mean.csv');ss.groupby('method').std(numeric_only=True).to_csv(a.reports/'tables/selected_seed_sd.csv');pd.DataFrame(admiss).to_csv(a.reports/'tables/selection_transfer.csv',index=False)
 for split in ['validation','final']:
  g=pd.read_csv(a.out/split/'curve_metrics.csv');g.to_csv(a.reports/f'tables/{split}_curve_seeds.csv',index=False);g.groupby(['policy','family','lam']).mean(numeric_only=True).to_csv(a.reports/f'tables/{split}_curves.csv')
 contrasts=[]
 for policy in ['pure','hybrid']:
  for c in [policy,policy+'_sft']+[n for n in names if n.startswith(policy+'_') and n not in [policy+'_rl',policy+'_sft']]:
   contrasts.extend(bootstrap_contrast({policy+'_rl':1,c:-1},cache,policy+'_rl - '+c))
 contrasts.extend(bootstrap_contrast({'pure_rl':1,'pure':-1,'hybrid_rl':-1,'hybrid':1},cache,'copy_interaction: (pureRL-pure)-(hybridRL-hybrid)'))
 pd.DataFrame(contrasts).to_csv(a.reports/'tables/paired_bootstrap.csv',index=False)
 sl=pd.read_csv(a.out/'final/slates.csv');per_seed,slmean=slate_aggregate(sl);per_seed.to_csv(a.reports/'tables/slate_seeds.csv');slmean.to_csv(a.reports/'tables/slate_mean.csv');sc={n:user_metrics([sl[sl.method==n]]) for n in sl.method.unique()};sr=[]
 for policy in ['pure','hybrid']:
  for c in [policy,policy+'_sft',f'{policy}_full_reward_match',f'{policy}_top100_reward_match',f'{policy}_full_tol05',f'{policy}_top100_tol05']:
   sr.extend(bootstrap_contrast({policy+'_rl':1,c:-1},sc,policy+'_rl - '+c))
 pd.DataFrame(sr).to_csv(a.reports/'tables/slate_bootstrap.csv',index=False)
 lat=pd.read_csv(a.out/'final/latency.csv');lat.to_csv(a.reports/'tables/latency_seeds.csv',index=False);lat.groupby('method').median(numeric_only=True).to_csv(a.reports/'tables/latency_median.csv')
 selectedlat=[r for r in lat.to_dict('records') if r['method'] in CORE]
 for pick in sel:
  r=lat[(lat.seed==pick['seed'])&(lat.method==pick['grid_method'])].iloc[0].to_dict();r['method']=selector_name(pick);selectedlat.append(r)
 pd.DataFrame(selectedlat).to_csv(a.reports/'tables/selected_latency_seeds.csv',index=False);pd.DataFrame(selectedlat).groupby('method').median(numeric_only=True).to_csv(a.reports/'tables/selected_latency_median.csv')
 times=[]
 for seed in seeds:
  for method,path in [('pure_rl',a.out/f'seed_{seed}/pure_rl_timing.json'),('pure_sft',a.out/f'seed_{seed}/sft_audit/timing.json'),('hybrid_rl',a.inputs/f'seed_{seed}/rl_h1_raw_timing.json'),('hybrid_sft',a.inputs/f'seed_{seed}/sft/timing.json')]:
   r=json.loads(path.read_text());r['supervised_best_ndcg_step']=r.pop('selected_step',None)
   if method=='pure_sft':
    chosen=json.loads((a.out/f'seed_{seed}/pure_sft_selection.json').read_text())['choice'];r['deployment_choice']=chosen;r['deployment_step']=0 if chosen=='initial' else (r['steps'] if chosen=='last' else r['supervised_best_ndcg_step'])
   if method=='pure_rl':r['deployment_step']=torch.load(a.out/f'seed_{seed}/pure_rl.pt',map_location='cpu',weights_only=False).get('step')
   times.append(dict(r,seed=seed,method=method,optimizer_seed=r.get('seed',seed+2000)))
 pd.DataFrame(times).to_csv(a.reports/'tables/training_cost.csv',index=False);rm_evidence(a.inputs,a.reports)
 dump(a.reports/'SUMMARY_COMPLETE.json',dict(assessment='exploratory_reassessment',policy_seeds=len(seeds),final_users=len(cache['pure']),all_tables_from_saved_results=True))
if __name__=='__main__':main()
