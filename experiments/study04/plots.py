"""Static scientific figures, all rows are frozen grids/validation selections."""
from runtime import *
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL={'full':'#2878b5','top100':'#e58b2b','RL':'#b3233d','SFT':'#587d4b'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=B/'runs/study_04');p.add_argument('--reports',type=Path,default=B.parents[1]/'validation/recomputed/study04');a=p.parse_args();fdir=a.reports/'figures';fdir.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for row,split in enumerate(['validation','final']):
  g=pd.read_csv(a.out/split/'curve_metrics.csv').groupby(['policy','family','lam']).mean(numeric_only=True);core=pd.read_csv(a.out/split/'core_metrics.csv').groupby('method').mean(numeric_only=True)
  for col,policy in enumerate(['pure','hybrid']):
   ax=axes[row,col];base=core.loc[policy]
   for family in ['full','top100']:
    z=g.loc[(policy,family)].sort_index();x=100*z.ndcg20/base.ndcg20;y=z.reward-base.reward
    ax.plot(x,y,marker='o' if family=='full' else 'x',ms=5,label=family,color=COL[family],alpha=.85)
    if family=='full':
     for lam in [0.,.5,2.,5.,10.]:ax.annotate(f'{lam:g}',(x.loc[lam],y.loc[lam]),xytext=(4,4),textcoords='offset points',fontsize=8,color=COL[family])
   for method,color,marker,label in [(policy+'_rl',COL['RL'],'*','RL'),(policy+'_sft',COL['SFT'],'s','selected SFT')]:
    z=core.loc[method];ax.scatter(100*z.ndcg20/base.ndcg20,z.reward-base.reward,s=115 if marker=='*' else 45,color=color,marker=marker,label=label,zorder=5)
   ax.axvline(95,color='gray',linestyle=':',lw=1);ax.scatter([100],[0],c='black',marker='D',label='reference',zorder=6,s=25);ax.set_title(f'{policy}: {"validation selection" if row==0 else "exploratory repeat-final"}');ax.set_xlabel('NDCG@20 / reference, %');ax.set_ylabel('Top20 proxy reward gain');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(fdir/'tradeoff_grid.png');fig.savefig(fdir/'tradeoff_grid.svg');plt.close(fig)

 diff=pd.read_csv(a.reports/'tables/paired_bootstrap.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.5))
 for ax,metric,title in zip(axes,['ndcg20','reward'],['Change in NDCG@20','Change in Top20 proxy reward']):
  for j,policy in enumerate(['pure','hybrid']):
   z=diff[(diff.comparison==f'{policy}_rl - {policy}')&(diff.metric==metric)].iloc[0];ax.errorbar(z.delta,j,xerr=[[z.delta-z.low],[z.high-z.delta]],fmt='o',capsize=4,color=['#2878b5','#b3233d'][j])
  ax.axvline(0,color='gray',lw=1);ax.set_yticks([0,1],['Pure GPT-2','Hybrid']);ax.set_title(title);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(fdir/'copy_effect.png');fig.savefig(fdir/'copy_effect.svg');plt.close(fig)

 rm=pd.read_csv(a.reports/'tables/RM_evidence.csv');fig,axes=plt.subplots(1,3,figsize=(12,4));ys=np.arange(len(rm))
 for ax,col,title in zip(axes[:2],['rmse','within_mse'],['Overall observed-response RMSE','Within-user centered MSE']):ax.barh(ys,rm[col],color=['#9d9d9d','#8ab0cd','#deb78a','#2878b5','#6ea082','#b3233d']);ax.set_yticks(ys,rm.kind);ax.set_title(title)
 axes[2].errorbar(rm.matched_concordance,ys,xerr=[rm.matched_concordance-rm.concordance_low,rm.concordance_high-rm.matched_concordance],fmt='o',capsize=3);axes[2].axvline(.5,color='gray',linestyle='--');axes[2].set_yticks(ys,rm.kind);axes[2].set_title('Matched concordance, user 95% CI')
 for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
 fig.suptitle('Frozen RM evidence: observed in-catalog interactions; no causal item effect');fig.tight_layout();fig.savefig(fdir/'rm_limits.png');fig.savefig(fdir/'rm_limits.svg');plt.close(fig)

 # All fixed points, including those failing relevance: no favorable point filtering.
 final=pd.read_csv(a.out/'final/curve_metrics.csv');lat=pd.read_csv(a.out/'final/latency.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,policy in zip(axes,['pure','hybrid']):
  for family in ['full','top100']:
   z=final[(final.policy==policy)&(final.family==family)].groupby('method').mean(numeric_only=True);c=lat.set_index(['seed','method']).reset_index();c=c[c.method.isin(z.index)].groupby('method').median(numeric_only=True);ax.scatter(1000*c.loc[z.index,'median_seconds'],z.reward,label=family,color=COL[family],s=28)
  core=pd.read_csv(a.out/'final/core_metrics.csv').groupby('method').mean(numeric_only=True);c=lat.groupby('method').median(numeric_only=True)
  for name,label,color,marker in [(policy, 'reference','black','D'),(policy+'_rl','RL',COL['RL'],'*')]:ax.scatter(1000*c.loc[name,'median_seconds'],core.loc[name,'reward'],s=110 if marker=='*' else 40,color=color,marker=marker,label=label)
  ax.set_title(policy+' — all fixed coefficients');ax.set_xlabel('Warm batch64 scoring + Top20, ms');ax.set_ylabel('Top20 proxy reward');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(fdir/'amortization.png');fig.savefig(fdir/'amortization.svg');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 core=pd.read_csv(a.out/'final/core_metrics.csv').groupby('method').mean(numeric_only=True)
 for ax,policy in zip(axes,['pure','hybrid']):
  z=final[(final.policy==policy)&(final.family=='full')].groupby('lam').mean(numeric_only=True).sort_index();base=core.loc[policy]
  ax.plot(100*z.ndcg20/base.ndcg20,z.expected_reward-base.expected_reward,'o-',color=COL['full'],label='full-catalog q(lambda)')
  for lam in [0.,.5,3.,5.,10.]:ax.annotate(f'{lam:g}',(100*z.loc[lam,'ndcg20']/base.ndcg20,z.loc[lam,'expected_reward']-base.expected_reward),xytext=(4,4),textcoords='offset points',fontsize=8)
  rl=core.loc[policy+'_rl'];ax.scatter(100*rl.ndcg20/base.ndcg20,rl.expected_reward-base.expected_reward,s=120,marker='*',color=COL['RL'],label='RL');ax.axvline(95,color='gray',linestyle=':');ax.set_title(policy+' — exploratory repeat-final');ax.set_xlabel('NDCG@20 / reference, %');ax.set_ylabel('Expected categorical proxy reward gain');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(fdir/'distribution_reward.png');fig.savefig(fdir/'distribution_reward.svg');plt.close(fig)
 print('5 figure pairs saved',flush=True)
if __name__=='__main__':main()
