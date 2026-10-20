"""One controlled pipeline from public raw data or from published frozen checkpoints.

No new search space: all configurations reproduce the completed protocols.
"""
import argparse,os,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')

def execute(script,*args):
 subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,env=ENV,check=True)

def study03(root,reports,device):
 for script,args in [
  ('prepare.py',[]),('distributions.py',[]),('train.py',['pilot']),('run.py',['reward']),('select_reward.py',[]),('previous_control.py',['marginal']),('train.py',['main']),('run.py',['policies']),('run.py',['final']),('previous_control.py',['assess'])]:
  extra=['--device',device] if script in ['train.py','run.py','previous_control.py'] else []
  execute('experiments/study03/'+script,*args,'--out',root,*extra)
 summarize03(root,reports)

def summarize03(root,reports):
 execute('experiments/study03/summarize.py','--out',root,'--reports',reports)
 execute('experiments/study03/reward_domain_analysis.py','--out',root,'--reports',reports)
 execute('experiments/study03/plot_data.py','--out',root,'--reports',reports)
 execute('experiments/study03/augment_horizon.py','--out',root,'--reports',reports)
 execute('experiments/study03/examples.py','--out',root,'--reports',reports)
 execute('experiments/study03/verify.py','--out',root)
 shutil.copy2(root/'data/target_distributions.csv',reports/'tables/target_distributions.csv')

def study04(inputs,root,reports,device):
 execute('experiments/study04/run.py','all','--inputs',inputs,'--out',root,'--device',device)
 summarize04(inputs,root,reports)

def summarize04(inputs,root,reports):
 execute('experiments/study04/summarize.py','--inputs',inputs,'--out',root,'--reports',reports)
 execute('experiments/study04/plots.py','--out',root,'--reports',reports)
 execute('experiments/study04/examples.py','--inputs',inputs,'--out',root,'--reports',reports)
 execute('experiments/study04/verify.py','--inputs',inputs,'--out',root,'--reports',reports)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['full','supplement','smoke','tables']);p.add_argument('--out',type=Path,default=ROOT/'runs/reproduce');p.add_argument('--reports',type=Path,default=ROOT/'validation/recomputed');p.add_argument('--inputs',type=Path,default=ROOT/'experiments/study04/inputs');p.add_argument('--device',choices=['cpu','mps'],default='mps');a=p.parse_args()
 if a.mode=='smoke':
  execute('experiments/study03/smoke.py','--out',a.out/'smoke03');execute('experiments/study04/smoke.py','--out',a.out/'smoke04');return
 if a.mode=='tables':
  summarize03(ROOT/'experiments/study03/runs/study_03',a.reports/'study03');summarize04(a.inputs,ROOT/'experiments/study04/runs/study_04',a.reports/'study04');return
 if a.out.exists():raise FileExistsError('Use a new output folder; saved experiments are never overwritten')
 a.out.mkdir(parents=True)
 if a.mode=='full':
  execute('scripts/download_data.py');study03(a.out/'study03',a.reports/'study03',a.device)
  inputs=a.out/'supplement_inputs';execute('experiments/study04/prepare_inputs.py','--study03',a.out/'study03','--out',inputs)
 else:inputs=a.inputs
 study04(inputs,a.out/'study04',a.reports/'study04',a.device)
if __name__=='__main__':main()
