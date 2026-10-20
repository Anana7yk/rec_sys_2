"""Validate documentation, committed-size policy and saved-result consistency; no fits."""
import argparse,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
METHODS=['pure','pure_sft','pure_rl','pure_full_reward_match','hybrid','hybrid_sft','hybrid_rl','hybrid_full_reward_match']

def files():
 if (ROOT/'.git').exists():
  result=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode()
  return [ROOT/p for p in result.split('\0') if p]
 return [p for folder in ['recsys','experiments','configs','results','docs','scripts','tests'] for p in (ROOT/folder).rglob('*') if p.is_file() and not any(x in p.parts for x in ['runs','inputs','__pycache__'])]+list(ROOT.glob('*.md'))

def check_links(paths):
 count=0
 for path in paths:
  if path.suffix!='.md':continue
  for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
   if re.match(r'^(https?://|mailto:|#)',link):continue
   target=unquote(link.split('#')[0].strip('<>'))
   resolved=(path.parent/target).resolve()
   assert resolved.is_relative_to(ROOT) and resolved.exists(),f'Broken local link: {path.relative_to(ROOT)} -> {link}'
   count+=1
 return count

def check_main_table():
 table=(ROOT/'README.md').read_text().split('<!-- key-results:start -->')[1].split('<!-- key-results:end -->')[0]
 rows=[line for line in table.splitlines() if re.match(r'^\|[^|]+\|\s+0\.',line)]
 assert len(rows)==len(METHODS)
 saved=pd.read_csv(ROOT/'results/study04/tables/selected_mean.csv').set_index('method')
 for method,line in zip(METHODS,rows):
  actual=[float(x.strip()) for x in line.split('|')[2:-1]]
  expected=[float(f'{saved.loc[method,col]:.6f}') for col in ['ndcg20','recall20','reward','evaluator_reward']]
  assert actual==expected,(method,actual,expected)
 return len(rows)

def compare_tables(directory):
 results=[]
 for study in ['study03','study04']:
  for reference in sorted((ROOT/f'results/{study}/tables').glob('*.csv')):
   candidate=directory/study/'tables'/reference.name
   if not candidate.exists():
    raise AssertionError(f'Missing regenerated table: {study}/{reference.name}')
   left=pd.read_csv(reference);right=pd.read_csv(candidate)
   assert list(left.columns)==list(right.columns),(reference.name,'columns')
   assert left.shape==right.shape,(reference.name,'shape')
   for col in left:
    if pd.api.types.is_numeric_dtype(left[col]):assert np.allclose(left[col],right[col],rtol=0,atol=1e-12,equal_nan=True),(study,reference.name,col)
    else:assert left[col].fillna('<NA>').equals(right[col].fillna('<NA>')),(study,reference.name,col)
   results.append(f'{study}/{reference.name}')
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument('--recomputed',type=Path);a=p.parse_args();paths=files();checks=dict(readme_rows_equal_saved_results=check_main_table(),local_links=check_links(paths))
 for path in paths:
  relative=path.relative_to(ROOT)
  assert path.stat().st_size<10*1024*1024,f'Large tracked file: {relative}'
  assert path.suffix not in ['.pt','.cbm','.parquet','.zip'],f'Heavy tracked payload: {relative}'
  if path.suffix in ['.py','.md','.json','.txt','.toml','.ini']:
   text=path.read_text()
   assert not re.search(r'/' + r'Users/[^/\s]+/|/' + r'home/[^/\s]+/',text),f'Personal machine path: {relative}'
   assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',text),f'Potential credential: {relative}'
 checks.update(files=len(paths),bytes=sum(p.stat().st_size for p in paths))
 if a.recomputed:checks['identical_tables_at_1e12']=compare_tables(a.recomputed)
 print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
