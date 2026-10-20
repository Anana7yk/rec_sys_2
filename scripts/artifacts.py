"""Import only useful payloads from independently SHA-verified historical archives."""
import argparse,json,sys,zipfile,shutil
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'recsys'))
from provenance import digest,safe_child
from download_data import download


def destination(name,study):
 parts=PurePosixPath(name)
 if parts.is_absolute() or '..' in parts.parts:raise ValueError('Unsafe archive member')
 if study=='03':
  prefix='gpt2_rl_study03/experiments/gpt2_rl_v3/runs/study_03/'
  if not name.startswith(prefix):return None
  rel=PurePosixPath(name[len(prefix):])
  # Pilot weights and full supervised last weights are unnecessary for result replay.
  if rel.parts[0]=='pilot' and rel.suffix in ['.pt','.npy']:return None
  if 'pure' in rel.parts and rel.name=='last.pt':return None
  return Path('experiments/study03/runs/study_03')/str(rel)
 prefix='gpt2_rl_submission/experiments/gpt2_rl_v4/'
 if not name.startswith(prefix):return None
 rel=PurePosixPath(name[len(prefix):])
 if rel.parts[0]=='inputs':return Path('experiments/study04')/str(rel)
 if str(rel).startswith('runs/study_04/'):
  if 'engineering' in rel.parts:return None
  if 'sft_audit' in rel.parts and rel.suffix=='.pt':return None
  return Path('experiments/study04')/str(rel)
 return None

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',choices=['03','04'],required=True);p.add_argument('--file',type=Path);p.add_argument('--download',action='store_true');p.add_argument('--url');a=p.parse_args();entry=json.loads((ROOT/'configs/artifacts.json').read_text())[a.study]
 path=a.file or ROOT/'downloads'/entry['filename']
 if a.download:
  url=a.url or entry['url']
  if not url:p.error('Public artifact URL is not published. Supply the matching archive with --file or an author-provided --url.')
  if not url.startswith('https://'):p.error('A downloadable HTTPS URL is required')
  download(url,path,entry['bytes'],entry['sha256'])
 if not path.is_file():p.error('Archive unavailable. See docs/ARTIFACTS.md; there is no bundled public download URL.')
 assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],'Archive size/SHA256 mismatch'
 count=0
 with zipfile.ZipFile(path) as z:
  if z.testzip() is not None:raise ValueError('Archive CRC mismatch')
  for member in z.infolist():
   if member.is_dir():continue
   relative=destination(member.filename,a.study)
   if relative is None:continue
   dest=safe_child(ROOT,relative);dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    with z.open(member) as source:
     import hashlib
     h=hashlib.sha256()
     while block:=source.read(4*1024*1024):h.update(block)
    assert digest(dest)==h.hexdigest(),f'Refusing to overwrite a different payload: {relative}'
   else:
    with z.open(member) as source,dest.open('wb') as target:shutil.copyfileobj(source,target,4*1024*1024)
   count+=1
 receipt=dict(study=a.study,archive_sha256=entry['sha256'],archive_bytes=entry['bytes'],imported_files=count,source_seals='Preserved archival metadata; refactored source is verified separately by Git and tests, never through old absolute paths')
 (ROOT/'artifacts').mkdir(exist_ok=True);(ROOT/f'artifacts/study{a.study}_import.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
