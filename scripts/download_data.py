"""Download the pinned public Yambda snapshot; commit only its manifest, not files."""
import argparse,hashlib,json,os
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def download(url,dest,size,sha):
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if dest.stat().st_size!=size or sha256(dest)!=sha:raise ValueError(f'Existing file has wrong checksum: {dest.name}; remove it explicitly')
  return
 part=dest.with_suffix(dest.suffix+'.part')
 try:
  h=hashlib.sha256();count=0
  with requests.get(url,stream=True,timeout=(20,120)) as response:
   response.raise_for_status()
   with part.open('wb') as f:
    for block in response.iter_content(4*1024*1024):
     f.write(block);h.update(block);count+=len(block)
  if count!=size or h.hexdigest()!=sha:raise ValueError(f'Checksum mismatch: {dest.name}')
  os.replace(part,dest)
 finally:
  part.unlink(missing_ok=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'data/raw');p.add_argument('--verify-only',action='store_true');a=p.parse_args();manifest=json.loads((ROOT/'configs/data_sources.json').read_text())
 for record in manifest['files']:
  dest=a.out/record['filename']
  if a.verify_only:
   assert dest.is_file() and dest.stat().st_size==record['bytes'] and sha256(dest)==record['sha256'],record['filename']
  else:download(record['url'],dest,record['bytes'],record['sha256'])
  print('Verified',record['filename'],flush=True)
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
