"""Portable new seals; legacy archives verify payloads without following old paths.

Old source hashes remain archival evidence. They cannot certify refactored source;
that distinction is returned explicitly rather than silently accessing another checkout.
"""
import hashlib,json,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def safe_child(root,relative):
 relative=Path(relative)
 if relative.is_absolute() or '..' in relative.parts:raise ValueError(f'Unsafe manifest path: {relative}')
 path=(Path(root)/relative).resolve()
 if not path.is_relative_to(Path(root).resolve()):raise ValueError('Manifest escapes its root')
 return path

def create_seal(files,run,inputs=None):
 run=Path(run).resolve();roots={'run':run,'repository':ROOT}
 if inputs is not None:roots['inputs']=Path(inputs).resolve()
 records=[]
 for p in sorted(set(Path(p).resolve() for p in files)):
  choices=[(key,base) for key,base in roots.items() if p.is_relative_to(base)]
  if not choices:raise ValueError(f'Cannot seal a file outside declared roots: {p.name}')
  role,base=max(choices,key=lambda pair:len(pair[1].parts))
  records.append(dict(role=role,path=str(p.relative_to(base)),sha256=digest(p)))
 result=dict(format=2,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=records)
 if inputs is not None:result['inputs_relative_to_run']=os.path.relpath(roots['inputs'],run)
 return result

def verify_seal(path,run,inputs=None):
 path=Path(path);run=Path(run).resolve();d=json.loads(path.read_text());checked=0;skipped=0
 if 'parent_seal_sha256' in d:
  assert digest(run/'SEAL.json')==d['parent_seal_sha256'],'Legacy parent seal changed'
 if inputs is None and 'inputs_relative_to_run' in d:inputs=(run/d['inputs_relative_to_run']).resolve()
 roots={'repository':ROOT,'run':run,'inputs':Path(inputs).resolve() if inputs else None}
 for record in d['files']:
  if d.get('format')==2:
   base=roots[record['role']]
   if base is None:raise ValueError('Missing input root')
   target=safe_child(base,record['path'])
  elif 'role' in record: # Immutable study04 archive, never read its old source paths.
   if record['role']=='source':skipped+=1;continue
   base=roots.get(record['role'])
   if base is None:raise ValueError(f'Unknown legacy role: {record["role"]}')
   target=safe_child(base,record['path'])
  else: # Immutable study03 archive with historical absolute metadata only.
   marker='/runs/study_03/'
   if marker not in record['path']:skipped+=1;continue
   target=safe_child(run,record['path'].split(marker,1)[1])
  assert digest(target)==record['sha256'],f'Changed payload: {record["path"]}'
  checked+=1
 return dict(verified_files=checked,archival_source_records_not_reverified=skipped,portable_format=d.get('format')==2)
