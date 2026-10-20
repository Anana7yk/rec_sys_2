"""Portable manifests and downloads must fail closed, without another checkout."""
import hashlib,json
from pathlib import Path
import importlib.util
import pytest
from provenance import safe_child,verify_seal,create_seal

def test_manifest_rejects_traversal_and_symlink_escape(tmp_path):
 outside=tmp_path/'outside';outside.mkdir();root=tmp_path/'root';root.mkdir()
 (root/'link').symlink_to(outside,target_is_directory=True)
 for path in ['../outside/file','/tmp/absolute','link/file']:
  with pytest.raises(ValueError):safe_child(root,path)

def test_portable_seal_detects_changed_payload(tmp_path):
 run=tmp_path/'run';run.mkdir();p=run/'weight.bin';p.write_bytes(b'fixed')
 seal=run/'SEAL.json';seal.write_text(json.dumps(create_seal([p],run)))
 assert verify_seal(seal,run)['verified_files']==1
 p.write_bytes(b'changed')
 with pytest.raises(AssertionError):verify_seal(seal,run)

def test_legacy_source_path_is_metadata_only(tmp_path):
 run=tmp_path/'run';run.mkdir();(run/'payload.bin').write_bytes(b'fixed')
 records=[dict(path='/unavailable/source/train.py',sha256='unavailable'),dict(path='/unavailable/runs/study_03/payload.bin',sha256=hashlib.sha256(b'fixed').hexdigest())]
 seal=run/'seal.json';seal.write_text(json.dumps(dict(files=records)))
 status=verify_seal(seal,run)
 assert status==dict(verified_files=1,archival_source_records_not_reverified=1,portable_format=False)

def test_download_checksum_failure_leaves_no_file(tmp_path,monkeypatch):
 script=Path(__file__).resolve().parents[1]/'scripts/download_data.py'
 spec=importlib.util.spec_from_file_location('download_test',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 class Response:
  def __enter__(self):return self
  def __exit__(self,*args):return False
  def raise_for_status(self):pass
  def iter_content(self,size):yield b'corrupt'
 monkeypatch.setattr(m.requests,'get',lambda *a,**k:Response())
 target=tmp_path/'raw.parquet'
 with pytest.raises(ValueError):m.download('https://example.invalid/pinned',target,7,'0'*64)
 assert not target.exists() and not target.with_suffix('.parquet.part').exists()
