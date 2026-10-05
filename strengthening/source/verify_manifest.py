import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SHA256_MANIFEST.json').read_text());errors=[]
for rel,expected in manifest.items():
 p=root/rel
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append(rel)
if errors:raise SystemExit('Manifest failures: '+str(errors))
print('All',len(manifest),'files match SHA256 manifest')
