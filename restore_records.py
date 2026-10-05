"""Verify and restore the complete archived studies; uses Python standard library only."""
from pathlib import Path
import hashlib, json, tempfile, zipfile
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'archive_manifest.json').read_text())
with tempfile.TemporaryFile() as archive:
    digest = hashlib.sha256()
    for part in manifest['parts']:
        content = (root / part['name']).read_bytes()
        if hashlib.sha256(content).hexdigest() != part['sha256']:
            raise ValueError('Checksum mismatch: ' + part['name'])
        archive.write(content)
        digest.update(content)
    if digest.hexdigest() != manifest['sha256']:
        raise ValueError('Combined archive checksum mismatch')
    archive.seek(0)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Unsafe archive path: ' + name)
            if target.exists() and target.read_bytes() != z.read(name):
                raise ValueError('Existing modified file; restore in a fresh checkout: ' + name)
        z.extractall(root)
for study in ['strengthening', 'sharing_study']:
    base = root / study
    entries = json.loads((base / 'SHA256_MANIFEST.json').read_text())
    for name, expected in entries.items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Study checksum mismatch: ' + study + '/' + name)
    print(study, ': verified', len(entries), 'manifest entries')
print('Complete experimental records restored. Follow README.md for analysis commands.')
