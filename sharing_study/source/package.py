from pathlib import Path
import json,shutil,zipfile,hashlib,importlib.metadata
ROOT=Path('work/sharing_study');OLD=Path('work/strengthening');OUT=Path('outputs/sharing_study');DEST=OUT/'reproduction';DEST.mkdir(exist_ok=True)
assert (ROOT/'COMPLETE.json').exists()
for f in ['PROTOCOL.md','checks.json','TRAINING_COMPLETE.json','COMPLETE.json','TRAINING_SOURCE_MANIFEST.json']:shutil.copy2(ROOT/f,DEST/f)
shutil.copytree(ROOT/'source',DEST/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for filename in ['analysis.json','RESULTS.md','interpretation.json','Technical_supplement.md','figure3_sharing.png','figure3_sharing.pdf','LITERATURE_NOTES.md','EXPERIMENT_AUDIT.json','MANUSCRIPT_AUDIT.json']:
 shutil.copy2(OUT/filename,DEST/filename)
# Exact records for all new fits and the reused reference, with ranks and validation logs.
checkpoints={}
for source,target in [(ROOT,DEST),(OLD,DEST/'original_reference')]:
 for ds in ['movielens','amazon']:
  rp=source/'runs'/ds;tp=target/'runs'/ds;tp.mkdir(parents=True,exist_ok=True);shutil.copy2(rp/'selection.json',tp/'selection.json')
  for phase in ['tuning','confirm','evaluation']:
   for f in (rp/phase).rglob('*'):
    if not f.is_file():continue
    rel=f.relative_to(rp/phase)
    if source==OLD and rel.parts[0]!='full':continue
    if f.suffix=='.pt':checkpoints[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();continue
    if f.suffix not in ['.json','.npz']:continue
    dest=tp/phase/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
(DEST/'checkpoint_hashes.json').write_text(json.dumps(checkpoints,indent=2))
(DEST/'environment.json').write_text(json.dumps({p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','matplotlib']},indent=2))
(DEST/'README.md').write_text('''# E2E-PREF representation sharing study

This is a post hoc extension of the existing evaluation. It includes all 32 new fitting records, five-seed test ranks for each new model/dataset, and the reused original full model records. No test-driven variant selection was performed. Read PROTOCOL.md and RESULTS.md together.

## Recompute reported results

With NumPy installed, from this extracted directory:

    python source/analyze_sharing.py --root . --original original_reference --output recomputed
    python source/plot_sharing.py --output recomputed

The first command verifies sampled ranks independently from stored scores, validates checkpoint and LR selections, checks paired user order, and recomputes all metrics and crossed bootstrap intervals. Full-catalog ranks were independently checked against sorted full score vectors during model evaluation; source/evaluation.py contains that check. The compact archive does not retain every full-catalog score.

## Repeat model fitting

Use the pinned data preparation recipe in the previous E2E_PREF_independent_evaluation_reproduction.zip archive to reconstruct the original data.pt files. Place the rebuilt data at a sibling directory named strengthening/data/{movielens,amazon}/data.pt, as referenced in source/sharing.py. Use a fresh copy of this directory without runs or completion markers, keeping source and PROTOCOL.md. Install the versions in environment.json in an isolated environment, run source/checks.py and then source/sharing.py. The runner uses four worker processes, completes both datasets before test evaluation, and resumes from done.json records. Existing output records must not be mistaken for a fresh retraining. Reuse or retrain the original full reference separately according to its original protocol.

Prepared data, pretrained language-model weights, raw datasets and trained checkpoints are not redistributed here. The original datasets/models remain subject to their providers' terms. Trained checkpoints remain in the local work/sharing_study/runs directory; hashes for every checkpoint are retained. Training used fixed cached text features, not downloaded or newly fine-tuned language models.

The two new variants have identical initial parameter tensors and parameter counts. The shared-average control isolates a routing/averaging choice at equal allocated capacity, not parameter tying alone. Original-versus-separate also changes capacity. Runtime is wall time on a shared local CPU and is not a serving-latency benchmark.
''')
manifest={str(f.relative_to(DEST)):hashlib.sha256(f.read_bytes()).hexdigest() for f in DEST.rglob('*') if f.is_file() and f.name!='SHA256_MANIFEST.json'}
(DEST/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(OUT/'E2E_PREF_sharing_reproduction.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in DEST.rglob('*'):
  if f.is_file():z.write(f,f.relative_to(DEST))
print('Packaged',len(manifest),'files')
