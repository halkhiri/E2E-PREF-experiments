from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path('work/sharing_study');OUT=Path('outputs/sharing_study');assert (ROOT/'COMPLETE.json').exists()
a=json.loads((OUT/'analysis.json').read_text());locked=json.loads((ROOT/'TRAINING_SOURCE_MANIFEST.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in locked.items())
fits=list((ROOT/'runs').glob('*/tuning/*/*/done.json'))+list((ROOT/'runs').glob('*/confirm/*/*/done.json'))
evals=list((ROOT/'runs').glob('*/evaluation/*/*/metrics.json'))
assert len(fits)==32 and len(evals)==20
assert min(f.stat().st_mtime for f in evals)>max(f.stat().st_mtime for f in fits)
for f in fits:
 d=json.loads(f.read_text());logs=json.loads((f.parent/'training.json').read_text());assert len(logs)==50
 assert all(np.isfinite(r['loss']) and np.isfinite(r['validation_ndcg']) for r in logs)
 assert d['best_epoch']==max(logs,key=lambda r:r['validation_ndcg'])['epoch']
for ds in ['movielens','amazon']:
 expected=1000 if ds=='movielens' else 1092
 for kind in ['separate','shared_dual']:
  for seed in [101,102,103,104,105]:
   p=ROOT/'runs'/ds/'evaluation'/kind/str(seed);saved=json.loads((p/'metrics.json').read_text())
   for proto in ['catalog','sampled']:
    z=np.load(p/(proto+'.npz'));r=z['ranks'];assert len(r)==expected
    derived={'NDCG@10':float(np.where(r<=10,1/np.log2(r+1),0).mean()),'Recall@20':float((r<=20).mean()),'MRR':float((1/r).mean())}
    for k,v in derived.items():assert abs(v-saved[proto][k])<1e-12
   ck=ROOT/'runs'/ds/'confirm'/kind/str(seed)/'best.pt';prov=json.loads((p/'provenance.json').read_text());assert hashlib.sha256(ck.read_bytes()).hexdigest()==prov['checkpoint_sha256']
 assert next(x['parameters'] for x in a['training'] if x['dataset']==ds and x['model']=='separate')==next(x['parameters'] for x in a['training'] if x['dataset']==ds and x['model']=='shared_dual')
assert len(a['comparisons'])==8
report={'new_fitting_runs':32,'new_evaluated_models':20,'all_sources_unchanged_during_training':True,'all_fitting_completed_before_new_test_evaluation':True,'all_training_losses_finite':True,'all_metric_values_recomputed_from_ranks':True,'all_evaluated_checkpoint_hashes_match':True,'all_eight_planned_contrasts_reported':True,'paired_user_order_and_sample_rank_checks':'performed in analyze_sharing.py','full_rank_sort_and_direct_attention_checks':'performed in evaluation.py and checks.py'}
(OUT/'EXPERIMENT_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
