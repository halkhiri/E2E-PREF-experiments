"""Final evidence audit, run only after completed test evaluation."""
import json,hashlib
from pathlib import Path
import numpy as np,torch,pandas as pd
import core,study
root=study.ROOT;report={};hashes={core.sha(root/'source/study.py'),core.sha(root/'source/study_dense_original.py'),core.sha(root/'source/sasrec_baseline.py')}
for ds in ['movielens','amazon']:
 d=torch.load(root/'data'/ds/'data.pt',weights_only=False);rr=root/'runs'/ds;done=list(rr.rglob('done.json'));assert len(done)==103,(ds,len(done));assert (rr/'EVALUATION_COMPLETE.json').exists()
 for f in done:
  z=json.loads(f.read_text());assert z['source_sha256'] in hashes;hist=json.loads((f.parent/'training.json').read_text());assert len(hist)==(25 if 'matched' in f.parts else 50)
  best=max(hist,key=lambda x:x['validation_ndcg']);assert z['best_epoch']==best['epoch'] and z['best_validation']==best['validation_ndcg'];assert np.isfinite([h['loss'] for h in hist]).all()
 knn=json.loads((rr/'knn_selection.json').read_text());assert knn['selected']==max(knn['trials'],key=lambda z:z['ndcg'])
 if (root/'deterministic_selection_checks.json').exists():
  prior=json.loads((root/'deterministic_selection_checks.json').read_text());assert knn==prior['selections'][ds]
 selection=json.loads((rr/'selection.json').read_text())
 for k,lr in selection.items():
  trials=[json.loads((rr/'tuning'/k/str(rate)/'done.json').read_text()) for rate in study.LRS];assert lr==max(trials,key=lambda x:x['best_validation'])['lr']
 count=0
 for f in (rr/'evaluation').glob('*/*/metrics.json'):
  result=json.loads(f.read_text())
  for proto in ['sampled','catalog']:
   z=np.load(f.parent/(proto+'.npz'));assert np.array_equal(z['user_ids'],d['user_ids']);r=z['ranks'];assert len(r)==len(d['seq']);assert np.all((r>=1)&(r<=len(d['item_ids'])))
   for k,v in core.metrics(r).items():assert abs(result[proto][k]-v)<1e-12
   if proto=='sampled':assert np.array_equal(z['candidates'],d['slates']['test']) and np.array_equal(core.ranks(z['scores'],z['candidates']),r)
  count+=1
 assert count==67,count
 from evaluation import knn_matrix
 mat=core.sp.load_npz(root/'data'/ds/'train.npz');sim=knn_matrix(mat,knn['selected']['k']);ids=np.arange(len(d['item_ids'])+1)
 for kind in ['popularity','itemknn']:
  saved=root/'runs'/ds/'evaluation'/kind/'deterministic';full=np.load(saved/'catalog.npz')['ranks'];sample=np.load(saved/'sampled.npz')['ranks']
  for u,seq in enumerate(d['seq']):
   scores=d['pop'].copy() if kind=='popularity' else np.asarray(sim[seq[:-1]].sum(0)).ravel();c=d['slates']['test'][u]
   assert sample[u]==1+np.flatnonzero(np.lexsort((c,-scores[c]))==0)[0]
   scores[0]=-np.inf;scores[seq[:-1]]=-np.inf;target=seq[-1];positive=scores[target];assert np.isfinite(positive)
   assert full[u]==1+((scores>positive)|((scores==positive)&(ids<target))).sum()
 report[ds]=dict(fitting_runs=len(done),evaluated_models=count,metrics_recomputed=True,validation_selection_verified=True,source_hashes_verified=True,deterministic_scores_independently_recomputed=True)
analysis=json.loads((Path('outputs/strengthened_study')/'analysis.json').read_text())
for row in analysis['metrics']:
 files=sorted((root/'runs'/row['dataset']/'evaluation'/row['model']).glob('*/metrics.json'))
 values=np.array([json.loads(f.read_text())[row['protocol']][row['metric']] for f in files]);assert len(values)==row['n_runs'];assert abs(values.mean()-row['mean'])<1e-12
 assert abs((values.std(ddof=1) if len(values)>1 else 0)-row['std'])<1e-12
for row in analysis['comparisons']:
 def mean(k):return next(m['mean'] for m in analysis['metrics'] if (m['dataset'],m['model'],m['protocol'],m['metric'])==(row['dataset'],k,row['protocol'],'NDCG@10'))
 assert abs(row['difference']-(mean(row['left'])-mean(row['right'])))<1e-12
 assert row['low_975']<=row['low']<=row['high']<=row['high_975']
 assert len(row['per_seed'])==5 and sum(x>0 for x in row['per_seed'])==row['positive_seeds']
report['analysis_aggregations_verified']=True
report['checkpoint_sha256']={str(p.relative_to(root)):core.sha(p) for p in (root/'runs').glob('*/*/*/*/*.pt')}
report['prepared_data_sha256']={str(p.relative_to(root)):core.sha(p) for p in (root/'data').glob('*/*') if p.suffix in ['.pt','.npz']}
core.dump(Path('outputs/strengthened_study')/'FINAL_AUDIT.json',report);print(report)
