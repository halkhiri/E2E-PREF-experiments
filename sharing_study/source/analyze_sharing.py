"""Recompute every planned contrast from saved ranks; no model fitting."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
P=argparse.ArgumentParser();P.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);P.add_argument('--original',type=Path);P.add_argument('--output',type=Path,default=Path('outputs/sharing_study'));args=P.parse_args()
ROOT=args.root;ORIGINAL=args.original or ROOT.parent/'strengthening';OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
assert (ROOT/'COMPLETE.json').exists()
SEEDS=[101,102,103,104,105];KINDS=['full','separate','shared_dual'];metrics=[];comparisons=[];training=[];hashes={}
def metric(r,name):
 return np.where(r<=10,1/np.log2(r+1),0) if name=='NDCG@10' else (r<=20).astype(float) if name=='Recall@20' else 1/r
def ci(a,b):
 delta=metric(a,'NDCG@10')-metric(b,'NDCG@10');rng=np.random.default_rng(4102026);draws=[]
 for _ in range(5000):draws.append(delta[np.ix_(rng.integers(len(delta),size=len(delta)),rng.integers(delta.shape[1],size=delta.shape[1]))].mean())
 return dict(difference=float(delta.mean()),low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),per_seed=delta.mean(1).tolist(),positive_seeds=int((delta.mean(1)>0).sum()))
for ds in ['movielens','amazon']:
 arrays={};user_order=None
 for k in KINDS:
  root=(ORIGINAL if k=='full' else ROOT)/'runs'/ds
  selection=json.loads((root/'selection.json').read_text())[k]
  for seed in SEEDS:
   path=root/'confirm'/k/str(seed);record=json.loads((path/'done.json').read_text());logs=json.loads((path/'training.json').read_text())
   assert len(logs)==50 and record['lr']==selection
   best=max(logs,key=lambda z:z['validation_ndcg']);assert best['epoch']==record['best_epoch'] and best['validation_ndcg']==record['best_validation']
   training.append(dict(dataset=ds,model=k,**record))
  if k!='full':
   trials=[json.loads((root/'tuning'/k/str(lr)/'done.json').read_text()) for lr in [.001,.003,.01]]
   assert max(trials,key=lambda z:z['best_validation'])['lr']==selection
  for protocol in ['catalog','sampled']:
   rr=[]
   for seed in SEEDS:
    path=root/'evaluation'/k/str(seed)/(protocol+'.npz');z=np.load(path);ids=z['user_ids'];r=z['ranks']
    if user_order is None:user_order=ids.copy()
    assert np.array_equal(user_order,ids)
    assert len(r)==len(ids) and np.isfinite(r).all() and (r>=1).all()
    if protocol=='sampled':
     scores=z['scores'];c=z['candidates'];rank=np.array([int(np.flatnonzero(np.lexsort((cc,-ss))==0)[0])+1 for ss,cc in zip(scores,c)])
     assert np.array_equal(rank,r)
    rr.append(r);hashes[str(path.relative_to(root))+'|'+ds]=hashlib.sha256(path.read_bytes()).hexdigest()
   arrays[k,protocol]=np.stack(rr)
   for name in ['NDCG@10','Recall@20','MRR']:
    per=metric(arrays[k,protocol],name).mean(1)
    metrics.append(dict(dataset=ds,model=k,protocol=protocol,metric=name,mean=float(per.mean()),std=float(per.std(ddof=1)),per_seed=per.tolist()))
 for protocol in ['catalog','sampled']:
  for left in ['full','shared_dual']:comparisons.append(dict(dataset=ds,protocol=protocol,left=left,right='separate',**ci(arrays[left,protocol],arrays['separate',protocol])))
result=dict(metrics=metrics,comparisons=comparisons,training=training,rank_hashes=hashes,protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest())
(OUT/'analysis.json').write_text(json.dumps(result,indent=2))
text=['# Representation sharing experiment','', 'Post hoc extension on previously evaluated datasets; all planned results reported. No new holdout. Intervals are exploratory conditional seed/user bootstrap intervals.','']
for ds in ['movielens','amazon']:
 text+=['## '+ds,'','| Model | Parameters | Catalog NDCG@10 | Sampled NDCG@10 |','|---|---:|---:|---:|']
 for k in KINDS:
  def fmt(p):
   r=next(z for z in metrics if (z['dataset'],z['model'],z['protocol'],z['metric'])==(ds,k,p,'NDCG@10'));return f"{r['mean']:.5f} ± {r['std']:.5f}"
  n=next(z['parameters'] for z in training if z['dataset']==ds and z['model']==k)
  text.append(f'| {k} | {n:,} | {fmt("catalog")} | {fmt("sampled")} |')
 text+=['','| Contrast | Protocol | Difference | 95% conditional interval | Positive seeds |','|---|---|---:|---|---:|']
 for z in comparisons:
  if z['dataset']==ds:text.append(f"| {z['left']} minus separate | {z['protocol']} | {z['difference']:+.5f} | [{z['low']:+.5f}, {z['high']:+.5f}] | {z['positive_seeds']}/5 |")
 text+=['','### Selected settings','']
 for k in KINDS:
  records=[z for z in training if z['dataset']==ds and z['model']==k]
  text.append(f"{k}: LR {records[0]['lr']}; selected epochs (seeds 101–105): "+', '.join(str(z['best_epoch']) for z in records)+'.')
 text+=['','Both new variants select the upper grid boundary (0.01); the fixed grid was not expanded after seeing results.','']
(OUT/'RESULTS.md').write_text('\n'.join(text)+'\n');print('\n'.join(text))
