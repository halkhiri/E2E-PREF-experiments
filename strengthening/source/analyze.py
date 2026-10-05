import json,math
from pathlib import Path
import numpy as np,pandas as pd
import core,study
OUT=Path('outputs/strengthened_study');OUT.mkdir(parents=True,exist_ok=True)
LABELS={'full':'E2E-PREF ranking only','no_graph':'Without graph encoder','no_temporal':'Mean history instead of temporal encoder','no_text':'Without frozen text features','no_candidate':'Candidate-independent scoring','aux_all':'All auxiliary losses','aux_con':'Contrastive loss only','aux_tem':'Temporal loss only','aux_sem':'Semantic loss only','bpr_mf':'BPR-MF','sasrec':'SASRec port','popularity':'Popularity','itemknn':'Item neighborhood','matched_joint':'Matched joint continuation','matched_frozen':'Matched frozen continuation'}
def values(r):return np.where(r<=10,1/np.log2(r+1),0)
def interval(a,b,seed=290926):
 delta=values(a)-values(b);rng=np.random.default_rng(seed);draws=[]
 for _ in range(5000):draws.append(delta[np.ix_(rng.integers(len(delta),size=len(delta)),rng.integers(delta.shape[1],size=delta.shape[1]))].mean())
 return dict(difference=float(delta.mean()),low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),low_975=float(np.quantile(draws,.0125)),high_975=float(np.quantile(draws,.9875)),positive_seeds=int((delta.mean(1)>0).sum()),per_seed=delta.mean(1).tolist())
rows=[];comparisons=[];training=[];arrays={}
for dataset in ['movielens','amazon']:
 root=study.RUN/dataset;assert (root/'EVALUATION_COMPLETE.json').exists()
 for kind in LABELS:
  dirs=sorted((root/'evaluation'/kind).glob('*'));assert len(dirs)==(1 if kind in ['popularity','itemknn'] else 5),(dataset,kind,len(dirs))
  for protocol in ['sampled','catalog']:
   rr=np.stack([np.load(p/(protocol+'.npz'))['ranks'] for p in dirs]);arrays[dataset,kind,protocol]=rr
   for metric in ['NDCG@10','Recall@20','MRR']:
    vv=np.array([core.metrics(r)[metric] for r in rr]);rows.append(dict(dataset=dataset,model=kind,label=LABELS[kind],protocol=protocol,metric=metric,mean=vv.mean(),std=vv.std(ddof=1) if len(vv)>1 else 0,n_runs=len(vv)))
  if kind not in ['popularity','itemknn']:
   group='matched' if kind.startswith('matched_') else 'confirm';sub=kind.replace('matched_','') if group=='matched' else kind
   for seed in study.SEEDS:
    record=dict(dataset=dataset,model=kind,**json.loads((root/group/sub/str(seed)/'done.json').read_text()))
    record['pretraining_seconds']=json.loads((root/'matched'/'pretrain'/str(seed)/'done.json').read_text())['seconds'] if group=='matched' else 0
    record['seconds_including_shared_pretraining']=record['seconds']+record['pretraining_seconds'];training.append(record)
 for protocol in ['sampled','catalog']:
  for left,right in [('matched_joint','matched_frozen')]+[('full',x) for x in ['no_graph','no_temporal','no_text','no_candidate','bpr_mf','sasrec']]+[(x,'full') for x in ['aux_all','aux_con','aux_tem','aux_sem']]:
   comparisons.append(dict(dataset=dataset,protocol=protocol,left=left,right=right,**interval(arrays[dataset,left,protocol],arrays[dataset,right,protocol])))
pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(training).to_csv(OUT/'training_costs.csv',index=False);pd.DataFrame(comparisons).drop(columns='per_seed').to_csv(OUT/'paired_comparisons.csv',index=False);core.dump(OUT/'analysis.json',dict(metrics=rows,comparisons=comparisons,training=training))
text=['# Independent evaluation and component study','', 'All planned results are reported. Model and checkpoint selection used validation only; primary outcomes use the complete eligible static catalog. Intervals describe conditional resampling of the observed five-seed/user grid.','']
for dataset in ['movielens','amazon']:
 text+=['## '+dataset,'', '| Model | Full catalog NDCG@10 | Sampled NDCG@10 |','|---|---:|---:|']
 for kind,label in LABELS.items():
  def val(p):
   r=next(r for r in rows if r['dataset']==dataset and r['model']==kind and r['protocol']==p and r['metric']=='NDCG@10');return f"{r['mean']:.5f} ± {r['std']:.5f}" if r['n_runs']>1 else f"{r['mean']:.5f}"
  text.append(f'| {label} | {val("catalog")} | {val("sampled")} |')
 text+=['','### Matched representation updating','']
 for p in ['catalog','sampled']:
  r=next(c for c in comparisons if c['dataset']==dataset and c['protocol']==p and c['left']=='matched_joint');text.append(f"{p}: joint minus frozen = {r['difference']:+.5f}, 95% conditional interval [{r['low']:+.5f}, {r['high']:+.5f}], positive in {r['positive_seeds']}/5 seeds. The 97.5% interval is [{r['low_975']:+.5f}, {r['high_975']:+.5f}].")
 text+=['','### Exploratory full-catalog contrasts','','| Left minus right | Difference | 95% conditional interval | Positive seeds |','|---|---:|---:|---:|']
 for r in comparisons:
  if r['dataset']==dataset and r['protocol']=='catalog' and r['left']!='matched_joint':text.append(f"| {LABELS[r['left']]} minus {LABELS[r['right']]} | {r['difference']:+.5f} | [{r['low']:+.5f}, {r['high']:+.5f}] | {r['positive_seeds']}/5 |")
 text+=['']
text+=['## Scope','', 'MovieLens uses 1,000 previously unused users, with zero overlap with the earlier pilot. Amazon Musical Instruments is an independent product domain with 1,092 eligible users and 900 products. This is a small 2014 5-core benchmark; the result does not imply evaluation on the entire Amazon catalog. Both datasets use per-user chronological splits and static side information rather than a globally time-causal deployment simulation.','', 'SASRec is a source-informed PyTorch port using left padding, all valid next-item positions and one negative per position, with its original Adam/logistic training structure. Search equality covers learning-rate trials, maximum epochs and per-user batch opportunities; positive-pair counts and computational costs differ. BPR samples all training positives. Six preliminary MovieLens baseline validation trials were archived before test access following the implementation audit. Hyperparameters other than learning rate and item-neighborhood size are fixed. All component and auxiliary-loss comparisons are exploratory, with no multiplicity-controlled significance claim.']
(OUT/'RESULTS.md').write_text('\n'.join(text)+'\n');print('\n'.join(text))
