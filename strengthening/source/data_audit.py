import torch,numpy as np,json,sys
import study,core
report={}
for ds in ['movielens','amazon']:
 d=torch.load(study.ROOT/'data'/ds/'data.pt',weights_only=False);times=d['times'];seq=d['seq'];hist=np.array([len(s)-2 for s in seq]);eligible=np.array([len(d['item_ids'])-len(set(s[:-1])) for s in seq]);stats=dict(training_history_median=float(np.median(hist)),training_history_min=int(hist.min()),training_history_max=int(hist.max()),training_history_exceeds30=int((hist>30).sum()),test_history_exceeds30=int((hist+1>30).sum()),users_with_timestamp_ties=sum(bool((np.diff(t)==0).any()) for t in times),tied_adjacent_events=sum(int((np.diff(t)==0).sum()) for t in times),total_adjacent_events=sum(len(t)-1 for t in times),test_eligible_catalog_min=int(eligible.min()),test_eligible_catalog_max=int(eligible.max()),test_eligible_catalog_mean=float(eligible.mean()),items_without_training_events=int((d['pop'][1:]==0).sum()),nonself_edges=int((d['nb'][:,1:]!=0).sum()))
 assert all(len(s)==len(set(s)) for s in seq)
 assert all((np.diff(t)>=0).all() for t in times)
 report[ds]=stats
core.dump(study.ROOT/'data_audit.json',report);print(json.dumps(report,indent=2))
