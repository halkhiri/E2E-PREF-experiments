import torch,numpy as np,core,study,sasrec_baseline,baseline_plan
report={}
for ds in ['movielens','amazon']:
 d=torch.load(study.ROOT/'data'/ds/'data.pt',weights_only=False)
 users,x,pos,neg=next(sasrec_baseline.examples(d,501,1))
 for u in users:
  train=set(d['seq'][u][:-2]);assert set(pos[u][pos[u]!=0])<=train and not(set(neg[u][neg[u]!=0])&train);assert np.array_equal(pos[u]!=0,x[u]!=0)
 report[ds+'_all_position_pairs']=int((pos!=0).sum())
 m=sasrec_baseline.SASRec(len(d['item_ids'])).eval();xx=torch.tensor(x[:2]);h=m.features(xx);alter=xx.clone();alter[:,-1]=((alter[:,-1]+13)%(len(d['item_ids'])))+1;hh=m.features(alter);assert torch.allclose(h[:,:-1],hh[:,:-1],atol=1e-6)
 rr,tt=core.pack(d['seq'],d['times'],[0,1],[len(d['seq'][u])-2 for u in [0,1]]);c=torch.tensor(d['slates']['validation'][:2]);s=m(torch.tensor([0,1]),rr,tt,c);per=torch.randperm(100);assert torch.allclose(m(torch.tensor([0,1]),rr,tt,c[:,per]),s[:,per],atol=1e-6);assert torch.isfinite(s).all();s.sum().backward();assert m.item.weight.grad.abs().sum()>0
 for us,en,cs in baseline_plan.bpr_plan(d,501,1):
  for u,c in zip(us,cs):assert c[0] in d['seq'][u][:-2] and not set(c[1:])&set(d['seq'][u][:-2])
 report[ds+'_checks']=True
core.dump(study.ROOT/'baseline_checks.json',report);print(report)
