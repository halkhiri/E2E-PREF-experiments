import torch,numpy as np,copy
import core,study
from pathlib import Path
checks={}
for dataset in ['movielens','amazon']:
 d=torch.load(study.ROOT/'data'/dataset/'data.pt',weights_only=False)
 for split,off in [('validation',2),('test',1)]:
  for s,c in zip(d['seq'],d['slates'][split]):
   assert c[0]==s[-off] and len(set(c))==100 and not(set(c[1:])&set(s[:-off]))
 plan=core.schedule(d,501,2)
 for users,ends,cs in plan:
  for u,end,c in zip(users,ends,cs):assert end<len(d['seq'][u])-2 and c[0]==d['seq'][u][end] and not set(c[1:])&set(d['seq'][u][:end])
 nb,w,pop,mat=core.make_graph([s[:-2] for s in d['seq']],len(d['item_ids']));assert np.array_equal(nb,d['nb']) and np.array_equal(pop,d['pop'])
 checks[dataset+'_split_and_graph']=True
 if dataset=='amazon':
  x,t=core.pack(d['seq'],d['times'],[0,1],[len(d['seq'][i])-2 for i in [0,1]]);c=torch.tensor(d['slates']['validation'][:2]);perm=torch.randperm(100)
  for kind in study.KINDS:
   m=study.model(d,kind,7).eval()
   if isinstance(m,core.E2E):
    g,v=m.items();u=m.encode(x,t,g);s=m.score(u,v,c);s2=m.score(u,v,c[:,perm]);assert torch.allclose(s2,s[:,perm],atol=2e-5)
    if kind=='no_candidate':assert torch.allclose(m.score(u,v,c[:,:1]),s[:,:1],atol=1e-5)
    if kind=='no_text':
     old=s.detach();m.text.add_(torch.randn_like(m.text));g,v=m.items();assert torch.equal(old,m.score(m.encode(x,t,g),v,c))
   else:s=m(torch.tensor([0,1]),x,t,c)
   F=study.F;loss=F.cross_entropy(s,torch.zeros(2,dtype=torch.long));loss.backward();assert torch.isfinite(s).all() and any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
   checks[kind+'_behavior']=True
assert core.metrics([1,21])['NDCG@10']==.5
core.dump(study.ROOT/'checks.json',checks);print(checks)
