import json
import numpy as np,torch
import sharing,core,study,evaluation
reports=[]
for ds in sharing.DATASETS:
 d=torch.load(sharing.DATA/ds/'data.pt',weights_only=False)
 a=sharing.model(d,'separate',101);b=sharing.model(d,'shared_dual',101)
 assert a.state_dict().keys()==b.state_dict().keys()
 assert all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items())
 torch.manual_seed(101);orig=study.Variant(d,'full')
 assert all(torch.equal(x,a.state_dict()[k]) for k,x in orig.state_dict().items())
 ids=torch.tensor([0,1,2,3,7,11,22]);a.eval();b.eval()
 with torch.no_grad():
  for m in [a,b]:
   ga=m.emb.weight;gb=m.emb_second.weight
   for l in m.gat:ga=l(ga,m.nb,m.w)
   for l in m.gat_second:gb=l(gb,m.nb,m.w)
   assert torch.allclose(m.graph(ids,m.emb,m.gat),ga[ids],atol=2e-6)
   assert torch.allclose(m.graph(ids,m.emb_second,m.gat_second),gb[ids],atol=2e-6)
   gs,vs=m.items(ids);g,v=m.items()
   assert torch.allclose(gs[ids],g[ids],atol=2e-6) and torch.allclose(vs[ids],v[ids],atol=2e-6)
 for m in [a,b]:
  for branch in [0,1]:
   m.zero_grad(set_to_none=True);z=m.items(ids)[branch];z[ids[1:],0].sum().backward()
   gradA=m.emb.weight.grad;gradB=m.emb_second.weight.grad
   activeA=gradA is not None and gradA.abs().sum()>0
   activeB=gradB is not None and gradB.abs().sum()>0
   assert (bool(activeA),bool(activeB))==((True,True) if m.kind=='shared_dual' else ((True,False) if branch==0 else (False,True)))
 # Verify full-catalog scoring using permitted validation histories only.
 for m in [a,b]:
  m.eval()
  with torch.no_grad():
   g,v=m.items();users,scores=next(evaluation.all_scores(m,d,'validation',2))
   for j,u in enumerate(users):
    c=np.setdiff1d(np.arange(1,len(d['item_ids'])+1),d['seq'][u][:-2]);x,t=core.pack(d['seq'],d['times'],[u],[len(d['seq'][u])-2])
    direct=m.score(m.encode(x,t,g),v,torch.tensor(c[None]))[0].numpy()
    assert np.allclose(scores[j,c],direct,atol=2e-5)
 reports.append({'dataset':ds,'original_parameters':sum(p.numel() for p in orig.parameters()),'separate_parameters':sum(p.numel() for p in a.parameters()),'shared_dual_parameters':sum(p.numel() for p in b.parameters()),'identical_initial_tensors':True,'gradient_routing':True,'sparse_dense_values':True,'full_slate_scoring':True})
core.dump(sharing.ROOT/'checks.json',reports);print(json.dumps(reports,indent=2))
