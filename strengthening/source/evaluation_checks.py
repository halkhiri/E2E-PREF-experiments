import torch,numpy as np,core,study,evaluation
D=torch.load(study.ROOT/'data/amazon/data.pt',weights_only=False);report={}
for kind in ['full','no_graph','no_temporal','no_text','no_candidate','bpr_mf','sasrec']:
 m=evaluation.make_model(D,kind,123).eval();users,scores=next(evaluation.all_scores(m,D,'validation',2));errors=[]
 with torch.no_grad():
  if isinstance(m,core.E2E):g,v=m.items()
  for j,u in enumerate(users):
   c=np.setdiff1d(np.arange(1,len(D['item_ids'])+1),D['seq'][u][:-2]);x,t=core.pack(D['seq'],D['times'],[u],[len(D['seq'][u])-2]);ct=torch.tensor(c[None]);s=m.score(m.encode(x,t,g),v,ct)[0] if isinstance(m,core.E2E) else m(torch.tensor([u]),x,t,ct)[0];err=float(np.max(np.abs(scores[j,c]-s.numpy())));assert err<2e-5;errors.append(err)
 report[kind]=max(errors)
core.dump(study.ROOT/'evaluation_checks.json',report);print(report)
