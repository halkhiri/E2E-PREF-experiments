import torch,core,study,numpy as np
D=torch.load(study.ROOT/'data/movielens/data.pt',weights_only=False);us,en,cs=core.schedule(D,999,1)[0];x,t=core.pack(D['seq'],D['times'],us[:16],en[:16]);c=torch.tensor(cs[:16]);report={}
for kind in ['full','no_graph','no_temporal','no_text','no_candidate']:
 a=study.model(D,kind,991).eval();b=study.model(D,kind,991).eval()
 g=a.emb.weight
 if kind!='no_graph':
  for layer in a.gat:g=layer(g,a.nb,a.w)
 cat=a.genres@a.genre_emb/a.genres.sum(1,keepdim=True).clamp_min(1);text=a.txt_proj(a.text) if kind!='no_text' else torch.zeros_like(g);v=a.fuse(torch.cat([g,text,cat,a.year_mlp(a.year)],-1));s=a.score(a.encode(x,t,g),v,c);study.F.cross_entropy(s,torch.zeros(len(x),dtype=torch.long)).backward()
 gg,vv=b.items(torch.cat([x.flatten(),c.flatten()]));ss=b.score(b.encode(x,t,gg),vv,c);study.F.cross_entropy(ss,torch.zeros(len(x),dtype=torch.long)).backward()
 err=float((s-ss).abs().max());ge=max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()) if p.grad is not None);assert err<2e-5 and ge<2e-5;report[kind]=dict(score_max_error=err,gradient_max_error=ge)
core.dump(study.ROOT/'equivalence_checks.json',report);print(report)
