import json,time
from pathlib import Path
import numpy as np,torch,scipy.sparse as sp
import core,study

def make_model(data,kind,seed):
 if kind=='sasrec':
  import sasrec_baseline
  torch.manual_seed(seed);return sasrec_baseline.SASRec(len(data['item_ids']))
 return study.model(data,kind,seed)

def all_scores(m,d,split,batch=8):
 m.eval();n=len(d['item_ids']);off=2 if split=='validation' else 1
 with torch.no_grad():
  if isinstance(m,core.E2E):
   g,v=m.items()
   if m.kind!='no_candidate':
    qw,kw,vw=m.attn.in_proj_weight.chunk(3);qb,kb,vb=m.attn.in_proj_bias.chunk(3);keys=torch.nn.functional.linear(v,kw,kb).reshape(n+1,4,8).permute(1,0,2);vals=torch.nn.functional.linear(v,vw,vb).reshape(n+1,4,8).permute(1,0,2)
  for start in range(0,len(d['seq']),batch):
   users=np.arange(start,min(start+batch,len(d['seq'])));ends=[len(d['seq'][u])-off for u in users];x,t=core.pack(d['seq'],d['times'],users,ends)
   mask=torch.zeros((len(users),n+1),dtype=torch.bool);mask[:,0]=True
   for j,u in enumerate(users):mask[j,d['seq'][u][:-off]]=True
   c=torch.arange(n+1).expand(len(users),-1)
   if isinstance(m,core.E2E):
    u=m.encode(x,t,g)
    if m.kind=='no_candidate':uc=u
    else:
     queries=torch.nn.functional.linear(u,qw,qb).reshape(len(users),4,1,8)
     attention=torch.nn.functional.scaled_dot_product_attention(queries,keys[None],vals[None],attn_mask=(~mask)[:,None,None,:],dropout_p=0).reshape(len(users),32)
     uc=m.attn.out_proj(attention)
    scores=(uc*m.w1)@v.T+(uc*m.w2).sum(-1)[:,None]+(v*m.w2).sum(-1)[None]+m.bias.weight[:,0][None]
   else:
    u=m.encode(x) if isinstance(m,study.SASRec) else m.user(torch.tensor(users));scores=u@m.item.weight.T
   scores=scores.masked_fill(mask,-torch.inf).numpy();yield users,scores

def ranks_sorted(scores,targets):
 ids=np.arange(scores.shape[1]);r=[]
 for s,target in zip(scores,targets):
  order=np.lexsort((ids,-s));r.append(int(np.flatnonzero(order==target)[0])+1)
 return np.array(r)
def evaluate(m,d,out):
 out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():return
 sample_path=out/'sampled.npz';sample=core.evaluate(m,d,'test',save=sample_path);rs=[];positives=[]
 for users,scores in all_scores(m,d,'test'):
  targets=np.array([d['seq'][u][-1] for u in users]);positive=scores[np.arange(len(users)),targets];r=1+((scores>positive[:,None])|((scores==positive[:,None])&(np.arange(scores.shape[1])[None]<targets[:,None]))).sum(1)
  assert np.array_equal(r,ranks_sorted(scores,targets));assert np.isfinite(positive).all();rs.extend(r);positives.extend(positive)
 z=np.load(sample_path);assert np.array_equal(z['ranks'],np.array([1+np.flatnonzero(np.lexsort((c,-s))==0)[0] for s,c in zip(z['scores'],z['candidates'])]))
 np.savez_compressed(out/'catalog.npz',ranks=np.array(rs),positive_scores=np.array(positives),user_ids=d['user_ids']);core.dump(out/'metrics.json',dict(sampled=sample,catalog=core.metrics(rs)))
def knn_matrix(mat,k):
 norms=np.sqrt(np.asarray(mat.power(2).sum(0)).ravel());inv=np.divide(1,norms,out=np.zeros_like(norms),where=norms>0);sim=(sp.diags(inv)@(mat.T@mat)@sp.diags(inv)).tocsr();sim.setdiag(0);sim.eliminate_zeros();sim.sort_indices()
 for i in range(sim.shape[0]):
  a,b=sim.indptr[i:i+2];order=np.lexsort((sim.indices[a:b],-sim.data[a:b]));sim.data[a:b][order[k:]]=0
 sim.eliminate_zeros();return sim
def select_knn(dataset,d,out):
 mat=sp.load_npz(study.ROOT/'data'/dataset/'train.npz');trials=[]
 for k in [20,50,100]:
  sim=knn_matrix(mat,k);r=[]
  for u,s in enumerate(d['seq']):
   c=d['slates']['validation'][u];sc=np.asarray(sim[s[:-2]].sum(0)).ravel()[c];r.append(core.ranks(sc[None],c[None])[0])
  trials.append(dict(k=k,ndcg=core.metrics(r)['NDCG@10']))
 best=max(trials,key=lambda z:z['ndcg']);core.dump(out/'knn_selection.json',dict(selected=best,trials=trials))
def deterministic(dataset,d,out):
 selection=json.loads((out/'knn_selection.json').read_text());mat=sp.load_npz(study.ROOT/'data'/dataset/'train.npz');sim=knn_matrix(mat,selection['selected']['k'])
 for kind in ['popularity','itemknn']:
  target=out/'evaluation'/kind/'deterministic';target.mkdir(parents=True,exist_ok=True);rs=[];rfull=[];ss=[]
  for u,s in enumerate(d['seq']):
   scores=d['pop'].copy() if kind=='popularity' else np.asarray(sim[s[:-1]].sum(0)).ravel();c=d['slates']['test'][u];ss.append(scores[c]);rs.append(core.ranks(scores[c][None],c[None])[0]);scores[0]=-np.inf;scores[s[:-1]]=-np.inf;rfull.append(ranks_sorted(scores[None],np.array([s[-1]]))[0])
  np.savez_compressed(target/'sampled.npz',ranks=rs,scores=ss,candidates=d['slates']['test'],user_ids=d['user_ids']);np.savez_compressed(target/'catalog.npz',ranks=rfull,user_ids=d['user_ids']);core.dump(target/'metrics.json',dict(sampled=core.metrics(rs),catalog=core.metrics(rfull)))
def verify_full_attention():
 d=torch.load(study.ROOT/'data/amazon/data.pt',weights_only=False);m=study.model(d,'full',9).eval()
 with torch.no_grad():
  g,v=m.items();users,scores=next(all_scores(m,d,'validation',2))
  for j,u in enumerate(users):
   c=np.setdiff1d(np.arange(1,len(d['item_ids'])+1),d['seq'][u][:-2]);x,t=core.pack(d['seq'],d['times'],[u],[len(d['seq'][u])-2]);direct=m.score(m.encode(x,t,g),v,torch.tensor(c[None]))[0].numpy();assert np.allclose(scores[j,c],direct,atol=2e-5)
 return True
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('dataset');a=p.parse_args();assert verify_full_attention()
 # Gate: every planned learned-model tuning/confirmation run must finish for BOTH datasets.
 for dataset in ['movielens','amazon']:assert (study.RUN/dataset/'TRAINING_COMPLETE.json').exists(),'Training remains incomplete; no test access'
 for dataset in ['movielens','amazon']:
  if not (study.RUN/dataset/'knn_selection.json').exists():select_knn(dataset,torch.load(study.ROOT/'data'/dataset/'data.pt',weights_only=False),study.RUN/dataset)
 d=torch.load(study.ROOT/'data'/a.dataset/'data.pt',weights_only=False);out=study.RUN/a.dataset
 deterministic(a.dataset,d,out)
 for kind in study.KINDS:
  for seed in study.SEEDS:
   m=make_model(d,kind,seed);m.load_state_dict(torch.load(out/'confirm'/kind/str(seed)/'best.pt',weights_only=True));evaluate(m,d,out/'evaluation'/kind/str(seed));print('EVALUATED',a.dataset,kind,seed,flush=True)
 for kind in ['joint','frozen']:
  for seed in study.SEEDS:
   m=study.model(d,'full',seed);m.load_state_dict(torch.load(out/'matched'/kind/str(seed)/'best.pt',weights_only=True));evaluate(m,d,out/'evaluation'/('matched_'+kind)/str(seed));print('EVALUATED',a.dataset,kind,seed,flush=True)
 core.dump(out/'EVALUATION_COMPLETE.json',dict(complete=True,full_attention_verified=True,independent_rank_sort_verified=True))
