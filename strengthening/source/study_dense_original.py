"""Prospective experiments. Training never calls test evaluation."""
import argparse,copy,json,math,time,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import core
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs';SEEDS=[101,102,103,104,105];LRS=[.001,.003,.01]
KINDS=['full','no_graph','no_temporal','no_text','no_candidate','aux_all','aux_con','aux_tem','aux_sem','bpr_mf','sasrec']
torch.set_num_threads(4)
class Variant(core.E2E):
 def __init__(self,data,kind):super().__init__(data);self.kind=kind
 def items(self):
  g=self.emb.weight
  if self.kind!='no_graph':
   for layer in self.gat:g=layer(g,self.nb,self.w)
  text=self.txt_proj(self.text) if self.kind!='no_text' else torch.zeros_like(g)
  cat=self.genres@self.genre_emb/self.genres.sum(1,keepdim=True).clamp_min(1)
  return g,self.fuse(torch.cat([g,text,cat,self.year_mlp(self.year)],-1))
 def encode(self,x,t,g,augment=False):
  if self.kind!='no_temporal':return super().encode(x,t,g,augment)
  mask=x.ne(0)
  if augment:
   keep=torch.rand(x.shape)>.2;keep[:,0]=True;mask=mask&keep
  return (g[x]*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
 def score(self,u,v,c,dot=False):
  if self.kind!='no_candidate' or dot:return super().score(u,v,c,dot)
  cv=v[c];return ((u[:,None]*cv)*self.w1).sum(-1)+((u[:,None]+cv)*self.w2).sum(-1)+self.bias(c).squeeze(-1)
class SASRec(nn.Module):
 """Port of kang205/SASRec blocks; same-prefix adaptation of pointwise loss."""
 def __init__(self,n,d=32):
  super().__init__();self.item=nn.Embedding(n+1,d,padding_idx=0);nn.init.xavier_uniform_(self.item.weight);self.item.weight.data[0]=0
  self.pos=nn.Embedding(30,d);self.drop=nn.Dropout(.2);self.blocks=nn.ModuleList()
  for _ in range(2):self.blocks.append(nn.ModuleDict(dict(n1=nn.LayerNorm(d,eps=1e-8),q=nn.Linear(d,d),k=nn.Linear(d,d),v=nn.Linear(d,d),n2=nn.LayerNorm(d,eps=1e-8),f1=nn.Linear(d,d),f2=nn.Linear(d,d))))
  self.norm=nn.LayerNorm(d,eps=1e-8)
 def encode(self,x):
  mask=x.ne(0);h=self.drop(self.item(x)*math.sqrt(32)+self.pos(torch.arange(x.shape[1]))[None])*mask[...,None]
  for b in self.blocks:
   q=b['n1'](h);a=b['q'](q)@b['k'](h).transpose(1,2)/math.sqrt(32);bad=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),1)[None]|~mask[:,None,:];a=self.drop(a.masked_fill(bad,-1e9).softmax(-1));h=q+a@b['v'](h);z=b['n2'](h);h=(z+self.drop(b['f2'](self.drop(F.relu(b['f1'](z))))))*mask[...,None]
  h=self.norm(h);return h[torch.arange(len(x)),mask.sum(1)-1]
 def forward(self,users,x,t,c):return (self.encode(x)[:,None]*self.item(c)).sum(-1)
def model(data,kind,seed):
 torch.manual_seed(seed)
 return core.Baseline(kind,len(data['seq']),len(data['item_ids'])) if kind=='bpr_mf' else SASRec(len(data['item_ids'])) if kind=='sasrec' else Variant(data,kind)
def fit(data,kind,seed,lr,out,plan,initial=None,dot=False,frozen=False,continuation=False):
 out.mkdir(parents=True,exist_ok=True)
 if (out/'done.json').exists():return json.loads((out/'done.json').read_text())
 m=model(data,kind,seed)
 if initial is not None:m.load_state_dict(initial)
 if frozen:
  for n,p in m.named_parameters():p.requires_grad_(core.head_parameter(n))
  frozen_before={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=1e-5,betas=(.9,.98 if kind=='sasrec' else .999));best=-1;log=[];start=time.time();torch.manual_seed(seed+2026)
 for ep,(users,ends,cs) in enumerate(plan,1):
  m.train()
  if continuation:m.eval();m.attn.train()
  total=0
  for a in range(0,len(users),128):
   us=users[a:a+128];en=ends[a:a+128];x,t=core.pack(data['seq'],data['times'],us,en);c=torch.tensor(cs[a:a+128]);opt.zero_grad()
   if isinstance(m,core.E2E):g,v=m.items();u=m.encode(x,t,g);scores=m.score(u,v,c,dot)
   else:scores=m(torch.tensor(us),x,t,c)
   if kind=='bpr_mf':loss=-F.logsigmoid(scores[:,[0]]-scores[:,1:]).mean()
   elif kind=='sasrec':loss=F.softplus(-scores[:,0]).mean()+F.softplus(scores[:,1:]).mean()
   else:loss=F.cross_entropy(scores,torch.zeros(len(us),dtype=torch.long))
   if kind.startswith('aux'):
    aux=0
    if kind in ['aux_all','aux_con']:
     u1=m.encode(x,t,g,True);u2=m.encode(x,t,g,True);aux=aux+.1*F.cross_entropy(F.normalize(u1,dim=-1)@F.normalize(u2,dim=-1).T/.2,torch.arange(len(us)))
    if kind in ['aux_all','aux_tem']:
     px,pt=core.pack(data['seq'],data['times'],us,en-1);aux=aux+.05*((u-m.encode(px,pt,g))**2).sum(-1).mean()
    if kind in ['aux_all','aux_sem']:
     ix=c.unique();aux=aux+.2*(1-F.cosine_similarity(v[ix],m.prior[ix])).mean()
    loss=loss+min(ep/5,1)*aux
   assert torch.isfinite(loss), (kind,seed,lr,ep)
   loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();total+=float(loss)*len(us)
  val=core.evaluate(m,data,'validation',dot)['NDCG@10'];log.append(dict(epoch=ep,loss=total/len(users),validation_ndcg=val,seconds=time.time()-start))
  if val>best:best=val;bestep=ep;torch.save(m.state_dict(),out/'best.pt')
  if ep%10==0:print(out,ep,round(val,5),round(time.time()-start,1),flush=True)
 if frozen:assert all(torch.equal(p,frozen_before[n]) for n,p in m.named_parameters() if not p.requires_grad)
 torch.save(m.state_dict(),out/'final.pt');core.dump(out/'training.json',log)
 result=dict(kind=kind,seed=seed,lr=lr,best_epoch=bestep,best_validation=best,seconds=time.time()-start,parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),source_sha256=core.sha(__file__))
 core.dump(out/'done.json',result);return result
def train(dataset):
 data=torch.load(ROOT/'data'/dataset/'data.pt',weights_only=False);out=RUN/dataset;out.mkdir(parents=True,exist_ok=True);selections={}
 plan=core.schedule(data,501,50)
 for kind in KINDS:
  trials=[fit(data,kind,501,lr,out/'tuning'/kind/str(lr),plan) for lr in LRS];chosen=max(trials,key=lambda x:x['best_validation']);selections[kind]=chosen['lr'];core.dump(out/'selection.json',selections)
 for seed in SEEDS:
  plan=core.schedule(data,seed,50)
  for kind in KINDS:fit(data,kind,seed,selections[kind],out/'confirm'/kind/str(seed),plan)
  lr=selections['full'];pre=out/'matched'/'pretrain'/str(seed);fit(data,'full',seed,lr,pre,plan[:25],dot=True);initial=torch.load(pre/'final.pt',weights_only=True)
  for name,frozen in [('joint',False),('frozen',True)]:fit(data,'full',seed,lr,out/'matched'/name/str(seed),plan[25:],initial=initial,frozen=frozen,continuation=True)
 core.dump(out/'TRAINING_COMPLETE.json',dict(dataset=dataset,protocol_sha256=core.sha(ROOT/'PROTOCOL.md'),source_sha256=core.sha(__file__)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset');a=p.parse_args();train(a.dataset)
