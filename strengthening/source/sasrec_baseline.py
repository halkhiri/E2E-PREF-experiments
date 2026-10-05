"""SASRec source-informed port with original all-position logistic training.
See SASREC_LICENSE.txt and SOURCE_PROVENANCE.md for attribution and adaptations.
"""
import time,json
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import core,study
class SASRec(study.SASRec):
 def __init__(self,n,d=32):
  super().__init__(n,d)
  nn.init.xavier_uniform_(self.item.weight);self.item.weight.data[0]=0;nn.init.xavier_uniform_(self.pos.weight)
  for b in self.blocks:
   for name in ['q','k','v','f1','f2']:nn.init.xavier_uniform_(b[name].weight);nn.init.zeros_(b[name].bias)
 def features(self,x):
  mask=x.ne(0);h=self.drop(self.item(x)*np.sqrt(32)+self.pos(torch.arange(x.shape[1]))[None])*mask[...,None]
  for b in self.blocks:
   q=b['n1'](h);a=b['q'](q)@b['k'](h).transpose(1,2)/np.sqrt(32);bad=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),1)[None]|~mask[:,None,:];a=self.drop(a.masked_fill(bad,-1e9).softmax(-1));h=q+a@b['v'](h);z=b['n2'](h);h=(z+self.drop(b['f2'](self.drop(F.relu(b['f1'](z))))))*mask[...,None]
  return self.norm(h)
 def encode(self,x):
  # Common evaluators supply right-padded histories; original SASRec uses left padding.
  left=torch.zeros_like(x)
  for j in range(len(x)):
   n=int(x[j].ne(0).sum());left[j,-n:]=x[j,:n]
  return self.features(left)[:,-1]
def examples(data,seed,epochs=50):
 rng=np.random.default_rng(seed+1009);train=[s[:-2] for s in data['seq']];x=np.zeros((len(train),30),np.int64);pos=x.copy()
 for u,s in enumerate(train):
  n=min(len(s)-1,30);x[u,-n:]=s[-n-1:-1];pos[u,-n:]=s[-n:]
 for ep in range(epochs):
  users=rng.permutation(len(train));neg=np.zeros_like(pos)
  for u in users:
   known=set(map(int,train[u]))
   for j in np.flatnonzero(pos[u]):
    while True:
     item=int(rng.integers(1,len(data['item_ids'])+1))
     if item not in known:neg[u,j]=item;break
  yield users,x,pos,neg

def fit(data,seed,lr,out):
 out.mkdir(parents=True,exist_ok=True)
 if (out/'done.json').exists():return json.loads((out/'done.json').read_text())
 torch.manual_seed(seed);m=SASRec(len(data['item_ids']));torch.manual_seed(seed+2026);opt=torch.optim.Adam(m.parameters(),lr=lr,betas=(.9,.98));best=-1;hist=[];start=time.time()
 for ep,(users,x,pos,neg) in enumerate(examples(data,seed),1):
  m.train();total=0;pairs=0
  for a in range(0,len(users),128):
   us=users[a:a+128];xt=torch.tensor(x[us]);pt=torch.tensor(pos[us]);nt=torch.tensor(neg[us]);opt.zero_grad();h=m.features(xt);valid=pt.ne(0);sp=(h*m.item(pt)).sum(-1)[valid];sn=(h*m.item(nt)).sum(-1)[valid];loss=(F.softplus(-sp)+F.softplus(sn)).mean();assert torch.isfinite(loss);loss.backward();opt.step();total+=float(loss)*len(sp);pairs+=len(sp)
  val=core.evaluate(m,data,'validation')['NDCG@10'];hist.append(dict(epoch=ep,loss=total/pairs,validation_ndcg=val,seconds=time.time()-start,positive_positions=pairs))
  if val>best:best=val;bestep=ep;torch.save(m.state_dict(),out/'best.pt')
  if ep%10==0:print(out,ep,val,time.time()-start,flush=True)
 torch.save(m.state_dict(),out/'final.pt');core.dump(out/'training.json',hist);result=dict(kind='sasrec',seed=seed,lr=lr,best_epoch=bestep,best_validation=best,seconds=time.time()-start,parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters()),source_sha256=core.sha(__file__),objective='all nonpadding sequence positions, one negative each, Adam beta2=.98',positive_positions_per_epoch=pairs)
 core.dump(out/'done.json',result);return result
