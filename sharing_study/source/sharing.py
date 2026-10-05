"""Locked graph-sharing extension; no test access during training."""
import copy,json,sys,time,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
import core,study,evaluation
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'strengthening'/'data'
RUN=ROOT/'runs'
KINDS=['separate','shared_dual']; DATASETS=['movielens','amazon']
torch.set_num_threads(1)
class Sharing(study.Variant):
 def __init__(self,data,kind,seed):
  super().__init__(data,kind)
  with torch.random.fork_rng():
   torch.manual_seed(seed+40000)
   self.emb_second=nn.Embedding(len(self.emb.weight),32,padding_idx=0)
   nn.init.normal_(self.emb_second.weight,std=.05)
   self.gat_second=nn.ModuleList([core.GAT(32),core.GAT(32)])
 def graph(self,ids,emb,gat):
  required=[ids]
  for _ in gat:required.append(torch.unique(self.nb[required[-1]]))
  g=emb.weight[required[-1]]
  for layer,cur,prev in zip(gat,reversed(required[:-1]),reversed(required[1:])):
   local=torch.searchsorted(prev,cur);nb=self.nb[cur];nb_local=torch.searchsorted(prev,nb)
   h=layer.proj(g).reshape(len(g),4,-1);hn=h[nb_local]
   logits=F.leaky_relu((h[local,None]*layer.a).sum(-1)+(hn*layer.b).sum(-1),.2)+torch.log(self.w[cur].clamp_min(1e-9))[:,:,None]
   logits=logits.masked_fill((nb==0)[:,:,None],-1e9)
   z=(logits.softmax(1)[...,None]*hn).sum(1).reshape(len(cur),-1)
   g=layer.norm(g[local]+F.elu(z))*cur.ne(0)[:,None]
  return g
 def items(self,needed=None):
  ids=torch.arange(len(self.emb.weight)) if needed is None else torch.unique(needed)
  ga=self.graph(ids,self.emb,self.gat);gb=self.graph(ids,self.emb_second,self.gat_second)
  if self.kind=='shared_dual':ga=gb=(ga+gb)*.5
  cat=self.genres[ids]@self.genre_emb/self.genres[ids].sum(1,keepdim=True).clamp_min(1)
  v=self.fuse(torch.cat([gb,self.txt_proj(self.text[ids]),cat,self.year_mlp(self.year[ids])],-1))
  if needed is None:return ga,v
  return torch.zeros_like(self.emb.weight).index_copy(0,ids,ga),torch.zeros_like(self.emb.weight).index_copy(0,ids,v)
def model(data,kind,seed):
 torch.manual_seed(seed)
 return Sharing(data,kind,seed)
study.model=model

def worker(ds,phase,kind,seed=None):
 d=torch.load(DATA/ds/'data.pt',weights_only=False)
 if phase=='tuning':
  plan=core.schedule(d,501,50)
  for lr in study.LRS:study.fit(d,kind,501,lr,RUN/ds/'tuning'/kind/str(lr),plan)
 else:
  seed=int(seed);lr=json.loads((RUN/ds/'selection.json').read_text())[kind]
  study.fit(d,kind,seed,lr,RUN/ds/'confirm'/kind/str(seed),core.schedule(d,seed,50))
def call(job):
 log=ROOT/'logs'/('_'.join(map(str,job))+'.log');log.parent.mkdir(exist_ok=True)
 with log.open('a') as f:subprocess.run([sys.executable,__file__,'worker',*map(str,job)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('FINISHED',*job,flush=True)
def phase(jobs):
 with ThreadPoolExecutor(max_workers=4) as pool:
  for fut in as_completed([pool.submit(call,j) for j in jobs]):fut.result()
def main():
 phase([(ds,'tuning',k) for ds in DATASETS for k in KINDS])
 for ds in DATASETS:
  select={}
  for k in KINDS:
   trials=[json.loads((RUN/ds/'tuning'/k/str(lr)/'done.json').read_text()) for lr in study.LRS]
   select[k]=max(trials,key=lambda x:x['best_validation'])['lr']
  core.dump(RUN/ds/'selection.json',select)
 phase([(ds,'confirm',k,s) for ds in DATASETS for k in KINDS for s in study.SEEDS])
 core.dump(ROOT/'TRAINING_COMPLETE.json',{'complete':True,'protocol_sha256':core.sha(ROOT/'PROTOCOL.md'),'sources':{p.name:core.sha(p) for p in Path(__file__).parent.glob('*.py')},'data':{ds:core.sha(DATA/ds/'data.pt') for ds in DATASETS}})
 # All fitting and selection finished before test access.
 for ds in DATASETS:
  d=torch.load(DATA/ds/'data.pt',weights_only=False)
  for k in KINDS:
   for s in study.SEEDS:
    m=model(d,k,s);checkpoint=RUN/ds/'confirm'/k/str(s)/'best.pt';m.load_state_dict(torch.load(checkpoint,weights_only=True))
    evaluation.evaluate(m,d,RUN/ds/'evaluation'/k/str(s))
    core.dump(RUN/ds/'evaluation'/k/str(s)/'provenance.json',{'checkpoint_sha256':core.sha(checkpoint),'data_sha256':core.sha(DATA/ds/'data.pt')})
    print('EVALUATED',ds,k,s,flush=True)
 core.dump(ROOT/'COMPLETE.json',{'complete':True,'time':time.time()})
if __name__=='__main__':
 if len(sys.argv)>1:worker(*sys.argv[2:])
 else:main()
