"""Corrected static-catalog E2E-PREF pilot. See ARCHITECTURE_AND_PROTOCOL.md."""
import argparse,copy,hashlib,json,math,time,zipfile,re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent

def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metrics(r):
 r=np.asarray(r);return {'NDCG@10':float(np.where(r<=10,1/np.log2(r+1),0).mean()),'Recall@20':float((r<=20).mean()),'MRR':float((1/r).mean())}
def ranks(scores,c):return 1+((scores>scores[:,[0]])|((scores==scores[:,[0]])&(c<c[:,[0]]))).sum(1)

def candidates(history,target,n_items,k,rng):
 """Only permitted history and current target enter exclusion. IDs are 1-based."""
 excluded=set(map(int,history));excluded.add(int(target))
 if n_items-len(excluded)<k:raise ValueError('Insufficient eligible negatives')
 neg=[]; used=set(excluded)
 while len(neg)<k:
  j=int(rng.integers(1,n_items+1))
  if j not in used:neg.append(j);used.add(j)
 return np.array([target]+neg,dtype=np.int64)

def make_graph(train,n_items):
 rows=[];cols=[]
 for u,s in enumerate(train):rows.extend([u]*len(s));cols.extend(s)
 mat=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(train),n_items+1))
 co=(mat.T@mat).tocsr();co.setdiag(0);co.eliminate_zeros();co.sort_indices()
 nb=np.zeros((n_items+1,9),np.int64);w=np.zeros_like(nb,np.float32)
 for i in range(1,n_items+1):
  nb[i,0]=i;w[i,0]=1;a,b=co.indptr[i:i+2];ix=co.indices[a:b];val=co.data[a:b]
  keep=np.flatnonzero(val>=2);keep=keep[np.lexsort((ix[keep],-val[keep]))[:8]]
  nb[i,1:1+len(keep)]=ix[keep];w[i,1:1+len(keep)]=np.log1p(val[keep])
 return nb,w,np.asarray(mat.sum(0)).ravel(),mat

def prepare(datazip,cache):
 cache.mkdir(parents=True,exist_ok=True)
 if (cache/'data.pt').exists():return torch.load(cache/'data.pt',weights_only=False)
 with zipfile.ZipFile(datazip) as z:
  movies=pd.read_csv(z.open('ml-20m/movies.csv')).sort_values('movieId');ids=movies.movieId.to_numpy();mapping={int(x):i+1 for i,x in enumerate(ids)}
  chunks=[]
  for x in pd.read_csv(z.open('ml-20m/ratings.csv'),chunksize=1000000):chunks.append(x[x.rating>=4])
  df=pd.concat(chunks,ignore_index=True);counts=df.groupby('userId').size();eligible=counts[counts>=10].index.to_numpy();users=np.sort(np.random.default_rng(2026).choice(eligible,1000,False))
  df=df[df.userId.isin(users)].sort_values(['userId','timestamp','movieId']);seq=[];times=[]
  for _,g in df.groupby('userId',sort=True):seq.append(np.array([mapping[int(i)] for i in g.movieId]));times.append(g.timestamp.to_numpy())
 genres=sorted({x for g in movies.genres for x in g.split('|')});genre=np.zeros((len(ids)+1,len(genres)),np.float32);year=np.zeros((len(ids)+1,1),np.float32);texts=[]
 for j,row in enumerate(movies.itertuples(),1):
  for g in row.genres.split('|'):genre[j,genres.index(g)]=1
  y=re.search(r'\((\d{4})\)\s*$',row.title);year[j,0]=(int(y[1])-1950)/100 if y else 0
  texts.append(row.title+' '+row.genres.replace('|',' '))
 # Pin the resolved Hub revision before downloading. No rating data leaves this machine.
 from huggingface_hub import HfApi
 from transformers import AutoTokenizer,AutoModel
 model_id='sentence-transformers/all-MiniLM-L6-v2';modelcache=cache/'huggingface';revision=HfApi().model_info(model_id).sha
 tok=AutoTokenizer.from_pretrained(model_id,revision=revision,cache_dir=modelcache)
 enc=AutoModel.from_pretrained(model_id,revision=revision,cache_dir=modelcache,use_safetensors=True).eval()
 text=np.zeros((len(ids)+1,384),np.float32)
 with torch.no_grad():
  for start in range(0,len(texts),64):
   inputs=tok(texts[start:start+64],padding=True,truncation=True,max_length=256,return_tensors='pt');h=enc(**inputs).last_hidden_state;mask=inputs['attention_mask'][...,None]
   v=F.normalize((h*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1);text[start+1:start+1+len(v)]=v.numpy()
   if start%2048==0:print('TEXT',start,len(texts),flush=True)
 del enc
 train=[s[:-2] for s in seq];nb,w,pop,mat=make_graph(train,len(ids));slates={}
 for split,offset,seed in [('validation',2,777),('test',1,778)]:
  rng=np.random.default_rng(seed);slates[split]=np.stack([candidates(s[:-offset],s[-offset],len(ids),99,rng) for s in seq])
 summary={'users':len(users),'catalog_items':len(ids),'positive_interactions':sum(map(len,seq)),'training_interactions':sum(map(len,train)),'validation_interactions':len(seq),'test_interactions':len(seq),'eligible_users':len(eligible),'test_targets_without_training_positive':int(sum(pop[s[-1]]==0 for s in seq)),'dataset_sha256':sha(datazip),'text_model':model_id,'text_revision':revision,'protocol':'static catalog, training-only graph, prefix-plus-current-target exclusions, no global temporal availability claim'}
 data={'seq':seq,'times':times,'user_ids':users,'item_ids':ids,'genres':genre,'year':year,'text':text,'nb':nb,'w':w,'pop':pop,'slates':slates,'summary':summary}
 torch.save(data,cache/'data.pt');dump(cache/'data_summary.json',summary);sp.save_npz(cache/'train_matrix.npz',mat)
 np.savez_compressed(cache/'split_manifest.npz',user_ids=users,item_ids=ids,validation_candidates=slates['validation'],test_candidates=slates['test'])
 return data

def pack(seq,times,users,ends,length=30):
 x=np.zeros((len(users),length),np.int64);t=np.zeros_like(x)
 for j,(u,end) in enumerate(zip(users,ends)):
  start=max(0,int(end)-length);s=seq[u][start:end];ts=times[u][start:end];x[j,:len(s)]=s
  t[j,:len(s)]=np.minimum(31,np.floor(np.log2(1+np.r_[0,np.diff(ts)]/3600))).astype(np.int64)
 return torch.tensor(x),torch.tensor(t)

class GAT(nn.Module):
 def __init__(self,d):
  super().__init__();self.proj=nn.Linear(d,d,bias=False);self.a=nn.Parameter(torch.randn(4,d//4)*.1);self.b=nn.Parameter(torch.randn(4,d//4)*.1);self.norm=nn.LayerNorm(d)
 def forward(self,x,nb,w):
  h=self.proj(x).reshape(len(x),4,-1);hn=h[nb];logits=F.leaky_relu((h[:,None]*self.a).sum(-1)+(hn*self.b).sum(-1),.2)+torch.log(w.clamp_min(1e-9))[:,:,None]
  logits=logits.masked_fill((nb==0)[:,:,None],-1e9);out=(logits.softmax(1)[...,None]*hn).sum(1).reshape(len(x),-1)
  return self.norm(x+F.elu(out))*torch.arange(len(x)).ne(0)[:,None]

class E2E(nn.Module):
 def __init__(self,data,d=32):
  super().__init__();n=len(data['text']);self.d=d
  for key in ['text','genres','year','nb','w']:self.register_buffer(key,torch.tensor(data[key]),persistent=False)
  self.emb=nn.Embedding(n,d,padding_idx=0);nn.init.normal_(self.emb.weight,std=.05)
  self.gat=nn.ModuleList([GAT(d),GAT(d)]);self.pos=nn.Embedding(30,d);self.gap=nn.Embedding(32,d)
  layer=nn.TransformerEncoderLayer(d,4,d*4,.2,batch_first=True,activation='gelu');self.transformer=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
  self.pool=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,1));self.txt_proj=nn.Linear(384,d)
  self.genre_emb=nn.Parameter(torch.randn(data['genres'].shape[1],d)*.05);self.year_mlp=nn.Sequential(nn.Linear(1,d),nn.GELU(),nn.Linear(d,d))
  self.fuse=nn.Sequential(nn.Linear(4*d,2*d),nn.GELU(),nn.Linear(2*d,2*d),nn.GELU(),nn.Linear(2*d,d))
  self.attn=nn.MultiheadAttention(d,4,dropout=0,batch_first=True);self.w1=nn.Parameter(torch.ones(d)/math.sqrt(d));self.w2=nn.Parameter(torch.zeros(d));self.bias=nn.Embedding(n,1);nn.init.zeros_(self.bias.weight)
  self.register_buffer('prior',F.normalize(self.text@torch.randn(384,d),dim=-1))
 def items(self):
  g=self.emb.weight
  for layer in self.gat:g=layer(g,self.nb,self.w)
  cat=self.genres@self.genre_emb/self.genres.sum(1,keepdim=True).clamp_min(1)
  v=self.fuse(torch.cat([g,self.txt_proj(self.text),cat,self.year_mlp(self.year)],-1));return g,v
 def encode(self,x,t,g,augment=False):
  valid=x.ne(0)
  if augment:
   keep=torch.rand(x.shape)>.2;keep[:,0]=True;valid=valid&keep
  h=(g[x]+self.pos(torch.arange(x.shape[1]))[None]+self.gap(t))*valid[...,None]
  h=self.transformer(h,mask=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),1),src_key_padding_mask=~valid)
  a=self.pool(h).squeeze(-1).masked_fill(~valid,-1e9).softmax(-1);return (h*a[...,None]).sum(1)
 def score(self,u,v,c,dot=False):
  cv=v[c]
  if dot:return (u[:,None]*cv).sum(-1)/math.sqrt(self.d)+self.bias(c).squeeze(-1)
  uc=self.attn(u[:,None],cv,cv,need_weights=False)[0][:,0]
  return ((uc[:,None]*cv)*self.w1).sum(-1)+((uc[:,None]+cv)*self.w2).sum(-1)+self.bias(c).squeeze(-1)

class Baseline(nn.Module):
 """Audited simplified family controls, not official baseline reproductions."""
 def __init__(self,kind,n_users,n_items,d=32):
  super().__init__();self.kind=kind;self.item=nn.Embedding(n_items+1,d,padding_idx=0)
  if kind in ['bpr_mf','mlp_cf']:self.user=nn.Embedding(n_users,d)
  if kind=='bpr_mf':nn.init.normal_(self.item.weight,std=.02);nn.init.normal_(self.user.weight,std=.02)
  if kind=='mlp_cf':self.mlp=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,1))
  if kind=='gru':self.gru=nn.GRU(d,d,batch_first=True);self.drop=nn.Dropout(.2)
  if kind=='self_attention':
   self.pos=nn.Embedding(30,d);layer=nn.TransformerEncoderLayer(d,4,d*4,.2,batch_first=True,activation='gelu',norm_first=True);self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.norm=nn.LayerNorm(d)
 def forward(self,users,x,t,c):
  v=self.item(c)
  if self.kind in ['bpr_mf','mlp_cf']:u=self.user(users)
  else:
   lengths=x.ne(0).sum(1)
   if self.kind=='gru':
    packed=nn.utils.rnn.pack_padded_sequence(self.drop(self.item(x)),lengths.cpu(),batch_first=True,enforce_sorted=False);u=self.gru(packed)[1][-1]
   else:
    h=self.item(x)+self.pos(torch.arange(x.shape[1]))[None];h=self.encoder(h,mask=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),1),src_key_padding_mask=x.eq(0));u=self.norm(h[torch.arange(len(x)),lengths-1])
  if self.kind=='mlp_cf':return self.mlp(torch.cat([u[:,None].expand_as(v),v],-1)).squeeze(-1)
  return (u[:,None]*v).sum(-1)

def schedule(data,seed,epochs=20):
 # No held-out values in training schedule construction: only training partition.
 rng=np.random.default_rng(seed+1009);train=[s[:-2] for s in data['seq']];n_items=len(data['item_ids']);plan=[]
 for epoch in range(epochs):
  users=rng.permutation(len(train));ends=np.array([rng.integers(2,len(train[u])) for u in users]);cs=np.stack([candidates(train[u][:end],train[u][end],n_items,31,rng) for u,end in zip(users,ends)])
  plan.append((users,ends,cs))
 return plan

def evaluate(model,data,split,dot=False,save=None):
 model.eval();ss=[];c=data['slates'][split]
 with torch.no_grad():
  if isinstance(model,E2E):g,v=model.items()
  for start in range(0,len(c),128):
   users=np.arange(start,min(start+128,len(c)));ends=[len(data['seq'][u])-(2 if split=='validation' else 1) for u in users];x,t=pack(data['seq'],data['times'],users,ends);cc=torch.tensor(c[users])
   score=model.score(model.encode(x,t,g),v,cc,dot) if isinstance(model,E2E) else model(torch.tensor(users),x,t,cc)
   ss.append(score.numpy())
 scores=np.concatenate(ss);r=ranks(scores,c);m=metrics(r)
 if not np.isfinite(scores).all() or m['NDCG@10']>m['Recall@20']+1e-12:raise ValueError('Invalid metrics/scores')
 if save:np.savez_compressed(save,scores=scores,candidates=c,ranks=r,user_ids=data['user_ids'])
 return m

def head_parameter(name):return name.startswith(('attn.','w1','w2','bias.'))
def fit(model,data,plan,name,seed,out,aux=False,dot=False,continuation=False,frozen=False):
 torch.manual_seed(seed+2026);before={k:v.clone() for k,v in model.state_dict().items() if not head_parameter(k)} if frozen else None
 if frozen:
  for k,p in model.named_parameters():p.requires_grad_(head_parameter(k))
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-5);best=-1;state=None;history=[];start=time.time()
 for epoch,(users,ends,cs) in enumerate(plan,1):
  model.train()
  if continuation:model.eval();model.attn.train() # both branches share disabled encoder dropout
  total=0
  for startbatch in range(0,len(users),128):
   us=users[startbatch:startbatch+128];en=ends[startbatch:startbatch+128];x,t=pack(data['seq'],data['times'],us,en);c=torch.tensor(cs[startbatch:startbatch+128]);opt.zero_grad()
   if isinstance(model,E2E):g,v=model.items();u=model.encode(x,t,g);score=model.score(u,v,c,dot)
   else:score=model(torch.tensor(us),x,t,c)
   if isinstance(model,Baseline) and model.kind=='bpr_mf':loss=-F.logsigmoid(score[:,[0]]-score[:,1:]).mean()
   else:loss=F.cross_entropy(score,torch.zeros(len(us),dtype=torch.long))
   if aux:
    a=model.encode(x,t,g,True);b=model.encode(x,t,g,True);con=F.cross_entropy(F.normalize(a,dim=-1)@F.normalize(b,dim=-1).T/.2,torch.arange(len(us)))
    px,pt=pack(data['seq'],data['times'],us,en-1);prev=model.encode(px,pt,g);tem=((u-prev)**2).sum(-1).mean();ix=c.unique();sem=(1-F.cosine_similarity(v[ix],model.prior[ix])).mean();loss=loss+min(epoch/5,1)*(.1*con+.05*tem+.2*sem)
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(us)
  val=evaluate(model,data,'validation',dot);row={'epoch':epoch,'loss':total/len(users),'validation':val,'seconds':time.time()-start};history.append(row)
  if val['NDCG@10']>best:best=val['NDCG@10'];state=copy.deepcopy(model.state_dict());bestepoch=epoch
  if epoch%5==0:print(name,seed,row,flush=True)
 final=copy.deepcopy(model.state_dict())
 if frozen:
  assert all(torch.equal(before[k],final[k]) for k in before),'Frozen encoder changed'
 model.load_state_dict(state);torch.save(state,out/f'{name}_{seed}.pt');dump(out/f'{name}_{seed}_training.json',history)
 if dot:return final
 m=evaluate(model,data,'test',save=out/f'{name}_{seed}_predictions.npz');row={'model':name,'seed':seed,'best_epoch':bestepoch,'validation_ndcg':best,'seconds':time.time()-start,**m};dump(out/f'{name}_{seed}_result.json',row);print('RESULT',row,flush=True);return row

def deterministic(data,cache,out):
 mat=sp.load_npz(cache/'train_matrix.npz');norm=np.sqrt(np.asarray(mat.power(2).sum(0)).ravel());inv=np.divide(1,norm,out=np.zeros_like(norm),where=norm>0);sim=(sp.diags(inv)@(mat.T@mat)@sp.diags(inv)).tocsr();sim.setdiag(0);sim.eliminate_zeros();sim.sort_indices()
 for i in range(sim.shape[0]):
  a,b=sim.indptr[i:i+2];ix=sim.indices[a:b];v=sim.data[a:b];order=np.lexsort((ix,-v));v[order[50:]]=0
 sim.eliminate_zeros();sp.save_npz(out/'itemknn_similarity.npz',sim)
 c=data['slates']['test'];scores={k:[] for k in ['popularity','itemknn','content_average']}
 for i,s in enumerate(data['seq']):
  hist=s[:-1];scores['popularity'].append(data['pop'][c[i]]);scores['itemknn'].append(np.asarray(sim[hist].sum(0)).ravel()[c[i]]);profile=data['text'][hist[-30:]].mean(0);scores['content_average'].append(data['text'][c[i]]@profile)
 for name,ss in scores.items():
  ss=np.stack(ss);r=ranks(ss,c);np.savez_compressed(out/f'{name}_predictions.npz',scores=ss,candidates=c,ranks=r,user_ids=data['user_ids']);dump(out/f'{name}_result.json',{'model':name,**metrics(r)})

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='work/data/ml-20m.zip');p.add_argument('--cache',default='work/clean_e2e_pref/data');p.add_argument('--out',default=str(ROOT/'runs'));p.add_argument('--prepare-only',action='store_true');p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);args=p.parse_args();torch.set_num_threads(4);cache=Path(args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 data=prepare(args.data,cache);dump(out/'data_summary.json',data['summary']);np.savez_compressed(out/'split_manifest.npz',user_ids=data['user_ids'],item_ids=data['item_ids'],**{k+'_candidates':v for k,v in data['slates'].items()})
 if args.prepare_only:return
 dump(out/'run_config.json',{'seeds':args.seeds,'architecture_contract_sha256':sha(ROOT/'ARCHITECTURE_AND_PROTOCOL.md'),'source_sha256':sha(__file__),'epochs':20,'pretrain_epochs':10,'continuation_epochs':10,'learning_rate':.001,'dimensions':32,'selection':'best sampled validation NDCG, ties earliest','test_not_used_for_tuning':True})
 deterministic(data,cache,out)
 for seed in args.seeds:
  plan=schedule(data,seed);np.savez_compressed(out/f'training_plan_{seed}.npz',users=np.stack([x[0] for x in plan]),ends=np.stack([x[1] for x in plan]),candidates=np.stack([x[2] for x in plan]))
  torch.manual_seed(seed);initial=E2E(data).state_dict()
  for name,aux in [('e2e_joint',True),('e2e_rank_only',False)]:
   torch.manual_seed(seed);m=E2E(data);m.load_state_dict(initial);fit(m,data,plan,name,seed,out,aux=aux)
  torch.manual_seed(seed);m=E2E(data);m.load_state_dict(initial);pre=fit(m,data,plan[:10],'shared_pretraining',seed,out,dot=True);torch.save(pre,out/f'shared_pretraining_final_{seed}.pt')
  for name,frozen in [('matched_frozen',True),('matched_joint',False)]:
   torch.manual_seed(seed);m=E2E(data);m.load_state_dict(pre);fit(m,data,plan[10:],name,seed,out,continuation=True,frozen=frozen)
  for name in ['bpr_mf','mlp_cf','gru','self_attention']:
   torch.manual_seed(seed);m=Baseline(name,len(data['seq']),len(data['item_ids']));fit(m,data,plan,name,seed,out)
 dump(out/'completed.json',{'status':'complete','seeds':args.seeds})
if __name__=='__main__':main()
