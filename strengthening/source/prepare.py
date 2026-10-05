import ast,gzip,json,sys,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
import core
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
def finish(name,df,ids,features,extra):
 out=ROOT/'data'/name;out.mkdir(exist_ok=True)
 mapping={str(x):i+1 for i,x in enumerate(ids)};seq=[];times=[];users=[]
 for u,g in df.sort_values(['user','time','item']).groupby('user',sort=True):
  users.append(u);seq.append(np.array([mapping[str(x)] for x in g.item],dtype=np.int64));times.append(g.time.to_numpy(dtype=np.int64))
 nb,w,pop,mat=core.make_graph([s[:-2] for s in seq],len(ids));slates={}
 for split,off,seed in [('validation',2,2909777),('test',1,2909778)]:
  rng=np.random.default_rng(seed);slates[split]=np.stack([core.candidates(s[:-off],s[-off],len(ids),99,rng) for s in seq])
 summary=dict(dataset=name,users=len(users),catalog_items=len(ids),positive_events=sum(map(len,seq)),training_events=sum(len(s)-2 for s in seq),test_without_train=int(sum(pop[s[-1]]==0 for s in seq)),**extra)
 data=dict(seq=seq,times=times,user_ids=np.array(users),item_ids=np.array(ids),nb=nb,w=w,pop=pop,slates=slates,summary=summary,**features)
 torch.save(data,out/'data.pt');core.dump(out/'summary.json',summary);core.sp.save_npz(out/'train.npz',mat);print(summary,flush=True)
if __name__=='__main__':
 old=torch.load('work/clean_e2e_pref/data/data.pt',weights_only=False)
 if not (ROOT/'data/movielens/data.pt').exists():
  with zipfile.ZipFile('work/data/ml-20m.zip') as z:
   df=pd.concat([c[c.rating>=4] for c in pd.read_csv(z.open('ml-20m/ratings.csv'),chunksize=1000000)])
  sizes=df.groupby('userId').size();eligible=np.setdiff1d(sizes[sizes>=10].index,old['user_ids']);users=np.sort(np.random.default_rng(29092026).choice(eligible,1000,False));assert not set(users)&set(old['user_ids'])
  df=df[df.userId.isin(users)].rename(columns={'userId':'user','movieId':'item','timestamp':'time'})
  finish('movielens',df,old['item_ids'],{k:old[k] for k in ['text','genres','year']},dict(old_pilot_user_overlap=0,sampling_seed=29092026,dataset_sha256=old['summary']['dataset_sha256']))
 if not (ROOT/'data/amazon/data.pt').exists():
  raw=[json.loads(x) for x in gzip.open(ROOT/'data/reviews_Musical_Instruments_5.json.gz','rt')];ids=sorted({r['asin'] for r in raw});rows=[dict(user=r['reviewerID'],item=r['asin'],time=r['unixReviewTime']) for r in raw if r['overall']>=4];df=pd.DataFrame(rows).sort_values('time').drop_duplicates(['user','item']);counts=df.groupby('user').size();df=df[df.user.isin(counts[counts>=5].index)]
  meta={}
  for line in gzip.open(ROOT/'data/meta_Musical_Instruments.json.gz','rt'):
   r=ast.literal_eval(line)
   if r['asin'] in ids:meta[r['asin']]=r
  cats=sorted({c for i in ids for group in meta.get(i,{}).get('categories',[]) for c in group});genre=np.zeros((len(ids)+1,len(cats)),np.float32);year=np.zeros((len(ids)+1,1),np.float32);texts=[]
  for j,i in enumerate(ids,1):
   r=meta.get(i,{});cs=sorted({c for group in r.get('categories',[]) for c in group})
   for c in cs:genre[j,cats.index(c)]=1
   p=r.get('price',0);year[j,0]=np.log1p(max(0,float(p))) if isinstance(p,(int,float)) else 0
   texts.append(r.get('title','')+' '+' '.join(cs))
  from transformers import AutoTokenizer,AutoModel
  snapshot=Path('work/clean_e2e_pref/data/huggingface/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
  tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True);enc=AutoModel.from_pretrained(snapshot,local_files_only=True).eval();text=np.zeros((len(ids)+1,384),np.float32)
  with torch.no_grad():
   for start in range(0,len(ids),64):
    inp=tok(texts[start:start+64],padding=True,truncation=True,max_length=256,return_tensors='pt');h=enc(**inp).last_hidden_state;mask=inp['attention_mask'][...,None];v=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=-1);text[start+1:start+1+len(v)]=v.numpy()
  finish('amazon',df,ids,dict(text=text,genres=genre,year=year),dict(raw_reviews=len(raw),minimum_positive_history=5,metadata_items_found=len(meta),empty_titles=sum(not meta.get(i,{}).get('title') for i in ids),source_sha256=core.sha(ROOT/'data/reviews_Musical_Instruments_5.json.gz'),metadata_sha256=core.sha(ROOT/'data/meta_Musical_Instruments.json.gz')))
