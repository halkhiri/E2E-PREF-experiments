"""Rebuild the exact study data from separately downloaded public source files."""
import argparse,ast,gzip,json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
import core,prepare
from transformers import AutoTokenizer,AutoModel

def embed(texts,model_path):
 tok=AutoTokenizer.from_pretrained(model_path);enc=AutoModel.from_pretrained(model_path).eval();arr=np.zeros((len(texts)+1,384),np.float32)
 with torch.no_grad():
  for a in range(0,len(texts),64):
   inp=tok(texts[a:a+64],padding=True,truncation=True,max_length=256,return_tensors='pt');h=enc(**inp).last_hidden_state;mask=inp['attention_mask'][...,None];v=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=-1);arr[a+1:a+1+len(v)]=v.numpy()
 return arr
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ml20m',required=True);p.add_argument('--amazon-reviews',required=True);p.add_argument('--amazon-metadata',required=True);p.add_argument('--model-snapshot',required=True,help='Local Hugging Face checkpoint revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41');p.add_argument('--output',required=True);a=p.parse_args();prepare.ROOT=Path(a.output);(prepare.ROOT/'data').mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 with zipfile.ZipFile(a.ml20m) as z:
  movies=pd.read_csv(z.open('ml-20m/movies.csv')).sort_values('movieId');df=pd.concat([c[c.rating>=4] for c in pd.read_csv(z.open('ml-20m/ratings.csv'),chunksize=1000000)],ignore_index=True)
 counts=df.groupby('userId').size();eligible=counts[counts>=10].index.to_numpy();old=np.sort(np.random.default_rng(2026).choice(eligible,1000,False));eligible=np.setdiff1d(eligible,old);users=np.sort(np.random.default_rng(29092026).choice(eligible,1000,False));df=df[df.userId.isin(users)].rename(columns={'userId':'user','movieId':'item','timestamp':'time'});ids=movies.movieId.to_numpy();cats=sorted({x for g in movies.genres for x in g.split('|')});genre=np.zeros((len(ids)+1,len(cats)),np.float32);year=np.zeros((len(ids)+1,1),np.float32);texts=[]
 for j,r in enumerate(movies.itertuples(),1):
  for c in r.genres.split('|'):genre[j,cats.index(c)]=1
  y=re.search(r'\((\d{4})\)\s*$',r.title);year[j,0]=(int(y[1])-1950)/100 if y else 0;texts.append(r.title+' '+r.genres.replace('|',' '))
 prepare.finish('movielens',df,ids,dict(text=embed(texts,a.model_snapshot),genres=genre,year=year),dict(old_pilot_user_overlap=0,sampling_seed=29092026,dataset_sha256=core.sha(a.ml20m)))
 raw=[json.loads(x) for x in gzip.open(a.amazon_reviews,'rt')];ids=sorted({r['asin'] for r in raw});df=pd.DataFrame([dict(user=r['reviewerID'],item=r['asin'],time=r['unixReviewTime']) for r in raw if r['overall']>=4]).sort_values('time').drop_duplicates(['user','item']);counts=df.groupby('user').size();df=df[df.user.isin(counts[counts>=5].index)];meta={}
 for line in gzip.open(a.amazon_metadata,'rt'):
  r=ast.literal_eval(line)
  if r['asin'] in ids:meta[r['asin']]=r
 cats=sorted({c for i in ids for group in meta.get(i,{}).get('categories',[]) for c in group});genre=np.zeros((len(ids)+1,len(cats)),np.float32);year=np.zeros((len(ids)+1,1),np.float32);texts=[]
 for j,i in enumerate(ids,1):
  r=meta.get(i,{});cs=sorted({c for group in r.get('categories',[]) for c in group})
  for c in cs:genre[j,cats.index(c)]=1
  price=r.get('price',0);year[j,0]=np.log1p(max(0,float(price))) if isinstance(price,(int,float)) else 0;texts.append(r.get('title','')+' '+' '.join(cs))
 prepare.finish('amazon',df,ids,dict(text=embed(texts,a.model_snapshot),genres=genre,year=year),dict(raw_reviews=len(raw),minimum_positive_history=5,metadata_items_found=len(meta),empty_titles=sum(not meta.get(i,{}).get('title') for i in ids),source_sha256=core.sha(a.amazon_reviews),metadata_sha256=core.sha(a.amazon_metadata)))
