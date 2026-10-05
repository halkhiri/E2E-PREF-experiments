"""Resumable bounded parallel scheduler. Same protocol and per-run code."""
import os,sys,subprocess,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import study,core,torch
ROOT=study.ROOT;DATASETS=['movielens','amazon'];MAX_WORKERS=6

def worker(ds,phase,kind):
 d=torch.load(ROOT/'data'/ds/'data.pt',weights_only=False);out=study.RUN/ds;out.mkdir(parents=True,exist_ok=True)
 if kind=='sasrec' and phase in ['tuning','confirm']:
  import sasrec_baseline
  if phase=='tuning':
   for lr in study.LRS:sasrec_baseline.fit(d,501,lr,out/'tuning'/kind/str(lr))
  else:
   lr=json.loads((out/'selection.json').read_text())[kind]
   for seed in study.SEEDS:sasrec_baseline.fit(d,seed,lr,out/'confirm'/kind/str(seed))
  return
 if phase=='tuning':
  plan=core.schedule(d,501,50)
  if kind=='bpr_mf':
   from baseline_plan import bpr_plan
   plan=bpr_plan(d,501,50)
  trainer=study
  if ds=='amazon' and kind=='full':
   import study_dense_original as trainer
  for lr in study.LRS:trainer.fit(d,kind,501,lr,out/'tuning'/kind/str(lr),plan)
 elif phase=='confirm':
  lr=json.loads((out/'selection.json').read_text())[kind]
  for seed in study.SEEDS:
   plan=core.schedule(d,seed,50)
   if kind=='bpr_mf':
    from baseline_plan import bpr_plan
    plan=bpr_plan(d,seed,50)
   study.fit(d,kind,seed,lr,out/'confirm'/kind/str(seed),plan)
 elif phase=='matched':
  seed=int(kind);plan=core.schedule(d,seed,50);lr=json.loads((out/'selection.json').read_text())['full'];pre=out/'matched'/'pretrain'/str(seed);study.fit(d,'full',seed,lr,pre,plan[:25],dot=True);initial=torch.load(pre/'final.pt',weights_only=True)
  for name,frozen in [('joint',False),('frozen',True)]:study.fit(d,'full',seed,lr,out/'matched'/name/str(seed),plan[25:],initial=initial,frozen=frozen,continuation=True)
def call(job):
 ds,phase,kind=job;logpath=ROOT/'parallel_logs'/f'{ds}_{phase}_{kind}.log';logpath.parent.mkdir(exist_ok=True)
 with open(logpath,'a') as log:
  ret=subprocess.run([sys.executable,__file__,'worker',ds,phase,str(kind)],stdout=log,stderr=subprocess.STDOUT)
 if ret.returncode:raise RuntimeError(f'{job} failed {ret.returncode}; see {logpath}')
 print('FINISHED',*job,flush=True)
def phase(jobs):
 with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
  for fut in as_completed([pool.submit(call,j) for j in jobs]):fut.result()
if __name__=='__main__':
 if len(sys.argv)>1:worker(*sys.argv[2:]);sys.exit()
 phase([(ds,'tuning',k) for ds in DATASETS for k in study.KINDS])
 for ds in DATASETS:
  selections={}
  for k in study.KINDS:
   trials=[json.loads((study.RUN/ds/'tuning'/k/str(lr)/'done.json').read_text()) for lr in study.LRS];selections[k]=max(trials,key=lambda x:x['best_validation'])['lr']
  core.dump(study.RUN/ds/'selection.json',selections)
 phase([(ds,'confirm',k) for ds in DATASETS for k in study.KINDS]+[(ds,'matched',s) for ds in DATASETS for s in study.SEEDS])
 for ds in DATASETS:core.dump(study.RUN/ds/'TRAINING_COMPLETE.json',dict(dataset=ds,protocol_sha256=core.sha(ROOT/'PROTOCOL.md'),source_sha256=core.sha(ROOT/'source/study.py')))
 for ds in DATASETS:
  with open(ROOT/(ds+'_evaluation.log'),'a') as log:subprocess.run([sys.executable,str(ROOT/'source/evaluation.py'),ds],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([sys.executable,str(ROOT/'source/analyze.py')],check=True)
 core.dump(ROOT/'COMPLETED.json',dict(complete=True,time=time.time()))
