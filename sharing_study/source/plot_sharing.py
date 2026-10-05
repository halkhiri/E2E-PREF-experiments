from pathlib import Path
import json,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/sharing_study'));args=p.parse_args();out=args.output
a=json.loads((out/'analysis.json').read_text());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(9,3.1),layout='constrained')
for ax,ds,title in zip(axs,['movielens','amazon'],['MovieLens','Amazon Musical Instruments']):
 for j,k in enumerate(['full','shared_dual']):
  r=next(r for r in a['comparisons'] if (r['dataset'],r['protocol'],r['left'])==(ds,'catalog',k))
  ax.errorbar(r['difference'],j,xerr=[[r['difference']-r['low']],[r['high']-r['difference']]],fmt='o',color=['#147D92','#AC5D22'][j],capsize=4)
 ax.axvline(0,color='#666',linestyle='--',lw=1);ax.set_yticks([0,1],['Original shared\nminus separate','Shared average\nminus separate'] if ds=='movielens' else []);ax.set_ylim(1.5,-.5);ax.set_title(title);ax.set_xlabel('Full-catalog NDCG@10 difference');ax.grid(axis='x',alpha=.2)
lo=min(ax.get_xlim()[0] for ax in axs);hi=max(ax.get_xlim()[1] for ax in axs)
for ax in axs:ax.set_xlim(lo,hi);ax.xaxis.set_major_locator(plt.MaxNLocator(4))
for ext in ['png','pdf']:fig.savefig(out/('figure3_sharing.'+ext),dpi=300,bbox_inches='tight')
plt.close(fig)
