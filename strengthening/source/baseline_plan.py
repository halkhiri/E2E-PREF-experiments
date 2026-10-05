"""Standard static BPR positive/negative sampling, training partition only."""
import numpy as np
import core

def bpr_plan(data,seed,epochs=50):
 rng=np.random.default_rng(seed+1009);train=[s[:-2] for s in data['seq']];plan=[]
 for _ in range(epochs):
  users=rng.permutation(len(train));targets=[int(train[u][rng.integers(len(train[u]))]) for u in users]
  cs=np.stack([core.candidates(train[u],target,len(data['item_ids']),31,rng) for u,target in zip(users,targets)])
  # BPR uses user/item IDs, not packed history. A valid dummy prefix avoids empty arrays.
  plan.append((users,np.ones(len(users),dtype=np.int64),cs))
 return plan
