import numpy as np,torch,core,study
D=torch.load(study.ROOT/'data/amazon/data.pt',weights_only=False);users=np.array([u for u,s in enumerate(D['seq']) if len(s)-2<=10][:3]);ends=[len(D['seq'][u])-2 for u in users];x,t=core.pack(D['seq'],D['times'],users,ends);length=max(ends);result={}
for kind in ['full','no_graph','no_temporal','no_text','no_candidate']:
 m=study.model(D,kind,907).eval()
 with torch.no_grad():
  g,v=m.items();a=m.encode(x,t,g);b=m.encode(x[:,:length],t[:,:length],g);error=float((a-b).abs().max());assert error<2e-5;result[kind]=error
core.dump(study.ROOT/'padding_checks.json',result);print(result)
