"""Recompute ListNet KL by candidate-count stratum and normalize by log(C)."""
import argparse,json
from collections import defaultdict
import numpy as np
from scipy.special import logit,softmax
from src.arr.data import load_groups
from src.arr.utils import read_jsonl

def analyze(pred,data):
 rows=list(read_jsonl(pred)); groups=load_groups(data); out=defaultdict(list); i=0
 for g in groups:
  n=len(g.candidates); a=np.array([r['metadata']['head_scores'] for r in rows[i:i+n]],float); i+=n
  p=softmax(logit(np.clip(a,1e-6,1-1e-6)),axis=0); m=p.mean(1); hbar=-(m*np.log(np.clip(m,1e-12,None))).sum(); hm=-(p*np.log(np.clip(p,1e-12,None))).sum(0).mean(); js=float(hbar-hm); out[n].append(js)
 return {str(c):{'groups':len(v),'mean_kl':float(np.mean(v)),'mean_kl_over_log_c':float(np.mean(v)/np.log(c))} for c,v in sorted(out.items())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--id-pred',required=True); ap.add_argument('--id-data',required=True); ap.add_argument('--ood-pred',required=True); ap.add_argument('--ood-data',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
 result={'id':analyze(a.id_pred,a.id_data),'ood':analyze(a.ood_pred,a.ood_data)}; json.dump(result,open(a.out,'w'),indent=2); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
