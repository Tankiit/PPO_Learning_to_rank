"""Cleaner, narrative ARR figures from shared/independent summaries."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'axes.axisbelow':True})

def main():
 p=argparse.ArgumentParser(); p.add_argument('--shared',required=True); p.add_argument('--independent',required=True); p.add_argument('--out',default='arr_figures'); a=p.parse_args(); out=Path(a.out); out.mkdir(exist_ok=True)
 s=json.load(open(a.shared))['listnet']; ind=json.load(open(a.independent));
 arms=[('Independent',ind),('Shared baseline',s['baseline']),('Shared bootstrap',s['bootstrap']),('Shared masks',s['features']),('Shared boot+mask',s['bootstrap_features']),('Shared λ=.01',s['lambda_0p01']),('Shared λ=.1',s['lambda_0p1']),('Shared λ=1',s['lambda_1'])]
 names=[x[0] for x in arms]; colors=['#c44e52']+['#4c78a8']*7
 def get(m,scope,key): return m[scope].get(key,np.nan)
 # Overview: one coherent figure
 fig,axs=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'hspace':.38,'wspace':.28})
 # A: KL ID/OOD dumbbell
 ax=axs[0,0]; y=np.arange(len(names)); kid=np.array([get(m,'in_domain','mean_kl_to_consensus') for _,m in arms],float); koo=np.array([get(m,'cross_domain_esnli','mean_kl_to_consensus') for _,m in arms],float); ax.plot([kid,koo], [y,y],color='#bdbdbd',lw=1); ax.scatter(kid,y,s=35,label='ID',color='#4c78a8',zorder=3); ax.scatter(koo,y,s=35,label='e-SNLI',color='#dd8452',zorder=3); ax.set_xscale('log'); ax.set_yticks(y,names); ax.invert_yaxis(); ax.set_xlabel('Mean KL-to-consensus (nats)'); ax.set_title('A  Disagreement magnitude'); ax.legend(frameon=False,ncol=2,loc='lower right')
 # B: NDCG ranked bars
 ax=axs[0,1]; nd=np.array([m.get('ndcg5',m['in_domain'].get('ndcg5',np.nan)) for _,m in arms]); order=np.argsort(nd); ax.barh(np.arange(len(names)),nd[order],color=np.array(colors)[order]); ax.set_yticks(np.arange(len(names)),np.array(names)[order]); ax.set_xlim(nd.min()-.008,nd.max()+.003); ax.set_xlabel('NDCG@5'); ax.set_title('B  Ranking quality');
 # C: OOD association lollipop
 ax=axs[1,0]; rho=np.array([get(m,'cross_domain_esnli','partial_width_error_given_confidence') for _,m in arms],float); order=np.argsort(rho); ax.hlines(np.arange(len(names)),0,rho[order],color='#bdbdbd'); ax.scatter(rho[order],np.arange(len(names)),color=np.array(colors)[order],s=42); ax.axvline(0,color='#333',lw=.8); ax.set_yticks(np.arange(len(names)),np.array(names)[order]); ax.set_xlabel('Partial Spearman ρ (error/width | confidence)'); ax.set_title('C  OOD association');
 # D: PR versus KL normalized
 ax=axs[1,1]; pr=np.array([get(m,'in_domain','effective_members') for _,m in arms]); kl=np.array([get(m,'in_domain','mean_kl_to_consensus') for _,m in arms]); ax.scatter(pr,kl,color=colors,s=48); ax.set_yscale('log'); ax.set_xlabel('Effective members'); ax.set_ylabel('Mean KL (nats, log)'); ax.set_title('D  Direction versus magnitude');
 for i,n in enumerate(names):
  if n in ('Independent','Shared baseline'): ax.annotate(n,(pr[i],kl[i]),xytext=(5,5),textcoords='offset points',fontsize=8)
 fig.suptitle('ARR epistemic ensemble comparison',fontsize=16,y=.98); fig.savefig(out/'arr_overview.png',dpi=320,bbox_inches='tight'); plt.close(fig)
 # Dedicated control plot: 98% shrink conceptual evidence
 fig,ax=plt.subplots(figsize=(7,4.5)); scale=np.array([1,.02]); prc=np.array([4.0,4.0]); klc=np.array([1.0,.0004]); ax.plot(scale,prc,'o-',label='Participation ratio',color='#4c78a8'); ax2=ax.twinx(); ax2.plot(scale,klc,'s--',label='KL-to-consensus',color='#c44e52'); ax.set_xscale('log'); ax2.set_yscale('log'); ax.set_xlabel('Relative disagreement scale'); ax.set_ylabel('Participation ratio (unchanged)'); ax2.set_ylabel('Relative KL (log scale)'); ax.set_title('98% collapse control: directional vs magnitude diversity'); ax.set_xticks([.02,1],['2%','100%']); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(out/'shrink_control.png',dpi=320); plt.close(fig)
 print('Wrote',out/'arr_overview.png','and',out/'shrink_control.png')
if __name__=='__main__': main()
