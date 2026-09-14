"""Create report-ready ARR epistemic figures from Modal summary JSON files."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.18,'axes.axisbelow':True})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shared',required=True); ap.add_argument('--independent',required=True); ap.add_argument('--out',default='arr_figures'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    shared=json.load(open(a.shared)); ind=json.load(open(a.independent))
    rows=[]
    for loss, arms in shared.items():
        for arm,m in arms.items():
            if arm=='mc_dropout_8': continue
            rows.append((f'shared {loss}\n{arm}',m['in_domain']['effective_members'],m['in_domain'].get('mean_kl_to_consensus',np.nan),m['cross_domain_esnli'].get('mean_kl_to_consensus',np.nan),m['ndcg5'],m['cross_domain_esnli'].get('partial_width_error_given_confidence',np.nan)))
    rows.append(('independent ListNet',ind['in_domain']['effective_members'],np.nan,np.nan,ind['in_domain']['ndcg5'],ind['cross_domain_esnli'].get('partial_width_error_given_confidence',np.nan)))
    # PR versus KL
    num=lambda v: np.nan if v is None else float(v)
    x=np.array([num(r[1]) for r in rows],dtype=float); y=np.array([num(r[2]) for r in rows],dtype=float); labels=[r[0] for r in rows]
    fig,ax=plt.subplots(figsize=(8.5,5.2)); mask=np.isfinite(y); ax.scatter(x[mask],y[mask],s=68,c=['#4878a8' if 'shared' in labels[i] else '#c44e52' for i in np.where(mask)[0]],edgecolor='white',linewidth=.7)
    for i in np.where(mask)[0]: ax.annotate(labels[i],(x[i],y[i]),fontsize=7,xytext=(4,4),textcoords='offset points')
    ax.set_xlabel('Effective members (directional participation)'); ax.set_ylabel('Mean KL-to-consensus (nats, log scale)'); ax.set_yscale('log'); ax.set_title('Participation and disagreement magnitude'); fig.tight_layout(); fig.savefig(out/'pr_vs_kl.png',dpi=300); plt.close(fig)
    # Heatmap of shared arms
    names=[]; vals=[]
    for loss,arms in shared.items():
      for arm,m in arms.items():
       if arm=='mc_dropout_8': continue
       names.append(f'{loss}:{arm}'); vals.append([m['ndcg5'],m['in_domain']['effective_members'],m['in_domain'].get('mean_kl_to_consensus',np.nan),m['in_domain']['width_trend_spearman'],m['cross_domain_esnli'].get('partial_width_error_given_confidence',np.nan)])
    A=np.asarray([[num(v) for v in row] for row in vals],float); fig,ax=plt.subplots(figsize=(10,6.5)); im=ax.imshow(A,aspect='auto',cmap='RdBu_r'); ax.set_yticks(range(len(names)),names,fontsize=8); ax.set_xticks(range(5),['NDCG@5','PR ID','KL ID','D3 rho','OOD partial rho'],rotation=25,ha='right');
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        if np.isfinite(A[i,j]): ax.text(j,i,f'{A[i,j]:.3g}',ha='center',va='center',fontsize=7)
    ax.set_title('Shared-backbone ablation matrix'); fig.colorbar(im,ax=ax,shrink=.8,label='metric value'); fig.tight_layout(); fig.savefig(out/'ablation_heatmap.png',dpi=300); plt.close(fig)
    # OOD partial rho and NDCG
    fig,ax=plt.subplots(figsize=(8,5)); rho=np.array([num(r[5]) for r in rows]); nd=np.array([num(r[4]) for r in rows]); mask=np.isfinite(rho); ax.scatter(nd[mask], rho[mask],s=55)
    for i in np.where(mask)[0]: ax.annotate(labels[i],(nd[i],rho[i]),fontsize=7,xytext=(4,4),textcoords='offset points')
    ax.axhline(0,color='#333',lw=.9); ax.set_xlabel('NDCG@5'); ax.set_ylabel('OOD partial Spearman (controlled for confidence)'); ax.set_title('Ranking quality versus OOD association'); fig.tight_layout(); fig.savefig(out/'ood_partial_vs_ndcg.png',dpi=300); plt.close(fig)
    # KL bars by arm
    valid=[r for r in rows if np.isfinite(num(r[3]))]; names=[r[0] for r in valid]; kl=[num(r[3]) for r in valid]
    order=np.argsort(kl); names=[names[i] for i in order]; kl=[kl[i] for i in order]; fig,ax=plt.subplots(figsize=(9,6)); ax.barh(range(len(names)),kl,color='#4c78a8'); ax.set_xscale('log'); ax.set_yticks(range(len(names)),names,fontsize=8); ax.set_xlabel('Mean KL-to-consensus (nats, log scale)'); ax.set_title('OOD disagreement magnitude by arm'); fig.tight_layout(); fig.savefig(out/'ood_kl_by_arm.png',dpi=300); plt.close(fig)
    print(f'Wrote figures to {out}')
if __name__=='__main__': main()
