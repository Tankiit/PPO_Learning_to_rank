"""Evidence figures from actual saved checkpoints/paired exposure comparisons."""
import json
from pathlib import Path
import numpy as np
from .metrics import query_metrics, probability_metrics


def trajectories(paths, output):
    import matplotlib.pyplot as plt
    from scipy.special import softmax
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    means=[]; reference=None; identities=None
    for path in paths:
        rows=json.loads(Path(path).read_text())
        current={r['group_id']:r['candidate_ids'] for r in rows}
        if identities is not None and current!=identities:
            raise ValueError('Checkpoint trajectory must use fixed test questions/candidates')
        identities=current
        measured=[query_metrics(r['scores'],r['targets'],r.get('temperature',1.)) for r in rows]
        valid=[r for r in measured if r['participation_ratio'] is not None]
        if not valid: continue
        x=np.array([r['covariance_trace'] for r in valid]); y=np.array([r['participation_ratio'] for r in valid])
        axes[0].scatter(x,y,s=8,alpha=.25,label=Path(path).stem)
        means.append((np.median(x),np.median(y)))
        if reference is None: reference=rows[0]
    if means:
        axes[0].plot(*np.asarray(means).T,color='black',marker='o',label='Checkpoint medians')
        p=softmax(np.asarray(reference['scores'])/reference.get('temperature',1.),axis=1)
        diagnostic=[probability_metrics(p.mean(0)+a*(p-p.mean(0))) for a in np.geomspace(.001,1,50)]
        axes[1].plot([r['covariance_trace'] for r in diagnostic],[r['participation_ratio'] for r in diagnostic])
    for ax in axes:
        ax.set(xscale='log',xlabel='Probability covariance trace',ylabel='Participation ratio',ylim=(0,max(4,max((p[1] for p in means),default=4)+.5)))
    axes[0].set_title('Actual training checkpoints'); axes[0].legend(fontsize=7)
    axes[1].set_title('Controlled contraction of first query')
    fig.savefig(output,dpi=180); plt.close(fig)


def exposure_figure(paths, output):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,path in enumerate(paths):
        result=json.loads(Path(path).read_text())
        label=Path(path).stem
        for key,marker in [('target','o'),('other','s')]:
            r=result[key]
            axes[0].plot([0,1],[r['before_mean'],r['after_mean']],marker=marker,label=f'{label}: {key}')
        r=result['target_specific_reduction']; lo,hi=r['ci95']; point=r['difference']
        axes[1].plot([lo,hi],[i,i],color='black')
        axes[1].scatter([point],[i],color='black')
    axes[0].set(xticks=[0,1],xticklabels=['Absent','Exposed'],ylabel='Mean candidate probability variance')
    axes[0].legend(fontsize=7)
    axes[1].axvline(0,color='gray',linewidth=.8)
    axes[1].set(yticks=range(len(paths)),yticklabels=[Path(p).stem for p in paths],xlabel='Target-specific reduction (95% question CI)')
    fig.savefig(output,dpi=180); plt.close(fig)
