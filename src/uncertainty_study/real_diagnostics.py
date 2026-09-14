"""R1/R2 evidence exclusively from actual held-out-question member predictions.

Input run directories must contain a completed manifest, selection.json and
checkpoint test predictions from the controlled study trainer.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from src.arr.losses import get_loss
from src.arr.utils import stable_hash
from .metrics import query_metrics, probability_metrics
from .train import save_json


def objective(scores, targets, loss, seed=2026):
    scores=torch.tensor(scores,dtype=torch.float64)
    targets=torch.tensor(np.tile(targets,(len(scores),1)),dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return float(get_loss(loss)(scores,targets,torch.ones_like(targets,dtype=torch.bool)))


def offset_audit(rows, loss, scales=(0.,1.,5.,20.), seed=2026):
    results=[]
    for r in rows:
        s=np.asarray(r['scores'],float); y=r['targets']; t=r.get('temperature',1.)
        if len(s)<2: raise ValueError('R1/R2 shared-head evidence requires multiple members')
        rng=np.random.default_rng(int(stable_hash(seed,r['group_id'],length=8),16))
        shifts=rng.normal(size=(len(s),1))
        base=query_metrics(s,y,t); p=softmax(s/t,axis=1)
        loss0=objective(s,y,loss)
        for scale in scales:
            transformed=s+scale*shifts
            metrics=query_metrics(transformed,y,t)
            value=objective(transformed,y,loss)
            results.append({'group_id':r['group_id'],'loss':loss,'offset_scale':scale,
                'candidate_count':len(y),'raw_variance':metrics['raw_variance'],
                'raw_width':metrics['raw_width'],'centered_variance':metrics['centered_variance'],
                'js':metrics['js'],'objective':value,'objective_before':loss0,
                'objective_absolute_change':abs(value-loss0),
                'objective_relative_change':abs(value-loss0)/max(abs(loss0),1e-12),
                'max_probability_change':float(np.abs(softmax(transformed/t,axis=1)-p).max()),
                'js_absolute_change':abs(metrics['js']-base['js']),
                'centered_variance_absolute_change':abs(metrics['centered_variance']-base['centered_variance']),
                'rankings_unchanged':bool(np.array_equal(np.argsort(s,axis=1),np.argsort(transformed,axis=1))),
                'regret_change':metrics['regret']-base['regret']})
    return results


def summarize_r1(rows):
    result=[]
    for scale in sorted(set(r['offset_scale'] for r in rows)):
        rs=[r for r in rows if r['offset_scale']==scale]
        result.append({'offset_scale':scale,'questions':len(rs),
            'changed_rankings':sum(not r['rankings_unchanged'] for r in rs),
            **{f'mean_{key}':float(np.mean([r[key] for r in rs])) for key in ('raw_variance','raw_width','centered_variance','js','objective')},
            **{f'max_{key}':max(r[key] for r in rs) for key in ('objective_absolute_change','objective_relative_change','max_probability_change','js_absolute_change','centered_variance_absolute_change')}})
    return result


def checkpoint_summary(rows, loss, step):
    metrics=[dict(r,**query_metrics(r['scores'],r['targets'],r.get('temperature',1.))) for r in rows]
    valid=[r for r in metrics if r['participation_ratio'] is not None]
    def summary(rs):
        return {'questions':len(rs),**{f'mean_{key}':float(np.mean([r[key] for r in rs])) for key in
            ('js','probability_width','covariance_trace','regret','ndcg_error')},
            'median_covariance_trace':float(np.median([r['covariance_trace'] for r in rs])),
            'median_participation_ratio':float(np.median([r['participation_ratio'] for r in rs if r['participation_ratio'] is not None])) if any(r['participation_ratio'] is not None for r in rs) else None,
            'mean_member_regret':float(np.mean([np.mean(r['member_regret']) for r in rs])),
            'mean_member_ndcg':float(np.mean([np.mean(r['member_ndcg']) for r in rs]))}
    out={'loss':loss,'step':step,**summary(metrics),'undefined_pr_questions':len(metrics)-len(valid),
         'by_candidate_count':{str(n):summary([r for r in metrics if r['candidate_count']==n]) for n in sorted(set(r['candidate_count'] for r in metrics))}}
    return out,metrics


def contraction(rows,loss):
    result=[]
    for r in rows:
        p=softmax(np.asarray(r['scores'])/r.get('temperature',1.),axis=1)
        mean=p.mean(0)
        for alpha in (1.,.5,.1,.01,.001):
            result.append({'group_id':r['group_id'],'loss':loss,'alpha':alpha,
                **probability_metrics(mean+alpha*(p-mean))})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+'); parser.add_argument('--output',required=True)
    args=parser.parse_args(); root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    colors={'listnet':'#2471a3','listmle':'#ca6f1e','mse':'#148f77'}
    r1_all=[]; checkpoint_all=[]; contractions=[]; selected_rows={}; provenance={}
    identity=None
    for path in args.runs:
        path=Path(path); manifest=json.loads((path/'manifest.json').read_text())
        if manifest['status']!='complete': raise ValueError('Completed training required')
        cfg=manifest['config']; loss=cfg['loss']
        if loss in selected_rows: raise ValueError('One run per loss; analyze seeds/folds separately')
        selection=json.loads((path/'selection.json').read_text()); selected=selection['step']
        rows=json.loads((path/f'test-{selected}.json').read_text())
        current={r['group_id']:(r['candidate_ids'],r['targets'],r['data_fingerprint']) for r in rows}
        if identity is not None and current!=identity: raise ValueError('Matched losses must use identical test questions/candidates')
        identity=current
        selected_rows[loss]=rows
        r1=offset_audit(rows,loss); r1_all+=r1
        contractions+=contraction(rows,loss)
        files=sorted(path.glob('test-*.json'),key=lambda p:int(p.stem.split('-')[1]))
        for file in files:
            step=int(file.stem.split('-')[1]); checkpoint=json.loads(file.read_text())
            if {r['group_id']:(r['candidate_ids'],r['targets'],r['data_fingerprint']) for r in checkpoint}!=identity:
                raise ValueError('Checkpoint test set changed')
            summary,metrics=checkpoint_summary(checkpoint,loss,step)
            checkpoint_all.append(summary)
            save_json(root/f'{loss}-queries-{step}.json',metrics)
        provenance[loss]={'run':str(path),'selected_step':selected,'training_seed':cfg['seed'],
            'fold':cfg['fold'],'base_model':cfg['base_model'],'updates':cfg['updates'],
            'data_hashes':cfg['data_hashes'],'resolved_revision':manifest['model'].get('resolved_revision'),
            'selected_prediction_hash':stable_hash(rows)}
    save_json(root/'provenance.json',provenance); save_json(root/'r1-query-audit.json',r1_all)
    summary={loss:summarize_r1([r for r in r1_all if r['loss']==loss]) for loss in selected_rows}
    save_json(root/'r1-summary.json',summary); save_json(root/'r2-checkpoints.json',checkpoint_all)
    save_json(root/'r2-real-prediction-contraction.json',contractions)

    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for loss,rs in summary.items():
        x=[r['offset_scale'] for r in rs]
        axes[0].plot(x,[r['mean_raw_variance'] for r in rs],marker='o',label=loss,color=colors[loss])
        axes[1].plot(x,[r['mean_js'] for r in rs],marker='o',label=loss,color=colors[loss])
        axes[2].plot(x,[r['mean_objective']/rs[0]['mean_objective'] for r in rs],marker='o',label=loss,color=colors[loss])
    for ax in axes: ax.set(xlabel='Member-specific offset scale'); ax.legend(frameon=False)
    axes[0].set(yscale='log',ylabel='Mean raw-score variance')
    axes[1].set(yscale='log',ylabel='Mean query JS disagreement')
    axes[2].set(yscale='log',ylabel='Objective / original objective')
    fig.suptitle('R1 · Trained DeBERTa rankers on held-out DS-Critique questions')
    fig.savefig(root/'r1-real-data.png',dpi=200); fig.savefig(root/'r1-real-data.pdf'); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for loss,rows in selected_rows.items():
        metrics=[query_metrics(r['scores'],r['targets'],r.get('temperature',1.)) for r in rows]
        valid=[r for r in metrics if r['participation_ratio'] is not None]
        axes[0].scatter([r['covariance_trace'] for r in valid],[r['participation_ratio'] for r in valid],s=14,alpha=.5,label=loss,color=colors[loss])
        checkpoints=[r for r in checkpoint_all if r['loss']==loss]
        axes[1].plot([r['median_covariance_trace'] for r in checkpoints],[r['median_participation_ratio'] for r in checkpoints],marker='o',label=loss,color=colors[loss])
        for r in checkpoints:
            axes[1].annotate(str(r['step']),(r['median_covariance_trace'],r['median_participation_ratio']),fontsize=7,xytext=(3,3),textcoords='offset points',color=colors[loss])
        cs=[r for r in contractions if r['loss']==loss]
        points=[]
        for a in (1.,.5,.1,.01,.001):
            block=[r for r in cs if r['alpha']==a and r['participation_ratio'] is not None]
            points.append((np.median([r['covariance_trace'] for r in block]),np.median([r['participation_ratio'] for r in block])))
        axes[2].plot(*np.asarray(points).T,marker='o',label=loss,color=colors[loss])
    for ax in axes:
        ax.set(xscale='log',ylim=(0,4.2),xlabel='Probability covariance trace',ylabel='Participation ratio'); ax.legend(frameon=False,fontsize=8)
    axes[0].set_title('Held-out queries · selected checkpoint')
    axes[1].set_title('Actual training · checkpoint medians')
    axes[2].set_title('Contraction of real predictions')
    fig.suptitle('R2 · Magnitude and dimensionality are separate measurements')
    fig.savefig(root/'r2-real-data.png',dpi=200); fig.savefig(root/'r2-real-data.pdf'); plt.close(fig)
    print(json.dumps({'r1':summary,'checkpoints':checkpoint_all},indent=2))

if __name__=='__main__': main()
